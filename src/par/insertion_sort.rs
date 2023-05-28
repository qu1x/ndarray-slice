//! Derivative work of [`core::slice::sort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice::sort`]: https://doc.rust-lang.org/src/core/slice/sort.rs.html

use crate::insertion_sort::InsertionHole;
use core::{mem, ptr};
use ndarray::{s, ArrayViewMut1, IndexLonger};

/// Inserts `v[v.len() - 1]` into pre-sorted sequence `v[..v.len() - 1]` so that whole `v[..]`
/// becomes sorted.
unsafe fn insert_tail<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &F)
where
	F: Fn(&T, &T) -> bool,
{
	debug_assert!(v.len() >= 2);

	let i = v.len() - 1;

	// SAFETY: caller must ensure v is at least len 2.
	unsafe {
		let w = v.view();
		let w = w.raw_view().deref_into_view();
		let mut v = v.raw_view_mut().deref_into_view_mut();
		// See insert_head which talks about why this approach is beneficial.

		// It's important that we use i_ptr here. If this check is positive and we continue,
		// We want to make sure that no other copy of the value was seen by is_less.
		// Otherwise we would have to copy it back.
		//if is_less(&*i_ptr, &*i_ptr.sub(1)) {
		if is_less(w.uget(i), w.uget(i - 1)) {
			// It's important, that we use tmp for comparison from now on. As it is the value that
			// will be copied back. And notionally we could have created a divergence if we copy
			// back the wrong value.
			let tmp = mem::ManuallyDrop::new(ptr::read(v.view_mut().uget(i)));
			// Intermediate state of the insertion process is always tracked by `hole`, which
			// serves two purposes:
			// 1. Protects integrity of `v` from panics in `is_less`.
			// 2. Fills the remaining hole in `v` in the end.
			//
			// Panic safety:
			//
			// If `is_less` panics at any point during the process, `hole` will get dropped and
			// fill the hole in `v` with `tmp`, thus ensuring that `v` still holds every object it
			// initially held exactly once.
			let mut hole = InsertionHole::new(&*tmp, v.view_mut().uget(i - 1));
			ptr::copy_nonoverlapping(hole.dest, v.view_mut().uget(i), 1);

			// SAFETY: We know i is at least 1.
			for j in (0..(i - 1)).rev() {
				let j_ptr = v.view_mut().uget(j);
				if !is_less(&*tmp, &*j_ptr) {
					break;
				}

				ptr::copy_nonoverlapping(j_ptr, hole.dest, 1);
				hole.dest = j_ptr;
			}
			// `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
		}
	}
}

/// Inserts `v[0]` into pre-sorted sequence `v[1..]` so that whole `v[..]` becomes sorted.
///
/// This is the integral subroutine of insertion sort.
unsafe fn insert_head<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &F)
where
	F: Fn(&T, &T) -> bool,
{
	debug_assert!(v.len() >= 2);

	// SAFETY: caller must ensure v is at least len 2.
	unsafe {
		let w = v.view();
		let w = w.raw_view().deref_into_view();
		if is_less(w.uget(1), w.uget(0)) {
			let mut v = v.raw_view_mut().deref_into_view_mut();
			//let arr_ptr = v.as_mut_ptr();

			// There are three ways to implement insertion here:
			//
			// 1. Swap adjacent elements until the first one gets to its final destination.
			//	However, this way we copy data around more than is necessary. If elements are big
			//	structures (costly to copy), this method will be slow.
			//
			// 2. Iterate until the right place for the first element is found. Then shift the
			//	elements succeeding it to make room for it and finally place it into the
			//	remaining hole. This is a good method.
			//
			// 3. Copy the first element into a temporary variable. Iterate until the right place
			//	for it is found. As we go along, copy every traversed element into the slot
			//	preceding it. Finally, copy data from the temporary variable into the remaining
			//	hole. This method is very good. Benchmarks demonstrated slightly better
			//	performance than with the 2nd method.
			//
			// All methods were benchmarked, and the 3rd showed best results. So we chose that one.
			let tmp = mem::ManuallyDrop::new(ptr::read(v.view_mut().uget(0)));

			// Intermediate state of the insertion process is always tracked by `hole`, which
			// serves two purposes:
			// 1. Protects integrity of `v` from panics in `is_less`.
			// 2. Fills the remaining hole in `v` in the end.
			//
			// Panic safety:
			//
			// If `is_less` panics at any point during the process, `hole` will get dropped and
			// fill the hole in `v` with `tmp`, thus ensuring that `v` still holds every object it
			// initially held exactly once.
			let dest = v.view_mut().uget(1);
			let mut hole = InsertionHole::new(&*tmp, dest);
			ptr::copy_nonoverlapping(dest, v.view_mut().uget(0), 1);

			for i in 2..v.len() {
				if !is_less(w.uget(i), &*tmp) {
					break;
				}
				//ptr::copy_nonoverlapping(arr_ptr.add(i), arr_ptr.add(i - 1), 1);
				ptr::copy_nonoverlapping(w.uget(i), v.view_mut().uget(i - 1), 1);
				hole.dest = v.view_mut().uget(i) as *mut T;
			}
			// `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
		}
	}
}

/// Sort `v` assuming `v[..offset]` is already sorted.
///
/// Never inline this function to avoid code bloat. It still optimizes nicely and has practically no
/// performance impact. Even improving performance in some cases.
#[inline(never)]
pub(super) fn insertion_sort_shift_left<T, F>(
	mut v: ArrayViewMut1<'_, T>,
	offset: usize,
	is_less: &F,
) where
	F: Fn(&T, &T) -> bool,
{
	let len = v.len();

	// Using assert here improves performance.
	assert!(offset != 0 && offset <= len);

	// Shift each element of the unsorted region v[i..] as far left as is needed to make v sorted.
	for i in offset..len {
		// SAFETY: we tested that `offset` must be at least 1, so this loop is only entered if len
		// >= 2. The range is exclusive and we know `i` must be at least 1 so this slice has at
		// >least len 2.
		unsafe {
			insert_tail(v.slice_mut(s![..=i]), is_less);
		}
	}
}

/// Sort `v` assuming `v[offset..]` is already sorted.
///
/// Never inline this function to avoid code bloat. It still optimizes nicely and has practically no
/// performance impact. Even improving performance in some cases.
#[inline(never)]
pub(super) fn insertion_sort_shift_right<T, F>(
	mut v: ArrayViewMut1<'_, T>,
	offset: usize,
	is_less: &F,
) where
	F: Fn(&T, &T) -> bool,
{
	let len = v.len();

	// Using assert here improves performance.
	assert!(offset != 0 && offset <= len && len >= 2);

	// Shift each element of the unsorted region v[..i] as far left as is needed to make v sorted.
	for i in (0..offset).rev() {
		// SAFETY: we tested that `offset` must be at least 1, so this loop is only entered if len
		// >= 2.We ensured that the slice length is always at least 2 long. We know that start_found
		// will be at least one less than end, and the range is exclusive. Which gives us i always
		// <= (end - 2).
		unsafe {
			insert_head(v.slice_mut(s![i..len]), is_less);
		}
	}
}

/// Partially sorts a slice by shifting several out-of-order elements around.
///
/// Returns `true` if the slice is sorted at the end. This function is *O*(*n*) worst-case.
#[cold]
pub fn partial_insertion_sort<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &F) -> bool
where
	F: Fn(&T, &T) -> bool,
{
	// Maximum number of adjacent out-of-order pairs that will get shifted.
	const MAX_STEPS: usize = 5;
	// If the slice is shorter than this, don't shift any elements.
	const SHORTEST_SHIFTING: usize = 50;

	let len = v.len();
	let mut i = 1;

	for _ in 0..MAX_STEPS {
		// SAFETY: We already explicitly did the bound checking with `i < len`.
		// All our subsequent indexing is only in the range `0 <= index < len`
		unsafe {
			let v = v.view();
			// Find the next pair of adjacent out-of-order elements.
			while i < len && !is_less(v.uget(i), v.uget(i - 1)) {
				i += 1;
			}
		}

		// Are we done?
		if i == len {
			return true;
		}

		// Don't shift elements on short arrays, that has a performance cost.
		if len < SHORTEST_SHIFTING {
			return false;
		}

		// Swap the found pair of elements. This puts them in correct order.
		v.swap(i - 1, i);

		if i >= 2 {
			// Shift the smaller element to the left.
			insertion_sort_shift_left(v.slice_mut(s![..i]), i - 1, is_less);

			// Shift the greater element to the right.
			insertion_sort_shift_right(v.slice_mut(s![..i]), 1, is_less);
		}
	}

	// Didn't manage to sort the slice in the limited number of steps.
	false
}

#[cfg(test)]
mod test {
	use super::insertion_sort_shift_left;
	use ndarray::Array1;
	use quickcheck_macros::quickcheck;

	#[quickcheck]
	fn sorted(xs: Vec<u32>) {
		let mut array = Array1::from_vec(xs);
		if !array.is_empty() {
			insertion_sort_shift_left(array.view_mut(), 1, &mut u32::lt);
		}
		for i in 1..array.len() {
			assert!(array[i - 1] <= array[i]);
		}
	}
}
