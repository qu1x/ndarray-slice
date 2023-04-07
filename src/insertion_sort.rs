//! Derivative work of [`core::slice::sort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice::sort`]: https://doc.rust-lang.org/src/core/slice/sort.rs.html

use crate::partition::CopyOnDrop;
use core::{mem, mem::ManuallyDrop, ptr};
use ndarray::{s, ArrayViewMut1, IndexLonger};

/// Sorts a slice using insertion sort, which is *O*(*n*^2) worst-case.
pub fn insertion_sort<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &mut F)
where
	F: FnMut(&T, &T) -> bool,
{
	for i in 1..v.len() {
		shift_tail(v.slice_mut(s![..i + 1]), is_less);
	}
}

/// Partially sorts a slice by shifting several out-of-order elements around.
///
/// Returns `true` if the slice is sorted at the end. This function is *O*(*n*) worst-case.
#[cold]
pub fn partial_insertion_sort<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &mut F) -> bool
where
	F: FnMut(&T, &T) -> bool,
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

		// Shift the smaller element to the left.
		shift_tail(v.slice_mut(s![..i]), is_less);
		// Shift the greater element to the right.
		shift_head(v.slice_mut(s![i..]), is_less);
	}

	// Didn't manage to sort the slice in the limited number of steps.
	false
}

/// Shifts the first element to the right until it encounters a greater or equal element.
fn shift_head<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &mut F)
where
	F: FnMut(&T, &T) -> bool,
{
	let len = v.len();
	// SAFETY: The unsafe operations below involves indexing without a bounds check (by offsetting a
	// pointer) and copying memory (`ptr::copy_nonoverlapping`).
	//
	// a. Indexing:
	//  1. We checked the size of the array to >=2.
	//  2. All the indexing that we will do is always between {0 <= index < len} at most.
	//
	// b. Memory copying
	//  1. We are obtaining pointers to references which are guaranteed to be valid.
	//  2. They cannot overlap because we obtain pointers to difference indices of the slice.
	//     Namely, `i` and `i-1`.
	//  3. If the slice is properly aligned, the elements are properly aligned.
	//     It is the caller's responsibility to make sure the slice is properly aligned.
	//
	// See comments below for further detail.
	unsafe {
		let w = v.view();
		// If the first two elements are out-of-order...
		if len >= 2 && is_less(w.uget(1), w.uget(0)) {
			// Read the first element into a stack-allocated variable. If a following comparison
			// operation panics, `hole` will get dropped and automatically write the element back
			// into the slice.
			let tmp = mem::ManuallyDrop::new(ptr::read(w.uget(0)));
			let src = v.view().index(1) as *const T;
			let dst = v.view_mut().index(0) as *mut T;
			let mut hole = CopyOnDrop {
				src: &*tmp,
				dest: src as *mut T,
			};
			ptr::copy_nonoverlapping(src, dst, 1);

			for i in 2..len {
				let w = v.view();
				if !is_less(w.uget(i), &*tmp) {
					break;
				}

				// Move `i`-th element one place to the left, thus shifting the hole to the right.
				ptr::copy_nonoverlapping(w.uget(i), v.view_mut().uget(i - 1), 1);
				hole.dest = v.view_mut().uget(i) as *mut T;
			}
			// `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
		}
	}
}

/// Shifts the last element to the left until it encounters a smaller or equal element.
fn shift_tail<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &mut F)
where
	F: FnMut(&T, &T) -> bool,
{
	let len = v.len();
	// SAFETY: The unsafe operations below involves indexing without a bound check (by offsetting a
	// pointer) and copying memory (`ptr::copy_nonoverlapping`).
	//
	// a. Indexing:
	//  1. We checked the size of the array to >= 2.
	//  2. All the indexing that we will do is always between `0 <= index < len-1` at most.
	//
	// b. Memory copying
	//  1. We are obtaining pointers to references which are guaranteed to be valid.
	//  2. They cannot overlap because we obtain pointers to difference indices of the slice.
	//     Namely, `i` and `i+1`.
	//  3. If the slice is properly aligned, the elements are properly aligned.
	//     It is the caller's responsibility to make sure the slice is properly aligned.
	//
	// See comments below for further detail.
	unsafe {
		// If the last two elements are out-of-order...
		if len >= 2 {
			let w = v.view();
			if is_less(w.uget(len - 1), w.uget(len - 2)) {
				// Read the last element into a stack-allocated variable. If a following comparison
				// operation panics, `hole` will get dropped and automatically write the element back
				// into the slice.
				let tmp = ManuallyDrop::new(ptr::read(w.uget(len - 1)));
				let mut hole = CopyOnDrop {
					src: &*tmp,
					dest: v.view_mut().index(len - 2),
				};
				let src = v.view().index(len - 2) as *const T;
				let dst = v.view_mut().index(len - 1) as *mut T;
				ptr::copy_nonoverlapping(src, dst, 1);

				for i in (0..len - 2).rev() {
					let src = v.view_mut().index(i) as *mut T;
					if !is_less(&*tmp, &*src) {
						break;
					}

					// Move `i`-th element one place to the right, thus shifting the hole to the left.
					let dst = v.view_mut().index(i + 1) as *mut T;
					ptr::copy_nonoverlapping(src, dst, 1);
					hole.dest = src;
				}
				// `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
			}
		}
	}
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
	use super::insertion_sort;
	use ndarray::Array1;
	use quickcheck_macros::quickcheck;

	#[quickcheck]
	fn sorted(xs: Vec<u32>) {
		let mut array = Array1::from_vec(xs);
		insertion_sort(array.view_mut(), &mut u32::lt);
		for i in 1..array.len() {
			assert!(array[i - 1] <= array[i]);
		}
	}
}
