//! Derivative work of [`core::slice::sort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice::sort`]: https://doc.rust-lang.org/src/core/slice/sort.rs.html

use crate::partition::reverse;
use core::{mem, ptr};
use ndarray::{s, ArrayViewMut1, IndexLonger};

/// Inserts `v[0]` into pre-sorted sequence `v[1..]` so that whole `v[..]` becomes sorted.
///
/// This is the integral subroutine of insertion sort.
fn insert_head<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &mut F)
where
	F: FnMut(&T, &T) -> bool,
{
	if v.len() >= 2 && is_less(&v[1], &v[0]) {
		// SAFETY: Copy tmp back even if panic, and ensure unique observation.
		unsafe {
			// There are three ways to implement insertion here:
			//
			// 1. Swap adjacent elements until the first one gets to its final destination.
			//    However, this way we copy data around more than is necessary. If elements are big
			//    structures (costly to copy), this method will be slow.
			//
			// 2. Iterate until the right place for the first element is found. Then shift the
			//    elements succeeding it to make room for it and finally place it into the
			//    remaining hole. This is a good method.
			//
			// 3. Copy the first element into a temporary variable. Iterate until the right place
			//    for it is found. As we go along, copy every traversed element into the slot
			//    preceding it. Finally, copy data from the temporary variable into the remaining
			//    hole. This method is very good. Benchmarks demonstrated slightly better
			//    performance than with the 2nd method.
			//
			// All methods were benchmarked, and the 3rd showed best results. So we chose that one.
			let tmp = mem::ManuallyDrop::new(ptr::read(&v[0]));

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
			let mut hole = InsertionHole {
				src: &*tmp,
				dest: &mut v[1],
			};
			ptr::copy_nonoverlapping(&v[1], &mut v[0], 1);

			for i in 2..v.len() {
				if !is_less(&v[i], &*tmp) {
					break;
				}
				ptr::copy_nonoverlapping(&v[i], &mut v[i - 1], 1);
				hole.dest = &mut v[i];
			}
			// `hole` gets dropped and thus copies `tmp` into the remaining hole in `v`.
		}
	}

	// When dropped, copies from `src` into `dest`.
	struct InsertionHole<T> {
		src: *const T,
		dest: *mut T,
	}

	impl<T> Drop for InsertionHole<T> {
		fn drop(&mut self) {
			// SAFETY: The caller must ensure that src and dest are correctly set.
			unsafe {
				ptr::copy_nonoverlapping(self.src, self.dest, 1);
			}
		}
	}
}

/// Merges non-decreasing runs `v[..mid]` and `v[mid..]` using `buf` as temporary storage, and
/// stores the result into `v[..]`.
///
/// # Safety
///
/// The two slices must be non-empty and `mid` must be in bounds. Buffer `buf` must be long enough
/// to hold a copy of the shorter slice. Also, `T` must not be a zero-sized type.
#[warn(unsafe_op_in_unsafe_fn)]
unsafe fn merge<T, F>(v: ArrayViewMut1<'_, T>, mid: usize, buf: *mut T, is_less: &mut F)
where
	F: FnMut(&T, &T) -> bool,
{
	let len = v.len();
	//let v = 0;//v.as_mut_ptr();

	// SAFETY: mid and len must be in-bounds of v.
	//let (v_mid, v_end) = (mid, len);//unsafe { (v.add(mid), v.add(len)) };

	// The merge process first copies the shorter run into `buf`. Then it traces the newly copied
	// run and the longer run forwards (or backwards), comparing their next unconsumed elements and
	// copying the lesser (or greater) one into `v`.
	//
	// As soon as the shorter run is fully consumed, the process is done. If the longer run gets
	// consumed first, then we must copy whatever is left of the shorter run into the remaining
	// hole in `v`.
	//
	// Intermediate state of the process is always tracked by `hole`, which serves two purposes:
	// 1. Protects integrity of `v` from panics in `is_less`.
	// 2. Fills the remaining hole in `v` if the longer run gets consumed first.
	//
	// Panic safety:
	//
	// If `is_less` panics at any point during the process, `hole` will get dropped and fill the
	// hole in `v` with the unconsumed range in `buf`, thus ensuring that `v` still holds every
	// object it initially held exactly once.
	let mut hole;

	if mid <= len - mid {
		// The left run is shorter.

		//let src = v.view_mut().index(0);
		// SAFETY: buf must have enough capacity for `v[..mid]`.
		unsafe {
			for i in 0..mid {
				ptr::copy_nonoverlapping(&v[i], buf.add(i), 1);
			}
			hole = MergeHole {
				buf,
				start: 0,
				end: mid,
				dest: 0,
				v,
			};
		}

		// Initially, these pointers point to the beginnings of their arrays.
		//let left = &mut hole.start;
		let mut right = mid; //v_mid
					 //let out = &mut hole.dest;

		while hole.start < hole.end && right < len {
			// Consume the lesser side.
			// If equal, prefer the left run to maintain stability.

			// SAFETY: left and right must be valid and part of v same for out.
			unsafe {
				let w = hole.v.view();
				let to_copy = if is_less(w.uget(right), &*hole.buf.add(hole.start)) {
					let idx = &mut right;
					let old = hole.v.view_mut().index(*idx);

					// SAFETY: ptr.add(1) must still be a valid pointer and part of `v`.
					*idx += 1; //unsafe { ptr.add(1) };
					old
				} else {
					let idx = &mut hole.start;
					let old = hole.buf.add(*idx);

					// SAFETY: ptr.add(1) must still be a valid pointer and part of `v`.
					*idx += 1; //unsafe { ptr.add(1) };
					old
				};
				let idx = &mut hole.dest;
				let old = hole.v.view_mut().index(*idx);

				// SAFETY: ptr.add(1) must still be a valid pointer and part of `v`.
				*idx += 1; //unsafe { ptr.add(1) };
				let dst = old;
				ptr::copy_nonoverlapping(to_copy, dst, 1);
			}
		}
	} else {
		// The right run is shorter.

		// SAFETY: buf must have enough capacity for `v[mid..]`.
		unsafe {
			for i in 0..len - mid {
				ptr::copy_nonoverlapping(&v[mid + i], buf.add(i), 1);
			}
			hole = MergeHole {
				buf,
				start: 0,
				end: len - mid,
				dest: mid,
				v,
			};
		}

		// Initially, these pointers point past the ends of their arrays.
		//let left = &mut hole.dest;
		//let right = &mut hole.end;
		let mut out = len; //v_end;

		while 0 < hole.dest && 0 < hole.end {
			// Consume the greater side.
			// If equal, prefer the right run to maintain stability.

			// SAFETY: left and right must be valid and part of v same for out.
			unsafe {
				let w = hole.v.view();
				let to_copy = if is_less(&*hole.buf.add(hole.end - 1), w.uget(hole.dest - 1)) {
					let idx = &mut hole.dest;
					// SAFETY: ptr.sub(1) must still be a valid pointer and part of `v`.
					*idx -= 1; //unsafe { ptr.sub(1) };
					hole.v.view_mut().index(*idx)
				} else {
					let idx = &mut hole.end;
					// SAFETY: ptr.sub(1) must still be a valid pointer and part of `v`.
					*idx -= 1; //unsafe { ptr.sub(1) };
					hole.buf.add(*idx)
				};
				let idx = &mut out;
				// SAFETY: ptr.sub(1) must still be a valid pointer and part of `v`.
				*idx -= 1; //unsafe { ptr.sub(1) };
				let dst = hole.v.view_mut().index(*idx);
				ptr::copy_nonoverlapping(to_copy, dst, 1);
			}
		}
	}
	// Finally, `hole` gets dropped. If the shorter run was not fully consumed, whatever remains of
	// it will now be copied into the hole in `v`.

	// When dropped, copies the range `start..end` into `dest..`.
	struct MergeHole<'a, T> {
		buf: *mut T,
		start: usize,
		end: usize,

		v: ArrayViewMut1<'a, T>,
		dest: usize,
	}
	//impl<'a, T> MergeHole<'a, T> {
	//    unsafe fn buf_get_and_increment(&mut self, idx: &mut usize) -> *mut T {
	//        let old = self.buf.add(*idx);

	//        // SAFETY: ptr.add(1) must still be a valid pointer and part of `v`.
	//        *idx = *idx + 1;//unsafe { ptr.add(1) };
	//        old
	//    }

	//    unsafe fn buf_decrement_and_get(&mut self, idx: &mut usize) -> *mut T {
	//        // SAFETY: ptr.sub(1) must still be a valid pointer and part of `v`.
	//        *idx = *idx - 1;//unsafe { ptr.sub(1) };
	//        self.buf.add(*idx)
	//    }

	//    unsafe fn out_get_and_increment(&mut self, idx: &mut usize) -> *mut T {
	//        let old = self.v.view_mut().index(*idx);

	//        // SAFETY: ptr.add(1) must still be a valid pointer and part of `v`.
	//        *idx = *idx + 1;//unsafe { ptr.add(1) };
	//        old
	//    }

	//    unsafe fn out_decrement_and_get(&mut self, idx: &mut usize) -> *mut T {
	//        // SAFETY: ptr.sub(1) must still be a valid pointer and part of `v`.
	//        *idx = *idx - 1;//unsafe { ptr.sub(1) };
	//        self.v.view_mut().index(*idx)
	//    }
	//}

	impl<'a, T> Drop for MergeHole<'a, T> {
		fn drop(&mut self) {
			// SAFETY: `T` is not a zero-sized type, and these are pointers into a slice's elements.
			unsafe {
				let len = self.end - self.start; //self.end.sub_ptr(self.start);
				for i in 0..len {
					let src = self.buf.add(self.start + i);
					let dst = self.v.view_mut().index(self.dest + i);
					ptr::copy_nonoverlapping(src, dst, 1);
				}
			}
		}
	}
}

/// This merge sort borrows some (but not all) ideas from TimSort, which used to be described in
/// detail [here](https://github.com/python/cpython/blob/main/Objects/listsort.txt). However Python
/// has switched to a Powersort based implementation.
///
/// The algorithm identifies strictly descending and non-descending subsequences, which are called
/// natural runs. There is a stack of pending runs yet to be merged. Each newly found run is pushed
/// onto the stack, and then some pairs of adjacent runs are merged until these two invariants are
/// satisfied:
///
/// 1. for every `i` in `1..runs.len()`: `runs[i - 1].len > runs[i].len`
/// 2. for every `i` in `2..runs.len()`: `runs[i - 2].len > runs[i - 1].len + runs[i].len`
///
/// The invariants ensure that the total running time is *O*(*n* \* log(*n*)) worst-case.
pub fn merge_sort<T, CmpF, ElemAllocF, ElemDeallocF, RunAllocF, RunDeallocF>(
	mut v: ArrayViewMut1<'_, T>,
	is_less: &mut CmpF,
	elem_alloc_fn: ElemAllocF,
	elem_dealloc_fn: ElemDeallocF,
	run_alloc_fn: RunAllocF,
	run_dealloc_fn: RunDeallocF,
) where
	CmpF: FnMut(&T, &T) -> bool,
	ElemAllocF: Fn(usize) -> *mut T,
	ElemDeallocF: Fn(*mut T, usize),
	RunAllocF: Fn(usize) -> *mut TimSortRun,
	RunDeallocF: Fn(*mut TimSortRun, usize),
{
	// Slices of up to this length get sorted using insertion sort.
	const MAX_INSERTION: usize = 20;
	// Very short runs are extended using insertion sort to span at least this many elements.
	const MIN_RUN: usize = 10;

	// The caller should have already checked that.
	debug_assert!(mem::size_of::<T>() > 0);

	let len = v.len();

	// Short arrays get sorted in-place via insertion sort to avoid allocations.
	if len <= MAX_INSERTION {
		if len >= 2 {
			for i in (0..len - 1).rev() {
				insert_head(v.slice_mut(s![i..]), is_less);
			}
		}
		return;
	}

	// Allocate a buffer to use as scratch memory. We keep the length 0 so we can keep in it
	// shallow copies of the contents of `v` without risking the dtors running on copies if
	// `is_less` panics. When merging two sorted runs, this buffer holds a copy of the shorter run,
	// which will always have length at most `len / 2`.
	let buf = BufGuard::new(len / 2, elem_alloc_fn, elem_dealloc_fn);
	let buf_ptr = buf.buf_ptr;

	let mut runs = RunVec::new(run_alloc_fn, run_dealloc_fn);

	// In order to identify natural runs in `v`, we traverse it backwards. That might seem like a
	// strange decision, but consider the fact that merges more often go in the opposite direction
	// (forwards). According to benchmarks, merging forwards is slightly faster than merging
	// backwards. To conclude, identifying runs by traversing backwards improves performance.
	let mut end = len;
	while end > 0 {
		// Find the next natural run, and reverse it if it's strictly descending.
		let mut start = end - 1;
		if start > 0 {
			start -= 1;

			// SAFETY: The v.get_unchecked must be fed with correct inbound indicies.
			unsafe {
				let w = v.view();
				if is_less(w.uget(start + 1), w.uget(start)) {
					while start > 0 && is_less(w.uget(start), w.uget(start - 1)) {
						start -= 1;
					}
					reverse(v.slice_mut(s![start..end]));
				} else {
					let w = v.view();
					while start > 0 && !is_less(w.uget(start), w.uget(start - 1)) {
						start -= 1;
					}
				}
			}
		}

		// Insert some more elements into the run if it's too short. Insertion sort is faster than
		// merge sort on short sequences, so this significantly improves performance.
		while start > 0 && end - start < MIN_RUN {
			start -= 1;
			insert_head(v.slice_mut(s![start..end]), is_less);
		}

		// Push this run onto the stack.
		runs.push(TimSortRun {
			start,
			len: end - start,
		});
		end = start;

		// Merge some pairs of adjacent runs to satisfy the invariants.
		while let Some(r) = collapse(runs.as_slice()) {
			let left = runs[r + 1];
			let right = runs[r];
			// SAFETY: `buf_ptr` must hold enough capacity for the shorter of the two sides, and
			// neither side may be on length 0.
			unsafe {
				merge(
					v.slice_mut(s![left.start..right.start + right.len]),
					left.len,
					buf_ptr,
					is_less,
				);
			}
			runs[r] = TimSortRun {
				start: left.start,
				len: left.len + right.len,
			};
			runs.remove(r + 1);
		}
	}

	// Finally, exactly one run must remain in the stack.
	debug_assert!(runs.len() == 1 && runs[0].start == 0 && runs[0].len == len);

	// Examines the stack of runs and identifies the next pair of runs to merge. More specifically,
	// if `Some(r)` is returned, that means `runs[r]` and `runs[r + 1]` must be merged next. If the
	// algorithm should continue building a new run instead, `None` is returned.
	//
	// TimSort is infamous for its buggy implementations, as described here:
	// http://envisage-project.eu/timsort-specification-and-verification/
	//
	// The gist of the story is: we must enforce the invariants on the top four runs on the stack.
	// Enforcing them on just top three is not sufficient to ensure that the invariants will still
	// hold for *all* runs in the stack.
	//
	// This function correctly checks invariants for the top four runs. Additionally, if the top
	// run starts at index 0, it will always demand a merge operation until the stack is fully
	// collapsed, in order to complete the sort.
	#[inline]
	fn collapse(runs: &[TimSortRun]) -> Option<usize> {
		let n = runs.len();
		if n >= 2
			&& (runs[n - 1].start == 0
				|| runs[n - 2].len <= runs[n - 1].len
				|| (n >= 3 && runs[n - 3].len <= runs[n - 2].len + runs[n - 1].len)
				|| (n >= 4 && runs[n - 4].len <= runs[n - 3].len + runs[n - 2].len))
		{
			if n >= 3 && runs[n - 3].len < runs[n - 1].len {
				Some(n - 3)
			} else {
				Some(n - 2)
			}
		} else {
			None
		}
	}

	// Extremely basic versions of Vec.
	// Their use is super limited and by having the code here, it allows reuse between the sort
	// implementations.
	struct BufGuard<T, ElemDeallocF>
	where
		ElemDeallocF: Fn(*mut T, usize),
	{
		buf_ptr: *mut T,
		capacity: usize,
		elem_dealloc_fn: ElemDeallocF,
	}

	impl<T, ElemDeallocF> BufGuard<T, ElemDeallocF>
	where
		ElemDeallocF: Fn(*mut T, usize),
	{
		fn new<ElemAllocF>(
			len: usize,
			elem_alloc_fn: ElemAllocF,
			elem_dealloc_fn: ElemDeallocF,
		) -> Self
		where
			ElemAllocF: Fn(usize) -> *mut T,
		{
			Self {
				buf_ptr: elem_alloc_fn(len),
				capacity: len,
				elem_dealloc_fn,
			}
		}
	}

	impl<T, ElemDeallocF> Drop for BufGuard<T, ElemDeallocF>
	where
		ElemDeallocF: Fn(*mut T, usize),
	{
		fn drop(&mut self) {
			(self.elem_dealloc_fn)(self.buf_ptr, self.capacity);
		}
	}

	struct RunVec<RunAllocF, RunDeallocF>
	where
		RunAllocF: Fn(usize) -> *mut TimSortRun,
		RunDeallocF: Fn(*mut TimSortRun, usize),
	{
		buf_ptr: *mut TimSortRun,
		capacity: usize,
		len: usize,
		run_alloc_fn: RunAllocF,
		run_dealloc_fn: RunDeallocF,
	}

	impl<RunAllocF, RunDeallocF> RunVec<RunAllocF, RunDeallocF>
	where
		RunAllocF: Fn(usize) -> *mut TimSortRun,
		RunDeallocF: Fn(*mut TimSortRun, usize),
	{
		fn new(run_alloc_fn: RunAllocF, run_dealloc_fn: RunDeallocF) -> Self {
			// Most slices can be sorted with at most 16 runs in-flight.
			const START_RUN_CAPACITY: usize = 16;

			Self {
				buf_ptr: run_alloc_fn(START_RUN_CAPACITY),
				capacity: START_RUN_CAPACITY,
				len: 0,
				run_alloc_fn,
				run_dealloc_fn,
			}
		}

		fn push(&mut self, val: TimSortRun) {
			if self.len == self.capacity {
				let old_capacity = self.capacity;
				let old_buf_ptr = self.buf_ptr;

				self.capacity *= 2;
				self.buf_ptr = (self.run_alloc_fn)(self.capacity);

				// SAFETY: buf_ptr new and old were correctly allocated and old_buf_ptr has
				// old_capacity valid elements.
				unsafe {
					ptr::copy_nonoverlapping(old_buf_ptr, self.buf_ptr, old_capacity);
				}

				(self.run_dealloc_fn)(old_buf_ptr, old_capacity);
			}

			// SAFETY: The invariant was just checked.
			unsafe {
				self.buf_ptr.add(self.len).write(val);
			}
			self.len += 1;
		}

		fn remove(&mut self, index: usize) {
			if index >= self.len {
				panic!("Index out of bounds");
			}

			// SAFETY: buf_ptr needs to be valid and len invariant upheld.
			unsafe {
				// the place we are taking from.
				let ptr = self.buf_ptr.add(index);

				// Shift everything down to fill in that spot.
				ptr::copy(ptr.add(1), ptr, self.len - index - 1);
			}
			self.len -= 1;
		}

		fn as_slice(&self) -> &[TimSortRun] {
			// SAFETY: Safe as long as buf_ptr is valid and len invariant was upheld.
			unsafe { &*ptr::slice_from_raw_parts(self.buf_ptr, self.len) }
		}

		fn len(&self) -> usize {
			self.len
		}
	}

	impl<RunAllocF, RunDeallocF> core::ops::Index<usize> for RunVec<RunAllocF, RunDeallocF>
	where
		RunAllocF: Fn(usize) -> *mut TimSortRun,
		RunDeallocF: Fn(*mut TimSortRun, usize),
	{
		type Output = TimSortRun;

		fn index(&self, index: usize) -> &Self::Output {
			if index < self.len {
				// SAFETY: buf_ptr and len invariant must be upheld.
				unsafe {
					return &*(self.buf_ptr.add(index));
				}
			}

			panic!("Index out of bounds");
		}
	}

	impl<RunAllocF, RunDeallocF> core::ops::IndexMut<usize> for RunVec<RunAllocF, RunDeallocF>
	where
		RunAllocF: Fn(usize) -> *mut TimSortRun,
		RunDeallocF: Fn(*mut TimSortRun, usize),
	{
		fn index_mut(&mut self, index: usize) -> &mut Self::Output {
			if index < self.len {
				// SAFETY: buf_ptr and len invariant must be upheld.
				unsafe {
					return &mut *(self.buf_ptr.add(index));
				}
			}

			panic!("Index out of bounds");
		}
	}

	impl<RunAllocF, RunDeallocF> Drop for RunVec<RunAllocF, RunDeallocF>
	where
		RunAllocF: Fn(usize) -> *mut TimSortRun,
		RunDeallocF: Fn(*mut TimSortRun, usize),
	{
		fn drop(&mut self) {
			// As long as TimSortRun is Copy we don't need to drop them individually but just the
			// whole allocation.
			(self.run_dealloc_fn)(self.buf_ptr, self.capacity);
		}
	}
}

/// Internal type used by merge_sort.
#[derive(Clone, Copy, Debug)]
pub struct TimSortRun {
	len: usize,
	start: usize,
}