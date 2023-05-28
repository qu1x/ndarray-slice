//! Derivative work of [`rayon::slice::mergesort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`rayon::slice::mergesort`]: https://docs.rs/rayon/latest/src/rayon/slice/mergesort.rs.html

use crate::partition::reverse;
use core::{
	mem::{self, size_of},
	ptr,
};
use ndarray::{s, ArrayView1, ArrayViewMut1, Axis, IndexLonger};
use rayon::iter::{ParallelBridge, ParallelIterator};

/// We need to transmit raw pointers across threads. It is possible to do this
/// without any unsafe code by converting pointers to `usize` or to `AtomicPtr<T>`
/// then back to a raw pointer for use. We prefer this approach because code
/// that uses this type is more explicit.
///
/// Unsafe code is still required to dereference the pointer, so this type is
/// not unsound on its own, although it does partly lift the unconditional
/// `!Send` and `!Sync` on raw pointers. As always, dereference with care.
struct SendPtr<T>(*mut T);

// SAFETY: !Send for raw pointers is not for safety, just as a lint
unsafe impl<T: Send> Send for SendPtr<T> {}

// SAFETY: !Sync for raw pointers is not for safety, just as a lint
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
	// Helper to avoid disjoint captures of `send_ptr.0`
	fn get(self) -> *mut T {
		self.0
	}
}

// Implement Clone without the T: Clone bound from the derive
impl<T> Clone for SendPtr<T> {
	fn clone(&self) -> Self {
		Self(self.0)
	}
}

// Implement Copy without the T: Copy bound from the derive
impl<T> Copy for SendPtr<T> {}

/// Inserts `v[0]` into pre-sorted sequence `v[1..]` so that whole `v[..]` becomes sorted.
///
/// This is the integral subroutine of insertion sort.
fn insert_head<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: &F)
where
	F: Fn(&T, &T) -> bool,
{
	if v.len() >= 2 && is_less(&v[1], &v[0]) {
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
unsafe fn merge<T, F>(v: ArrayViewMut1<'_, T>, mid: usize, buf: *mut T, is_less: &F)
where
	F: Fn(&T, &T) -> bool,
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

/// The result of merge sort.
#[must_use]
#[derive(Clone, Copy, PartialEq, Eq)]
enum MergesortResult {
	/// The slice has already been sorted.
	NonDescending,
	/// The slice has been descending and therefore it was left intact.
	Descending,
	/// The slice was sorted.
	Sorted,
}

/// A sorted run that starts at index `start` and is of length `len`.
#[derive(Clone, Copy)]
struct Run {
	start: usize,
	len: usize,
}

/// Examines the stack of runs and identifies the next pair of runs to merge. More specifically,
/// if `Some(r)` is returned, that means `runs[r]` and `runs[r + 1]` must be merged next. If the
/// algorithm should continue building a new run instead, `None` is returned.
///
/// TimSort is infamous for its buggy implementations, as described here:
/// <http://envisage-project.eu/timsort-specification-and-verification/>
///
/// The gist of the story is: we must enforce the invariants on the top four runs on the stack.
/// Enforcing them on just top three is not sufficient to ensure that the invariants will still
/// hold for *all* runs in the stack.
///
/// This function correctly checks invariants for the top four runs. Additionally, if the top
/// run starts at index 0, it will always demand a merge operation until the stack is fully
/// collapsed, in order to complete the sort.
#[inline]
fn collapse(runs: &[Run]) -> Option<usize> {
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

/// Sorts a slice using merge sort, unless it is already in descending order.
///
/// This function doesn't modify the slice if it is already non-descending or descending.
/// Otherwise, it sorts the slice into non-descending order.
///
/// This merge sort borrows some (but not all) ideas from TimSort, which is described in detail
/// [here](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
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
///
/// # Safety
///
/// The argument `buf` is used as a temporary buffer and must be at least as long as `v`.
unsafe fn merge_sort<T, F>(mut v: ArrayViewMut1<'_, T>, buf: *mut T, is_less: &F) -> MergesortResult
where
	T: Send,
	F: Fn(&T, &T) -> bool,
{
	// Very short runs are extended using insertion sort to span at least this many elements.
	const MIN_RUN: usize = 10;

	let len = v.len();

	// In order to identify natural runs in `v`, we traverse it backwards. That might seem like a
	// strange decision, but consider the fact that merges more often go in the opposite direction
	// (forwards). According to benchmarks, merging forwards is slightly faster than merging
	// backwards. To conclude, identifying runs by traversing backwards improves performance.
	let mut runs = vec![];
	let mut end = len;
	while end > 0 {
		// Find the next natural run, and reverse it if it's strictly descending.
		let mut start = end - 1;

		if start > 0 {
			start -= 1;

			let w = v.view();
			if is_less(w.uget(start + 1), w.uget(start)) {
				while start > 0 && is_less(w.uget(start), w.uget(start - 1)) {
					start -= 1;
				}

				// If this descending run covers the whole slice, return immediately.
				if start == 0 && end == len {
					return MergesortResult::Descending;
				} else {
					reverse(v.slice_mut(s![start..end]));
				}
			} else {
				while start > 0 && !is_less(w.uget(start), w.uget(start - 1)) {
					start -= 1;
				}

				// If this non-descending run covers the whole slice, return immediately.
				if end - start == len {
					return MergesortResult::NonDescending;
				}
			}
		}

		// Insert some more elements into the run if it's too short. Insertion sort is faster than
		// merge sort on short sequences, so this significantly improves performance.
		while start > 0 && end - start < MIN_RUN {
			start -= 1;
			insert_head(v.slice_mut(s![start..end]), &is_less);
		}

		// Push this run onto the stack.
		runs.push(Run {
			start,
			len: end - start,
		});
		end = start;

		// Merge some pairs of adjacent runs to satisfy the invariants.
		while let Some(r) = collapse(&runs) {
			let left = runs[r + 1];
			let right = runs[r];
			merge(
				v.slice_mut(s![left.start..right.start + right.len]),
				left.len,
				buf,
				&is_less,
			);

			runs[r] = Run {
				start: left.start,
				len: left.len + right.len,
			};
			runs.remove(r + 1);
		}
	}

	// Finally, exactly one run must remain in the stack.
	debug_assert!(runs.len() == 1 && runs[0].start == 0 && runs[0].len == len);

	// The original order of the slice was neither non-descending nor descending.
	MergesortResult::Sorted
}

////////////////////////////////////////////////////////////////////////////
// Everything above this line is copied from `std::slice::sort` (with very minor tweaks).
// Everything below this line is parallelization.
////////////////////////////////////////////////////////////////////////////

/// Splits two sorted slices so that they can be merged in parallel.
///
/// Returns two indices `(a, b)` so that slices `left[..a]` and `right[..b]` come before
/// `left[a..]` and `right[b..]`.
fn split_for_merge<T, F>(
	left: ArrayView1<'_, T>,
	right: ArrayView1<'_, T>,
	is_less: &F,
) -> (usize, usize)
where
	F: Fn(&T, &T) -> bool,
{
	let left_len = left.len();
	let right_len = right.len();

	if left_len >= right_len {
		let left_mid = left_len / 2;

		// Find the first element in `right` that is greater than or equal to `left[left_mid]`.
		let mut a = 0;
		let mut b = right_len;
		while a < b {
			let m = a + (b - a) / 2;
			if is_less(&right[m], &left[left_mid]) {
				a = m + 1;
			} else {
				b = m;
			}
		}

		(left_mid, a)
	} else {
		let right_mid = right_len / 2;

		// Find the first element in `left` that is greater than `right[right_mid]`.
		let mut a = 0;
		let mut b = left_len;
		while a < b {
			let m = a + (b - a) / 2;
			if is_less(&right[right_mid], &left[m]) {
				b = m;
			} else {
				a = m + 1;
			}
		}

		(a, right_mid)
	}
}

/// Merges slices `left` and `right` in parallel and stores the result into `dest`.
///
/// # Safety
///
/// The `dest` pointer must have enough space to store the result.
///
/// Even if `is_less` panics at any point during the merge process, this function will fully copy
/// all elements from `left` and `right` into `dest` (not necessarily in sorted order).
#[warn(unsafe_op_in_unsafe_fn)]
unsafe fn par_merge<T, F>(
	mut left: ArrayViewMut1<'_, T>,
	mut right: ArrayViewMut1<'_, T>,
	mut dest: ArrayViewMut1<'_, T>,
	is_less: &F,
) where
	T: Send,
	F: Fn(&T, &T) -> bool + Sync,
{
	// Slices whose lengths sum up to this value are merged sequentially. This number is slightly
	// larger than `CHUNK_LENGTH`, and the reason is that merging is faster than merge sorting, so
	// merging needs a bit coarser granularity in order to hide the overhead of Rayon's task
	// scheduling.
	const MAX_SEQUENTIAL: usize = 5000;

	let left_raw = left.raw_view_mut();
	let right_raw = right.raw_view_mut();
	let dest_raw = dest.raw_view_mut();

	let left_len = left.len();
	let right_len = right.len();

	// Intermediate state of the merge process, which serves two purposes:
	// 1. Protects integrity of `dest` from panics in `is_less`.
	// 2. Copies the remaining elements as soon as one of the two sides is exhausted.
	//
	// Panic safety:
	//
	// If `is_less` panics at any point during the merge process, `s` will get dropped and copy the
	// remaining parts of `left` and `right` into `dest`.
	let left = unsafe { left_raw.deref_into_view_mut() };
	let right = unsafe { right_raw.deref_into_view_mut() };
	let dest = unsafe { dest_raw.deref_into_view_mut() };
	let mut s = State {
		//left_start: left.as_mut_ptr(),
		//left_end: left.as_mut_ptr().add(left_len),
		//right_start: right.as_mut_ptr(),
		//right_end: right.as_mut_ptr().add(right_len),
		//dest,
		left,
		left_start: 0,
		right,
		right_start: 0,
		dest,
		dest_start: 0,
	};

	if left_len == 0 || right_len == 0 || left_len + right_len < MAX_SEQUENTIAL {
		while s.left_start < s.left.len() && s.right_start < s.right.len() {
			// Consume the lesser side.
			// If equal, prefer the left run to maintain stability.
			if is_less(&s.right[s.right_start], &s.left[s.left_start]) {
				unsafe {
					ptr::copy_nonoverlapping(&s.right[s.right_start], &mut s.dest[s.dest_start], 1)
				};
				s.right_start += 1;
			} else {
				unsafe {
					ptr::copy_nonoverlapping(&s.left[s.left_start], &mut s.dest[s.dest_start], 1)
				};
				s.left_start += 1;
			};
			s.dest_start += 1;
		}
	} else {
		let left = unsafe { left_raw.deref_into_view_mut() };
		let right = unsafe { right_raw.deref_into_view_mut() };
		let dest = unsafe { dest_raw.deref_into_view_mut() };

		// Function `split_for_merge` might panic. If that happens, `s` will get destructed and copy
		// the whole `left` and `right` into `dest`.
		let (left_mid, right_mid) = split_for_merge(left.view(), right.view(), is_less);
		let (left_l, left_r) = left.split_at(Axis(0), left_mid);
		let (right_l, right_r) = right.split_at(Axis(0), right_mid);

		// Prevent the destructor of `s` from running. Rayon will ensure that both calls to
		// `par_merge` happen. If one of the two calls panics, they will ensure that elements still
		// get copied into `dest_left` and `dest_right``.
		mem::forget(s);

		// Wrap pointers in SendPtr so that they can be sent to another thread
		// See the documentation of SendPtr for a full explanation
		//let dest_l = SendPtr(dest);
		//let dest_r = SendPtr(dest.add(left_l.len() + right_l.len()));
		let (dest_l, dest_r) = dest.split_at(Axis(0), left_l.len() + right_l.len());
		rayon::join(
			move || unsafe { par_merge(left_l, right_l, dest_l, is_less) },
			move || unsafe { par_merge(left_r, right_r, dest_r, is_less) },
		);
	}
	// Finally, `s` gets dropped if we used sequential merge, thus copying the remaining elements
	// all at once.

	// When dropped, copies arrays `left_start..left_end` and `right_start..right_end` into `dest`,
	// in that order.
	struct State<'a, T> {
		//left_start: *mut T,
		//left_end: *mut T,
		//right_start: *mut T,
		//right_end: *mut T,
		//dest: *mut T,
		left: ArrayViewMut1<'a, T>,
		left_start: usize,
		right: ArrayViewMut1<'a, T>,
		right_start: usize,
		dest: ArrayViewMut1<'a, T>,
		dest_start: usize,
	}

	impl<'a, T> Drop for State<'a, T> {
		fn drop(&mut self) {
			//let size = size_of::<T>();
			//let left_len = (self.left_end as usize - self.left_start as usize) / size;
			//let right_len = (self.right_end as usize - self.right_start as usize) / size;

			// Copy array `left`, followed by `right`.
			unsafe {
				let left_len = self.left.len() - self.left_start;
				for i in 0..left_len {
					ptr::copy_nonoverlapping(
						&self.left[i + self.left_start],
						&mut self.dest[i + self.dest_start],
						1,
					);
				}
				//self.dest = self.dest.add(left_len);
				let right_len = self.right.len() - self.right_start;
				for i in 0..right_len {
					ptr::copy_nonoverlapping(
						&self.right[i + self.right_start],
						&mut self.dest[i + self.dest_start + left_len],
						1,
					);
				}
			}
		}
	}
}

/// Recursively merges pre-sorted chunks inside `v`.
///
/// Chunks of `v` are stored in `chunks` as intervals (inclusive left and exclusive right bound).
/// Argument `buf` is an auxiliary buffer that will be used during the procedure.
/// If `into_buf` is true, the result will be stored into `buf`, otherwise it will be in `v`.
///
/// # Safety
///
/// The number of chunks must be positive and they must be adjacent: the right bound of each chunk
/// must equal the left bound of the following chunk.
///
/// The buffer must be at least as long as `v`.
#[warn(unsafe_op_in_unsafe_fn)]
unsafe fn recurse<T, F>(
	mut v: ArrayViewMut1<'_, T>,
	mut buf: ArrayViewMut1<'_, T>,
	chunks: &[(usize, usize)],
	into_buf: bool,
	is_less: &F,
) where
	T: Send,
	F: Fn(&T, &T) -> bool + Sync,
{
	let v_raw = v.raw_view_mut();
	let buf_raw = buf.raw_view_mut();

	let len = chunks.len();
	debug_assert!(len > 0);
	// Base case of the algorithm.
	// If only one chunk is remaining, there's no more work to split and merge.
	if len == 1 {
		if into_buf {
			// Copy the chunk from `v` into `buf`.
			let (start, end) = chunks[0];
			//let src = v.add(start);
			//let dest = buf.add(start);
			for i in start..end {
				unsafe { ptr::copy_nonoverlapping(&v[i], &mut buf[i], 1) };
			}
		}
		return;
	}

	// Split the chunks into two halves.
	let (start, _) = chunks[0];
	let (mid, _) = chunks[len / 2];
	let (_, end) = chunks[len - 1];
	let (left, right) = chunks.split_at(len / 2);

	// After recursive calls finish we'll have to merge chunks `(start, mid)` and `(mid, end)` from
	// `src` into `dest`. If the current invocation has to store the result into `buf`, we'll
	// merge chunks from `v` into `buf`, and vice versa.
	//
	// Recursive calls flip `into_buf` at each level of recursion. More concretely, `par_merge`
	// merges chunks from `buf` into `v` at the first level, from `v` into `buf` at the second
	// level etc.
	let v = unsafe { v_raw.deref_into_view_mut() };
	let buf = unsafe { buf_raw.deref_into_view_mut() };
	let (mut src, mut dest) = if into_buf { (v, buf) } else { (buf, v) };

	// Panic safety:
	//
	// If `is_less` panics at any point during the recursive calls, the destructor of `guard` will
	// be executed, thus copying everything from `src` into `dest`. This way we ensure that all
	// chunks are in fact copied into `dest`, even if the merge process doesn't finish.
	let guard = CopyOnDrop {
		src: src.view_mut(),
		dest: dest.view_mut(),
		src_start: start,
		dest_start: start,
		len: end - start,
	};

	let v_left = unsafe { v_raw.deref_into_view_mut() };
	let buf_left = unsafe { buf_raw.deref_into_view_mut() };
	let v_right = unsafe { v_raw.deref_into_view_mut() };
	let buf_right = unsafe { buf_raw.deref_into_view_mut() };
	// Wrap pointers in SendPtr so that they can be sent to another thread
	// See the documentation of SendPtr for a full explanation
	//let v = SendPtr(v);
	//let buf = SendPtr(buf);
	rayon::join(
		move || {
			unsafe {
				recurse(
					v_left, buf_left, /*v.get(), buf.get(),*/ left, !into_buf, is_less,
				)
			}
		},
		move || {
			unsafe {
				recurse(
					v_right, buf_right, /*v.get(), buf.get(),*/ right, !into_buf, is_less,
				)
			}
		},
	);

	// Everything went all right - recursive calls didn't panic.
	// Forget the guard in order to prevent its destructor from running.
	mem::forget(guard);

	// Merge chunks `(start, mid)` and `(mid, end)` from `src` into `dest`.
	//let src_left = slice::from_raw_parts_mut(src.add(start), mid - start);
	//let src_right = slice::from_raw_parts_mut(src.add(mid), end - mid);
	//par_merge(src_left, src_right, dest.add(start), is_less);
	let (src_left, src_right) = src.multi_slice_mut((s![start..mid], s![mid..end]));
	unsafe { par_merge(src_left, src_right, dest.slice_mut(s![start..]), is_less) };

	/// When dropped, copies from `src` into `dest` a sequence of length `len`.
	struct CopyOnDrop<'a, T> {
		src: ArrayViewMut1<'a, T>,
		dest: ArrayViewMut1<'a, T>,
		src_start: usize,
		dest_start: usize,
		len: usize,
	}

	impl<'a, T> Drop for CopyOnDrop<'a, T> {
		fn drop(&mut self) {
			unsafe {
				for i in 0..self.len {
					let a = self.src_start + i;
					let b = self.dest_start + i;
					ptr::copy_nonoverlapping(&self.src[a], &mut self.dest[b], 1);
				}
			}
		}
	}
}

/// Sorts `v` using merge sort in parallel.
///
/// The algorithm is stable, allocates memory, and `O(n log n)` worst-case.
/// The allocated temporary buffer is of the same length as is `v`.
pub fn par_merge_sort<T, F>(mut v: ArrayViewMut1<'_, T>, is_less: F)
where
	T: Send,
	F: Fn(&T, &T) -> bool + Sync,
{
	// Slices of up to this length get sorted using insertion sort in order to avoid the cost of
	// buffer allocation.
	const MAX_INSERTION: usize = 20;
	// The length of initial chunks. This number is as small as possible but so that the overhead
	// of Rayon's task scheduling is still negligible.
	const CHUNK_LENGTH: usize = 2000;

	// Sorting has no meaningful behavior on zero-sized types.
	if size_of::<T>() == 0 {
		return;
	}

	let len = v.len();

	// Short slices get sorted in-place via insertion sort to avoid allocations.
	if len <= MAX_INSERTION {
		if len >= 2 {
			for i in (0..len - 1).rev() {
				insert_head(v.slice_mut(s![i..]), &is_less);
			}
		}
		return;
	}

	// Allocate a buffer to use as scratch memory. We keep the length 0 so we can keep in it
	// shallow copies of the contents of `v` without risking the dtors running on copies if
	// `is_less` panics.
	let mut buf = Vec::<T>::with_capacity(len);
	let buf = buf.as_mut_ptr();

	// If the slice is not longer than one chunk would be, do sequential merge sort and return.
	if len <= CHUNK_LENGTH {
		let res = unsafe { merge_sort(v.view_mut(), buf, &is_less) };
		if res == MergesortResult::Descending {
			reverse(v.view_mut());
		}
		return;
	}

	// Split the slice into chunks and merge sort them in parallel.
	// However, descending chunks will not be sorted - they will be simply left intact.
	let mut iter = {
		// Wrap pointer in SendPtr so that it can be sent to another thread
		// See the documentation of SendPtr for a full explanation
		let buf = SendPtr(buf);
		let is_less = &is_less;

		// akin par_bridge().map().collect() but order-preserving
		let chunks_iter = v.axis_chunks_iter_mut(Axis(0), CHUNK_LENGTH);
		let len = chunks_iter.len();
		let mut chunks = Vec::with_capacity(len);
		chunks_iter
			.enumerate()
			.zip(chunks.spare_capacity_mut())
			.par_bridge()
			//.with_max_len(1)
			.for_each(move |((i, chunk), out)| {
				let l = CHUNK_LENGTH * i;
				let r = l + chunk.len();
				unsafe {
					let buf = buf.get().add(l);
					out.write((l, r, merge_sort(chunk, buf, is_less)));
				}
			});
		unsafe { chunks.set_len(len) };
		chunks.into_iter().peekable()
	};

	// Now attempt to concatenate adjacent chunks that were left intact.
	let mut chunks = Vec::with_capacity(iter.len());

	while let Some((a, mut b, res)) = iter.next() {
		// If this chunk was not modified by the sort procedure...
		if res != MergesortResult::Sorted {
			while let Some(&(x, y, r)) = iter.peek() {
				// If the following chunk is of the same type and can be concatenated...
				if r == res && (r == MergesortResult::Descending) == is_less(&v[x], &v[x - 1]) {
					// Concatenate them.
					b = y;
					iter.next();
				} else {
					break;
				}
			}
		}

		// Descending chunks must be reversed.
		if res == MergesortResult::Descending {
			reverse(v.slice_mut(s![a..b]));
		}

		chunks.push((a, b));
	}

	// All chunks are properly sorted.
	// Now we just have to merge them together.
	unsafe {
		let buf = ArrayViewMut1::from_shape_ptr(len, buf);
		recurse(v, buf, &chunks, false, &is_less);
	}
}

#[cfg(test)]
mod test {
	use super::{par_merge_sort, split_for_merge};
	use core::cmp::Ordering;
	use ndarray::{s, Array1, ArrayView1};
	use quickcheck_macros::quickcheck;
	use rand::distributions::Uniform;
	use rand::{thread_rng, Rng};

	#[test]
	fn split() {
		fn check(left: &[u32], right: &[u32]) {
			let left = ArrayView1::from_shape(left.len(), left).unwrap();
			let right = ArrayView1::from_shape(right.len(), right).unwrap();
			let (l, r) = split_for_merge(left, right, &|&a, &b| a < b);
			assert!(left
				.slice(s![..l])
				.iter()
				.all(|&x| right.slice(s![r..]).iter().all(|&y| x <= y)));
			assert!(right
				.slice(s![..r])
				.iter()
				.all(|&x| left.slice(s![l..]).iter().all(|&y| x < y)));
		}

		check(&[1, 2, 2, 2, 2, 3], &[1, 2, 2, 2, 2, 3]);
		check(&[1, 2, 2, 2, 2, 3], &[]);
		check(&[], &[1, 2, 2, 2, 2, 3]);

		let rng = &mut thread_rng();

		for _ in 0..100 {
			let limit: u32 = rng.gen_range(1..21);
			let left_len: usize = rng.gen_range(0..20);
			let right_len: usize = rng.gen_range(0..20);

			let mut left = rng
				.sample_iter(&Uniform::new(0, limit))
				.take(left_len)
				.collect::<Vec<_>>();
			let mut right = rng
				.sample_iter(&Uniform::new(0, limit))
				.take(right_len)
				.collect::<Vec<_>>();

			left.sort();
			right.sort();
			check(&left, &right);
		}
	}

	#[derive(Debug, Clone, Copy)]
	struct Item {
		index: usize,
		value: u32,
	}

	impl Eq for Item {}

	impl PartialEq for Item {
		fn eq(&self, other: &Self) -> bool {
			self.value == other.value
		}
	}

	impl Ord for Item {
		fn cmp(&self, other: &Self) -> Ordering {
			self.value.cmp(&other.value)
		}
	}

	impl PartialOrd for Item {
		fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
			Some(self.cmp(other))
		}
	}

	impl From<(usize, u32)> for Item {
		fn from((index, value): (usize, u32)) -> Self {
			Self { index, value }
		}
	}

	#[cfg_attr(miri, ignore)]
	#[quickcheck]
	fn stably_sorted(xs: Vec<u32>) {
		let xs = xs
			.into_iter()
			.enumerate()
			.map(Item::from)
			.collect::<Vec<Item>>();
		let mut sorted = xs.clone();
		sorted.sort();
		let sorted = Array1::from_vec(sorted);
		let mut array = Array1::from_vec(xs);
		par_merge_sort(array.view_mut(), Item::lt);
		for (a, s) in array.iter().zip(&sorted) {
			assert_eq!(a.index, s.index);
			assert_eq!(a.value, s.value);
		}
	}
}
