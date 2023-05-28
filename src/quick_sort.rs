//! Derivative work of [`core::slice::sort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice::sort`]: https://doc.rust-lang.org/src/core/slice/sort.rs.html

use crate::{
	heap_sort::heap_sort,
	insertion_sort::insertion_sort_shift_left,
	insertion_sort::partial_insertion_sort,
	partition::{break_patterns, choose_pivot, partition, partition_equal},
};
use core::{cmp, mem};
use ndarray::{ArrayViewMut1, Axis, IndexLonger};

/// Sorts `v` using pattern-defeating quicksort, which is *O*(*n* \* log(*n*)) worst-case.
pub fn quick_sort<T, F>(v: ArrayViewMut1<'_, T>, mut is_less: F)
where
	F: FnMut(&T, &T) -> bool,
{
	// Sorting has no meaningful behavior on zero-sized types.
	if mem::size_of::<T>() == 0 {
		return;
	}

	// Limit the number of imbalanced partitions to `floor(log2(len)) + 1`.
	let limit = usize::BITS - v.len().leading_zeros();

	recurse(v, &mut is_less, None, limit);
}

/// Sorts `v` recursively.
///
/// If the slice had a predecessor in the original array, it is specified as `pred`.
///
/// `limit` is the number of allowed imbalanced partitions before switching to `heap_sort`. If zero,
/// this function will immediately switch to heapsort.
fn recurse<'a, T, F>(
	mut v: ArrayViewMut1<'a, T>,
	is_less: &mut F,
	mut pred: Option<&'a T>,
	mut limit: u32,
) where
	F: FnMut(&T, &T) -> bool,
{
	// Slices of up to this length get sorted using insertion sort.
	const MAX_INSERTION: usize = 20;

	// True if the last partitioning was reasonably balanced.
	let mut was_balanced = true;
	// True if the last partitioning didn't shuffle elements (the slice was already partitioned).
	let mut was_partitioned = true;

	loop {
		let len = v.len();

		// Very short slices get sorted using insertion sort.
		if len <= MAX_INSERTION {
			if len >= 2 {
				insertion_sort_shift_left(v, 1, is_less);
			}
			return;
		}

		// If too many bad pivot choices were made, simply fall back to heapsort in order to
		// guarantee `O(n * log(n))` worst-case.
		if limit == 0 {
			heap_sort(v, is_less);
			return;
		}

		// If the last partitioning was imbalanced, try breaking patterns in the slice by shuffling
		// some elements around. Hopefully we'll choose a better pivot this time.
		if !was_balanced {
			break_patterns(v.view_mut());
			limit -= 1;
		}

		// Choose a pivot and try guessing whether the slice is already sorted.
		let (pivot, likely_sorted) = choose_pivot(v.view_mut(), is_less);

		// If the last partitioning was decently balanced and didn't shuffle elements, and if pivot
		// selection predicts the slice is likely already sorted...
		if was_balanced && was_partitioned && likely_sorted {
			// Try identifying several out-of-order elements and shifting them to correct
			// positions. If the slice ends up being completely sorted, we're done.
			if partial_insertion_sort(v.view_mut(), is_less) {
				return;
			}
		}

		// If the chosen pivot is equal to the predecessor, then it's the smallest element in the
		// slice. Partition the slice into elements equal to and elements greater than the pivot.
		// This case is usually hit when the slice contains many duplicate elements.
		if let Some(p) = pred {
			if !is_less(p, &v[pivot]) {
				let mid = partition_equal(v.view_mut(), pivot, is_less);

				// Continue sorting elements greater than the pivot.
				let (_, new_v) = v.split_at(Axis(0), mid);
				v = new_v;
				continue;
			}
		}

		// Partition the slice.
		let (mid, was_p) = partition(v.view_mut(), pivot, is_less);
		was_balanced = cmp::min(mid, len - mid) >= len / 8;
		was_partitioned = was_p;

		// Split the slice into `left`, `pivot`, and `right`.
		let (left, right) = v.split_at(Axis(0), mid);
		let (pivot, right) = right.split_at(Axis(0), 1);
		let pivot = pivot.index(0);

		// Recurse into the shorter side only in order to minimize the total number of recursive
		// calls and consume less stack space. Then just continue with the longer side (this is
		// akin to tail recursion).
		if left.len() < right.len() {
			recurse(left, is_less, pred, limit);
			v = right;
			pred = Some(pivot);
		} else {
			recurse(right, is_less, Some(pivot), limit);
			v = left;
		}
	}
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
	use super::quick_sort;
	use ndarray::Array1;
	use quickcheck_macros::quickcheck;

	#[quickcheck]
	fn sorted(xs: Vec<u32>) {
		let mut sorted = xs.clone();
		sorted.sort_unstable();
		let sorted = Array1::from_vec(sorted);
		let mut array = Array1::from_vec(xs);
		quick_sort(array.view_mut(), &mut u32::lt);
		assert_eq!(array, sorted);
	}
}
