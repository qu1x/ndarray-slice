//! Fast and robust slice-based algorithms (e.g., [sorting], [selection], [search]) for
//! non-contiguous (sub)views into *n*-dimensional arrays. Reimplements algorithms in [`slice`] and
#![cfg_attr(feature = "rayon", doc = "[`rayon::slice`]")]
#![cfg_attr(not(feature = "rayon"), doc = "`rayon::slice`")]
//! for [`ndarray`] with arbitrary memory layout (e.g., non-contiguous).
//!
//! # Example
//!
//! ```
//! use ndarray_slice::{ndarray::arr2, Slice1Ext};
//!
//! // 2-dimensional array of 4 rows and 5 columns.
//! let mut v = arr2(&[[-5, 4, 1, -3,  2],   // row 0, axis 0
//!                    [ 8, 3, 2,  4,  8],   // row 1, axis 0
//!                    [38, 9, 3,  0,  3],   // row 2, axis 0
//!                    [ 4, 9, 0,  8, -1]]); // row 3, axis 0
//! //                    \     \       \
//! //                  column 0 \    column 4         axis 1
//! //                         column 2                axis 1
//!
//! // Mutable subview into the last column.
//! let mut column = v.column_mut(4);
//!
//! // Due to row-major memory layout, columns are non-contiguous
//! // and hence cannot be sorted by viewing them as mutable slices.
//! assert_eq!(column.as_slice_mut(), None);
//!
//! // Instead, sorting is specifically implemented for non-contiguous
//! // mutable (sub)views.
//! column.sort_unstable();
//!
//! assert!(v == arr2(&[[-5, 4, 1, -3, -1],
//!                     [ 8, 3, 2,  4,  2],
//!                     [38, 9, 3,  0,  3],
//!                     [ 4, 9, 0,  8,  8]]));
//! //                                   \
//! //                                 column 4 sorted, others untouched
//! ```
//!
//! # Current Implementation
//!
//! Complexities where *n* is the length of the (sub)view and *m* the count of indices to select.
//!
//! | Resource | Complexity | Sorting (stable) | Sorting (unstable)  | Selection (unstable) | Bulk Selection (unstable) |
//! |----------|------------|------------------|---------------------|----------------------|---------------------------|
//! | Time     | Best       | *O*(*n*)         | *O*(*n*)            | *O*(*n*)             | *O*(*n* log *m*)          |
//! | Time     | Average    | *O*(*n* log *n*) | *O*(*n* log *n*)    | *O*(*n*)             | *O*(*n* log *m*)          |
//! | Time     | Worst      | *O*(*n* log *n*) | *O*(*n* log *n*)    | *O*(*n*)             | *O*(*n* log *m*)          |
//! | Space    | Best       | *O*(1)           | *O*(1)              | *O*(1)               | *O*(*m*)                  |
//! | Space    | Average    | *O*(*n*/2)       | *O*(log *n*)        | *O*(1)               | *O*(*m*+log *m*)          |
//! | Space    | Worst      | *O*(*n*/2)       | *O*(log *n*)        | *O*(1)               | *O*(*m*+log *m*)          |
//!
//!
//! [sorting]: https://en.wikipedia.org/wiki/Sorting_algorithm
//! [selection]: https://en.wikipedia.org/wiki/Selection_algorithm
//! [search]: https://en.wikipedia.org/wiki/Search_algorithm
//!
//! [`sort`]: Slice1Ext::sort
//! [`sort_unstable`]: Slice1Ext::sort_unstable
//! [`select_nth_unstable`]: Slice1Ext::select_nth_unstable
//!
//! # Roadmap
//!
//!   * Add `SliceExt` trait for *n*-dimensional array or (sub)view with methods expecting `Axis` as
//!     their first argument. Comparing methods will always be suffixed with `_by` or `_by_key`
//!     defining how to compare multi-dimensional elements (e.g., rows) along the provided axis of
//!     interest (e.g., columns).
//!
//! # Features
//!
//!   * `alloc` for stable `sort`/`sort_by`/`sort_by_key`. Enabled by `std`.
//!   * `std` for stable `sort_by_cached_key`. Enabled by `default` or `rayon`.
//!   * `rayon` for parallel `par_sort*`/`par_select_many_nth_unstable*`.

#![deny(
	missing_docs,
	rustdoc::broken_intra_doc_links,
	rustdoc::missing_crate_level_docs
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![cfg_attr(miri, feature(strict_provenance), feature(maybe_uninit_slice))]

mod heap_sort;
mod insertion_sort;
mod merge_sort;
mod partition;
mod partition_dedup;
mod quick_sort;
mod stable_sort;

#[cfg(feature = "rayon")]
mod par;
#[cfg(feature = "rayon")]
use par::{
	merge_sort::par_merge_sort, partition::par_partition_at_indices, quick_sort::par_quick_sort,
};

#[cfg(feature = "alloc")]
use crate::stable_sort::stable_sort;

use crate::{
	partition::{is_sorted, partition_at_index, partition_at_indices, reverse},
	partition_dedup::partition_dedup,
	quick_sort::quick_sort,
};
use core::cmp::Ordering::{self, Greater, Less};
use ndarray::{ArrayBase, ArrayViewMut1, Data, DataMut, Ix1};

pub use ndarray;

// Derivative work of [`core::slice`] licensed under `MIT OR Apache-2.0`.
//
// [`core::slice`]: https://doc.rust-lang.org/src/core/slice/mod.rs.html

/// Extension trait for 1-dimensional [`ArrayBase<S, Ix1>`](`ArrayBase`) array or (sub)view with
/// arbitrary memory layout (e.g., non-contiguous) providing methods (e.g., [sorting], [selection],
/// [search]) similar to [`slice`] and
#[cfg_attr(feature = "rayon", doc = "[`rayon::slice`].")]
#[cfg_attr(not(feature = "rayon"), doc = "`rayon::slice`.")]
///
/// [sorting]: https://en.wikipedia.org/wiki/Sorting_algorithm
/// [selection]: https://en.wikipedia.org/wiki/Selection_algorithm
/// [search]: https://en.wikipedia.org/wiki/Search_algorithm
pub trait Slice1Ext<A, S>
where
	S: Data<Elem = A>,
{
	/// Sorts the array in parallel.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* log *n*) worst-case.
	///
	/// When applicable, unstable sorting is preferred because it is generally faster than stable
	/// sorting and it doesn't allocate auxiliary memory.
	/// See [`par_sort_unstable`](Slice1Ext::par_sort_unstable).
	///
	/// # Current Implementation
	///
	/// The current algorithm is an adaptive, iterative merge sort inspired by
	/// [timsort](https://en.wikipedia.org/wiki/Timsort).
	/// It is designed to be very fast in cases where the array is nearly sorted, or consists of
	/// two or more sorted sequences concatenated one after another.
	///
	/// Also, it allocates temporary storage half the size of `self`, but for short arrays a
	/// non-allocating insertion sort is used instead.
	///
	/// In order to sort the array in parallel, the array is first divided into smaller chunks and
	/// all chunks are sorted in parallel. Then, adjacent chunks that together form non-descending
	/// or descending runs are concatenated. Finally, the remaining chunks are merged together using
	/// parallel subdivision of chunks and parallel merge operation.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5, 4, 1, -3, 2]);
	///
	/// v.par_sort();
	/// assert!(v == arr1(&[-5, -3, 1, 2, 4]));
	/// ```
	#[cfg(feature = "rayon")]
	fn par_sort(&mut self)
	where
		A: Ord + Send,
		S: DataMut;
	/// Sorts the array in parallel with a comparator function.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* log *n*) worst-case.
	///
	/// The comparator function must define a total ordering for the elements in the array. If
	/// the ordering is not total, the order of the elements is unspecified. An order is a
	/// total order if it is (for all `a`, `b` and `c`):
	///
	/// * total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true, and
	/// * transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
	///
	/// For example, while [`f64`] doesn't implement [`Ord`] because `NaN != NaN`, we can use
	/// `partial_cmp` as our sort function when we know the array doesn't contain a `NaN`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut floats = arr1(&[5f64, 4.0, 1.0, 3.0, 2.0]);
	/// floats.par_sort_by(|a, b| a.partial_cmp(b).unwrap());
	/// assert_eq!(floats, arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]));
	/// ```
	///
	/// When applicable, unstable sorting is preferred because it is generally faster than stable
	/// sorting and it doesn't allocate auxiliary memory.
	/// See [`par_sort_unstable_by`](Slice1Ext::par_sort_unstable_by).
	///
	/// # Current Implementation
	///
	/// The current algorithm is an adaptive, iterative merge sort inspired by
	/// [timsort](https://en.wikipedia.org/wiki/Timsort).
	/// It is designed to be very fast in cases where the array is nearly sorted, or consists of
	/// two or more sorted sequences concatenated one after another.
	///
	/// Also, it allocates temporary storage half the size of `self`, but for short arrays a
	/// non-allocating insertion sort is used instead.
	///
	/// In order to sort the array in parallel, the array is first divided into smaller chunks and
	/// all chunks are sorted in parallel. Then, adjacent chunks that together form non-descending
	/// or descending runs are concatenated. Finally, the remaining chunks are merged together using
	/// parallel subdivision of chunks and parallel merge operation.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[5, 4, 1, 3, 2]);
	/// v.par_sort_by(|a, b| a.cmp(b));
	/// assert!(v == arr1(&[1, 2, 3, 4, 5]));
	///
	/// // reverse sorting
	/// v.par_sort_by(|a, b| b.cmp(a));
	/// assert!(v == arr1(&[5, 4, 3, 2, 1]));
	/// ```
	#[cfg(feature = "rayon")]
	fn par_sort_by<F>(&mut self, compare: F)
	where
		A: Send,
		F: Fn(&A, &A) -> Ordering + Sync,
		S: DataMut;
	/// Sorts the array in parallel with a key extraction function.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*mn* log *n*)
	/// worst-case, where the key function is *O*(*m*).
	///
	/// For expensive key functions (e.g. functions that are not simple property accesses or
	/// basic operations), [`par_sort_by_cached_key`](Slice1Ext::par_sort_by_cached_key) is likely to be
	/// significantly faster, as it does not recompute element keys."
	///
	/// When applicable, unstable sorting is preferred because it is generally faster than stable
	/// sorting and it doesn't allocate auxiliary memory.
	/// See [`par_sort_unstable_by_key`](Slice1Ext::par_sort_unstable_by_key).
	///
	/// # Current Implementation
	///
	/// The current algorithm is an adaptive, iterative merge sort inspired by
	/// [timsort](https://en.wikipedia.org/wiki/Timsort).
	/// It is designed to be very fast in cases where the array is nearly sorted, or consists of
	/// two or more sorted sequences concatenated one after another.
	///
	/// Also, it allocates temporary storage half the size of `self`, but for short arrays a
	/// non-allocating insertion sort is used instead.
	///
	/// In order to sort the array in parallel, the array is first divided into smaller chunks and
	/// all chunks are sorted in parallel. Then, adjacent chunks that together form non-descending
	/// or descending runs are concatenated. Finally, the remaining chunks are merged together using
	/// parallel subdivision of chunks and parallel merge operation.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// v.par_sort_by_key(|k| k.abs());
	/// assert!(v == arr1(&[1, 2, -3, 4, -5]));
	/// ```
	#[cfg(feature = "rayon")]
	fn par_sort_by_key<K, F>(&mut self, f: F)
	where
		A: Send,
		K: Ord,
		F: Fn(&A) -> K + Sync,
		S: DataMut;
	/// Sorts the array in parallel with a key extraction function.
	///
	/// During sorting, the key function is called at most once per element, by using
	/// temporary storage to remember the results of key evaluation.
	/// The order of calls to the key function is unspecified and may change in future versions
	/// of the standard library.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*mn* + *n* log *n*)
	/// worst-case, where the key function is *O*(*m*).
	///
	/// For simple key functions (e.g., functions that are property accesses or
	/// basic operations), [`par_sort_by_key`](Slice1Ext::par_sort_by_key) is likely to be
	/// faster.
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// In the worst case, the algorithm allocates temporary storage in a `Vec<(K, usize)>` the
	/// length of the array.
	///
	/// In order to sort the array in parallel, the array is first divided into smaller chunks and
	/// all chunks are sorted in parallel. Then, adjacent chunks that together form non-descending
	/// or descending runs are concatenated. Finally, the remaining chunks are merged together using
	/// parallel subdivision of chunks and parallel merge operation.
	///
	/// # Examples
	///
	/// ```
	/// # if !cfg!(miri) {
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 32, -3, 2]);
	///
	/// v.par_sort_by_cached_key(|k| k.to_string());
	/// assert!(v == arr1(&[-3, -5, 2, 32, 4]));
	/// # }
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	#[cfg(feature = "rayon")]
	fn par_sort_by_cached_key<K, F>(&mut self, f: F)
	where
		A: Send + Sync,
		F: Fn(&A) -> K + Sync,
		K: Ord + Send,
		S: DataMut;

	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* log *n*) worst-case.
	///
	/// When applicable, unstable sorting is preferred because it is generally faster than stable
	/// sorting and it doesn't allocate auxiliary memory.
	/// See [`sort_unstable`](Slice1Ext::sort_unstable).
	///
	/// # Current Implementation
	///
	/// The current algorithm is an adaptive, iterative merge sort inspired by
	/// [timsort](https://en.wikipedia.org/wiki/Timsort).
	/// It is designed to be very fast in cases where the array is nearly sorted, or consists of
	/// two or more sorted sequences concatenated one after another.
	///
	/// Also, it allocates temporary storage half the size of `self`, but for short arrays a
	/// non-allocating insertion sort is used instead.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5, 4, 1, -3, 2]);
	///
	/// v.sort();
	/// assert!(v == arr1(&[-5, -3, 1, 2, 4]));
	/// ```
	#[cfg(feature = "alloc")]
	fn sort(&mut self)
	where
		A: Ord,
		S: DataMut;
	/// Sorts the array with a comparator function.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*n* log *n*) worst-case.
	///
	/// The comparator function must define a total ordering for the elements in the array. If
	/// the ordering is not total, the order of the elements is unspecified. An order is a
	/// total order if it is (for all `a`, `b` and `c`):
	///
	/// * total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true, and
	/// * transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
	///
	/// For example, while [`f64`] doesn't implement [`Ord`] because `NaN != NaN`, we can use
	/// `partial_cmp` as our sort function when we know the array doesn't contain a `NaN`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut floats = arr1(&[5f64, 4.0, 1.0, 3.0, 2.0]);
	/// floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
	/// assert_eq!(floats, arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]));
	/// ```
	///
	/// When applicable, unstable sorting is preferred because it is generally faster than stable
	/// sorting and it doesn't allocate auxiliary memory.
	/// See [`sort_unstable_by`](Slice1Ext::sort_unstable_by).
	///
	/// # Current Implementation
	///
	/// The current algorithm is an adaptive, iterative merge sort inspired by
	/// [timsort](https://en.wikipedia.org/wiki/Timsort).
	/// It is designed to be very fast in cases where the array is nearly sorted, or consists of
	/// two or more sorted sequences concatenated one after another.
	///
	/// Also, it allocates temporary storage half the size of `self`, but for short arrays a
	/// non-allocating insertion sort is used instead.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[5, 4, 1, 3, 2]);
	/// v.sort_by(|a, b| a.cmp(b));
	/// assert!(v == arr1(&[1, 2, 3, 4, 5]));
	///
	/// // reverse sorting
	/// v.sort_by(|a, b| b.cmp(a));
	/// assert!(v == arr1(&[5, 4, 3, 2, 1]));
	/// ```
	#[cfg(feature = "alloc")]
	fn sort_by<F>(&mut self, compare: F)
	where
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut;
	/// Sorts the array with a key extraction function.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*mn* log *n*)
	/// worst-case, where the key function is *O*(*m*).
	///
	#[cfg_attr(
		feature = "std",
		doc = "\
	For expensive key functions (e.g. functions that are not simple property accesses or
	basic operations), [`sort_by_cached_key`](Slice1Ext::sort_by_cached_key) is likely to be
	significantly faster, as it does not recompute element keys."
	)]
	///
	/// When applicable, unstable sorting is preferred because it is generally faster than stable
	/// sorting and it doesn't allocate auxiliary memory.
	/// See [`sort_unstable_by_key`](Slice1Ext::sort_unstable_by_key).
	///
	/// # Current Implementation
	///
	/// The current algorithm is an adaptive, iterative merge sort inspired by
	/// [timsort](https://en.wikipedia.org/wiki/Timsort).
	/// It is designed to be very fast in cases where the array is nearly sorted, or consists of
	/// two or more sorted sequences concatenated one after another.
	///
	/// Also, it allocates temporary storage half the size of `self`, but for short arrays a
	/// non-allocating insertion sort is used instead.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// v.sort_by_key(|k| k.abs());
	/// assert!(v == arr1(&[1, 2, -3, 4, -5]));
	/// ```
	#[cfg(feature = "alloc")]
	fn sort_by_key<K, F>(&mut self, f: F)
	where
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut;

	/// Sorts the array with a key extraction function.
	///
	/// During sorting, the key function is called at most once per element, by using
	/// temporary storage to remember the results of key evaluation.
	/// The order of calls to the key function is unspecified and may change in future versions
	/// of the standard library.
	///
	/// This sort is stable (i.e., does not reorder equal elements) and *O*(*mn* + *n* log *n*)
	/// worst-case, where the key function is *O*(*m*).
	///
	/// For simple key functions (e.g., functions that are property accesses or
	/// basic operations), [`sort_by_key`](Slice1Ext::sort_by_key) is likely to be
	/// faster.
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// In the worst case, the algorithm allocates temporary storage in a `Vec<(K, usize)>` the
	/// length of the array.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 32, -3, 2]);
	///
	/// v.sort_by_cached_key(|k| k.to_string());
	/// assert!(v == arr1(&[-3, -5, 2, 32, 4]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	#[cfg(feature = "std")]
	fn sort_by_cached_key<K, F>(&mut self, f: F)
	where
		F: FnMut(&A) -> K,
		K: Ord,
		S: DataMut;

	/// Sorts the array in parallel, but might not preserve the order of equal elements.
	///
	/// This sort is unstable (i.e., may reorder equal elements), in-place
	/// (i.e., does not allocate), and *O*(*n* log *n*) worst-case.
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// It is typically faster than stable sorting, except in a few special cases, e.g., when the
	/// array consists of several concatenated sorted sequences.
	///
	/// All quicksorts work in two stages: partitioning into two halves followed by recursive
	/// calls. The partitioning phase is sequential, but the two recursive calls are performed in
	/// parallel.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5, 4, 1, -3, 2]);
	///
	/// v.par_sort_unstable();
	/// assert!(v == arr1(&[-5, -3, 1, 2, 4]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	#[cfg(feature = "rayon")]
	fn par_sort_unstable(&mut self)
	where
		A: Ord + Send,
		S: DataMut;
	/// Sorts the array in parallel with a comparator function, but might not preserve the order of equal
	/// elements.
	///
	/// This sort is unstable (i.e., may reorder equal elements), in-place
	/// (i.e., does not allocate), and *O*(*n* log *n*) worst-case.
	///
	/// The comparator function must define a total ordering for the elements in the array. If
	/// the ordering is not total, the order of the elements is unspecified. An order is a
	/// total order if it is (for all `a`, `b` and `c`):
	///
	/// * total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true, and
	/// * transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
	///
	/// For example, while [`f64`] doesn't implement [`Ord`] because `NaN != NaN`, we can use
	/// `partial_cmp` as our sort function when we know the array doesn't contain a `NaN`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut floats = arr1(&[5f64, 4.0, 1.0, 3.0, 2.0]);
	/// floats.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
	/// assert_eq!(floats, arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]));
	/// ```
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// It is typically faster than stable sorting, except in a few special cases, e.g., when the
	/// array consists of several concatenated sorted sequences.
	///
	/// All quicksorts work in two stages: partitioning into two halves followed by recursive
	/// calls. The partitioning phase is sequential, but the two recursive calls are performed in
	/// parallel.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[5, 4, 1, 3, 2]);
	/// v.par_sort_unstable_by(|a, b| a.cmp(b));
	/// assert!(v == arr1(&[1, 2, 3, 4, 5]));
	///
	/// // reverse sorting
	/// v.par_sort_unstable_by(|a, b| b.cmp(a));
	/// assert!(v == arr1(&[5, 4, 3, 2, 1]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	#[cfg(feature = "rayon")]
	fn par_sort_unstable_by<F>(&mut self, compare: F)
	where
		A: Send,
		F: Fn(&A, &A) -> Ordering + Sync,
		S: DataMut;
	/// Sorts the array in parallel with a key extraction function, but might not preserve the order of equal
	/// elements.
	///
	/// This sort is unstable (i.e., may reorder equal elements), in-place
	/// (i.e., does not allocate), and *O*(*mn* log *n*) worst-case, where the key function is
	/// *O*(*m*).
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// Due to its key calling strategy, [`par_sort_unstable_by_key`](#method.par_sort_unstable_by_key)
	/// is likely to be slower than [`par_sort_by_cached_key`](#method.par_sort_by_cached_key) in
	/// cases where the key function is expensive.
	///
	/// All quicksorts work in two stages: partitioning into two halves followed by recursive
	/// calls. The partitioning phase is sequential, but the two recursive calls are performed in
	/// parallel.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// v.par_sort_unstable_by_key(|k| k.abs());
	/// assert!(v == arr1(&[1, 2, -3, 4, -5]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	#[cfg(feature = "rayon")]
	fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
	where
		A: Send,
		K: Ord,
		F: Fn(&A) -> K + Sync,
		S: DataMut;

	/// Sorts the array, but might not preserve the order of equal elements.
	///
	/// This sort is unstable (i.e., may reorder equal elements), in-place
	/// (i.e., does not allocate), and *O*(*n* log *n*) worst-case.
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// It is typically faster than stable sorting, except in a few special cases, e.g., when the
	/// array consists of several concatenated sorted sequences.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5, 4, 1, -3, 2]);
	///
	/// v.sort_unstable();
	/// assert!(v == arr1(&[-5, -3, 1, 2, 4]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	fn sort_unstable(&mut self)
	where
		A: Ord,
		S: DataMut;
	/// Sorts the array with a comparator function, but might not preserve the order of equal
	/// elements.
	///
	/// This sort is unstable (i.e., may reorder equal elements), in-place
	/// (i.e., does not allocate), and *O*(*n* log *n*) worst-case.
	///
	/// The comparator function must define a total ordering for the elements in the array. If
	/// the ordering is not total, the order of the elements is unspecified. An order is a
	/// total order if it is (for all `a`, `b` and `c`):
	///
	/// * total and antisymmetric: exactly one of `a < b`, `a == b` or `a > b` is true, and
	/// * transitive, `a < b` and `b < c` implies `a < c`. The same must hold for both `==` and `>`.
	///
	/// For example, while [`f64`] doesn't implement [`Ord`] because `NaN != NaN`, we can use
	/// `partial_cmp` as our sort function when we know the array doesn't contain a `NaN`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut floats = arr1(&[5f64, 4.0, 1.0, 3.0, 2.0]);
	/// floats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
	/// assert_eq!(floats, arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]));
	/// ```
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// It is typically faster than stable sorting, except in a few special cases, e.g., when the
	/// array consists of several concatenated sorted sequences.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[5, 4, 1, 3, 2]);
	/// v.sort_unstable_by(|a, b| a.cmp(b));
	/// assert!(v == arr1(&[1, 2, 3, 4, 5]));
	///
	/// // reverse sorting
	/// v.sort_unstable_by(|a, b| b.cmp(a));
	/// assert!(v == arr1(&[5, 4, 3, 2, 1]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	fn sort_unstable_by<F>(&mut self, compare: F)
	where
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut;
	/// Sorts the array with a key extraction function, but might not preserve the order of equal
	/// elements.
	///
	/// This sort is unstable (i.e., may reorder equal elements), in-place
	/// (i.e., does not allocate), and *O*(*mn* log *n*) worst-case, where the key function is
	/// *O*(*m*).
	///
	/// # Current Implementation
	///
	/// The current algorithm is based on [pattern-defeating quicksort][pdqsort] by Orson Peters,
	/// which combines the fast average case of randomized quicksort with the fast worst case of
	/// heapsort, while achieving linear time on arrays with certain patterns. It uses some
	/// randomization to avoid degenerate cases, but with a fixed seed to always provide
	/// deterministic behavior.
	///
	/// Due to its key calling strategy, [`sort_unstable_by_key`](#method.sort_unstable_by_key)
	/// is likely to be slower than [`sort_by_cached_key`](#method.sort_by_cached_key) in
	/// cases where the key function is expensive.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// v.sort_unstable_by_key(|k| k.abs());
	/// assert!(v == arr1(&[1, 2, -3, 4, -5]));
	/// ```
	///
	/// [pdqsort]: https://github.com/orlp/pdqsort
	fn sort_unstable_by_key<K, F>(&mut self, f: F)
	where
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut;

	/// Checks if the elements of this array are sorted.
	///
	/// That is, for each element `a` and its following element `b`, `a <= b` must hold. If the
	/// array yields exactly zero or one element, `true` is returned.
	///
	/// Note that if `Self::Item` is only `PartialOrd`, but not `Ord`, the above definition
	/// implies that this function returns `false` if any two consecutive items are not
	/// comparable.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let empty: [i32; 0] = [];
	///
	/// assert!(arr1(&[1, 2, 2, 9]).is_sorted());
	/// assert!(!arr1(&[1, 3, 2, 4]).is_sorted());
	/// assert!(arr1(&[0]).is_sorted());
	/// assert!(arr1(&empty).is_sorted());
	/// assert!(!arr1(&[0.0, 1.0, f32::NAN]).is_sorted());
	/// ```
	#[must_use]
	fn is_sorted(&self) -> bool
	where
		A: PartialOrd;
	/// Checks if the elements of this array are sorted using the given comparator function.
	///
	/// Instead of using `PartialOrd::partial_cmp`, this function uses the given `compare`
	/// function to determine the ordering of two elements. Apart from that, it's equivalent to
	/// [`is_sorted`]; see its documentation for more information.
	///
	/// [`is_sorted`]: Slice1Ext::is_sorted
	#[must_use]
	fn is_sorted_by<F>(&self, compare: F) -> bool
	where
		F: FnMut(&A, &A) -> Option<Ordering>;
	/// Checks if the elements of this array are sorted using the given key extraction function.
	///
	/// Instead of comparing the array's elements directly, this function compares the keys of the
	/// elements, as determined by `f`. Apart from that, it's equivalent to [`is_sorted`]; see its
	/// documentation for more information.
	///
	/// [`is_sorted`]: Slice1Ext::is_sorted
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// assert!(arr1(&["c", "bb", "aaa"]).is_sorted_by_key(|s| s.len()));
	/// assert!(!arr1(&[-2i32, -1, 0, 3]).is_sorted_by_key(|n| n.abs()));
	/// ```
	#[must_use]
	fn is_sorted_by_key<F, K>(&self, f: F) -> bool
	where
		F: FnMut(&A) -> K,
		K: PartialOrd;

	/// Reorder the array in parallel such that the elements at `indices` are at their final sorted position.
	///
	/// Bulk version of [`select_nth_unstable`] extending `collection` with `&mut element`
	/// in the order of `indices`. The provided `indices` must be sorted and unique which can be
	/// achieved with [`par_sort_unstable`] followed by [`partition_dedup`].
	///
	/// # Current Implementation
	///
	/// The current algorithm chooses `at = indices.len() / 2` as pivot index and recurses in parallel into the
	/// left and right subviews of `indices` (i.e., `..at` and `at + 1..`) with corresponding left
	/// and right subviews of `self` (i.e., `..pivot` and `pivot + 1..`) where `pivot = indices[at]`.
	/// Requiring `indices` to be already sorted, reduces the time complexity in the length *m* of
	/// `indices` from *O*(*m*) to *O*(log *m*) compared to invoking [`select_nth_unstable`] on the
	/// full view of `self` for each index.
	///
	/// # Panics
	///
	/// Panics when any `indices[i] >= len()`, meaning it always panics on empty arrays. Panics
	/// when `indices` is unsorted or contains duplicates.
	///
	/// [`par_sort_unstable`]: Slice1Ext::par_sort_unstable
	/// [`partition_dedup`]: Slice1Ext::partition_dedup
	/// [`select_nth_unstable`]: Slice1Ext::select_nth_unstable
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2, 9, 3, 4, 0]);
	///
	/// // Find values at following indices.
	/// let indices = arr1(&[1, 4, 6]);
	///
	/// let mut values = Vec::new();
	/// v.par_select_many_nth_unstable(&indices, &mut values);
	///
	/// assert!(values == [&-3, &2, &4]);
	/// ```
	#[cfg(feature = "rayon")]
	fn par_select_many_nth_unstable<'a, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut Vec<&'a mut A>,
	) where
		A: Ord + Send,
		S: DataMut,
		S2: Data<Elem = usize> + Sync;
	/// Reorder the array in parallel with a comparator function such that the elements at `indices` are at
	/// their final sorted position.
	///
	/// Bulk version of [`select_nth_unstable_by`] extending `collection` with `&mut element`
	/// in the order of `indices`. The provided `indices` must be sorted and unique which can be
	/// achieved with [`par_sort_unstable`] followed by [`partition_dedup`].
	///
	/// # Current Implementation
	///
	/// The current algorithm chooses `at = indices.len() / 2` as pivot index and recurses in parallel into the
	/// left and right subviews of `indices` (i.e., `..at` and `at + 1..`) with corresponding left
	/// and right subviews of `self` (i.e., `..pivot` and `pivot + 1..`) where `pivot = indices[at]`.
	/// Requiring `indices` to be already sorted, reduces the time complexity in the length *m* of
	/// `indices` from *O*(*m*) to *O*(log *m*) compared to invoking [`select_nth_unstable_by`] on the
	/// full view of `self` for each index.
	///
	/// # Panics
	///
	/// Panics when any `indices[i] >= len()`, meaning it always panics on empty arrays. Panics
	/// when `indices` is unsorted or contains duplicates.
	///
	/// [`par_sort_unstable`]: Slice1Ext::par_sort_unstable
	/// [`partition_dedup`]: Slice1Ext::partition_dedup
	/// [`select_nth_unstable_by`]: Slice1Ext::select_nth_unstable_by
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	/// use std::collections::HashMap;
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2, 9, 3, 4, 0]);
	///
	/// // Find values at following indices.
	/// let indices = arr1(&[1, 4, 6]);
	///
	/// let mut values = Vec::new();
	/// v.par_select_many_nth_unstable_by(&indices, &mut values, |a, b| b.cmp(a));
	///
	/// assert!(values == [&4, &2, &0]);
	/// ```
	#[cfg(feature = "rayon")]
	fn par_select_many_nth_unstable_by<'a, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut Vec<&'a mut A>,
		compare: F,
	) where
		A: Send,
		F: Fn(&A, &A) -> Ordering + Sync,
		S: DataMut,
		S2: Data<Elem = usize> + Sync;
	/// Reorder the array in parallel with a key extraction function such that the elements at `indices` are at
	/// their final sorted position.
	///
	/// Bulk version of [`select_nth_unstable_by_key`] extending `collection` with `&mut element`
	/// in the order of `indices`. The provided `indices` must be sorted and unique which can be
	/// achieved with [`par_sort_unstable`] followed by [`partition_dedup`].
	///
	/// # Current Implementation
	///
	/// The current algorithm chooses `at = indices.len() / 2` as pivot index and recurses in parallel into the
	/// left and right subviews of `indices` (i.e., `..at` and `at + 1..`) with corresponding left
	/// and right subviews of `self` (i.e., `..pivot` and `pivot + 1..`) where `pivot = indices[at]`.
	/// Requiring `indices` to be already sorted, reduces the time complexity in the length *m* of
	/// `indices` from *O*(*m*) to *O*(log *m*) compared to invoking [`select_nth_unstable_by_key`] on the
	/// full view of `self` for each index.
	///
	/// # Panics
	///
	/// Panics when any `indices[i] >= len()`, meaning it always panics on empty arrays. Panics
	/// when `indices` is unsorted or contains duplicates.
	///
	/// [`par_sort_unstable`]: Slice1Ext::par_sort_unstable
	/// [`partition_dedup`]: Slice1Ext::partition_dedup
	/// [`select_nth_unstable_by_key`]: Slice1Ext::select_nth_unstable_by_key
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	/// use std::collections::HashMap;
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2, 9, 3, 4, 0]);
	///
	/// // Find values at following indices.
	/// let indices = arr1(&[1, 4, 6]);
	///
	/// let mut values = Vec::new();
	/// v.par_select_many_nth_unstable_by_key(&indices, &mut values, |&a| a.abs());
	///
	/// assert!(values == [&1, &3, &4]);
	/// ```
	#[cfg(feature = "rayon")]
	fn par_select_many_nth_unstable_by_key<'a, K, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut Vec<&'a mut A>,
		f: F,
	) where
		A: Send,
		K: Ord,
		F: Fn(&A) -> K + Sync,
		S: DataMut,
		S2: Data<Elem = usize> + Sync;

	/// Reorder the array such that the elements at `indices` are at their final sorted position.
	///
	/// Bulk version of [`select_nth_unstable`] extending `collection` with `(index, &mut element)`
	/// tuples in the order of `indices`. The provided `indices` must be sorted and unique which can
	/// be achieved with [`sort_unstable`] followed by [`partition_dedup`].
	///
	/// # Current Implementation
	///
	/// The current algorithm chooses `at = indices.len() / 2` as pivot index and recurses into the
	/// left and right subviews of `indices` (i.e., `..at` and `at + 1..`) with corresponding left
	/// and right subviews of `self` (i.e., `..pivot` and `pivot + 1..`) where `pivot = indices[at]`.
	/// Requiring `indices` to be already sorted, reduces the time complexity in the length *m* of
	/// `indices` from *O*(*m*) to *O*(log *m*) compared to invoking [`select_nth_unstable`] on the
	/// full view of `self` for each index.
	///
	/// # Panics
	///
	/// Panics when any `indices[i] >= len()`, meaning it always panics on empty arrays. Panics
	/// when `indices` is unsorted or contains duplicates.
	///
	/// [`sort_unstable`]: Slice1Ext::sort_unstable
	/// [`partition_dedup`]: Slice1Ext::partition_dedup
	/// [`select_nth_unstable`]: Slice1Ext::select_nth_unstable
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	/// use std::collections::HashMap;
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2, 9, 3, 4, 0]);
	///
	/// // Find values at following indices.
	/// let indices = arr1(&[1, 4, 6]);
	///
	/// let mut map = HashMap::new();
	/// v.select_many_nth_unstable(&indices, &mut map);
	/// let values = indices.map(|index| *map[index]);
	///
	/// assert!(values == arr1(&[-3, 2, 4]));
	/// ```
	fn select_many_nth_unstable<'a, E, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut E,
	) where
		A: Ord + 'a,
		E: Extend<(usize, &'a mut A)>,
		S: DataMut,
		S2: Data<Elem = usize>;
	/// Reorder the array with a comparator function such that the elements at `indices` are at
	/// their final sorted position.
	///
	/// Bulk version of [`select_nth_unstable_by`] extending `collection` with `(index, &mut element)`
	/// tuples in the order of `indices`. The provided `indices` must be sorted and unique which can
	/// be achieved with [`sort_unstable`] followed by [`partition_dedup`].
	///
	/// # Current Implementation
	///
	/// The current algorithm chooses `at = indices.len() / 2` as pivot index and recurses into the
	/// left and right subviews of `indices` (i.e., `..at` and `at + 1..`) with corresponding left
	/// and right subviews of `self` (i.e., `..pivot` and `pivot + 1..`) where `pivot = indices[at]`.
	/// Requiring `indices` to be already sorted, reduces the time complexity in the length *m* of
	/// `indices` from *O*(*m*) to *O*(log *m*) compared to invoking [`select_nth_unstable_by`] on the
	/// full view of `self` for each index.
	///
	/// # Panics
	///
	/// Panics when any `indices[i] >= len()`, meaning it always panics on empty arrays. Panics
	/// when `indices` is unsorted or contains duplicates.
	///
	/// [`sort_unstable`]: Slice1Ext::sort_unstable
	/// [`partition_dedup`]: Slice1Ext::partition_dedup
	/// [`select_nth_unstable_by`]: Slice1Ext::select_nth_unstable_by
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	/// use std::collections::HashMap;
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2, 9, 3, 4, 0]);
	///
	/// // Find values at following indices.
	/// let indices = arr1(&[1, 4, 6]);
	///
	/// let mut map = HashMap::new();
	/// v.select_many_nth_unstable_by(&indices, &mut map, |a, b| b.cmp(a));
	/// let values = indices.map(|index| *map[index]);
	///
	/// assert!(values == arr1(&[4, 2, 0]));
	/// ```
	fn select_many_nth_unstable_by<'a, E, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut E,
		compare: F,
	) where
		A: 'a,
		E: Extend<(usize, &'a mut A)>,
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut,
		S2: Data<Elem = usize>;
	/// Reorder the array with a key extraction function such that the elements at `indices` are at
	/// their final sorted position.
	///
	/// Bulk version of [`select_nth_unstable_by_key`] extending `collection` with `(index, &mut element)`
	/// tuples in the order of `indices`. The provided `indices` must be sorted and unique which can
	/// be achieved with [`sort_unstable`] followed by [`partition_dedup`].
	///
	/// # Current Implementation
	///
	/// The current algorithm chooses `at = indices.len() / 2` as pivot index and recurses into the
	/// left and right subviews of `indices` (i.e., `..at` and `at + 1..`) with corresponding left
	/// and right subviews of `self` (i.e., `..pivot` and `pivot + 1..`) where `pivot = indices[at]`.
	/// Requiring `indices` to be already sorted, reduces the time complexity in the length *m* of
	/// `indices` from *O*(*m*) to *O*(log *m*) compared to invoking [`select_nth_unstable_by_key`] on the
	/// full view of `self` for each index.
	///
	/// # Panics
	///
	/// Panics when any `indices[i] >= len()`, meaning it always panics on empty arrays. Panics
	/// when `indices` is unsorted or contains duplicates.
	///
	/// [`sort_unstable`]: Slice1Ext::sort_unstable
	/// [`partition_dedup`]: Slice1Ext::partition_dedup
	/// [`select_nth_unstable_by_key`]: Slice1Ext::select_nth_unstable_by_key
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	/// use std::collections::HashMap;
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2, 9, 3, 4, 0]);
	///
	/// // Find values at following indices.
	/// let indices = arr1(&[1, 4, 6]);
	///
	/// let mut map = HashMap::new();
	/// v.select_many_nth_unstable_by_key(&indices, &mut map, |&a| a.abs());
	/// let values = indices.map(|index| *map[index]);
	///
	/// assert!(values == arr1(&[1, 3, 4]));
	/// ```
	fn select_many_nth_unstable_by_key<'a, E, K, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut E,
		f: F,
	) where
		A: 'a,
		E: Extend<(usize, &'a mut A)>,
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut,
		S2: Data<Elem = usize>;
	/// Reorder the array such that the element at `index` is at its final sorted position.
	///
	/// This reordering has the additional property that any value at position `i < index` will be
	/// less than or equal to any value at a position `j > index`. Additionally, this reordering is
	/// unstable (i.e. any number of equal elements may end up at position `index`), in-place
	/// (i.e. does not allocate), and runs in *O*(*n*) time.
	/// This function is also known as "kth element" in other libraries.
	///
	/// It returns a triplet of the following from the reordered array:
	/// the subarray prior to `index`, the element at `index`, and the subarray after `index`;
	/// accordingly, the values in those two subarrays will respectively all be less-than-or-equal-to
	/// and greater-than-or-equal-to the value of the element at `index`.
	///
	/// # Current Implementation
	///
	/// The current algorithm is an introselect implementation based on Pattern Defeating Quicksort, which is also
	/// the basis for [`sort_unstable`]. The fallback algorithm is Median of Medians using Tukey's Ninther for
	/// pivot selection, which guarantees linear runtime for all inputs.
	///
	/// [`sort_unstable`]: Slice1Ext::sort_unstable
	///
	/// # Panics
	///
	/// Panics when `index >= len()`, meaning it always panics on empty arrays.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// // Find the median
	/// v.select_nth_unstable(2);
	///
	/// // We are only guaranteed the array will be one of the following, based on the way we sort
	/// // about the specified index.
	/// assert!(v == arr1(&[-3, -5, 1, 2, 4]) ||
	///         v == arr1(&[-5, -3, 1, 2, 4]) ||
	///         v == arr1(&[-3, -5, 1, 4, 2]) ||
	///         v == arr1(&[-5, -3, 1, 4, 2]));
	/// ```
	#[must_use]
	fn select_nth_unstable(
		&mut self,
		index: usize,
	) -> (ArrayViewMut1<'_, A>, &mut A, ArrayViewMut1<'_, A>)
	where
		A: Ord,
		S: DataMut;
	/// Reorder the array with a comparator function such that the element at `index` is at its
	/// final sorted position.
	///
	/// This reordering has the additional property that any value at position `i < index` will be
	/// less than or equal to any value at a position `j > index` using the comparator function.
	/// Additionally, this reordering is unstable (i.e. any number of equal elements may end up at
	/// position `index`), in-place (i.e. does not allocate), and runs in *O*(*n*) time.
	/// This function is also known as "kth element" in other libraries.
	///
	/// It returns a triplet of the following from
	/// the array reordered according to the provided comparator function: the subarray prior to
	/// `index`, the element at `index`, and the subarray after `index`; accordingly, the values in
	/// those two subarrays will respectively all be less-than-or-equal-to and greater-than-or-equal-to
	/// the value of the element at `index`.
	///
	/// # Current Implementation
	///
	/// The current algorithm is an introselect implementation based on Pattern Defeating Quicksort, which is also
	/// the basis for [`sort_unstable`]. The fallback algorithm is Median of Medians using Tukey's Ninther for
	/// pivot selection, which guarantees linear runtime for all inputs.
	///
	/// [`sort_unstable`]: Slice1Ext::sort_unstable
	///
	/// # Panics
	///
	/// Panics when `index >= len()`, meaning it always panics on empty arrays.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// // Find the median as if the array were sorted in descending order.
	/// v.select_nth_unstable_by(2, |a, b| b.cmp(a));
	///
	/// // We are only guaranteed the array will be one of the following, based on the way we sort
	/// // about the specified index.
	/// assert!(v == arr1(&[2, 4, 1, -5, -3]) ||
	///         v == arr1(&[2, 4, 1, -3, -5]) ||
	///         v == arr1(&[4, 2, 1, -5, -3]) ||
	///         v == arr1(&[4, 2, 1, -3, -5]));
	/// ```
	#[must_use]
	fn select_nth_unstable_by<F>(
		&mut self,
		index: usize,
		compare: F,
	) -> (ArrayViewMut1<'_, A>, &mut A, ArrayViewMut1<'_, A>)
	where
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut;
	/// Reorder the array with a key extraction function such that the element at `index` is at its
	/// final sorted position.
	///
	/// This reordering has the additional property that any value at position `i < index` will be
	/// less than or equal to any value at a position `j > index` using the key extraction function.
	/// Additionally, this reordering is unstable (i.e. any number of equal elements may end up at
	/// position `index`), in-place (i.e. does not allocate), and runs in *O*(*n*) time.
	/// This function is also known as "kth element" in other libraries.
	///
	/// It returns a triplet of the following from
	/// the array reordered according to the provided key extraction function: the subarray prior to
	/// `index`, the element at `index`, and the subarray after `index`; accordingly, the values in
	/// those two subarrays will respectively all be less-than-or-equal-to and greater-than-or-equal-to
	/// the value of the element at `index`.
	///
	/// # Current Implementation
	///
	/// The current algorithm is an introselect implementation based on Pattern Defeating Quicksort, which is also
	/// the basis for [`sort_unstable`]. The fallback algorithm is Median of Medians using Tukey's Ninther for
	/// pivot selection, which guarantees linear runtime for all inputs.
	///
	/// [`sort_unstable`]: Slice1Ext::sort_unstable
	///
	/// # Panics
	///
	/// Panics when `index >= len()`, meaning it always panics on empty arrays.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[-5i32, 4, 1, -3, 2]);
	///
	/// // Return the median as if the array were sorted according to absolute value.
	/// v.select_nth_unstable_by_key(2, |a| a.abs());
	///
	/// // We are only guaranteed the array will be one of the following, based on the way we sort
	/// // about the specified index.
	/// assert!(v == arr1(&[1, 2, -3, 4, -5]) ||
	///         v == arr1(&[1, 2, -3, -5, 4]) ||
	///         v == arr1(&[2, 1, -3, 4, -5]) ||
	///         v == arr1(&[2, 1, -3, -5, 4]));
	/// ```
	#[must_use]
	fn select_nth_unstable_by_key<K, F>(
		&mut self,
		index: usize,
		f: F,
	) -> (ArrayViewMut1<'_, A>, &mut A, ArrayViewMut1<'_, A>)
	where
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut;

	/// Returns the index of the partition point according to the given predicate
	/// (the index of the first element of the second partition).
	///
	/// The array is assumed to be partitioned according to the given predicate.
	/// This means that all elements for which the predicate returns true are at the start of the array
	/// and all elements for which the predicate returns false are at the end.
	/// For example, `[7, 15, 3, 5, 4, 12, 6]` is partitioned under the predicate `x % 2 != 0`
	/// (all odd numbers are at the start, all even at the end).
	///
	/// If this array is not partitioned, the returned result is unspecified and meaningless,
	/// as this method performs a kind of binary search.
	///
	/// See also [`binary_search`], [`binary_search_by`], and [`binary_search_by_key`].
	///
	/// [`binary_search`]: Slice1Ext::binary_search
	/// [`binary_search_by`]: Slice1Ext::binary_search_by
	/// [`binary_search_by_key`]: Slice1Ext::binary_search_by_key
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{
	///     ndarray::{arr1, s},
	///     Slice1Ext,
	/// };
	///
	/// let v = arr1(&[1, 2, 3, 3, 5, 6, 7]);
	/// let i = v.partition_point(|&x| x < 5);
	///
	/// assert_eq!(i, 4);
	/// assert!(v.slice(s![..i]).iter().all(|&x| x < 5));
	/// assert!(v.slice(s![i..]).iter().all(|&x| !(x < 5)));
	/// ```
	///
	/// If all elements of the array match the predicate, including if the array
	/// is empty, then the length of the array will be returned:
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let a = arr1(&[2, 4, 8]);
	/// assert_eq!(a.partition_point(|x| x < &100), a.len());
	/// let a = arr1(&[0i32; 0]);
	/// assert_eq!(a.partition_point(|x| x < &100), 0);
	/// ```
	///
	/// If you want to insert an item to a sorted vector, while maintaining
	/// sort order:
	///
	/// ```
	/// use ndarray_slice::{ndarray::array, Slice1Ext};
	///
	/// let mut s = array![0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
	/// let num = 42;
	/// let idx = s.partition_point(|&x| x < num);
	/// let mut s = s.into_raw_vec();
	/// s.insert(idx, num);
	/// assert_eq!(s, [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 42, 55]);
	/// ```
	#[must_use]
	fn partition_point<P>(&self, pred: P) -> usize
	where
		P: FnMut(&A) -> bool;

	/// Returns `true` if the array contains an element with the given value.
	///
	/// This operation is *O*(*n*).
	///
	/// Note that if you have a sorted array, [`binary_search`] may be faster.
	///
	/// [`binary_search`]: Slice1Ext::binary_search
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let v = arr1(&[10, 40, 30]);
	/// assert!(v.contains(&30));
	/// assert!(!v.contains(&50));
	/// ```
	///
	/// If you do not have a `&A`, but some other value that you can compare
	/// with one (for example, `String` implements `PartialEq<str>`), you can
	/// use `iter().any`:
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let v = arr1(&[String::from("hello"), String::from("world")]); // array of `String`
	/// assert!(v.iter().any(|e| e == "hello")); // search with `&str`
	/// assert!(!v.iter().any(|e| e == "hi"));
	/// ```
	#[must_use]
	fn contains(&self, x: &A) -> bool
	where
		A: PartialEq;

	/// Binary searches this array for a given element.
	/// This behaves similarly to [`contains`] if this array is sorted.
	///
	/// If the value is found then [`Result::Ok`] is returned, containing the
	/// index of the matching element. If there are multiple matches, then any
	/// one of the matches could be returned. The index is chosen
	/// deterministically, but is subject to change in future versions of Rust.
	/// If the value is not found then [`Result::Err`] is returned, containing
	/// the index where a matching element could be inserted while maintaining
	/// sorted order.
	///
	/// See also [`binary_search_by`], [`binary_search_by_key`], and [`partition_point`].
	///
	/// [`contains`]: Slice1Ext::contains
	/// [`binary_search_by`]: Slice1Ext::binary_search_by
	/// [`binary_search_by_key`]: Slice1Ext::binary_search_by_key
	/// [`partition_point`]: Slice1Ext::partition_point
	///
	/// # Examples
	///
	/// Looks up a series of four elements. The first is found, with a
	/// uniquely determined position; the second and third are not
	/// found; the fourth could match any position in `[1, 4]`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let s = arr1(&[0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]);
	///
	/// assert_eq!(s.binary_search(&13),  Ok(9));
	/// assert_eq!(s.binary_search(&4),   Err(7));
	/// assert_eq!(s.binary_search(&100), Err(13));
	/// let r = s.binary_search(&1);
	/// assert!(match r { Ok(1..=4) => true, _ => false, });
	/// ```
	///
	/// If you want to find that whole *range* of matching items, rather than
	/// an arbitrary matching one, that can be done using [`partition_point`]:
	/// ```
	/// use ndarray_slice::{
	///     ndarray::{arr1, s},
	///     Slice1Ext,
	/// };
	///
	/// let s = arr1(&[0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]);
	///
	/// let low = s.partition_point(|x| x < &1);
	/// assert_eq!(low, 1);
	/// let high = s.partition_point(|x| x <= &1);
	/// assert_eq!(high, 5);
	/// let r = s.binary_search(&1);
	/// assert!((low..high).contains(&r.unwrap()));
	///
	/// assert!(s.slice(s![..low]).iter().all(|&x| x < 1));
	/// assert!(s.slice(s![low..high]).iter().all(|&x| x == 1));
	/// assert!(s.slice(s![high..]).iter().all(|&x| x > 1));
	///
	/// // For something not found, the "range" of equal items is empty
	/// assert_eq!(s.partition_point(|x| x < &11), 9);
	/// assert_eq!(s.partition_point(|x| x <= &11), 9);
	/// assert_eq!(s.binary_search(&11), Err(9));
	/// ```
	///
	/// If you want to insert an item to a sorted vector, while maintaining
	/// sort order, consider using [`partition_point`]:
	///
	/// ```
	/// use ndarray_slice::{
	///     ndarray::{arr1, array},
	///     Slice1Ext,
	/// };
	///
	/// let mut s = array![0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
	/// let num = 42;
	/// let idx = s.partition_point(|&x| x < num);
	/// // The above is equivalent to `let idx = s.binary_search(&num).unwrap_or_else(|x| x);`
	/// let mut s = s.into_raw_vec();
	/// s.insert(idx, num);
	/// assert_eq!(s, [0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 42, 55]);
	/// ```
	fn binary_search(&self, x: &A) -> Result<usize, usize>
	where
		A: Ord;
	/// Binary searches this array with a comparator function.
	/// This behaves similarly to [`contains`] if this array is sorted.
	///
	/// The comparator function should implement an order consistent
	/// with the sort order of the underlying array, returning an
	/// order code that indicates whether its argument is `Less`,
	/// `Equal` or `Greater` the desired target.
	///
	/// If the value is found then [`Result::Ok`] is returned, containing the
	/// index of the matching element. If there are multiple matches, then any
	/// one of the matches could be returned. The index is chosen
	/// deterministically, but is subject to change in future versions of Rust.
	/// If the value is not found then [`Result::Err`] is returned, containing
	/// the index where a matching element could be inserted while maintaining
	/// sorted order.
	///
	/// See also [`binary_search`], [`binary_search_by_key`], and [`partition_point`].
	///
	/// [`contains`]: Slice1Ext::contains
	/// [`binary_search`]: Slice1Ext::binary_search
	/// [`binary_search_by_key`]: Slice1Ext::binary_search_by_key
	/// [`partition_point`]: Slice1Ext::partition_point
	///
	/// # Examples
	///
	/// Looks up a series of four elements. The first is found, with a
	/// uniquely determined position; the second and third are not
	/// found; the fourth could match any position in `[1, 4]`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let s = arr1(&[0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]);
	///
	/// let seek = 13;
	/// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Ok(9));
	/// let seek = 4;
	/// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Err(7));
	/// let seek = 100;
	/// assert_eq!(s.binary_search_by(|probe| probe.cmp(&seek)), Err(13));
	/// let seek = 1;
	/// let r = s.binary_search_by(|probe| probe.cmp(&seek));
	/// assert!(match r { Ok(1..=4) => true, _ => false, });
	/// ```
	fn binary_search_by<F>(&self, f: F) -> Result<usize, usize>
	where
		F: FnMut(&A) -> Ordering;
	/// Binary searches this array with a key extraction function.
	/// This behaves similarly to [`contains`] if this array is sorted.
	///
	#[cfg_attr(
		feature = "alloc",
		doc = "\
	    Assumes that the array is sorted by the key, for instance with
	    [`sort_by_key`] using the same key extraction function.
	"
	)]
	///
	/// If the value is found then [`Result::Ok`] is returned, containing the
	/// index of the matching element. If there are multiple matches, then any
	/// one of the matches could be returned. The index is chosen
	/// deterministically, but is subject to change in future versions of Rust.
	/// If the value is not found then [`Result::Err`] is returned, containing
	/// the index where a matching element could be inserted while maintaining
	/// sorted order.
	///
	/// See also [`binary_search`], [`binary_search_by`], and [`partition_point`].
	///
	/// [`contains`]: Slice1Ext::contains
	#[cfg_attr(
		feature = "alloc",
		doc = "\
	    [`sort_by_key`]: Slice1Ext::sort_by_key
	"
	)]
	/// [`binary_search`]: Slice1Ext::binary_search
	/// [`binary_search_by`]: Slice1Ext::binary_search_by
	/// [`partition_point`]: Slice1Ext::partition_point
	///
	/// # Examples
	///
	/// Looks up a series of four elements in a array of pairs sorted by
	/// their second elements. The first is found, with a uniquely
	/// determined position; the second and third are not found; the
	/// fourth could match any position in `[1, 4]`.
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let s = arr1(&[(0, 0), (2, 1), (4, 1), (5, 1), (3, 1),
	///          (1, 2), (2, 3), (4, 5), (5, 8), (3, 13),
	///          (1, 21), (2, 34), (4, 55)]);
	///
	/// assert_eq!(s.binary_search_by_key(&13, |&(a, b)| b),  Ok(9));
	/// assert_eq!(s.binary_search_by_key(&4, |&(a, b)| b),   Err(7));
	/// assert_eq!(s.binary_search_by_key(&100, |&(a, b)| b), Err(13));
	/// let r = s.binary_search_by_key(&1, |&(a, b)| b);
	/// assert!(match r { Ok(1..=4) => true, _ => false, });
	/// ```
	fn binary_search_by_key<B, F>(&self, b: &B, f: F) -> Result<usize, usize>
	where
		F: FnMut(&A) -> B,
		B: Ord;

	/// Moves all consecutive repeated elements to the end of the array according to the
	/// [`PartialEq`] trait implementation.
	///
	/// Returns two arrays. The first contains no consecutive repeated elements.
	/// The second contains all the duplicates in no specified order.
	///
	/// If the array is sorted, the first returned array contains no duplicates.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut array = arr1(&[1, 2, 2, 3, 3, 2, 1, 1]);
	///
	/// let (dedup, duplicates) = array.partition_dedup();
	///
	/// assert_eq!(dedup, arr1(&[1, 2, 3, 2, 1]));
	/// assert_eq!(duplicates, arr1(&[2, 3, 1]));
	/// ```
	fn partition_dedup(&mut self) -> (ArrayViewMut1<'_, A>, ArrayViewMut1<'_, A>)
	where
		A: PartialEq,
		S: DataMut;
	/// Moves all but the first of consecutive elements to the end of the array satisfying
	/// a given equality relation.
	///
	/// Returns two arrays. The first contains no consecutive repeated elements.
	/// The second contains all the duplicates in no specified order.
	///
	/// The `same_bucket` function is passed references to two elements from the array and
	/// must determine if the elements compare equal. The elements are passed in opposite order
	/// from their order in the array, so if `same_bucket(a, b)` returns `true`, `a` is moved
	/// at the end of the array.
	///
	/// If the array is sorted, the first returned array contains no duplicates.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut array = arr1(&["foo", "Foo", "BAZ", "Bar", "bar", "baz", "BAZ"]);
	///
	/// let (dedup, duplicates) = array.partition_dedup_by(|a, b| a.eq_ignore_ascii_case(b));
	///
	/// assert_eq!(dedup, arr1(&["foo", "BAZ", "Bar", "baz"]));
	/// assert_eq!(duplicates, arr1(&["bar", "Foo", "BAZ"]));
	/// ```
	fn partition_dedup_by<F>(
		&mut self,
		same_bucket: F,
	) -> (ArrayViewMut1<'_, A>, ArrayViewMut1<'_, A>)
	where
		F: FnMut(&mut A, &mut A) -> bool,
		S: DataMut;
	/// Moves all but the first of consecutive elements to the end of the array that resolve
	/// to the same key.
	///
	/// Returns two arrays. The first contains no consecutive repeated elements.
	/// The second contains all the duplicates in no specified order.
	///
	/// If the array is sorted, the first returned array contains no duplicates.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut array = arr1(&[10, 20, 21, 30, 30, 20, 11, 13]);
	///
	/// let (dedup, duplicates) = array.partition_dedup_by_key(|i| *i / 10);
	///
	/// assert_eq!(dedup, arr1(&[10, 20, 30, 20, 11]));
	/// assert_eq!(duplicates, arr1(&[21, 30, 13]));
	/// ```
	fn partition_dedup_by_key<K, F>(
		&mut self,
		key: F,
	) -> (ArrayViewMut1<'_, A>, ArrayViewMut1<'_, A>)
	where
		F: FnMut(&mut A) -> K,
		K: PartialEq,
		S: DataMut;

	/// Reverses the order of elements in the array, in place.
	///
	/// # Examples
	///
	/// ```
	/// use ndarray_slice::{ndarray::arr1, Slice1Ext};
	///
	/// let mut v = arr1(&[1, 2, 3]);
	/// v.reverse();
	/// assert!(v == arr1(&[3, 2, 1]));
	/// ```
	fn reverse(&mut self)
	where
		S: DataMut;
}

impl<A, S> Slice1Ext<A, S> for ArrayBase<S, Ix1>
where
	S: Data<Elem = A>,
{
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_sort(&mut self)
	where
		A: Ord + Send,
		S: DataMut,
	{
		par_merge_sort(self.view_mut(), A::lt);
	}
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_sort_by<F>(&mut self, compare: F)
	where
		A: Send,
		F: Fn(&A, &A) -> Ordering + Sync,
		S: DataMut,
	{
		par_merge_sort(self.view_mut(), |a: &A, b: &A| compare(a, b) == Less)
	}
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_sort_by_key<K, F>(&mut self, f: F)
	where
		A: Send,
		K: Ord,
		F: Fn(&A) -> K + Sync,
		S: DataMut,
	{
		par_merge_sort(self.view_mut(), |a: &A, b: &A| f(a).lt(&f(b)))
	}
	#[cfg(feature = "rayon")]
	fn par_sort_by_cached_key<K, F>(&mut self, f: F)
	where
		A: Send + Sync,
		F: Fn(&A) -> K + Sync,
		K: Ord + Send,
		S: DataMut,
	{
		use core::mem;
		use rayon::{
			iter::{ParallelBridge, ParallelIterator},
			slice::ParallelSliceMut,
		};

		//let slice = self.as_parallel_slice_mut();
		let len = self.len();
		if len < 2 {
			return;
		}

		// Helper macro for indexing our vector by the smallest possible type, to reduce allocation.
		macro_rules! sort_by_key {
			($t:ty) => {{
				let mut indices: Vec<_> = self
					.iter()
					.enumerate()
					.par_bridge()
					.map(|(i, x)| (f(&*x), i as $t))
					.collect();
				// The elements of `indices` are unique, as they are indexed, so any sort will be
				// stable with respect to the original slice. We use `sort_unstable` here because
				// it requires less memory allocation.
				indices.par_sort_unstable();
				for i in 0..len {
					let mut index = indices[i].1;
					while (index as usize) < i {
						index = indices[index as usize].1;
					}
					indices[i].1 = index;
					self.swap(i, index as usize);
				}
			}};
		}

		let sz_u8 = mem::size_of::<(K, u8)>();
		let sz_u16 = mem::size_of::<(K, u16)>();
		let sz_u32 = mem::size_of::<(K, u32)>();
		let sz_usize = mem::size_of::<(K, usize)>();

		if sz_u8 < sz_u16 && len <= (std::u8::MAX as usize) {
			return sort_by_key!(u8);
		}
		if sz_u16 < sz_u32 && len <= (std::u16::MAX as usize) {
			return sort_by_key!(u16);
		}
		if sz_u32 < sz_usize && len <= (std::u32::MAX as usize) {
			return sort_by_key!(u32);
		}
		sort_by_key!(usize)
	}

	#[cfg(feature = "alloc")]
	#[inline]
	fn sort(&mut self)
	where
		A: Ord,
		S: DataMut,
	{
		stable_sort(self.view_mut(), A::lt);
	}
	#[cfg(feature = "alloc")]
	#[inline]
	fn sort_by<F>(&mut self, mut compare: F)
	where
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut,
	{
		stable_sort(self.view_mut(), &mut |a: &A, b: &A| compare(a, b) == Less)
	}
	#[cfg(feature = "alloc")]
	#[inline]
	fn sort_by_key<K, F>(&mut self, mut f: F)
	where
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut,
	{
		stable_sort(self.view_mut(), &mut |a: &A, b: &A| f(a).lt(&f(b)))
	}
	#[cfg(feature = "std")]
	fn sort_by_cached_key<K, F>(&mut self, f: F)
	where
		F: FnMut(&A) -> K,
		K: Ord,
		S: DataMut,
	{
		use core::mem;

		// Helper macro for indexing our vector by the smallest possible type, to reduce allocation.
		macro_rules! sort_by_key {
			($t:ty, $array:ident, $f:ident) => {{
				let mut indices: Vec<_> = $array
					.iter()
					.map($f)
					.enumerate()
					.map(|(i, k)| (k, i as $t))
					.collect();
				// The elements of `indices` are unique, as they are indexed, so any sort will be
				// stable with respect to the original array. We use `sort_unstable` here because
				// it requires less memory allocation.
				indices.sort_unstable();
				for i in 0..$array.len() {
					let mut index = indices[i].1;
					while (index as usize) < i {
						index = indices[index as usize].1;
					}
					indices[i].1 = index;
					$array.swap(i, index as usize);
				}
			}};
		}

		let sz_u8 = mem::size_of::<(K, u8)>();
		let sz_u16 = mem::size_of::<(K, u16)>();
		let sz_u32 = mem::size_of::<(K, u32)>();
		let sz_usize = mem::size_of::<(K, usize)>();

		let len = self.len();
		if len < 2 {
			return;
		}
		if sz_u8 < sz_u16 && len <= (u8::MAX as usize) {
			return sort_by_key!(u8, self, f);
		}
		if sz_u16 < sz_u32 && len <= (u16::MAX as usize) {
			return sort_by_key!(u16, self, f);
		}
		if sz_u32 < sz_usize && len <= (u32::MAX as usize) {
			return sort_by_key!(u32, self, f);
		}
		sort_by_key!(usize, self, f)
	}

	#[cfg(feature = "rayon")]
	#[inline]
	fn par_sort_unstable(&mut self)
	where
		A: Ord + Send,
		S: DataMut,
	{
		par_quick_sort(self.view_mut(), A::lt);
	}
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_sort_unstable_by<F>(&mut self, compare: F)
	where
		A: Send,
		F: Fn(&A, &A) -> Ordering + Sync,
		S: DataMut,
	{
		par_quick_sort(self.view_mut(), |a: &A, b: &A| compare(a, b) == Less)
	}
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_sort_unstable_by_key<K, F>(&mut self, f: F)
	where
		A: Send,
		K: Ord,
		F: Fn(&A) -> K + Sync,
		S: DataMut,
	{
		par_quick_sort(self.view_mut(), |a: &A, b: &A| f(a).lt(&f(b)))
	}

	#[inline]
	fn sort_unstable(&mut self)
	where
		A: Ord,
		S: DataMut,
	{
		quick_sort(self.view_mut(), A::lt);
	}
	#[inline]
	fn sort_unstable_by<F>(&mut self, mut compare: F)
	where
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut,
	{
		quick_sort(self.view_mut(), &mut |a: &A, b: &A| compare(a, b) == Less)
	}
	#[inline]
	fn sort_unstable_by_key<K, F>(&mut self, mut f: F)
	where
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut,
	{
		quick_sort(self.view_mut(), &mut |a: &A, b: &A| f(a).lt(&f(b)))
	}

	#[inline]
	fn is_sorted(&self) -> bool
	where
		A: PartialOrd,
	{
		is_sorted(self.view(), |a, b| a.partial_cmp(b))
	}
	#[inline]
	fn is_sorted_by<F>(&self, compare: F) -> bool
	where
		F: FnMut(&A, &A) -> Option<Ordering>,
	{
		is_sorted(self.view(), compare)
	}
	#[inline]
	fn is_sorted_by_key<F, K>(&self, mut f: F) -> bool
	where
		F: FnMut(&A) -> K,
		K: PartialOrd,
	{
		is_sorted(self.view(), |a, b| f(a).partial_cmp(&f(b)))
	}

	#[cfg(feature = "rayon")]
	#[inline]
	fn par_select_many_nth_unstable<'a, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut Vec<&'a mut A>,
	) where
		A: Ord + Send,
		S: DataMut,
		S2: Data<Elem = usize> + Sync,
	{
		collection.reserve_exact(indices.len());
		let values = collection.spare_capacity_mut();
		par_partition_at_indices(self.view_mut(), 0, indices.view(), values, &A::lt);
		unsafe { collection.set_len(collection.len() + indices.len()) };
	}
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_select_many_nth_unstable_by<'a, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut Vec<&'a mut A>,
		compare: F,
	) where
		A: Send,
		F: Fn(&A, &A) -> Ordering + Sync,
		S: DataMut,
		S2: Data<Elem = usize> + Sync,
	{
		collection.reserve_exact(indices.len());
		let values = collection.spare_capacity_mut();
		par_partition_at_indices(
			self.view_mut(),
			0,
			indices.view(),
			values,
			&|a: &A, b: &A| compare(a, b) == Less,
		);
		unsafe { collection.set_len(collection.len() + indices.len()) };
	}
	#[cfg(feature = "rayon")]
	#[inline]
	fn par_select_many_nth_unstable_by_key<'a, K, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut Vec<&'a mut A>,
		f: F,
	) where
		A: Send,
		K: Ord,
		F: Fn(&A) -> K + Sync,
		S: DataMut,
		S2: Data<Elem = usize> + Sync,
	{
		collection.reserve_exact(indices.len());
		let values = collection.spare_capacity_mut();
		par_partition_at_indices(
			self.view_mut(),
			0,
			indices.view(),
			values,
			&|a: &A, b: &A| f(a).lt(&f(b)),
		);
		unsafe { collection.set_len(collection.len() + indices.len()) };
	}

	#[inline]
	fn select_many_nth_unstable<'a, E, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut E,
	) where
		A: Ord + 'a,
		E: Extend<(usize, &'a mut A)>,
		S: DataMut,
		S2: Data<Elem = usize>,
	{
		partition_at_indices(self.view_mut(), 0, indices.view(), collection, &mut A::lt);
	}
	#[inline]
	fn select_many_nth_unstable_by<'a, E, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut E,
		mut compare: F,
	) where
		A: 'a,
		E: Extend<(usize, &'a mut A)>,
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut,
		S2: Data<Elem = usize>,
	{
		partition_at_indices(
			self.view_mut(),
			0,
			indices.view(),
			collection,
			&mut |a: &A, b: &A| compare(a, b) == Less,
		);
	}
	#[inline]
	fn select_many_nth_unstable_by_key<'a, E, K, F, S2>(
		&'a mut self,
		indices: &ArrayBase<S2, Ix1>,
		collection: &mut E,
		mut f: F,
	) where
		A: 'a,
		E: Extend<(usize, &'a mut A)>,
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut,
		S2: Data<Elem = usize>,
	{
		partition_at_indices(
			self.view_mut(),
			0,
			indices.view(),
			collection,
			&mut |a: &A, b: &A| f(a).lt(&f(b)),
		);
	}

	#[inline]
	fn select_nth_unstable(
		&mut self,
		index: usize,
	) -> (ArrayViewMut1<'_, A>, &mut A, ArrayViewMut1<'_, A>)
	where
		A: Ord,
		S: DataMut,
	{
		partition_at_index(self.view_mut(), index, &mut A::lt)
	}
	#[inline]
	fn select_nth_unstable_by<F>(
		&mut self,
		index: usize,
		mut compare: F,
	) -> (ArrayViewMut1<'_, A>, &mut A, ArrayViewMut1<'_, A>)
	where
		F: FnMut(&A, &A) -> Ordering,
		S: DataMut,
	{
		partition_at_index(self.view_mut(), index, &mut |a: &A, b: &A| {
			compare(a, b) == Less
		})
	}
	#[inline]
	fn select_nth_unstable_by_key<K, F>(
		&mut self,
		index: usize,
		mut f: F,
	) -> (ArrayViewMut1<'_, A>, &mut A, ArrayViewMut1<'_, A>)
	where
		K: Ord,
		F: FnMut(&A) -> K,
		S: DataMut,
	{
		partition_at_index(self.view_mut(), index, &mut |a: &A, b: &A| f(a).lt(&f(b)))
	}

	#[inline]
	fn partition_point<P>(&self, mut pred: P) -> usize
	where
		P: FnMut(&A) -> bool,
	{
		self.binary_search_by(|x| if pred(x) { Less } else { Greater })
			.unwrap_or_else(|i| i)
	}

	#[inline]
	fn contains(&self, x: &A) -> bool
	where
		A: PartialEq,
	{
		self.iter().any(|a| a == x)
	}

	#[inline]
	fn binary_search(&self, x: &A) -> Result<usize, usize>
	where
		A: Ord,
	{
		self.binary_search_by(|p| p.cmp(x))
	}
	fn binary_search_by<F>(&self, mut f: F) -> Result<usize, usize>
	where
		F: FnMut(&A) -> Ordering,
	{
		// INVARIANTS:
		// - 0 <= left <= left + size = right <= self.len()
		// - f returns Less for everything in self[..left]
		// - f returns Greater for everything in self[right..]
		let mut size = self.len();
		let mut left = 0;
		let mut right = size;
		while left < right {
			let mid = left + size / 2;

			// SAFETY: the while condition means `size` is strictly positive, so
			// `size/2 < size`. Thus `left + size/2 < left + size`, which
			// coupled with the `left + size <= self.len()` invariant means
			// we have `left + size/2 < self.len()`, and this is in-bounds.
			let cmp = f(unsafe { self.uget(mid) });

			// The reason why we use if/else control flow rather than match
			// is because match reorders comparison operations, which is perf sensitive.
			// This is x86 asm for u8: https://rust.godbolt.org/z/8Y8Pra.
			if cmp == Less {
				left = mid + 1;
			} else if cmp == Greater {
				right = mid;
			} else {
				// SAFETY: same as the `get_unchecked` above
				//unsafe { crate::intrinsics::assume(mid < self.len()) };
				debug_assert!(mid < self.len());
				return Ok(mid);
			}

			size = right - left;
		}

		// SAFETY: directly true from the overall invariant.
		// Note that this is `<=`, unlike the assume in the `Ok` path.
		//unsafe { crate::intrinsics::assume(left <= self.len()) };
		debug_assert!(left <= self.len());
		Err(left)
	}
	#[inline]
	fn binary_search_by_key<B, F>(&self, b: &B, mut f: F) -> Result<usize, usize>
	where
		F: FnMut(&A) -> B,
		B: Ord,
	{
		self.binary_search_by(|k| f(k).cmp(b))
	}

	#[inline]
	fn partition_dedup(&mut self) -> (ArrayViewMut1<'_, A>, ArrayViewMut1<'_, A>)
	where
		A: PartialEq,
		S: DataMut,
	{
		self.partition_dedup_by(|a, b| a == b)
	}
	#[inline]
	fn partition_dedup_by<F>(
		&mut self,
		same_bucket: F,
	) -> (ArrayViewMut1<'_, A>, ArrayViewMut1<'_, A>)
	where
		F: FnMut(&mut A, &mut A) -> bool,
		S: DataMut,
	{
		partition_dedup(self.view_mut(), same_bucket)
	}
	#[inline]
	fn partition_dedup_by_key<K, F>(
		&mut self,
		mut key: F,
	) -> (ArrayViewMut1<'_, A>, ArrayViewMut1<'_, A>)
	where
		F: FnMut(&mut A) -> K,
		K: PartialEq,
		S: DataMut,
	{
		self.partition_dedup_by(|a, b| key(a) == key(b))
	}

	#[inline]
	fn reverse(&mut self)
	where
		S: DataMut,
	{
		reverse(self.view_mut());
	}
}
