//! Derivative work of [`core::slice::sort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice::sort`]: https://doc.rust-lang.org/src/core/slice/sort.rs.html

use crate::{
	insertion_sort::InsertionHole,
	par::insertion_sort::insertion_sort_shift_left,
	partition::{break_patterns, reverse},
};
use core::{
	cmp::{
		self,
		Ordering::{Equal, Greater, Less},
	},
	mem::{self, ManuallyDrop, MaybeUninit},
	ptr,
};
use ndarray::{s, ArrayView1, ArrayViewMut1, Axis, IndexLonger};

// For slices of up to this length it's probably faster to simply sort them.
// Defined at the module scope because it's used in multiple functions.
const MAX_INSERTION: usize = 10;

pub fn par_partition_at_indices<'a, T, F>(
	mut v: ArrayViewMut1<'a, T>,
	mut offset: usize,
	mut indices: ArrayView1<usize>,
	mut values: &mut [MaybeUninit<&'a mut T>],
	is_less: &F,
) where
	T: Send,
	F: Fn(&T, &T) -> bool + Sync,
{
	// If both partitions are up to this length, we continue sequentially. This number is as small
	// as possible but so that the overhead of Rayon's task scheduling is still negligible.
	const MAX_SEQUENTIAL: usize = 2000;

	while !indices.is_empty() {
		let at = indices.len() / 2;

		let (left_indices, right_indices) = indices.split_at(Axis(0), at);
		let (index, right_indices) = right_indices.split_at(Axis(0), 1);
		let pivot = *index.index(0);

		let (left, value, right) = partition_at_index(v, pivot - offset, is_less);
		values[at].write(value);

		let (left_values, right_values) = values.split_at_mut(at);
		let right_values = &mut right_values[1..];
		if at == 0 || pivot - offset <= MAX_SEQUENTIAL {
			par_partition_at_indices(left, offset, left_indices, left_values, is_less);
			v = right;
			offset = pivot + 1;
			indices = right_indices;
			values = right_values;
		} else {
			rayon::join(
				|| par_partition_at_indices(left, offset, left_indices, left_values, is_less),
				|| par_partition_at_indices(right, pivot + 1, right_indices, right_values, is_less),
			);
			break;
		}
	}
}

/// Reorder the slice such that the element at `index` is at its final sorted position.
pub fn partition_at_index<'a, T, F>(
	mut v: ArrayViewMut1<'a, T>,
	index: usize,
	is_less: &F,
) -> (ArrayViewMut1<'a, T>, &'a mut T, ArrayViewMut1<'a, T>)
where
	F: Fn(&T, &T) -> bool,
{
	if index >= v.len() {
		panic!(
			"partition_at_index index {} greater than length of slice {}",
			index,
			v.len()
		);
	}

	if mem::size_of::<T>() == 0 {
		// Sorting has no meaningful behavior on zero-sized types. Do nothing.
	} else if index == v.len() - 1 {
		// Find max element and place it in the last position of the array. We're free to use
		// `unwrap()` here because we know v must not be empty.
		let (max_index, _) = v.iter().enumerate().max_by(from_is_less(is_less)).unwrap();
		v.swap(max_index, index);
	} else if index == 0 {
		// Find min element and place it in the first position of the array. We're free to use
		// `unwrap()` here because we know v must not be empty.
		let (min_index, _) = v.iter().enumerate().min_by(from_is_less(is_less)).unwrap();
		v.swap(min_index, index);
	} else {
		partition_at_index_loop(v.view_mut(), index, is_less, None);
	}

	let (left, right) = v.split_at(Axis(0), index);
	let (pivot, right) = right.split_at(Axis(0), 1);
	(left, pivot.index(0), right)
}

/// helper function used to find the index of the min/max element
/// using e.g. `slice.iter().enumerate().min_by(from_is_less(&mut is_less)).unwrap()`
fn from_is_less<T>(
	is_less: &impl Fn(&T, &T) -> bool,
) -> impl Fn(&(usize, &T), &(usize, &T)) -> cmp::Ordering + '_ {
	|&(_, x), &(_, y)| {
		if is_less(x, y) {
			cmp::Ordering::Less
		} else {
			cmp::Ordering::Greater
		}
	}
}

fn partition_at_index_loop<'a, T, F>(
	mut v: ArrayViewMut1<'a, T>,
	mut index: usize,
	is_less: &F,
	mut pred: Option<&'a T>,
) where
	F: Fn(&T, &T) -> bool,
{
	// Limit the amount of iterations and fall back to fast deterministic selection
	// to ensure O(n) worst case running time. This limit needs to be constant, because
	// using `ilog2(len)` like in `sort` would result in O(n log n) time complexity.
	// The exact value of the limit is chosen somewhat arbitrarily, but for most inputs bad pivot
	// selections should be relatively rare, so the limit usually shouldn't be reached
	// anyways.
	let mut limit = 16;

	// True if the last partitioning was reasonably balanced.
	let mut was_balanced = true;

	loop {
		if v.len() <= MAX_INSERTION {
			if !v.is_empty() {
				insertion_sort_shift_left(v.view_mut(), 1, is_less);
			}
			return;
		}

		if limit == 0 {
			median_of_medians(v.view_mut(), is_less, index);
			return;
		}

		// If the last partitioning was imbalanced, try breaking patterns in the slice by shuffling
		// some elements around. Hopefully we'll choose a better pivot this time.
		if !was_balanced {
			break_patterns(v.view_mut());
			limit -= 1;
		}

		// Choose a pivot
		let (pivot, _) = choose_pivot(v.view_mut(), is_less);

		// If the chosen pivot is equal to the predecessor, then it's the smallest element in the
		// slice. Partition the slice into elements equal to and elements greater than the pivot.
		// This case is usually hit when the slice contains many duplicate elements.
		if let Some(p) = pred {
			if !is_less(p, &v[pivot]) {
				let mid = partition_equal(v.view_mut(), pivot, is_less);

				// If we've passed our index, then we're good.
				if mid > index {
					return;
				}

				// Otherwise, continue sorting elements greater than the pivot.
				let (_, new_v) = v.split_at(Axis(0), mid);
				v = new_v;
				index -= mid;
				pred = None;
				continue;
			}
		}

		let (mid, _) = partition(v.view_mut(), pivot, is_less);
		was_balanced = cmp::min(mid, v.len() - mid) >= v.len() / 8;

		// Split the slice into `left`, `pivot`, and `right`.
		let (left, right) = v.split_at(Axis(0), mid);
		let (pivot, right) = right.split_at(Axis(0), 1);
		let pivot = pivot.index(0);

		match mid.cmp(&index) {
			Less => {
				v = right;
				index = index - mid - 1;
				pred = Some(pivot);
			}
			Greater => v = left,
			// If mid == index, then we're done, since partition() guaranteed that all elements
			// after mid are greater than or equal to mid.
			Equal => return,
		}
	}
}

/// Selection algorithm to select the k-th element from the slice in guaranteed O(n) time.
/// This is essentially a quickselect that uses Tukey's Ninther for pivot selection
fn median_of_medians<T, F: Fn(&T, &T) -> bool>(
	mut v: ArrayViewMut1<'_, T>,
	is_less: &F,
	mut k: usize,
) {
	// Since this function isn't public, it should never be called with an out-of-bounds index.
	debug_assert!(k < v.len());

	// If T is as ZST, `partition_at_index` will already return early.
	debug_assert!(mem::size_of::<T>() != 0);

	// We now know that `k < v.len() <= isize::MAX`
	loop {
		if v.len() <= MAX_INSERTION {
			if v.len() > 1 {
				insertion_sort_shift_left(v.view_mut(), 1, is_less);
			}
			return;
		}

		// `median_of_{minima,maxima}` can't handle the extreme cases of the first/last element,
		// so we catch them here and just do a linear search.
		if k == v.len() - 1 {
			// Find max element and place it in the last position of the array. We're free to use
			// `unwrap()` here because we know v must not be empty.
			let (max_index, _) = v.iter().enumerate().max_by(from_is_less(is_less)).unwrap();
			v.swap(max_index, k);
			return;
		} else if k == 0 {
			// Find min element and place it in the first position of the array. We're free to use
			// `unwrap()` here because we know v must not be empty.
			let (min_index, _) = v.iter().enumerate().min_by(from_is_less(is_less)).unwrap();
			v.swap(min_index, k);
			return;
		}

		let p = median_of_ninthers(v.view_mut(), is_less);

		match p.cmp(&k) {
			Equal => return,
			Greater => {
				let (left, _right) = v.split_at(Axis(0), p);
				v = left;
			}
			Less => {
				// Since `p < k < v.len()`, `p + 1` doesn't overflow and is
				// a valid index into the slice.
				let (_left, right) = v.split_at(Axis(0), p + 1);
				v = right;
				k -= p + 1;
			}
		}
	}
}

// Optimized for when `k` lies somewhere in the middle of the slice. Selects a pivot
// as close as possible to the median of the slice. For more details on how the algorithm
// operates, refer to the paper <https://drops.dagstuhl.de/opus/volltexte/2017/7612/pdf/LIPIcs-SEA-2017-24.pdf>.
fn median_of_ninthers<T, F: Fn(&T, &T) -> bool>(mut v: ArrayViewMut1<'_, T>, is_less: &F) -> usize {
	// use `saturating_mul` so the multiplication doesn't overflow on 16-bit platforms.
	let frac = if v.len() <= 1024 {
		v.len() / 12
	} else if v.len() <= 128_usize.saturating_mul(1024) {
		v.len() / 64
	} else {
		v.len() / 1024
	};

	let pivot = frac / 2;
	let lo = v.len() / 2 - pivot;
	let hi = frac + lo;
	let gap = (v.len() - 9 * frac) / 4;
	let mut a = lo - 4 * frac - gap;
	let mut b = hi + gap;
	for i in lo..hi {
		ninther(
			v.view_mut(),
			is_less,
			[a, i - frac, b, a + 1, i, b + 1, a + 2, i + frac, b + 2],
		);
		a += 3;
		b += 3;
	}

	median_of_medians(v.slice_mut(s![lo..lo + frac]), is_less, pivot);
	partition(v, lo + pivot, is_less).0
}

/// Moves around the 9 elements at the indices a..i, such that
/// `v[d]` contains the median of the 9 elements and the other
/// elements are partitioned around it.
fn ninther<T, F: Fn(&T, &T) -> bool>(mut v: ArrayViewMut1<'_, T>, is_less: &F, n: [usize; 9]) {
	let [a, mut b, c, mut d, e, mut f, g, mut h, i] = n;
	b = median_idx(v.view(), is_less, a, b, c);
	h = median_idx(v.view(), is_less, g, h, i);
	if is_less(&v[h], &v[b]) {
		mem::swap(&mut b, &mut h);
	}
	if is_less(&v[f], &v[d]) {
		mem::swap(&mut d, &mut f);
	}
	if is_less(&v[e], &v[d]) {
		// do nothing
	} else if is_less(&v[f], &v[e]) {
		d = f;
	} else {
		if is_less(&v[e], &v[b]) {
			v.swap(e, b);
		} else if is_less(&v[h], &v[e]) {
			v.swap(e, h);
		}
		return;
	}
	if is_less(&v[d], &v[b]) {
		d = b;
	} else if is_less(&v[h], &v[d]) {
		d = h;
	}

	v.swap(d, e);
}

/// returns the index pointing to the median of the 3
/// elements `v[a]`, `v[b]` and `v[c]`
fn median_idx<T, F: Fn(&T, &T) -> bool>(
	v: ArrayView1<'_, T>,
	is_less: &F,
	mut a: usize,
	b: usize,
	mut c: usize,
) -> usize {
	if is_less(&v[c], &v[a]) {
		mem::swap(&mut a, &mut c);
	}
	if is_less(&v[c], &v[b]) {
		return c;
	}
	if is_less(&v[b], &v[a]) {
		return a;
	}
	b
}

/// Partitions `v` into elements equal to `v[pivot]` followed by elements greater than `v[pivot]`.
///
/// Returns the number of elements equal to the pivot. It is assumed that `v` does not contain
/// elements smaller than the pivot.
pub fn partition_equal<T, F>(mut v: ArrayViewMut1<'_, T>, pivot: usize, is_less: &F) -> usize
where
	F: Fn(&T, &T) -> bool,
{
	// Place the pivot at the beginning of slice.
	v.swap(0, pivot);
	let (pivot, mut v) = v.split_at(Axis(0), 1);
	let pivot = pivot.index(0);

	// Read the pivot into a stack-allocated variable for efficiency. If a following comparison
	// operation panics, the pivot will be automatically written back into the slice.
	// SAFETY: The pointer here is valid because it is obtained from a reference to a slice.
	let tmp = ManuallyDrop::new(unsafe { ptr::read(pivot) });
	let _pivot_guard = unsafe { InsertionHole::new(&*tmp, pivot) };
	let pivot = &*tmp;

	let len = v.len();
	if len == 0 {
		return 0;
	}

	// Now partition the slice.
	let mut l = 0;
	let mut r = len;
	loop {
		// SAFETY: The unsafety below involves indexing an array.
		// For the first one: We already do the bounds checking here with `l < r`.
		// For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
		//                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
		unsafe {
			// Find the first element greater than the pivot.
			while l < r && !is_less(pivot, v.view().uget(l)) {
				l += 1;
			}

			// Find the last element equal to the pivot.
			loop {
				r -= 1;
				if l >= r || !is_less(pivot, v.view().uget(r)) {
					break;
				}
			}

			// Are we done?
			if l >= r {
				break;
			}

			// Swap the found pair of out-of-order elements.
			v.uswap(l, r);
			l += 1;
		}
	}

	// We found `l` elements equal to the pivot. Add 1 to account for the pivot itself.
	l + 1
	// `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated variable)
	// back into the slice where it originally was. This step is critical in ensuring safety!
}

/// Partitions `v` into elements smaller than `v[pivot]`, followed by elements greater than or
/// equal to `v[pivot]`.
///
/// Returns a tuple of:
///
/// 1. Number of elements smaller than `v[pivot]`.
/// 2. True if `v` was already partitioned.
pub fn partition<T, F>(mut v: ArrayViewMut1<'_, T>, pivot: usize, is_less: &F) -> (usize, bool)
where
	F: Fn(&T, &T) -> bool,
{
	let (mid, was_partitioned) = {
		let mut v = v.view_mut();
		// Place the pivot at the beginning of slice.
		v.swap(0, pivot);
		let (pivot, mut v) = v.split_at(Axis(0), 1);
		let pivot = pivot.index(0);

		// Read the pivot into a stack-allocated variable for efficiency. If a following comparison
		// operation panics, the pivot will be automatically written back into the slice.

		// SAFETY: `pivot` is a reference to the first element of `v`, so `ptr::read` is safe.
		let tmp = ManuallyDrop::new(unsafe { ptr::read(pivot) });
		let _pivot_guard = unsafe { InsertionHole::new(&*tmp, pivot) };
		let pivot = &*tmp;

		// Find the first pair of out-of-order elements.
		let mut l = 0;
		let mut r = v.len();

		// SAFETY: The unsafety below involves indexing an array.
		// For the first one: We already do the bounds checking here with `l < r`.
		// For the second one: We initially have `l == 0` and `r == v.len()` and we checked that `l < r` at every indexing operation.
		//                     From here we know that `r` must be at least `r == l` which was shown to be valid from the first one.
		unsafe {
			// Find the first element greater than or equal to the pivot.
			while l < r && is_less(v.view().uget(l), pivot) {
				l += 1;
			}

			// Find the last element smaller that the pivot.
			while l < r && !is_less(v.view().uget(r - 1), pivot) {
				r -= 1;
			}
		}

		(
			l + partition_in_blocks(v.slice_mut(s![l..r]), pivot, is_less),
			l >= r,
		)

		// `_pivot_guard` goes out of scope and writes the pivot (which is a stack-allocated
		// variable) back into the slice where it originally was. This step is critical in ensuring
		// safety!
	};

	// Place the pivot between the two partitions.
	v.swap(0, mid);

	(mid, was_partitioned)
}

/// Partitions `v` into elements smaller than `pivot`, followed by elements greater than or equal
/// to `pivot`.
///
/// Returns the number of elements smaller than `pivot`.
///
/// Partitioning is performed block-by-block in order to minimize the cost of branching operations.
/// This idea is presented in the [BlockQuicksort][pdf] paper.
///
/// [pdf]: https://drops.dagstuhl.de/opus/volltexte/2016/6389/pdf/LIPIcs-ESA-2016-38.pdf
fn partition_in_blocks<T, F>(mut v: ArrayViewMut1<'_, T>, pivot: &T, is_less: &F) -> usize
where
	F: Fn(&T, &T) -> bool,
{
	if v.is_empty() {
		return 0;
	}

	// Number of elements in a typical block.
	const BLOCK: usize = 128;

	// The partitioning algorithm repeats the following steps until completion:
	//
	// 1. Trace a block from the left side to identify elements greater than or equal to the pivot.
	// 2. Trace a block from the right side to identify elements smaller than the pivot.
	// 3. Exchange the identified elements between the left and right side.
	//
	// We keep the following variables for a block of elements:
	//
	// 1. `block` - Number of elements in the block.
	// 2. `start` - Start pointer into the `offsets` array.
	// 3. `end` - End pointer into the `offsets` array.
	// 4. `offsets - Indices of out-of-order elements within the block.

	// The current block on the left side (from `l` to `l.add(block_l)`).
	let mut l = 0; //v.as_mut_ptr();
	let mut block_l = BLOCK;
	let mut start_l = ptr::null_mut();
	let mut end_l = ptr::null_mut();
	let mut offsets_l = [MaybeUninit::<u8>::uninit(); BLOCK];

	// The current block on the right side (from `r.sub(block_r)` to `r`).
	// SAFETY: The documentation for .add() specifically mention that `vec.as_ptr().add(vec.len())` is always safe`
	let mut r = v.len(); //unsafe { l.add(v.len()) };
	let mut block_r = BLOCK;
	let mut start_r = ptr::null_mut();
	let mut end_r = ptr::null_mut();
	let mut offsets_r = [MaybeUninit::<u8>::uninit(); BLOCK];

	// FIXME: When we get VLAs, try creating one array of length `min(v.len(), 2 * BLOCK)` rather
	// than two fixed-size arrays of length `BLOCK`. VLAs might be more cache-efficient.

	// Returns the number of elements between pointers `l` (inclusive) and `r` (exclusive).
	fn ptr_width<T>(l: *mut T, r: *mut T) -> usize {
		assert!(mem::size_of::<T>() > 0);
		// FIXME: this should *likely* use `offset_from`, but more
		// investigation is needed (including running tests in miri).
		#[cfg(miri)]
		{
			(r.addr() - l.addr()) / mem::size_of::<T>()
		}
		#[cfg(not(miri))]
		{
			(r as usize - l as usize) / mem::size_of::<T>()
		}
	}
	fn width(l: usize, r: usize) -> usize {
		r - l
	}

	loop {
		// We are done with partitioning block-by-block when `l` and `r` get very close. Then we do
		// some patch-up work in order to partition the remaining elements in between.
		let is_done = width(l, r) <= 2 * BLOCK;

		if is_done {
			// Number of remaining elements (still not compared to the pivot).
			let mut rem = width(l, r);
			if start_l < end_l || start_r < end_r {
				rem -= BLOCK;
			}

			// Adjust block sizes so that the left and right block don't overlap, but get perfectly
			// aligned to cover the whole remaining gap.
			if start_l < end_l {
				block_r = rem;
			} else if start_r < end_r {
				block_l = rem;
			} else {
				// There were the same number of elements to switch on both blocks during the last
				// iteration, so there are no remaining elements on either block. Cover the remaining
				// items with roughly equally-sized blocks.
				block_l = rem / 2;
				block_r = rem - block_l;
			}
			debug_assert!(block_l <= BLOCK && block_r <= BLOCK);
			debug_assert!(width(l, r) == block_l + block_r);
		}

		if start_l == end_l {
			// Trace `block_l` elements from the left side.
			#[cfg(miri)]
			{
				start_l = MaybeUninit::slice_as_mut_ptr(&mut offsets_l);
			}
			#[cfg(not(miri))]
			{
				start_l = offsets_l[0].as_mut_ptr();
			}
			end_l = start_l;
			let mut elem = l;

			for i in 0..block_l {
				// SAFETY: The unsafety operations below involve the usage of the `offset`.
				//         According to the conditions required by the function, we satisfy them because:
				//         1. `offsets_l` is stack-allocated, and thus considered separate allocated object.
				//         2. The function `is_less` returns a `bool`.
				//            Casting a `bool` will never overflow `isize`.
				//         3. We have guaranteed that `block_l` will be `<= BLOCK`.
				//            Plus, `end_l` was initially set to the begin pointer of `offsets_` which was declared on the stack.
				//            Thus, we know that even in the worst case (all invocations of `is_less` returns false) we will only be at most 1 byte pass the end.
				//        Another unsafety operation here is dereferencing `elem`.
				//        However, `elem` was initially the begin pointer to the slice which is always valid.
				unsafe {
					// Branchless comparison.
					*end_l = i as u8;
					end_l = end_l.add(!is_less(v.view_mut().index(elem), pivot) as usize);
					elem += 1; //elem = elem.add(1);
				}
			}
		}

		if start_r == end_r {
			// Trace `block_r` elements from the right side.
			#[cfg(miri)]
			{
				start_r = MaybeUninit::slice_as_mut_ptr(&mut offsets_r);
			}
			#[cfg(not(miri))]
			{
				start_r = offsets_r[0].as_mut_ptr();
			}
			end_r = start_r;
			let mut elem = r;

			for i in 0..block_r {
				// SAFETY: The unsafety operations below involve the usage of the `offset`.
				//         According to the conditions required by the function, we satisfy them because:
				//         1. `offsets_r` is stack-allocated, and thus considered separate allocated object.
				//         2. The function `is_less` returns a `bool`.
				//            Casting a `bool` will never overflow `isize`.
				//         3. We have guaranteed that `block_r` will be `<= BLOCK`.
				//            Plus, `end_r` was initially set to the begin pointer of `offsets_` which was declared on the stack.
				//            Thus, we know that even in the worst case (all invocations of `is_less` returns true) we will only be at most 1 byte pass the end.
				//        Another unsafety operation here is dereferencing `elem`.
				//        However, `elem` was initially `1 * sizeof(T)` past the end and we decrement it by `1 * sizeof(T)` before accessing it.
				//        Plus, `block_r` was asserted to be less than `BLOCK` and `elem` will therefore at most be pointing to the beginning of the slice.
				unsafe {
					// Branchless comparison.
					elem -= 1;
					*end_r = i as u8;
					end_r = end_r.add(is_less(v.view_mut().index(elem), pivot) as usize);
				}
			}
		}

		// Number of out-of-order elements to swap between the left and right side.
		let count = cmp::min(ptr_width(start_l, end_l), ptr_width(start_r, end_r));

		if count > 0 {
			macro_rules! left {
                () => {
                    v.view_mut().index(l + usize::from(*start_l)) as *mut T //l.add(usize::from(*start_l))
                };
            }
			macro_rules! right {
                () => {
                    v.view_mut().index(r - (usize::from(*start_r) + 1)) as *mut T //r.sub(usize::from(*start_r) + 1)
                };
            }

			// Instead of swapping one pair at the time, it is more efficient to perform a cyclic
			// permutation. This is not strictly equivalent to swapping, but produces a similar
			// result using fewer memory operations.

			// SAFETY: The use of `ptr::read` is valid because there is at least one element in
			// both `offsets_l` and `offsets_r`, so `left!` is a valid pointer to read from.
			//
			// The uses of `left!` involve calls to `offset` on `l`, which points to the
			// beginning of `v`. All the offsets pointed-to by `start_l` are at most `block_l`, so
			// these `offset` calls are safe as all reads are within the block. The same argument
			// applies for the uses of `right!`.
			//
			// The calls to `start_l.offset` are valid because there are at most `count-1` of them,
			// plus the final one at the end of the unsafe block, where `count` is the minimum number
			// of collected offsets in `offsets_l` and `offsets_r`, so there is no risk of there not
			// being enough elements. The same reasoning applies to the calls to `start_r.offset`.
			//
			// The calls to `copy_nonoverlapping` are safe because `left!` and `right!` are guaranteed
			// not to overlap, and are valid because of the reasoning above.
			unsafe {
				let tmp = ptr::read(left!());
				ptr::copy_nonoverlapping(right!(), left!(), 1);

				for _ in 1..count {
					start_l = start_l.add(1);
					ptr::copy_nonoverlapping(left!(), right!(), 1);
					start_r = start_r.add(1);
					ptr::copy_nonoverlapping(right!(), left!(), 1);
				}

				ptr::copy_nonoverlapping(&tmp, right!(), 1);
				mem::forget(tmp);
				start_l = start_l.add(1);
				start_r = start_r.add(1);
			}
		}

		if start_l == end_l {
			// All out-of-order elements in the left block were moved. Move to the next block.

			// block-width-guarantee
			// SAFETY: if `!is_done` then the slice width is guaranteed to be at least `2*BLOCK` wide. There
			// are at most `BLOCK` elements in `offsets_l` because of its size, so the `offset` operation is
			// safe. Otherwise, the debug assertions in the `is_done` case guarantee that
			// `width(l, r) == block_l + block_r`, namely, that the block sizes have been adjusted to account
			// for the smaller number of remaining elements.
			l += block_l; //l = unsafe { l.add(block_l) };
		}

		if start_r == end_r {
			// All out-of-order elements in the right block were moved. Move to the previous block.

			// SAFETY: Same argument as [block-width-guarantee]. Either this is a full block `2*BLOCK`-wide,
			// or `block_r` has been adjusted for the last handful of elements.
			r -= block_r; //r = unsafe { r.sub(block_r) };
		}

		if is_done {
			break;
		}
	}

	// All that remains now is at most one block (either the left or the right) with out-of-order
	// elements that need to be moved. Such remaining elements can be simply shifted to the end
	// within their block.

	if start_l < end_l {
		// The left block remains.
		// Move its remaining out-of-order elements to the far right.
		debug_assert_eq!(width(l, r), block_l);
		while start_l < end_l {
			// remaining-elements-safety
			// SAFETY: while the loop condition holds there are still elements in `offsets_l`, so it
			// is safe to point `end_l` to the previous element.
			//
			// The `ptr::swap` is safe if both its arguments are valid for reads and writes:
			//  - Per the debug assert above, the distance between `l` and `r` is `block_l`
			//    elements, so there can be at most `block_l` remaining offsets between `start_l`
			//    and `end_l`. This means `r` will be moved at most `block_l` steps back, which
			//    makes the `r.offset` calls valid (at that point `l == r`).
			//  - `offsets_l` contains valid offsets into `v` collected during the partitioning of
			//    the last block, so the `l.offset` calls are valid.
			unsafe {
				end_l = end_l.sub(1);
				v.uswap(l + usize::from(*end_l), r - 1); //ptr::swap(l.add(usize::from(*end_l)), r.sub(1));
				r -= 1; //r = r.sub(1);
			}
		}
		width(0, r) //width(v.as_mut_ptr(), r)
	} else if start_r < end_r {
		// The right block remains.
		// Move its remaining out-of-order elements to the far left.
		debug_assert_eq!(width(l, r), block_r);
		while start_r < end_r {
			// SAFETY: See the reasoning in [remaining-elements-safety].
			unsafe {
				end_r = end_r.sub(1);
				v.uswap(l, r - (usize::from(*end_r) + 1)); //ptr::swap(l, r.sub(usize::from(*end_r) + 1));
				l += 1; //l = l.add(1);
			}
		}
		width(0, l) //width(v.as_mut_ptr(), l)
	} else {
		// Nothing else to do, we're done.
		width(0, l) //width(v.as_mut_ptr(), l)
	}
}

/// Chooses a pivot in `v` and returns the index and `true` if the slice is likely already sorted.
///
/// Elements in `v` might be reordered in the process.
pub fn choose_pivot<T, F>(v: ArrayViewMut1<'_, T>, is_less: &F) -> (usize, bool)
where
	F: Fn(&T, &T) -> bool,
{
	// Minimum length to choose the median-of-medians method.
	// Shorter slices use the simple median-of-three method.
	const SHORTEST_MEDIAN_OF_MEDIANS: usize = 50;
	// Maximum number of swaps that can be performed in this function.
	const MAX_SWAPS: usize = 4 * 3;

	let len = v.len();

	// Three indices near which we are going to choose a pivot.
	let mut a = len / 4;
	let mut b = len / 4 * 2;
	let mut c = len / 4 * 3;

	// Counts the total number of swaps we are about to perform while sorting indices.
	let mut swaps = 0;

	if len >= 8 {
		// Swaps indices so that `v[a] <= v[b]`.
		// SAFETY: `len >= 8` so there are at least two elements in the neighborhoods of
		// `a`, `b` and `c`. This means the three calls to `sort_adjacent` result in
		// corresponding calls to `sort3` with valid 3-item neighborhoods around each
		// pointer, which in turn means the calls to `sort2` are done with valid
		// references. Thus the `v.get_unchecked` calls are safe, as is the `ptr::swap`
		// call.
		let v = v.view();
		let mut sort2 = |a: &mut usize, b: &mut usize| unsafe {
			if is_less(v.uget(*b), v.uget(*a)) {
				ptr::swap(a, b);
				swaps += 1;
			}
		};

		// Swaps indices so that `v[a] <= v[b] <= v[c]`.
		let mut sort3 = |a: &mut usize, b: &mut usize, c: &mut usize| {
			sort2(a, b);
			sort2(b, c);
			sort2(a, b);
		};

		if len >= SHORTEST_MEDIAN_OF_MEDIANS {
			// Finds the median of `v[a - 1], v[a], v[a + 1]` and stores the index into `a`.
			let mut sort_adjacent = |a: &mut usize| {
				let tmp = *a;
				sort3(&mut (tmp - 1), a, &mut (tmp + 1));
			};

			// Find medians in the neighborhoods of `a`, `b`, and `c`.
			sort_adjacent(&mut a);
			sort_adjacent(&mut b);
			sort_adjacent(&mut c);
		}

		// Find the median among `a`, `b`, and `c`.
		sort3(&mut a, &mut b, &mut c);
	}

	if swaps < MAX_SWAPS {
		(b, swaps == 0)
	} else {
		// The maximum number of swaps was performed. Chances are the slice is descending or mostly
		// descending, so reversing will probably help sort it faster.
		reverse(v);
		(len - 1 - b, true)
	}
}

#[cfg(test)]
mod test {
	use super::{par_partition_at_indices, partition_at_index};
	use crate::{par::quick_sort::par_quick_sort, partition_dedup::partition_dedup};
	use ndarray::arr1;
	use quickcheck::TestResult;
	use quickcheck_macros::quickcheck;

	#[cfg_attr(miri, ignore)]
	#[quickcheck]
	fn at_indices(xs: Vec<u32>) -> TestResult {
		if xs.is_empty() {
			return TestResult::discard();
		}
		let mut array = arr1(&xs);
		let mut sorted = arr1(&xs);
		par_quick_sort(sorted.view_mut(), u32::lt);
		let mut indices = arr1(&[xs.len() - 1, xs.len() / 2, xs.len() / 3, xs.len() / 4, 0]);
		par_quick_sort(indices.view_mut(), usize::lt);
		let (indices, _duplicates) = partition_dedup(indices.view_mut(), |a, b| a.eq(&b));
		if indices.iter().any(|&index| index >= xs.len()) {
			return TestResult::discard();
		}
		let mut collection = Vec::with_capacity(indices.len());
		let values = collection.spare_capacity_mut();
		assert_eq!(indices.len(), values.len());
		par_partition_at_indices(array.view_mut(), 0, indices.view(), values, &u32::lt);
		unsafe { collection.set_len(collection.len() + indices.len()) };
		for (index, value) in indices.into_iter().zip(collection.into_iter()) {
			assert_eq!(*value, sorted[*index]);
		}
		TestResult::passed()
	}

	#[quickcheck]
	fn at_index(xs: Vec<u32>) -> TestResult {
		if xs.is_empty() {
			return TestResult::discard();
		}
		let mut array = arr1(&xs);
		let (left, value, right) = partition_at_index(array.view_mut(), xs.len() / 3, &u32::lt);
		for left in left {
			assert!(left <= value);
		}
		for right in right {
			assert!(value <= right);
		}
		TestResult::passed()
	}
}
