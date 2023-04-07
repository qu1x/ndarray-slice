//! Derivative work of [`core::slice::sort`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice::sort`]: https://doc.rust-lang.org/src/core/slice/sort.rs.html

use ndarray::{s, ArrayViewMut1};

/// Sorts `v` using heapsort, which guarantees *O*(*n* \* log(*n*)) worst-case.
#[cold]
pub fn heap_sort<T, F>(mut v: ArrayViewMut1<'_, T>, mut is_less: F)
where
	F: FnMut(&T, &T) -> bool,
{
	// This binary heap respects the invariant `parent >= child`.
	let mut sift_down = |mut v: ArrayViewMut1<'_, T>, mut node| {
		loop {
			// Children of `node`.
			let mut child = 2 * node + 1;
			if child >= v.len() {
				break;
			}

			// Choose the greater child.
			if child + 1 < v.len() && is_less(&v[child], &v[child + 1]) {
				child += 1;
			}

			// Stop if the invariant holds at `node`.
			if !is_less(&v[node], &v[child]) {
				break;
			}

			// Swap `node` with the greater child, move one step down, and continue sifting.
			v.swap(node, child);
			node = child;
		}
	};

	// Build the heap in linear time.
	for i in (0..v.len() / 2).rev() {
		sift_down(v.view_mut(), i);
	}

	// Pop maximal elements from the heap.
	for i in (1..v.len()).rev() {
		v.swap(0, i);
		sift_down(v.slice_mut(s![..i]), 0);
	}
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
	use super::heap_sort;
	use ndarray::Array1;
	use quickcheck_macros::quickcheck;

	#[quickcheck]
	fn sorted(xs: Vec<u32>) {
		let mut array = Array1::from_vec(xs);
		heap_sort(array.view_mut(), &mut u32::lt);
		for i in 1..array.len() {
			assert!(array[i - 1] <= array[i]);
		}
	}
}
