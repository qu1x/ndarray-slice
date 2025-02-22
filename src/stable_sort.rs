//! Derivative work of [`alloc::slice`] licensed under `MIT OR Apache-2.0`.
//!
//! [`alloc::slice`]: https://doc.rust-lang.org/src/alloc/slice.rs.html

#![cfg(feature = "alloc")]

use crate::merge_sort::{TimSortRun, merge_sort};
use core::{alloc::Layout, mem};
use ndarray::ArrayViewMut1;

#[cfg(not(feature = "std"))]
extern crate alloc as no_std_alloc;
#[cfg(not(feature = "std"))]
use no_std_alloc::alloc::{alloc, dealloc};
#[cfg(feature = "std")]
use std::alloc::{alloc, dealloc};

#[inline]
pub fn stable_sort<T, F>(v: ArrayViewMut1<'_, T>, mut is_less: F)
where
	F: FnMut(&T, &T) -> bool,
{
	if mem::size_of::<T>() == 0 {
		// Sorting has no meaningful behavior on zero-sized types. Do nothing.
		return;
	}

	let elem_alloc_fn = |len: usize| -> *mut T {
		// SAFETY: Creating the layout is safe as long as merge_sort never calls this with len >
		// v.len(). Alloc in general will only be used as 'shadow-region' to store temporary swap
		// elements.
		unsafe { alloc(Layout::array::<T>(len).unwrap_unchecked()) as *mut T }
	};

	let elem_dealloc_fn = |buf_ptr: *mut T, len: usize| {
		// SAFETY: Creating the layout is safe as long as merge_sort never calls this with len >
		// v.len(). The caller must ensure that buf_ptr was created by elem_alloc_fn with the same
		// len.
		unsafe {
			dealloc(
				buf_ptr as *mut u8,
				Layout::array::<T>(len).unwrap_unchecked(),
			);
		}
	};

	let run_alloc_fn = |len: usize| -> *mut TimSortRun {
		// SAFETY: Creating the layout is safe as long as merge_sort never calls this with an
		// obscene length or 0.
		unsafe { alloc(Layout::array::<TimSortRun>(len).unwrap_unchecked()) as *mut TimSortRun }
	};

	let run_dealloc_fn = |buf_ptr: *mut TimSortRun, len: usize| {
		// SAFETY: The caller must ensure that buf_ptr was created by elem_alloc_fn with the same
		// len.
		unsafe {
			dealloc(
				buf_ptr as *mut u8,
				Layout::array::<TimSortRun>(len).unwrap_unchecked(),
			);
		}
	};

	merge_sort(
		v,
		&mut is_less,
		elem_alloc_fn,
		elem_dealloc_fn,
		run_alloc_fn,
		run_dealloc_fn,
	);
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
	use super::stable_sort;
	use core::cmp::Ordering;
	use ndarray::Array1;
	use quickcheck_macros::quickcheck;

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
		stable_sort(array.view_mut(), &mut Item::lt);
		for (a, s) in array.iter().zip(&sorted) {
			assert_eq!(a.index, s.index);
			assert_eq!(a.value, s.value);
		}
	}
}
