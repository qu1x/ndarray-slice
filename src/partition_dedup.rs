//! Derivative work of [`core::slice`] licensed under `MIT OR Apache-2.0`.
//!
//! [`core::slice`]: https://doc.rust-lang.org/src/core/slice/mod.rs.html

use ndarray::{ArrayViewMut1, Axis};

pub fn partition_dedup<T, F>(
	mut v: ArrayViewMut1<'_, T>,
	mut same_bucket: F,
) -> (ArrayViewMut1<'_, T>, ArrayViewMut1<'_, T>)
where
	F: FnMut(&mut T, &mut T) -> bool,
{
	// Although we have a mutable reference to `v`, we cannot make
	// *arbitrary* changes. The `same_bucket` calls could panic, so we
	// must ensure that the slice is in a valid state at all times.
	//
	// The way that we handle this is by using swaps; we iterate
	// over all the elements, swapping as we go so that at the end
	// the elements we wish to keep are in the front, and those we
	// wish to reject are at the back. We can then split the slice.
	// This operation is still `O(n)`.
	//
	// Example: We start in this state, where `r` represents "next
	// read" and `w` represents "next_write".
	//
	//           r
	//     +---+---+---+---+---+---+
	//     | 0 | 1 | 1 | 2 | 3 | 3 |
	//     +---+---+---+---+---+---+
	//           w
	//
	// Comparing v[r] against v[w-1], this is not a duplicate, so
	// we swap v[r] and v[w] (no effect as r==w) and then increment both
	// r and w, leaving us with:
	//
	//               r
	//     +---+---+---+---+---+---+
	//     | 0 | 1 | 1 | 2 | 3 | 3 |
	//     +---+---+---+---+---+---+
	//               w
	//
	// Comparing v[r] against v[w-1], this value is a duplicate,
	// so we increment `r` but leave everything else unchanged:
	//
	//                   r
	//     +---+---+---+---+---+---+
	//     | 0 | 1 | 1 | 2 | 3 | 3 |
	//     +---+---+---+---+---+---+
	//               w
	//
	// Comparing v[r] against v[w-1], this is not a duplicate,
	// so swap v[r] and v[w] and advance r and w:
	//
	//                       r
	//     +---+---+---+---+---+---+
	//     | 0 | 1 | 2 | 1 | 3 | 3 |
	//     +---+---+---+---+---+---+
	//                   w
	//
	// Not a duplicate, repeat:
	//
	//                           r
	//     +---+---+---+---+---+---+
	//     | 0 | 1 | 2 | 3 | 1 | 3 |
	//     +---+---+---+---+---+---+
	//                       w
	//
	// Duplicate, advance r. End of slice. Split at w.

	let len = v.len();
	if len <= 1 {
		let (duplicates, dedup) = v.split_at(Axis(0), 0);
		return (dedup, duplicates);
	}

	let mut next_read: usize = 1;
	let mut next_write: usize = 1;

	// SAFETY: the `while` condition guarantees `next_read` and `next_write`
	// are less than `len`, thus are inside `v`. `prev_ptr_write` points to
	// one element before `ptr_write`, but `next_write` starts at 1, so
	// `prev_ptr_write` is never less than 0 and is inside the slice.
	// This fulfils the requirements for dereferencing `ptr_read`, `prev_ptr_write`
	// and `ptr_write`, and for using `ptr.add(next_read)`, `ptr.add(next_write - 1)`
	// and `prev_ptr_write.offset(1)`.
	//
	// `next_write` is also incremented at most once per loop at most meaning
	// no element is skipped when it may need to be swapped.
	//
	// `ptr_read` and `prev_ptr_write` never point to the same element. This
	// is required for `&mut *ptr_read`, `&mut *prev_ptr_write` to be safe.
	// The explanation is simply that `next_read >= next_write` is always true,
	// thus `next_read > next_write - 1` is too.
	unsafe {
		// Avoid bounds checks by using raw pointers.
		while next_read < len {
			let read = next_read;
			let prev_write = next_write - 1;
			let a = v.uget_mut(read) as *mut T;
			let b = v.uget_mut(prev_write) as *mut T;
			if !same_bucket(&mut *a, &mut *b) {
				if next_read != next_write {
					let write = prev_write + 1;
					v.uswap(read, write);
				}
				next_write += 1;
			}
			next_read += 1;
		}
	}

	v.split_at(Axis(0), next_write)
}

#[cfg(feature = "std")]
#[cfg(test)]
mod test {
	use super::partition_dedup;
	use ndarray::Array1;
	use quickcheck_macros::quickcheck;

	#[quickcheck]
	fn deduped(xs: Vec<u32>) {
		let mut array = Array1::from_vec(xs);
		let (dedup, duplicates) = partition_dedup(array.view_mut(), |a, b| a == b);
		for i in 1..dedup.len() {
			assert!(dedup[i - 1] != dedup[i]);
		}
		for duplicate in duplicates {
			assert!(dedup.iter().any(|dedup| dedup == duplicate));
		}
	}
}
