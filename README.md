# ndarray-slice

[![Build][]](https://github.com/qu1x/ndarray-slice/actions/workflows/build.yml)
[![Documentation][]](https://docs.rs/ndarray-slice)
[![Downloads][]](https://crates.io/crates/ndarray-slice)
[![Version][]](https://crates.io/crates/ndarray-slice)
[![Rust][]](https://www.rust-lang.org)
[![License][]](https://opensource.org/licenses)

[Build]: https://github.com/qu1x/ndarray-slice/actions/workflows/build.yml/badge.svg
[Documentation]: https://docs.rs/ndarray-slice/badge.svg
[Downloads]: https://img.shields.io/crates/d/ndarray-slice.svg
[Version]: https://img.shields.io/crates/v/ndarray-slice.svg
[Rust]: https://img.shields.io/badge/rust-v1.60-brightgreen.svg
[License]: https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg

Fast and robust slice-based algorithms (e.g., [sorting], [selection], [search]) for
non-contiguous (sub)views into *n*-dimensional arrays. Reimplements algorithms in [`slice`] and
[`rayon::slice`] for [`ndarray`] with arbitrary memory layout (e.g., non-contiguous).

[`slice`]: https://doc.rust-lang.org/std/primitive.slice.html
[`rayon::slice`]: https://docs.rs/rayon/latest/rayon/slice/index.html
[`ndarray`]: https://docs.rs/ndarray

## Example

```rust
use ndarray_slice::{ndarray::arr2, Slice1Ext};

// 2-dimensional array of 4 rows and 5 columns.
let mut v = arr2(&[[-5, 4, 1, -3,  2],   // row 0, axis 0
                   [ 8, 3, 2,  4,  8],   // row 1, axis 0
                   [38, 9, 3,  0,  3],   // row 2, axis 0
                   [ 4, 9, 0,  8, -1]]); // row 3, axis 0
//                    \     \       \
//                  column 0 \    column 4         axis 1
//                         column 2                axis 1

// Mutable subview into the last column.
let mut column = v.column_mut(4);

// Due to row-major memory layout, columns are non-contiguous
// and hence cannot be sorted by viewing them as mutable slices.
assert_eq!(column.as_slice_mut(), None);

// Instead, sorting is specifically implemented for non-contiguous
// mutable (sub)views.
column.sort_unstable();

assert!(v == arr2(&[[-5, 4, 1, -3, -1],
                    [ 8, 3, 2,  4,  2],
                    [38, 9, 3,  0,  3],
                    [ 4, 9, 0,  8,  8]]));
//                                   \
//                                 column 4 sorted, others untouched
```

## Current Implementation

Complexities where *n* is the length of the (sub)view and *m* the count of indices to select.

| Resource | Complexity | Sorting (stable) | Sorting (unstable)  | Selection (unstable) | Bulk Selection (unstable) |
|----------|------------|------------------|---------------------|----------------------|---------------------------|
| Time     | Best       | *O*(*n*)         | *O*(*n*)            | *O*(*n*)             | *O*(*n* log *m*)          |
| Time     | Average    | *O*(*n* log *n*) | *O*(*n* log *n*)    | *O*(*n*)             | *O*(*n* log *m*)          |
| Time     | Worst      | *O*(*n* log *n*) | *O*(*n* log *n*)    | *O*(*n*)             | *O*(*n* log *m*)          |
| Space    | Best       | *O*(1)           | *O*(1)              | *O*(1)               | *O*(*m*)                  |
| Space    | Average    | *O*(*n*/2)       | *O*(log *n*)        | *O*(1)               | *O*(*m*+log *m*)          |
| Space    | Worst      | *O*(*n*/2)       | *O*(log *n*)        | *O*(1)               | *O*(*m*+log *m*)          |


[sorting]: https://en.wikipedia.org/wiki/Sorting_algorithm
[selection]: https://en.wikipedia.org/wiki/Selection_algorithm
[search]: https://en.wikipedia.org/wiki/Search_algorithm

[`sort`]: Slice1Ext::sort
[`sort_unstable`]: Slice1Ext::sort_unstable
[`select_nth_unstable`]: Slice1Ext::select_nth_unstable

## Roadmap

  * Add `SliceExt` trait for *n*-dimensional array or (sub)view with methods expecting `Axis` as
    their first argument. Comparing methods will always be suffixed with `_by` or `_by_key`
    defining how to compare multi-dimensional elements (e.g., rows) along the provided axis of
    interest (e.g., columns).

See the [release history](RELEASES.md) to keep track of the development.

## Features

  * `alloc` for stable `sort`/`sort_by`/`sort_by_key`. Enabled by `std`.
  * `std` for stable `sort_by_cached_key`. Enabled by `default` or `rayon`.
  * `rayon` for parallel `par_sort*`/`par_select_many_nth_unstable*`.

# License

Copyright © 2023 Rouven Spreckels <rs@qu1x.dev>

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSES/Apache-2.0](LICENSES/Apache-2.0) or
   https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSES/MIT](LICENSES/MIT) or https://opensource.org/licenses/MIT)

at your option.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
