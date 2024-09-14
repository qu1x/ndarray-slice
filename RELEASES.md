# Version 0.4.0 (2024-09-14)

  * Bump dependencies.

# Version 0.3.1 (2024-06-13)

  * Spill recursion stack over to heap if necessary.

# Version 0.3.0 (2024-03-19)

  * Synchronize with Rust standard library.

# Version 0.2.4 (2024-01-23)

  * Slightly improve performance of unstable sorting.

# Version 0.2.3 (2023-05-28)

  * Lower worst-case time complexity from *O*(*n* log *n*) to *O*(*n*)
    for selection algorithms.
  * Improve overall sorting performance.

# Version 0.2.2 (2023-04-13)

  * Add `rayon` feature for parallel sorting and parallel bulk-selection.
  * Guarantee that bulk-selected elements are in the order of their indices.
  * Half recursive branching of bulk-selection via tail call elimination.

# Version 0.2.1 (2023-04-08)

  * Update docs.

# Version 0.2.0 (2023-04-08)

  * Reduce complexity of bulk selection.
  * Change function signature of bulk selection not requiring `std` nor `alloc`.

# Version 0.1.1 (2023-04-07)

  * Add `alloc` feature.

# Version 0.1.0 (2023-04-07)

  * Initial release.
