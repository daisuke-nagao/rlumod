<!-- SPDX-FileCopyrightText: 2026 Daisuke Nagao -->
<!-- SPDX-License-Identifier: MIT -->

# lumod-c test porting status

`rlumod` provides a safe, zero-based Rust API and a C-compatible LUmod API behind the
`lumod-c` feature. Both frontends use the same private algorithm core; validation and the
dense/packed storage layouts remain in their adapters.

The contract suite covers the complete update lifecycle, all four modification modes,
last-row/last-column swap deletion, shrink and regrow, ordinary and transposed solves,
dimensions 1 through 8, extreme scales, singular factors, storage bounds, and normalized
factorization and solve residuals.

All Rust-facing logical indices are zero-based. Dense vectors expose element `0` directly,
dense matrix storage uses zero-based row/column formulas, and a negative `LUback` position is
encoded as `-(index + 1)`.

The library is `no_std` with every feature combination. Tests and examples use `std` only in
their own targets.

## Examples

Run the safe Rust API example as:

```console
cargo run --example rlumod -- 8 2
```

Run the C-compatible example as:

```console
cargo run --features lumod-c --example lumod-c -- 8 2
```

The deterministic output of the two examples is compared after excluding elapsed time, which
varies between runs.

## Deliberately not translated one-for-one

- The C BLAS loader and unrelated Core string, file, and sorting utilities remain in the
  separate `lumod-c` project.
- C header-isolation, CMake package/subproject consumer, compile-command copying, and C smoke
  executables are C/CMake integration tests. Cargo compiles each public Rust module as part of
  the Rust checks.
