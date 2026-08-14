# rlumod

`rlumod` is a safe, `no_std` Rust adaptation of **LUmod**, Michael A.
Saunders' software for updating a square matrix as rows and columns are added,
removed, or replaced, and for solving systems with the updated matrix. It
maintains the required internal representation across changes instead of
rebuilding it from scratch.

The `rlumod` project is an independent Rust adaptation and is not affiliated
with, endorsed by, or maintained by Michael A. Saunders or SOL.

The crate provides:

- row-and-column insertion, row replacement, column replacement, and
  row-and-column removal;
- in-place solution of `A x = b` and `A^T x = b`;
- safe, zero-based APIs for both `f32` and `f64`;
- caller-owned factor and workspace storage, with no heap allocation during
  updates or solves; and
- an optional `lumod-c` feature exposing a lower-level Rust slice API with the
  original LUmod names and mode arguments.

## Example

```rust
use rlumod::{LuMod, Workspace, storage_lengths};

let capacity = 2;
let lengths = storage_lengths(capacity).unwrap();
let mut l = vec![0.0_f64; lengths.l];
let mut u = vec![0.0_f64; lengths.u];
let (mut y, mut z, mut w) = (
    vec![0.0; capacity],
    vec![0.0; capacity],
    vec![0.0; capacity],
);

let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
let mut factors =
    LuMod::from_storage(0, capacity, &mut l, &mut u).unwrap();

factors.push(&[], &[], 2.0, &mut workspace).unwrap();
factors.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();

// The represented matrix is [[2, 5], [3, 7]].
let mut rhs = [12.0, 17.0];
factors.solve_in_place(&mut rhs).unwrap();
assert!((rhs[0] - 1.0).abs() < 1.0e-12);
assert!((rhs[1] - 2.0).abs() < 1.0e-12);
```

## Documentation

The API documentation describes storage requirements, preconditions, update
semantics, and error behavior in detail. Build and open it locally with:

```console
cargo doc --open
```

To include the compatibility API in the generated documentation:

```console
cargo doc --all-features --open
```

## About LUmod and SOL

The original LUmod was written by Michael A. Saunders at Stanford University's
Systems Optimization Laboratory (SOL). It maintains the factorization
`L A = U`, where `L` is a product of stabilized elementary transformations and
`U` is upper triangular.

See the [original LUmod page at SOL](https://stanford.edu/group/SOL/software/lumod/new_index.html)
for the algorithm's provenance, original distributions, and references.

## License

This Rust adaptation is distributed under the [MIT License](LICENSE).
The original LUmod is used under SOL's MIT license option. Its copyright and
license notice are retained in [LICENSES/SOL-LUMOD-MIT.txt](LICENSES/SOL-LUMOD-MIT.txt).
