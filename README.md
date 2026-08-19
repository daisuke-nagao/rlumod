# rlumod

`rlumod` is a safe, `no_std` Rust adaptation of **LUmod**, Michael A.
Saunders' software for incrementally updating a dense square matrix and
solving systems with the updated matrix. It maintains the LUmod
representation `L A = U`; it does not compute a conventional `A = LU`
decomposition of an arbitrary completed matrix.

The crate provides:

- row-and-column insertion, row replacement, column replacement, and
  row-and-column removal;
- in-place solution of `A x = b` and `A^T x = b`;
- safe, zero-based APIs for both `f32` and `f64`;
- caller-owned factor and workspace storage, with no heap allocation during
  updates or solves; and
- an optional `lumod-c` feature exposing a lower-level Rust slice API with the
  original LUmod names and mode arguments; and
- an optional `c-ffi-one-based` feature exposing one-based C entry points for
  `LUmod`, `Lprod`, and `Usolve`.

`rlumod` supports dense square matrices only. It does not provide sparse
matrix storage or a one-shot factorization API.

## Installation

```console
cargo add rlumod
```

## C FFI

The optional `c-ffi-one-based` feature includes dense, one-based C entry points
for `LUmod`, `Lprod`, and `Usolve`, as declared by
`lumod-c/lumod_dense.h`. It automatically enables the lower-level `lumod-c`
Rust API. `rlumod` remains an `rlib`; the final `no_std` crate is responsible
for its panic handler and for selecting `staticlib` when a C-linkable archive
is needed. The header is `include/lumod-c/lumod_dense.h`.

The final crate must also reference `rlumod` so the linker retains the C ABI:

```toml
[lib]
crate-type = ["staticlib"]

[dependencies]
rlumod = { version = "0.1.2", features = ["c-ffi-one-based"] }
```

```rust
#![no_std]

extern crate rlumod; // Force-link the exported C ABI.

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}
```

This compatibility ABI uses caller-owned, one-based buffers. Element zero is
an unused dummy; L needs `maxmod * maxmod + 1` doubles and U needs
`maxmod * (maxmod + 1) / 2 + 1` doubles. Callers must provide aligned,
writable, sufficiently large, non-overlapping buffers. The ABI cannot verify
those C pointer properties. Sparse storage, a dynamic library, and a checked
length-and-status C API are not provided.

## Documentation and examples

The [crate documentation](https://docs.rs/rlumod/latest/rlumod/) is the main
user guide. It covers matrix construction, storage lifetime, update and
removal semantics, numerical behavior, and migration from the C LUmod API.

Small runnable programs and the larger validation drivers are listed in the
[examples guide](examples/README.md).

Build and open the documentation locally with:

```console
cargo doc --all-features --open
```

## About LUmod and SOL

The original LUmod was written by Michael A. Saunders at Stanford University's
Systems Optimization Laboratory (SOL). It maintains the factorization
`L A = U`, where `L` is a product of stabilized elementary transformations and
`U` is upper triangular.

See the [official LUmod page at SOL](https://web.stanford.edu/group/SOL/software/lumod/)
for the algorithm's provenance, original distributions, and references.

The `rlumod` project is an independent Rust adaptation and is not affiliated
with, endorsed by, or maintained by Michael A. Saunders or SOL.

## License

This Rust adaptation is distributed under the [MIT License](LICENSE).
The original LUmod is used under SOL's MIT license option. Its copyright and
license notice are retained in [LICENSES/SOL-LUMOD-MIT.txt](LICENSES/SOL-LUMOD-MIT.txt).
