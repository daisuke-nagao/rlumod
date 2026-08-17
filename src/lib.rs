// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![no_std]
#![warn(missing_docs)]

//! A safe, allocation-free Rust adaptation of the LUmod numerical software
//! made available by Stanford's Systems Optimization Laboratory.
//!
//! `rlumod` incrementally constructs and updates a dense square matrix while
//! maintaining the LUmod representation `L A = U`, where `L` is a product of
//! stabilized elementary transformations and `U` is upper triangular.
//!
//! # Scope and non-goals
//!
//! `rlumod` does not compute a conventional LU decomposition `A = L U` from an
//! arbitrary completed matrix. In particular:
//!
//! - [`LuMod`] represents dense square matrices only;
//! - `L` is not the lower-triangular factor from a conventional LU
//!   decomposition;
//! - there is no one-shot factorization function; construct a new matrix with
//!   repeated calls to [`LuMod::push`]; and
//! - there is no sparse matrix storage or sparse safe API.
//!
//! The optional `lumod_c` module provides low-level dense operations with the
//! original LUmod names. It does not add sparse support.
//!
//! # Quick start
//!
//! ```
//! use rlumod::{LuMod, Workspace, storage_lengths};
//!
//! let capacity = 2;
//! let lengths = storage_lengths(capacity).unwrap();
//! let mut l = vec![0.0_f64; lengths.l];
//! let mut u = vec![0.0_f64; lengths.u];
//! let mut y = vec![0.0; capacity];
//! let mut z = vec![0.0; capacity];
//! let mut w = vec![0.0; capacity];
//!
//! let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
//! let mut factors = LuMod::from_storage(0, capacity, &mut l, &mut u).unwrap();
//! factors.push(&[], &[], 2.0, &mut workspace).unwrap();
//! factors.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();
//!
//! // The represented matrix is [[2, 5], [3, 7]].
//! let mut rhs = [12.0, 17.0];
//! factors.solve_in_place(&mut rhs).unwrap();
//! assert!((rhs[0] - 1.0).abs() < 1.0e-12);
//! assert!((rhs[1] - 2.0).abs() < 1.0e-12);
//! ```
//!
//! The example uses `Vec` only to choose the capacity at run time. [`LuMod`]
//! does not allocate, and fixed arrays or static buffers may be used instead;
//! see `examples/static_storage.rs` in the source distribution.
//!
//! # Building a matrix
//!
//! For a new matrix, normally call [`LuMod::from_storage`] with a dimension of
//! zero and then append its rows and columns. To construct
//!
//! ```text
//! A = [[a00, a01, a02],
//!      [a10, a11, a12],
//!      [a20, a21, a22]],
//! ```
//!
//! make these calls in order:
//!
//! ```text
//! push([],         [],         a00)
//! push([a10],      [a01],      a11)
//! push([a20, a21], [a02, a12], a22)
//! ```
//!
//! Each row and column slice contains the entries that precede the new
//! diagonal. After construction, [`LuMod::replace_row`] and
//! [`LuMod::replace_column`] accept a complete row or column of length
//! [`LuMod::dimension`].
//!
//! # Storage lifecycle and `no_std`
//!
//! [`storage_lengths`] returns the required factor-buffer lengths for a fixed
//! capacity. The L buffer contains `capacity * capacity` factor entries plus
//! `capacity` entries used internally by solves. The U buffer uses packed
//! upper-triangular storage. [`Workspace`] wraps three equally sized buffers
//! used by updates.
//!
//! [`LuMod::from_storage`] adopts the buffers without initializing them. A zero
//! dimension starts a new representation. With a nonzero dimension, the
//! required prefixes must already contain factors previously produced for the
//! same capacity; this mathematical precondition cannot be validated. This is
//! useful when dropping a `LuMod` to release its borrows and later adopting the
//! retained buffers again.
//!
//! The crate is `no_std` and performs no heap allocation. Storage may still be
//! supplied by `Vec` in applications that have an allocator, or by fixed and
//! static arrays in allocation-free applications.
//!
//! # Updates and removal
//!
//! [`LuMod::remove`] has swap-remove semantics: it moves the last row into the
//! removed row position and the last column into the removed column position,
//! then decreases the dimension. The returned [`Removal`] reports the original
//! last indices that moved. External row and column identities should be
//! updated with the same semantics, for example with `Vec::swap_remove`. See
//! `examples/remove_with_ids.rs` in the source distribution.
//!
//! # Error and numerical behavior
//!
//! The safe API validates dimensions, indices, slice lengths, and workspace
//! capacity. A structural update error leaves the factors and dimension
//! unchanged. A solve error leaves the right-hand side unchanged.
//!
//! "Safe" does not mean numerically nonsingular or well-conditioned. Updates
//! accept singular and non-finite matrix values, and a successful update does
//! not assess conditioning. Solves reject zero or non-finite U-factor diagonal
//! entries, but do not classify a small finite diagonal as nearly singular or
//! compute a condition number.
//!
//! # Migrating from dense C LUmod
//!
//! The safe API replaces integer modes and raw buffers with explicit methods,
//! typed zero-based indices, and validated slices:
//!
//! | Dense C operation | Safe Rust operation |
//! |---|---|
//! | `LUmod(mode = 1, n = new_dimension, ...)` | [`LuMod::push`] with row and column slices of length `new_dimension - 1` plus a separate diagonal |
//! | `LUmod(mode = 2, kcol, ...)` | [`LuMod::replace_column`] with [`ColumnIndex`] |
//! | `LUmod(mode = 3, krow, ...)` | [`LuMod::replace_row`] with [`RowIndex`] |
//! | `LUmod(mode = 4, krow, kcol, ...)` | [`LuMod::remove`], which also decreases the stored dimension |
//! | `Lprod(mode = 1, ...)` then `Usolve(mode = 1, ...)` | [`LuMod::solve_in_place`] |
//! | `Usolve(mode = 2, ...)` then `Lprod(mode = 2, ...)` | [`LuMod::solve_transpose_in_place`] |
//!
//! Original C LUmod uses one-based row and column numbers and caller-managed
//! dimension changes. The safe API uses zero-based [`RowIndex`] and
//! [`ColumnIndex`] values and owns the logical dimension. Allocate factor
//! storage with [`storage_lengths`] rather than reproducing the C formulas:
//! unlike the C L array, the safe L buffer also contains private solve work.
//! Construct [`Workspace`] separately for update work. Safe operations support
//! `f32` and `f64` and return structural or solve errors instead of relying on
//! unchecked preconditions.
//!
//! For a closer port, enable the `lumod-c` feature. The `lumod_c` module keeps the
//! original names and mode arguments, but it is a zero-based Rust slice API,
//! not a C ABI, and its public numerical functions use `f64`. New Rust code
//! should normally prefer [`LuMod`]. The sparse C implementation has no
//! corresponding API in this crate.
//!
//! # Runnable examples
//!
//! The source distribution includes small examples for the complete safe API
//! lifecycle, fixed storage, and external identity tracking:
//!
//! ```console
//! cargo run --example basic_lifecycle
//! cargo run --example static_storage
//! cargo run --example remove_with_ids
//! ```
//!
//! `rlumod` is licensed under the MIT License; see `LICENSE`. The original
//! LUmod is used under SOL's MIT license option; its copyright and license
//! notice are retained separately in `LICENSES/SOL-LUMOD-MIT.txt`.

#[cfg(test)]
extern crate std;

#[cfg(all(test, target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::wasm_bindgen_test_configure;

#[cfg(all(test, target_arch = "wasm32", target_os = "unknown"))]
wasm_bindgen_test_configure!(run_in_browser);

mod algorithm;
mod api;
pub use api::*;

#[cfg(feature = "lumod-c")]
pub mod lumod_c;

#[cfg(all(test, feature = "lumod-c"))]
mod math_contract;
