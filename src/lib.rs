// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![no_std]
#![warn(missing_docs)]

//! A complete Rust adaptation of the LUmod numerical software made available
//! by Stanford's Systems Optimization Laboratory.
//!
//! The safe API updates a dense square matrix through caller-owned LU-factor
//! storage. It performs no heap allocation and maintains the invariant
//! `dimension <= capacity`. Successful updates leave the factors representing
//! the updated matrix; structural update errors leave both the factors and the
//! dimension unchanged.
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
//! [`LuMod::from_storage`] adopts rather than initializes storage. Reusing
//! buffers previously maintained by `LuMod` is valid; otherwise the caller must
//! ensure that their contents encode factors for the supplied dimension.
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
