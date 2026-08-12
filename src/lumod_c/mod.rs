// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

//! Low-level API compatible with the original LUmod calling conventions.
//!
//! Enable it with the `lumod-c` Cargo feature. This module preserves the
//! original names and mode arguments, but it is a Rust slice API rather than a
//! C ABI. New code should normally use [`crate::LuMod`], which validates
//! dimensions, indices, and workspace before modifying factors.
//!
//! All indices are zero-based. For a positive `maxmod`, `l` uses dense
//! row-major storage of at least `maxmod * maxmod` elements and `u` uses packed
//! upper-triangular storage of at least `maxmod * (maxmod + 1) / 2` elements.
//! These functions return no validation errors: callers must satisfy each
//! function's preconditions.
//!
//! # Example
//!
//! ```
//! use rlumod::lumod_c::{LUmod, Lprod, Usolve};
//!
//! let (mut l, mut u) = ([0.0; 1], [0.0; 1]);
//! let (mut y, mut z, mut w) = ([2.0], [2.0], [0.0]);
//! LUmod(1, 1, 1, 0, 0, &mut l, &mut u, &mut y, &mut z, &mut w);
//!
//! let (mut rhs, mut work) = ([6.0], [0.0]);
//! Lprod(1, 1, 1, &mut l, &mut rhs, &mut work);
//! Usolve(1, 1, 1, &mut u, &mut work);
//! assert_eq!(work, [3.0]);
//! ```

#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)] // Mirrors the indexed C storage formulas.

#[cfg(test)]
use crate::algorithm::Transform;
use crate::algorithm::{self, DenseL, Numerics, PackedU};

#[cfg(test)]
mod tests;

const MACHINE_PRECISION: f64 = 2.22e-16;
const TINY_NUMBER: f64 = 1.0e-4;

fn numerics(epsilon: f64) -> Numerics<f64> {
    Numerics {
        epsilon,
        update_tiny: TINY_NUMBER,
    }
}

/// Updates dense LU factors in place using an original LUmod operation mode.
///
/// | `mode` | Operation | Required input |
/// |---|---|---|
/// | `1` | Append the last row and column; `n` is the new dimension | `y[..n]` is the new row, `z[..n - 1]` is the new column, and `y[n - 1]` is the diagonal |
/// | `2` | Replace column `kcol` | `z[..n]` is the replacement column |
/// | `3` | Replace row `krow` | `y[..n]` is the replacement row |
/// | `4` | Move the last row/column into `krow`/`kcol`, then remove the last positions | `y`, `z`, and `w` are workspace; the resulting logical dimension is `n - 1` |
///
/// An unsupported mode or an out-of-range mode-specific index is a no-op. A
/// non-positive `n` is also a no-op. Singular and non-finite values are not
/// rejected.
///
/// # Preconditions
///
/// For a nonempty valid operation, `0 < n <= maxmod`, the factor buffers must
/// satisfy the module storage contract, and `y`, `z`, and `w` must each contain
/// at least `n` elements. Existing factor entries must represent the matrix at
/// the dimension expected by the selected mode.
///
/// # Panics
///
/// May panic if a required slice is too short or the dimension and capacity
/// relationships are invalid.
pub fn LUmod(
    mode: i32,
    maxmod: i32,
    n: i32,
    krow: i32,
    kcol: i32,
    l: &mut [f64],
    u: &mut [f64],
    y: &mut [f64],
    z: &mut [f64],
    w: &mut [f64],
) {
    let stride = maxmod.max(1) as usize;
    let n = n.max(0) as usize;
    if n == 0 {
        return;
    }
    let mut l = DenseL::new(l, stride);
    let mut u = PackedU::new(u, stride);
    let numerics = numerics(MACHINE_PRECISION);
    match mode {
        1 => algorithm::push(n, &mut l, &mut u, y, z, w, numerics),
        2 => {
            let Some(column) = usize::try_from(kcol).ok().filter(|&value| value < n) else {
                return;
            };
            algorithm::replace_column(column, n, &mut l, &mut u, y, z, w, numerics);
        }
        3 => {
            let Some(row) = usize::try_from(krow).ok().filter(|&value| value < n) else {
                return;
            };
            algorithm::replace_row(row, n, &mut l, &mut u, y, z, w, numerics);
        }
        4 => {
            let Some(row) = usize::try_from(krow).ok().filter(|&value| value < n) else {
                return;
            };
            let Some(column) = usize::try_from(kcol).ok().filter(|&value| value < n) else {
                return;
            };
            algorithm::remove(row, column, n, &mut l, &mut u, y, z, w, numerics);
        }
        _ => {}
    }
}

/// Multiplies a vector by the stored L factor.
///
/// With `mode == 1`, writes `L * y` to `z`; every other mode writes `Lᵀ * y`.
/// The first `n` output elements are overwritten and `y` is unchanged.
///
/// # Preconditions
///
/// `0 <= n <= maxmod`, `l` must satisfy the module storage contract, and `y`
/// and `z` must each contain at least `n` elements. The active part of `l` must
/// encode a valid L factor.
///
/// # Panics
///
/// May panic, possibly after partially writing `z`, if a required slice is too
/// short or the dimension and capacity relationships are invalid.
pub fn Lprod(mode: i32, maxmod: i32, n: i32, l: &mut [f64], y: &mut [f64], z: &mut [f64]) {
    let n = n.max(0) as usize;
    let l = DenseL::new(l, maxmod.max(1) as usize);
    algorithm::l_product(mode != 1, n, &l, y, z);
}

/// Applies a forward sequence of elementary transformations to `L`, `U`, and `y`.
///
/// Rows in `first..last` are eliminated using `eps` as the relative tolerance.
/// If `last < n`, the active portion of `y` is then stored as row `last` of
/// `U`. This is a low-level building block for [`LUmod`].
///
/// # Preconditions
///
/// `0 <= first <= last < n <= maxmod`, `0 <= nu <= maxmod`, both factor
/// buffers must satisfy the module storage contract, and `y` must contain at
/// least `max(n, nu)` elements. The active entries must describe a consistent
/// intermediate factorization.
///
/// # Panics
///
/// May panic if a required slice is too short or an index relationship is
/// invalid.
pub fn LUforw(
    first: i32,
    last: i32,
    n: i32,
    nu: i32,
    maxmod: i32,
    eps: f64,
    l: &mut [f64],
    u: &mut [f64],
    y: &mut [f64],
) {
    let mut l = DenseL::new(l, maxmod.max(1) as usize);
    let mut u = PackedU::new(u, maxmod.max(1) as usize);
    algorithm::forward(
        first.max(0) as usize,
        last.max(0) as usize,
        n.max(0) as usize,
        nu.max(0) as usize,
        &mut l,
        &mut u,
        y,
        numerics(eps),
    );
}

/// Applies a backward sequence of elementary transformations to `L`, `U`, and `z`.
///
/// On input, a nonnegative `*last` selects that row and enables trimming of
/// trailing `z` entries whose magnitude is at most `eps`. A negative value
/// `-k` selects row `k - 1` without trimming. On return, `*last` is the row
/// actually selected; `y` contains the extracted dense U row used by the
/// transformation. When `n <= 0`, this function only sets `*last` to zero.
///
/// # Preconditions
///
/// For `n > 0`, the selected row and `first` must lie in `0..n`, with
/// `first <= selected`, `n <= maxmod`, and `0 <= nu <= maxmod`. Both factor
/// buffers must satisfy the module storage contract, `y` and `z` must each
/// contain at least `max(n, nu)` elements, and the active entries must describe
/// a consistent intermediate factorization.
///
/// # Panics
///
/// May panic if a required slice is too short or an index relationship is
/// invalid.
pub fn LUback(
    first: i32,
    last: &mut i32,
    n: i32,
    nu: i32,
    maxmod: i32,
    eps: f64,
    l: &mut [f64],
    u: &mut [f64],
    y: &mut [f64],
    z: &mut [f64],
) {
    let n = n.max(0) as usize;
    if n == 0 {
        *last = 0;
        return;
    }
    let (selected, trim_trailing_zeros) = if *last >= 0 {
        ((*last as usize).min(n - 1), true)
    } else {
        (
            (last.unsigned_abs() as usize).saturating_sub(1).min(n - 1),
            false,
        )
    };
    let mut l = DenseL::new(l, maxmod.max(1) as usize);
    let mut u = PackedU::new(u, maxmod.max(1) as usize);
    *last = algorithm::backward(
        first.max(0) as usize,
        selected,
        trim_trailing_zeros,
        n,
        nu.max(0) as usize,
        &mut l,
        &mut u,
        y,
        z,
        numerics(eps),
    ) as i32;
}

/// Solves an upper-triangular system in place.
///
/// With `mode == 1`, replaces `y` with the solution of `U * x = y`; every
/// other mode solves `Uᵀ * x = y`. The diagonal is not validated, so zero or
/// non-finite entries can produce infinite or NaN output.
///
/// # Preconditions
///
/// `0 <= n <= maxmod`, `u` must satisfy the module storage contract, `y` must
/// contain at least `n` elements, and the active entries of `u` must encode the
/// intended U factor.
///
/// # Panics
///
/// May panic if a required slice is too short or the dimension and capacity
/// relationships are invalid.
pub fn Usolve(mode: i32, maxmod: i32, n: i32, u: &mut [f64], y: &mut [f64]) {
    let u = PackedU::new(u, maxmod.max(1) as usize);
    algorithm::u_solve(mode != 1, n.max(0) as usize, &u, y);
}

#[cfg(test)]
pub(crate) fn elmgen(x: &mut f64, y: &mut f64, eps: f64, cs: &mut f64, sn: &mut f64) {
    let transform = algorithm::elementary(x, y, numerics(eps));
    *cs = if transform.swap { -1.0 } else { 0.0 };
    *sn = transform.multiplier;
}

#[cfg(test)]
fn u_index(row: usize, column: usize, stride: usize) -> usize {
    algorithm::u_index(row, column, stride)
}

#[cfg(test)]
pub(crate) fn apply_pair(x: &mut f64, y: &mut f64, cs: f64, sn: f64) {
    algorithm::apply_pair(
        x,
        y,
        Transform {
            swap: cs < 0.0,
            multiplier: sn,
        },
    );
}
