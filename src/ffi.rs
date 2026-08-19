// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

//! Dense, one-based C compatibility ABI for `LUmod`, `Lprod`, and `Usolve`.
//!
//! Enable this module with the `c-ffi-one-based` Cargo feature.

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case, clippy::too_many_arguments)]

use core::ffi::{c_double, c_int};
use core::mem::{align_of, size_of};
use core::slice;

use crate::lumod_c as rust;

fn element_count_fits_slice(length: usize) -> bool {
    length <= isize::MAX as usize / size_of::<c_double>()
}

fn storage_lengths(maxmod: c_int) -> Option<(usize, usize)> {
    let capacity = usize::try_from(maxmod).ok().filter(|&value| value > 0)?;
    let l = capacity.checked_mul(capacity)?;
    let next = capacity.checked_add(1)?;
    let u = if capacity % 2 == 0 {
        (capacity / 2).checked_mul(next)?
    } else {
        capacity.checked_mul(next / 2)?
    };
    element_count_fits_slice(l)
        .then_some(())
        .and_then(|()| element_count_fits_slice(u).then_some((l, u)))
}

fn dimension(value: c_int, capacity: usize) -> Option<usize> {
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0 && value <= capacity)
}

fn index(value: c_int, dimension: usize) -> Option<c_int> {
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0 && value <= dimension)
        .and_then(|value| c_int::try_from(value - 1).ok())
}

fn pointer_is_aligned(pointer: *mut c_double) -> bool {
    !pointer.is_null() && (pointer as usize) % align_of::<c_double>() == 0
}

fn pointers_are_aligned<const N: usize>(pointers: [*mut c_double; N]) -> bool {
    pointers.into_iter().all(pointer_is_aligned)
}

unsafe fn one_based_slice<'a>(pointer: *mut c_double, length: usize) -> Option<&'a mut [c_double]> {
    let allocation_length = length.checked_add(1)?;
    if !pointer_is_aligned(pointer) || !element_count_fits_slice(allocation_length) {
        return None;
    }
    // SAFETY: The caller guarantees an additional writable dummy element at index zero.
    let first = unsafe { pointer.add(1) };
    // SAFETY: The caller guarantees that the elements after the dummy form the
    // required writable, initialized, non-overlapping region.
    Some(unsafe { slice::from_raw_parts_mut(first, length) })
}

/// Updates dense one-based LUmod factor storage.
///
/// # Safety
///
/// All pointers must satisfy the storage, alignment, and non-aliasing contract
/// declared in `lumod_dense.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LUmod(
    mode: c_int,
    maxmod: c_int,
    n: c_int,
    krow: c_int,
    kcol: c_int,
    l: *mut c_double,
    u: *mut c_double,
    y: *mut c_double,
    z: *mut c_double,
    w: *mut c_double,
) {
    if !(1..=4).contains(&mode) {
        return;
    }
    let Some((l_len, u_len)) = storage_lengths(maxmod) else {
        return;
    };
    let capacity = maxmod as usize;
    let Some(n_len) = dimension(n, capacity) else {
        return;
    };
    let (krow, kcol) = match mode {
        1 => (0, 0),
        2 => {
            let Some(kcol) = index(kcol, n_len) else {
                return;
            };
            (0, kcol)
        }
        3 => {
            let Some(krow) = index(krow, n_len) else {
                return;
            };
            (krow, 0)
        }
        4 => {
            let (Some(krow), Some(kcol)) = (index(krow, n_len), index(kcol, n_len)) else {
                return;
            };
            (krow, kcol)
        }
        _ => return,
    };
    if !pointers_are_aligned([l, u, y, z, w]) {
        return;
    }
    // SAFETY: This function's contract requires all five one-based buffers to
    // be valid and mutually non-overlapping for their computed lengths.
    let (Some(l), Some(u), Some(y), Some(z), Some(w)) = (unsafe {
        (
            one_based_slice(l, l_len),
            one_based_slice(u, u_len),
            one_based_slice(y, n_len),
            one_based_slice(z, n_len),
            one_based_slice(w, n_len),
        )
    }) else {
        return;
    };
    rust::LUmod(mode, maxmod, n, krow, kcol, l, u, y, z, w);
}

/// Multiplies a vector by a dense one-based L factor.
///
/// # Safety
///
/// All pointers must satisfy the storage, alignment, and non-aliasing contract
/// declared in `lumod_dense.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Lprod(
    mode: c_int,
    maxmod: c_int,
    n: c_int,
    l: *mut c_double,
    y: *mut c_double,
    z: *mut c_double,
) {
    let Some((l_len, _)) = storage_lengths(maxmod) else {
        return;
    };
    let Some(n_len) = dimension(n, maxmod as usize) else {
        return;
    };
    if !pointers_are_aligned([l, y, z]) {
        return;
    }
    // SAFETY: Required by this function's pointer contract.
    let (Some(l), Some(y), Some(z)) = (unsafe {
        (
            one_based_slice(l, l_len),
            one_based_slice(y, n_len),
            one_based_slice(z, n_len),
        )
    }) else {
        return;
    };
    rust::Lprod(mode, maxmod, n, l, y, z);
}

/// Solves a dense one-based upper-triangular system in place.
///
/// # Safety
///
/// Both pointers must satisfy the storage, alignment, and non-aliasing contract
/// declared in `lumod_dense.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Usolve(
    mode: c_int,
    maxmod: c_int,
    n: c_int,
    u: *mut c_double,
    y: *mut c_double,
) {
    let Some((_, u_len)) = storage_lengths(maxmod) else {
        return;
    };
    let Some(n_len) = dimension(n, maxmod as usize) else {
        return;
    };
    if !pointers_are_aligned([u, y]) {
        return;
    }
    // SAFETY: Required by this function's pointer contract.
    let (Some(u), Some(y)) = (unsafe { (one_based_slice(u, u_len), one_based_slice(y, n_len)) })
    else {
        return;
    };
    rust::Usolve(mode, maxmod, n, u, y);
}

#[cfg(test)]
mod tests;
