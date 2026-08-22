// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use core::{
    mem::{align_of, size_of},
    ptr::NonNull,
    slice,
};

use super::{
    RLUMOD_STATUS_MISALIGNED_POINTER, RLUMOD_STATUS_NULL_POINTER,
    RLUMOD_STATUS_OVERLAPPING_BUFFERS, RLUMOD_STATUS_SIZE_OVERFLOW, StatusResult,
};

#[derive(Clone, Copy)]
pub(super) struct Region {
    start: usize,
    end: usize,
}

impl Region {
    pub(super) fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

pub(super) fn region<T>(pointer: *const T, length: usize) -> StatusResult<Option<Region>> {
    if length == 0 {
        return Ok(None);
    }
    if pointer.is_null() {
        return Err(RLUMOD_STATUS_NULL_POINTER);
    }
    let start = pointer as usize;
    if start % align_of::<T>() != 0 {
        return Err(RLUMOD_STATUS_MISALIGNED_POINTER);
    }
    if length > isize::MAX as usize / size_of::<T>() {
        return Err(RLUMOD_STATUS_SIZE_OVERFLOW);
    }
    let bytes = length
        .checked_mul(size_of::<T>())
        .ok_or(RLUMOD_STATUS_SIZE_OVERFLOW)?;
    let end = start
        .checked_add(bytes)
        .ok_or(RLUMOD_STATUS_SIZE_OVERFLOW)?;
    Ok(Some(Region { start, end }))
}

pub(super) fn required_region<T>(pointer: *const T) -> StatusResult<Region> {
    region(pointer, 1)?.ok_or(RLUMOD_STATUS_NULL_POINTER)
}

pub(super) fn optional_region<T>(pointer: *const T) -> StatusResult<Option<Region>> {
    if pointer.is_null() {
        Ok(None)
    } else {
        required_region(pointer).map(Some)
    }
}

pub(super) fn ensure_disjoint<const N: usize>(regions: [Option<Region>; N]) -> StatusResult {
    for left in 0..N {
        for right in (left + 1)..N {
            if regions[left]
                .zip(regions[right])
                .is_some_and(|(a, b)| a.overlaps(b))
            {
                return Err(RLUMOD_STATUS_OVERLAPPING_BUFFERS);
            }
        }
    }
    Ok(())
}

pub(super) unsafe fn read_descriptor<T: Copy>(pointer: *const T) -> StatusResult<(T, Region)> {
    let descriptor_region = required_region(pointer)?;
    // SAFETY: The C contract requires a live initialized descriptor. Null and
    // alignment were checked before this unavoidable FFI-boundary read.
    Ok((unsafe { pointer.read() }, descriptor_region))
}

pub(super) unsafe fn mutable_slice<'a, T>(pointer: *mut T, length: usize) -> &'a mut [T] {
    let pointer = if length == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        pointer
    };
    // SAFETY: Callers validate the full region and conflicting aliases first.
    unsafe { slice::from_raw_parts_mut(pointer, length) }
}

pub(super) unsafe fn shared_slice<'a, T>(pointer: *const T, length: usize) -> &'a [T] {
    let pointer = if length == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        pointer
    };
    // SAFETY: Callers validate the region; immutable inputs may alias each other.
    unsafe { slice::from_raw_parts(pointer, length) }
}
