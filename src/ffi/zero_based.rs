// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

//! Checked zero-based C ABI.

#![allow(missing_docs)]

use core::{
    mem::{align_of, size_of},
    ptr::{self, NonNull},
    slice,
};

use crate::{
    ColumnIndex, LuMod, Real, Removal, RowIndex, SolveError, StorageError, UpdateError, Workspace,
    storage_lengths,
};

type Status = i32;
type StatusResult<T = ()> = Result<T, Status>;

const RLUMOD_STATUS_OK: Status = 0;
const RLUMOD_STATUS_NULL_POINTER: Status = 1;
const RLUMOD_STATUS_MISALIGNED_POINTER: Status = 2;
const RLUMOD_STATUS_SIZE_OVERFLOW: Status = 3;
const RLUMOD_STATUS_OVERLAPPING_BUFFERS: Status = 4;
const RLUMOD_STATUS_INVALID_DIMENSION: Status = 5;
const RLUMOD_STATUS_INSUFFICIENT_STORAGE: Status = 6;
const RLUMOD_STATUS_CAPACITY_EXCEEDED: Status = 7;
const RLUMOD_STATUS_ROW_OUT_OF_BOUNDS: Status = 8;
const RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS: Status = 9;
const RLUMOD_STATUS_LENGTH_MISMATCH: Status = 10;
const RLUMOD_STATUS_INSUFFICIENT_WORKSPACE: Status = 11;
const RLUMOD_STATUS_SINGULAR: Status = 12;
const RLUMOD_STATUS_NON_FINITE_DIAGONAL: Status = 13;

fn status(result: StatusResult) -> Status {
    result.map_or_else(|status| status, |()| RLUMOD_STATUS_OK)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factor<T> {
    pub dimension: usize,
    pub capacity: usize,
    pub l: *mut T,
    pub l_len: usize,
    pub u: *mut T,
    pub u_len: usize,
}

impl<T> Default for Factor<T> {
    fn default() -> Self {
        Self {
            dimension: 0,
            capacity: 0,
            l: ptr::null_mut(),
            l_len: 0,
            u: ptr::null_mut(),
            u_len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Work<T> {
    pub y: *mut T,
    pub y_len: usize,
    pub z: *mut T,
    pub z_len: usize,
    pub w: *mut T,
    pub w_len: usize,
}

impl<T> Default for Work<T> {
    fn default() -> Self {
        Self {
            y: ptr::null_mut(),
            y_len: 0,
            z: ptr::null_mut(),
            z_len: 0,
            w: ptr::null_mut(),
            w_len: 0,
        }
    }
}

pub type F32Factor = Factor<f32>;
pub type F64Factor = Factor<f64>;
pub type F32Workspace = Work<f32>;
pub type F64Workspace = Work<f64>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FfiRemoval {
    pub has_moved_row: u8,
    pub moved_row: usize,
    pub has_moved_column: u8,
    pub moved_column: usize,
}

#[derive(Clone, Copy)]
struct Region {
    start: usize,
    end: usize,
}

impl Region {
    fn overlaps(self, other: Self) -> bool {
        self.start < other.end && other.start < self.end
    }
}

fn region<T>(pointer: *const T, length: usize) -> StatusResult<Option<Region>> {
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

fn required_region<T>(pointer: *const T) -> StatusResult<Region> {
    region(pointer, 1)?.ok_or(RLUMOD_STATUS_NULL_POINTER)
}

fn optional_region<T>(pointer: *const T) -> StatusResult<Option<Region>> {
    if pointer.is_null() {
        Ok(None)
    } else {
        required_region(pointer).map(Some)
    }
}

fn ensure_disjoint<const N: usize>(regions: [Option<Region>; N]) -> StatusResult {
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

unsafe fn read_descriptor<T: Copy>(pointer: *const T) -> StatusResult<(T, Region)> {
    let descriptor_region = required_region(pointer)?;
    // SAFETY: The C contract requires a live initialized descriptor. Null and
    // alignment were checked before this unavoidable FFI-boundary read.
    Ok((unsafe { pointer.read() }, descriptor_region))
}

unsafe fn mutable_slice<'a, T>(pointer: *mut T, length: usize) -> &'a mut [T] {
    let pointer = if length == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        pointer
    };
    // SAFETY: Callers validate the full region and conflicting aliases first.
    unsafe { slice::from_raw_parts_mut(pointer, length) }
}

unsafe fn shared_slice<'a, T>(pointer: *const T, length: usize) -> &'a [T] {
    let pointer = if length == 0 {
        NonNull::<T>::dangling().as_ptr()
    } else {
        pointer
    };
    // SAFETY: Callers validate the region; immutable inputs may alias each other.
    unsafe { slice::from_raw_parts(pointer, length) }
}

fn storage_status(error: StorageError) -> Status {
    match error {
        StorageError::InvalidDimension => RLUMOD_STATUS_INVALID_DIMENSION,
        StorageError::InsufficientStorage => RLUMOD_STATUS_INSUFFICIENT_STORAGE,
        StorageError::SizeOverflow => RLUMOD_STATUS_SIZE_OVERFLOW,
    }
}

fn update_status(error: UpdateError) -> Status {
    match error {
        UpdateError::CapacityExceeded => RLUMOD_STATUS_CAPACITY_EXCEEDED,
        UpdateError::RowOutOfBounds => RLUMOD_STATUS_ROW_OUT_OF_BOUNDS,
        UpdateError::ColumnOutOfBounds => RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS,
        UpdateError::LengthMismatch => RLUMOD_STATUS_LENGTH_MISMATCH,
        UpdateError::InsufficientWorkspace => RLUMOD_STATUS_INSUFFICIENT_WORKSPACE,
    }
}

fn factor_regions<T: Real>(
    descriptor: Factor<T>,
) -> StatusResult<(Option<Region>, Option<Region>)> {
    if descriptor.dimension > descriptor.capacity {
        return Err(RLUMOD_STATUS_INVALID_DIMENSION);
    }
    let lengths = storage_lengths(descriptor.capacity).map_err(storage_status)?;
    if descriptor.l_len < lengths.l || descriptor.u_len < lengths.u {
        return Err(RLUMOD_STATUS_INSUFFICIENT_STORAGE);
    }
    Ok((
        region(descriptor.l, descriptor.l_len)?,
        region(descriptor.u, descriptor.u_len)?,
    ))
}

type WorkspaceRegions = (Option<Region>, Option<Region>, Option<Region>);

fn workspace_regions<T: Copy>(descriptor: Work<T>) -> StatusResult<WorkspaceRegions> {
    if descriptor.y_len != descriptor.z_len || descriptor.y_len != descriptor.w_len {
        return Err(RLUMOD_STATUS_INSUFFICIENT_STORAGE);
    }
    Ok((
        region(descriptor.y, descriptor.y_len)?,
        region(descriptor.z, descriptor.z_len)?,
        region(descriptor.w, descriptor.w_len)?,
    ))
}

unsafe fn make_factor<'a, T: Real>(descriptor: Factor<T>) -> StatusResult<LuMod<'a, T>> {
    // SAFETY: Descriptor regions and aliases were validated by the caller.
    let l = unsafe { mutable_slice(descriptor.l, descriptor.l_len) };
    // SAFETY: Descriptor regions and aliases were validated by the caller.
    let u = unsafe { mutable_slice(descriptor.u, descriptor.u_len) };
    LuMod::from_storage(descriptor.dimension, descriptor.capacity, l, u).map_err(storage_status)
}

unsafe fn make_workspace<'a, T: Copy>(descriptor: Work<T>) -> StatusResult<Workspace<'a, T>> {
    // SAFETY: Descriptor regions and aliases were validated by the caller.
    let y = unsafe { mutable_slice(descriptor.y, descriptor.y_len) };
    // SAFETY: Descriptor regions and aliases were validated by the caller.
    let z = unsafe { mutable_slice(descriptor.z, descriptor.z_len) };
    // SAFETY: Descriptor regions and aliases were validated by the caller.
    let w = unsafe { mutable_slice(descriptor.w, descriptor.w_len) };
    Workspace::new(y, z, w).map_err(storage_status)
}

unsafe fn factor_from_storage_impl<T: Real>(
    output: *mut Factor<T>,
    dimension: usize,
    capacity: usize,
    l: *mut T,
    l_len: usize,
    u: *mut T,
    u_len: usize,
) -> StatusResult {
    let output_region = required_region(output)?;
    let descriptor = Factor {
        dimension,
        capacity,
        l,
        l_len,
        u,
        u_len,
    };
    let (l_region, u_region) = factor_regions(descriptor)?;
    ensure_disjoint([Some(output_region), l_region, u_region])?;
    // SAFETY: All descriptor regions and aliases have been validated.
    unsafe { make_factor(descriptor)? };
    // SAFETY: `output` is valid, aligned, and disjoint from adopted storage.
    unsafe { output.write(descriptor) };
    Ok(())
}

unsafe fn workspace_init_impl<T: Copy>(
    output: *mut Work<T>,
    y: *mut T,
    y_len: usize,
    z: *mut T,
    z_len: usize,
    w: *mut T,
    w_len: usize,
) -> StatusResult {
    let output_region = required_region(output)?;
    let descriptor = Work {
        y,
        y_len,
        z,
        z_len,
        w,
        w_len,
    };
    let (y_region, z_region, w_region) = workspace_regions(descriptor)?;
    ensure_disjoint([Some(output_region), y_region, z_region, w_region])?;
    // SAFETY: All descriptor regions and aliases have been validated.
    unsafe { make_workspace(descriptor)? };
    // SAFETY: `output` is valid, aligned, and disjoint from workspace buffers.
    unsafe { output.write(descriptor) };
    Ok(())
}

struct UpdateContext<T> {
    factor: Factor<T>,
    workspace: Work<T>,
    regions: UpdateRegions,
}

struct UpdateRegions {
    factor_descriptor: Region,
    workspace_descriptor: Region,
    l: Option<Region>,
    u: Option<Region>,
    y: Option<Region>,
    z: Option<Region>,
    w: Option<Region>,
}

impl<T: Real> UpdateContext<T> {
    unsafe fn from_raw(
        factor_pointer: *mut Factor<T>,
        workspace_pointer: *const Work<T>,
    ) -> StatusResult<Self> {
        // SAFETY: Descriptors are read only after null/alignment validation.
        let (factor, factor_descriptor_region) = unsafe { read_descriptor(factor_pointer) }?;
        // SAFETY: Descriptors are read only after null/alignment validation.
        let (workspace, workspace_descriptor_region) =
            unsafe { read_descriptor(workspace_pointer) }?;
        let (l_region, u_region) = factor_regions(factor)?;
        let (y_region, z_region, w_region) = workspace_regions(workspace)?;
        Ok(Self {
            factor,
            workspace,
            regions: UpdateRegions {
                factor_descriptor: factor_descriptor_region,
                workspace_descriptor: workspace_descriptor_region,
                l: l_region,
                u: u_region,
                y: y_region,
                z: z_region,
                w: w_region,
            },
        })
    }

    fn validate_update_region(&self, additional_region: Option<Region>) -> StatusResult {
        ensure_disjoint([
            Some(self.regions.factor_descriptor),
            Some(self.regions.workspace_descriptor),
            self.regions.l,
            self.regions.u,
            self.regions.y,
            self.regions.z,
            self.regions.w,
            additional_region,
        ])
    }

    fn validate_push_inputs(
        &self,
        row_region: Option<Region>,
        column_region: Option<Region>,
    ) -> StatusResult {
        self.validate_update_region(row_region)?;
        self.validate_update_region(column_region)
    }

    fn validate_replace_input(&self, values_region: Option<Region>) -> StatusResult {
        self.validate_update_region(values_region)
    }

    fn validate_remove_output(&self, output_region: Option<Region>) -> StatusResult {
        self.validate_update_region(output_region)
    }

    unsafe fn make_parts<'a>(self) -> StatusResult<(LuMod<'a, T>, Workspace<'a, T>)> {
        // SAFETY: The operation validated all regions and conflicting aliases.
        let factor = unsafe { make_factor(self.factor) }?;
        // SAFETY: The operation validated all regions and conflicting aliases.
        let workspace = unsafe { make_workspace(self.workspace) }?;
        Ok((factor, workspace))
    }
}

unsafe fn push_impl<T: Real>(
    factor_pointer: *mut Factor<T>,
    row: *const T,
    row_len: usize,
    column: *const T,
    column_len: usize,
    diagonal: T,
    workspace_pointer: *const Work<T>,
) -> StatusResult {
    // SAFETY: Raw descriptors retain the caller's contract.
    let context = unsafe { UpdateContext::from_raw(factor_pointer, workspace_pointer) }?;
    let row_region = region(row, row_len)?;
    let column_region = region(column, column_len)?;
    context.validate_push_inputs(row_region, column_region)?;
    let dimension = {
        // SAFETY: Every region and conflicting alias was validated above.
        let (mut factor, mut workspace) = unsafe { context.make_parts() }?;
        // SAFETY: Input regions were validated and may alias only one another.
        let row = unsafe { shared_slice(row, row_len) };
        // SAFETY: Input regions were validated and may alias only one another.
        let column = unsafe { shared_slice(column, column_len) };
        factor
            .push(row, column, diagonal, &mut workspace)
            .map_err(update_status)?;
        factor.dimension()
    };
    // SAFETY: The descriptor is valid and disjoint from all live buffer regions.
    unsafe { ptr::addr_of_mut!((*factor_pointer).dimension).write(dimension) };
    Ok(())
}

#[derive(Clone, Copy)]
enum Replacement {
    Row,
    Column,
}

unsafe fn replace_impl<T: Real>(
    factor_pointer: *mut Factor<T>,
    index: usize,
    values: *const T,
    values_len: usize,
    workspace_pointer: *const Work<T>,
    replacement: Replacement,
) -> StatusResult {
    // SAFETY: Raw descriptors retain the caller's contract.
    let context = unsafe { UpdateContext::from_raw(factor_pointer, workspace_pointer) }?;
    let values_region = region(values, values_len)?;
    context.validate_replace_input(values_region)?;
    // SAFETY: Every region and conflicting alias was validated above.
    let (mut factor, mut workspace) = unsafe { context.make_parts() }?;
    // SAFETY: The immutable input region was validated above.
    let values = unsafe { shared_slice(values, values_len) };
    match replacement {
        Replacement::Row => factor.replace_row(RowIndex(index), values, &mut workspace),
        Replacement::Column => factor.replace_column(ColumnIndex(index), values, &mut workspace),
    }
    .map_err(update_status)
}

unsafe fn remove_impl<T: Real>(
    factor_pointer: *mut Factor<T>,
    row: usize,
    column: usize,
    workspace_pointer: *const Work<T>,
    output: *mut FfiRemoval,
) -> StatusResult {
    // SAFETY: Raw descriptors retain the caller's contract.
    let context = unsafe { UpdateContext::from_raw(factor_pointer, workspace_pointer) }?;
    let output_region = optional_region(output)?;
    context.validate_remove_output(output_region)?;
    let (removal, dimension) = {
        // SAFETY: Every region and conflicting alias was validated above.
        let (mut factor, mut workspace) = unsafe { context.make_parts() }?;
        let removal = factor
            .remove(RowIndex(row), ColumnIndex(column), &mut workspace)
            .map_err(update_status)?;
        (removal, factor.dimension())
    };
    // SAFETY: The descriptor is valid and no buffer references remain live.
    unsafe { ptr::addr_of_mut!((*factor_pointer).dimension).write(dimension) };
    if !output.is_null() {
        // SAFETY: Optional output was validated and is disjoint from all regions.
        unsafe { output.write(ffi_removal(removal)) };
    }
    Ok(())
}

fn ffi_removal(removal: Removal) -> FfiRemoval {
    FfiRemoval {
        has_moved_row: u8::from(removal.moved_row.is_some()),
        moved_row: removal.moved_row.map_or(0, |index| index.0),
        has_moved_column: u8::from(removal.moved_column.is_some()),
        moved_column: removal.moved_column.map_or(0, |index| index.0),
    }
}

unsafe fn solve_impl<T: Real>(
    factor_pointer: *const Factor<T>,
    rhs: *mut T,
    rhs_len: usize,
    error_index: *mut usize,
    transpose: bool,
) -> StatusResult {
    // SAFETY: Descriptor is read only after null/alignment validation.
    let (factor_descriptor, factor_descriptor_region) = unsafe { read_descriptor(factor_pointer) }?;
    let (l_region, u_region) = factor_regions(factor_descriptor)?;
    let rhs_region = region(rhs, rhs_len)?;
    let error_region = optional_region(error_index)?;
    ensure_disjoint([
        Some(factor_descriptor_region),
        l_region,
        u_region,
        rhs_region,
        error_region,
    ])?;
    let result = {
        // SAFETY: Every region and conflicting alias was validated above.
        let factor = unsafe { make_factor(factor_descriptor) }?;
        // SAFETY: The mutable RHS region was validated above.
        let rhs = unsafe { mutable_slice(rhs, rhs_len) };
        if transpose {
            factor.solve_transpose_in_place(rhs)
        } else {
            factor.solve_in_place(rhs)
        }
    };
    match result {
        Ok(()) => Ok(()),
        Err(SolveError::LengthMismatch) => Err(RLUMOD_STATUS_LENGTH_MISMATCH),
        Err(SolveError::Singular { index }) => {
            if !error_index.is_null() {
                // SAFETY: Optional output was validated and factor references are dropped.
                unsafe { error_index.write(index) };
            }
            Err(RLUMOD_STATUS_SINGULAR)
        }
        Err(SolveError::NonFiniteDiagonal { index }) => {
            if !error_index.is_null() {
                // SAFETY: Optional output was validated and factor references are dropped.
                unsafe { error_index.write(index) };
            }
            Err(RLUMOD_STATUS_NON_FINITE_DIAGONAL)
        }
    }
}

unsafe fn storage_lengths_impl(capacity: usize, l: *mut usize, u: *mut usize) -> StatusResult {
    let l_region = required_region(l)?;
    let u_region = required_region(u)?;
    if l_region.overlaps(u_region) {
        return Err(RLUMOD_STATUS_OVERLAPPING_BUFFERS);
    }
    let lengths = storage_lengths(capacity).map_err(storage_status)?;
    // SAFETY: Both outputs are valid, aligned, and disjoint.
    unsafe {
        l.write(lengths.l);
        u.write(lengths.u);
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlumod_storage_lengths(
    capacity: usize,
    l: *mut usize,
    u: *mut usize,
) -> Status {
    // SAFETY: Raw output pointers retain the caller's contract.
    status(unsafe { storage_lengths_impl(capacity, l, u) })
}

macro_rules! export_float_abi {
    (
        $ty:ty,
        $factor:ty,
        $workspace:ty,
        $from_storage:ident,
        $workspace_init:ident,
        $push:ident,
        $replace_row:ident,
        $replace_column:ident,
        $remove:ident,
        $solve:ident,
        $solve_transpose:ident
    ) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $from_storage(
            output: *mut $factor,
            dimension: usize,
            capacity: usize,
            l: *mut $ty,
            l_len: usize,
            u: *mut $ty,
            u_len: usize,
        ) -> Status {
            status(unsafe {
                factor_from_storage_impl(output, dimension, capacity, l, l_len, u, u_len)
            })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $workspace_init(
            output: *mut $workspace,
            y: *mut $ty,
            y_len: usize,
            z: *mut $ty,
            z_len: usize,
            w: *mut $ty,
            w_len: usize,
        ) -> Status {
            status(unsafe { workspace_init_impl(output, y, y_len, z, z_len, w, w_len) })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $push(
            factor: *mut $factor,
            row: *const $ty,
            row_len: usize,
            column: *const $ty,
            column_len: usize,
            diagonal: $ty,
            workspace: *const $workspace,
        ) -> Status {
            status(unsafe {
                push_impl(
                    factor, row, row_len, column, column_len, diagonal, workspace,
                )
            })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $replace_row(
            factor: *mut $factor,
            row: usize,
            values: *const $ty,
            values_len: usize,
            workspace: *const $workspace,
        ) -> Status {
            status(unsafe {
                replace_impl(factor, row, values, values_len, workspace, Replacement::Row)
            })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $replace_column(
            factor: *mut $factor,
            column: usize,
            values: *const $ty,
            values_len: usize,
            workspace: *const $workspace,
        ) -> Status {
            status(unsafe {
                replace_impl(
                    factor,
                    column,
                    values,
                    values_len,
                    workspace,
                    Replacement::Column,
                )
            })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $remove(
            factor: *mut $factor,
            row: usize,
            column: usize,
            workspace: *const $workspace,
            output: *mut FfiRemoval,
        ) -> Status {
            status(unsafe { remove_impl(factor, row, column, workspace, output) })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $solve(
            factor: *const $factor,
            rhs: *mut $ty,
            rhs_len: usize,
            error_index: *mut usize,
        ) -> Status {
            status(unsafe { solve_impl(factor, rhs, rhs_len, error_index, false) })
        }

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $solve_transpose(
            factor: *const $factor,
            rhs: *mut $ty,
            rhs_len: usize,
            error_index: *mut usize,
        ) -> Status {
            status(unsafe { solve_impl(factor, rhs, rhs_len, error_index, true) })
        }
    };
}

export_float_abi!(
    f32,
    F32Factor,
    F32Workspace,
    rlumod_f32_factor_from_storage,
    rlumod_f32_workspace_init,
    rlumod_f32_push,
    rlumod_f32_replace_row,
    rlumod_f32_replace_column,
    rlumod_f32_remove,
    rlumod_f32_solve_in_place,
    rlumod_f32_solve_transpose_in_place
);

export_float_abi!(
    f64,
    F64Factor,
    F64Workspace,
    rlumod_f64_factor_from_storage,
    rlumod_f64_workspace_init,
    rlumod_f64_push,
    rlumod_f64_replace_row,
    rlumod_f64_replace_column,
    rlumod_f64_remove,
    rlumod_f64_solve_in_place,
    rlumod_f64_solve_transpose_in_place
);

#[cfg(test)]
mod tests;
