// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use core::ptr;

use super::{
    F32Factor, F32Workspace, F64Factor, F64Workspace, FfiRemoval, RLUMOD_STATUS_CAPACITY_EXCEEDED,
    RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS, RLUMOD_STATUS_INSUFFICIENT_STORAGE,
    RLUMOD_STATUS_LENGTH_MISMATCH, RLUMOD_STATUS_MISALIGNED_POINTER,
    RLUMOD_STATUS_NON_FINITE_DIAGONAL, RLUMOD_STATUS_NULL_POINTER, RLUMOD_STATUS_OK,
    RLUMOD_STATUS_OVERLAPPING_BUFFERS, RLUMOD_STATUS_ROW_OUT_OF_BOUNDS, RLUMOD_STATUS_SINGULAR,
    RLUMOD_STATUS_SIZE_OVERFLOW, rlumod_f32_factor_from_storage, rlumod_f32_push,
    rlumod_f32_solve_in_place, rlumod_f32_workspace_init, rlumod_f64_factor_from_storage,
    rlumod_f64_push, rlumod_f64_remove, rlumod_f64_replace_column, rlumod_f64_replace_row,
    rlumod_f64_solve_in_place, rlumod_f64_solve_transpose_in_place, rlumod_f64_workspace_init,
    rlumod_storage_lengths,
};

#[test]
fn reports_storage_lengths_atomically() {
    let (mut l, mut u) = (usize::MAX, usize::MAX);
    unsafe {
        assert_eq!(rlumod_storage_lengths(3, &mut l, &mut u), RLUMOD_STATUS_OK);
    }
    assert_eq!((l, u), (12, 6));

    let before = (l, u);
    unsafe {
        assert_eq!(
            rlumod_storage_lengths(usize::MAX, &mut l, &mut u),
            RLUMOD_STATUS_SIZE_OVERFLOW
        );
    }
    assert_eq!((l, u), before);

    unsafe {
        assert_eq!(
            rlumod_storage_lengths(3, ptr::null_mut(), &mut u),
            RLUMOD_STATUS_NULL_POINTER
        );
        assert_eq!(
            rlumod_storage_lengths(3, &mut l, &mut l),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
    }
}

#[test]
fn exports_the_complete_zero_based_f64_lifecycle() {
    let (mut l, mut u) = ([0.0; 12], [0.0; 6]);
    let (mut y, mut z, mut w) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    let mut factor = F64Factor::default();
    let mut workspace = F64Workspace::default();

    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                3,
                l.as_mut_ptr(),
                l.len(),
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_workspace_init(
                &mut workspace,
                y.as_mut_ptr(),
                y.len(),
                z.as_mut_ptr(),
                z.len(),
                w.as_mut_ptr(),
                w.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_push(&mut factor, ptr::null(), 0, ptr::null(), 0, 2.0, &workspace,),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_push(
                &mut factor,
                [3.0].as_ptr(),
                1,
                [5.0].as_ptr(),
                1,
                7.0,
                &workspace,
            ),
            RLUMOD_STATUS_OK
        );
    }
    assert_eq!((factor.dimension, factor.capacity), (2, 3));

    let mut rhs = [12.0, 17.0];
    unsafe {
        assert_eq!(
            rlumod_f64_solve_in_place(&factor, rhs.as_mut_ptr(), rhs.len(), ptr::null_mut(),),
            RLUMOD_STATUS_OK
        );
    }
    assert!((rhs[0] - 1.0).abs() <= 32.0 * f64::EPSILON);
    assert!((rhs[1] - 2.0).abs() <= 32.0 * f64::EPSILON);

    let mut transposed_rhs = [8.0, 19.0];
    unsafe {
        assert_eq!(
            rlumod_f64_solve_transpose_in_place(
                &factor,
                transposed_rhs.as_mut_ptr(),
                transposed_rhs.len(),
                ptr::null_mut(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_replace_row(&mut factor, 0, [6.0, 5.0].as_ptr(), 2, &workspace),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_replace_column(&mut factor, 0, [6.0, 8.0].as_ptr(), 2, &workspace),
            RLUMOD_STATUS_OK
        );
    }
    assert!((transposed_rhs[0] - 1.0).abs() <= 32.0 * f64::EPSILON);
    assert!((transposed_rhs[1] - 2.0).abs() <= 32.0 * f64::EPSILON);

    let mut removal = FfiRemoval::default();
    unsafe {
        assert_eq!(
            rlumod_f64_remove(&mut factor, 0, 1, &workspace, &mut removal),
            RLUMOD_STATUS_OK
        );
    }
    assert_eq!(factor.dimension, 1);
    assert_eq!((removal.has_moved_row, removal.moved_row), (1, 1));
    assert_eq!(removal.has_moved_column, 0);
}

#[test]
fn exports_f32_and_accepts_null_for_empty_buffers() {
    let mut factor = F32Factor::default();
    let mut workspace = F32Workspace::default();
    unsafe {
        assert_eq!(
            rlumod_f32_factor_from_storage(
                &mut factor,
                0,
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f32_workspace_init(
                &mut workspace,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f32_push(&mut factor, ptr::null(), 0, ptr::null(), 0, 1.0, &workspace,),
            RLUMOD_STATUS_CAPACITY_EXCEEDED
        );
        assert_eq!(
            rlumod_f32_solve_in_place(&factor, ptr::null_mut(), 0, ptr::null_mut(),),
            RLUMOD_STATUS_OK
        );
    }
}

#[test]
fn maps_errors_without_mutating_outputs() {
    let (mut l, mut u) = ([0.0; 2], [0.0; 1]);
    let mut factor = F64Factor::default();
    let before = factor;
    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                l.as_mut_ptr(),
                1,
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_INSUFFICIENT_STORAGE
        );
    }
    assert_eq!(factor, before);

    let (mut y, mut z, mut w) = ([0.0; 1], [0.0; 1], [0.0; 1]);
    let mut workspace = F64Workspace::default();
    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                l.as_mut_ptr(),
                l.len(),
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_workspace_init(
                &mut workspace,
                y.as_mut_ptr(),
                y.len(),
                z.as_mut_ptr(),
                z.len(),
                w.as_mut_ptr(),
                w.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_push(&mut factor, ptr::null(), 0, ptr::null(), 0, 0.0, &workspace,),
            RLUMOD_STATUS_OK
        );
    }

    let mut rhs = [9.0];
    let mut error_index = usize::MAX;
    unsafe {
        assert_eq!(
            rlumod_f64_solve_in_place(&factor, rhs.as_mut_ptr(), 1, &mut error_index),
            RLUMOD_STATUS_SINGULAR
        );
    }
    assert_eq!((rhs, error_index), ([9.0], 0));

    unsafe { factor.u.write(f64::NAN) };
    error_index = usize::MAX;
    unsafe {
        assert_eq!(
            rlumod_f64_solve_transpose_in_place(&factor, rhs.as_mut_ptr(), 1, &mut error_index,),
            RLUMOD_STATUS_NON_FINITE_DIAGONAL
        );
        error_index = 37;
        assert_eq!(
            rlumod_f64_solve_in_place(&factor, rhs.as_mut_ptr(), 0, &mut error_index),
            RLUMOD_STATUS_LENGTH_MISMATCH
        );
    }
    assert_eq!((rhs, error_index), ([9.0], 37));
}

#[test]
fn rejects_invalid_pointer_regions_before_touching_memory() {
    let mut bytes = [0_u8; 128];
    let misaligned = unsafe { bytes.as_mut_ptr().add(1).cast::<f64>() };
    let mut factor = F64Factor::default();
    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                misaligned,
                2,
                bytes.as_mut_ptr().add(64).cast(),
                1,
            ),
            RLUMOD_STATUS_MISALIGNED_POINTER
        );
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                ptr::null_mut(),
                2,
                bytes.as_mut_ptr().add(64).cast(),
                1,
            ),
            RLUMOD_STATUS_NULL_POINTER
        );

        let mut aligned_storage = [0.0_f64; 3];
        let aligned = aligned_storage.as_mut_ptr();
        assert_eq!(
            rlumod_f64_factor_from_storage(&mut factor, 0, 1, aligned, 2, aligned.add(1), 1,),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
    }
}

#[test]
fn rejects_overflow_and_every_mutable_alias_class() {
    let (mut l, mut u) = ([0.0_f64; 2], [0.0_f64; 1]);
    let mut factor = F64Factor::default();
    let impossible = (usize::MAX & !(core::mem::align_of::<f64>() - 1)) as *mut f64;
    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                impossible,
                2,
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_SIZE_OVERFLOW
        );
        assert_eq!(
            rlumod_f64_factor_from_storage(
                l.as_mut_ptr().cast(),
                0,
                1,
                l.as_mut_ptr(),
                l.len(),
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                l.as_mut_ptr(),
                l.len(),
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_OK
        );
    }

    let (mut y, mut z) = ([0.0; 1], [0.0; 1]);
    let mut workspace = F64Workspace::default();
    let before = workspace;
    unsafe {
        assert_eq!(
            rlumod_f64_workspace_init(
                &mut workspace,
                y.as_mut_ptr(),
                y.len(),
                y.as_mut_ptr(),
                y.len(),
                z.as_mut_ptr(),
                z.len(),
            ),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
    }
    assert_eq!(workspace, before);

    let mut rhs_canary = [41.0];
    unsafe {
        assert_eq!(
            rlumod_f64_solve_in_place(&factor, factor.l, 1, ptr::null_mut()),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_solve_in_place(
                &factor,
                rhs_canary.as_mut_ptr(),
                1,
                rhs_canary.as_mut_ptr().cast(),
            ),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
    }
    assert_eq!(rhs_canary, [41.0]);
}

#[test]
fn update_errors_and_remove_to_zero_preserve_state() {
    let (mut l, mut u) = ([0.0_f64; 2], [0.0_f64; 1]);
    let (mut y, mut z, mut w) = ([0.0; 1], [0.0; 1], [0.0; 1]);
    let mut factor = F64Factor::default();
    let mut workspace = F64Workspace::default();
    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                1,
                l.as_mut_ptr(),
                l.len(),
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_workspace_init(
                &mut workspace,
                y.as_mut_ptr(),
                1,
                z.as_mut_ptr(),
                1,
                w.as_mut_ptr(),
                1,
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_push(&mut factor, ptr::null(), 0, ptr::null(), 0, 3.0, &workspace,),
            RLUMOD_STATUS_OK
        );
    }
    let factor_before = factor;
    let l_before = l;
    let u_before = u;
    let mut removal = FfiRemoval {
        has_moved_row: 9,
        moved_row: 9,
        has_moved_column: 9,
        moved_column: 9,
    };
    unsafe {
        assert_eq!(
            rlumod_f64_replace_row(&mut factor, usize::MAX, [1.0].as_ptr(), 1, &workspace),
            RLUMOD_STATUS_ROW_OUT_OF_BOUNDS
        );
        assert_eq!(
            rlumod_f64_replace_column(&mut factor, usize::MAX, [1.0].as_ptr(), 1, &workspace,),
            RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS
        );
        assert_eq!(
            rlumod_f64_remove(&mut factor, 0, 1, &workspace, &mut removal),
            RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS
        );
    }
    assert_eq!((factor, l, u), (factor_before, l_before, u_before));
    assert_eq!(removal.has_moved_row, 9);

    unsafe {
        assert_eq!(
            rlumod_f64_remove(&mut factor, 0, 0, &workspace, ptr::null_mut()),
            RLUMOD_STATUS_OK
        );
        assert_eq!(factor.dimension, 0);
        assert_eq!(
            rlumod_f64_push(&mut factor, ptr::null(), 0, ptr::null(), 0, 4.0, &workspace,),
            RLUMOD_STATUS_OK
        );
    }
    assert_eq!(factor.dimension, 1);
}

#[test]
fn enforces_operation_specific_update_alias_rules() {
    let (mut l, mut u) = ([0.0_f64; 6], [0.0_f64; 3]);
    let (mut y, mut z, mut w) = ([0.0_f64; 4], [0.0_f64; 2], [0.0_f64; 2]);
    let mut factor = F64Factor::default();
    let mut workspace = F64Workspace::default();
    unsafe {
        assert_eq!(
            rlumod_f64_factor_from_storage(
                &mut factor,
                0,
                2,
                l.as_mut_ptr(),
                l.len(),
                u.as_mut_ptr(),
                u.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_workspace_init(
                &mut workspace,
                y.as_mut_ptr(),
                2,
                z.as_mut_ptr(),
                z.len(),
                w.as_mut_ptr(),
                w.len(),
            ),
            RLUMOD_STATUS_OK
        );
        assert_eq!(
            rlumod_f64_push(&mut factor, ptr::null(), 0, ptr::null(), 0, 3.0, &workspace),
            RLUMOD_STATUS_OK
        );
    }

    let factor_before = factor;
    let workspace_before = workspace;
    let (l_before, u_before, y_before, z_before, w_before) = (l, u, y, z, w);
    let input = [5.0_f64];
    let factor_l = factor.l;
    let workspace_y = workspace.y;
    unsafe {
        assert_eq!(
            rlumod_f64_push(&mut factor, factor_l, 1, input.as_ptr(), 1, 7.0, &workspace,),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_push(
                &mut factor,
                input.as_ptr(),
                1,
                workspace_y,
                1,
                7.0,
                &workspace,
            ),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_replace_row(&mut factor, 0, factor_l, 1, &workspace),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_replace_column(&mut factor, 0, workspace_y, 1, &workspace),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_remove(&mut factor, 0, 0, &workspace, factor_l.cast::<FfiRemoval>(),),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
        assert_eq!(
            rlumod_f64_remove(
                &mut factor,
                0,
                0,
                &workspace,
                workspace_y.cast::<FfiRemoval>(),
            ),
            RLUMOD_STATUS_OVERLAPPING_BUFFERS
        );
    }
    assert_eq!((factor, workspace), (factor_before, workspace_before));
    assert_eq!(
        (l, u, y, z, w),
        (l_before, u_before, y_before, z_before, w_before)
    );

    unsafe {
        assert_eq!(
            rlumod_f64_push(
                &mut factor,
                input.as_ptr(),
                input.len(),
                input.as_ptr(),
                input.len(),
                7.0,
                &workspace,
            ),
            RLUMOD_STATUS_OK
        );
    }
    assert_eq!(factor.dimension, 2);
}
