// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use rlumod::{
    ColumnIndex, LuMod, Real, Removal, RowIndex, SolveError, StorageError, StorageLengths,
    UpdateError, Workspace, storage_lengths,
};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::{wasm_bindgen_test as test, wasm_bindgen_test_configure};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
wasm_bindgen_test_configure!(run_in_browser);

const STATIC_LENGTHS: StorageLengths = match storage_lengths(3) {
    Ok(lengths) => lengths,
    Err(_) => panic!("unexpected storage overflow"),
};
const STATIC_L: [f64; STATIC_LENGTHS.l] = [0.0; STATIC_LENGTHS.l];
const STATIC_U: [f64; STATIC_LENGTHS.u] = [0.0; STATIC_LENGTHS.u];
const STATIC_Y: [f64; 3] = [0.0; 3];
const STATIC_Z: [f64; 3] = [0.0; 3];
const STATIC_W: [f64; 3] = [0.0; 3];

#[test]
fn supports_const_storage_lengths() {
    assert_eq!(STATIC_LENGTHS, StorageLengths { l: 12, u: 6 });
    assert_eq!(STATIC_L, [0.0; 12]);
    assert_eq!(STATIC_U, [0.0; 6]);
    assert_eq!(STATIC_Y, [0.0; 3]);
    assert_eq!(STATIC_Z, [0.0; 3]);
    assert_eq!(STATIC_W, [0.0; 3]);
}

const MACRO_CAPACITY: usize = 3;

#[test]
fn stack_storage_macro_initializes_fixed_storage() {
    let macro_storage_f64 = rlumod::stack_storage!(f64; MACRO_CAPACITY);
    let macro_storage_f32_zero = rlumod::stack_storage!(f32; 0);

    assert_eq!(macro_storage_f64.0, [0.0; 12]);
    assert_eq!(macro_storage_f64.1, [0.0; 6]);
    assert_eq!(macro_storage_f64.2, [0.0; 3]);
    assert_eq!(macro_storage_f64.3, [0.0; 3]);
    assert_eq!(macro_storage_f64.4, [0.0; 3]);
    assert!(macro_storage_f32_zero.0.is_empty());
    assert!(macro_storage_f32_zero.1.is_empty());
    assert!(macro_storage_f32_zero.2.is_empty());
    assert!(macro_storage_f32_zero.3.is_empty());
    assert!(macro_storage_f32_zero.4.is_empty());

    let repeated = rlumod::stack_storage!(f64; 3);
    assert_eq!(repeated.0, [0.0; 12]);
    assert_eq!(repeated.1, [0.0; 6]);
}

#[test]
fn calculates_storage_lengths_without_overflow() {
    for (capacity, l, u) in [(0, 0, 0), (1, 2, 1), (2, 6, 3), (3, 12, 6)] {
        let lengths: StorageLengths = storage_lengths(capacity).unwrap();
        assert_eq!(lengths.l, l);
        assert_eq!(lengths.u, u);
    }

    assert_eq!(storage_lengths(usize::MAX), Err(StorageError::SizeOverflow));
}

#[test]
fn validates_storage_length_overflow_boundary() {
    let max_capacity = usize::MAX.isqrt();
    let lengths = storage_lengths(max_capacity).unwrap();

    assert_eq!(lengths.l, max_capacity * (max_capacity + 1));
    assert_eq!(lengths.u, max_capacity * max_capacity.div_ceil(2));
    assert_eq!(
        storage_lengths(max_capacity + 1),
        Err(StorageError::SizeOverflow)
    );
}

fn storage<T: Copy>(capacity: usize, zero: T) -> (Vec<T>, Vec<T>) {
    let lengths = storage_lengths(capacity).unwrap();
    (vec![zero; lengths.l], vec![zero; lengths.u])
}

macro_rules! lifecycle {
    ($name:ident, $float:ty) => {
        #[test]
        fn $name() {
            let (mut l, mut u) = storage(3, 0.0 as $float);
            let (mut y, mut z, mut w) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3]);
            let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
            let mut factor = LuMod::from_storage(0, 3, &mut l, &mut u).unwrap();

            factor.push(&[], &[], 2.0, &mut workspace).unwrap();
            factor.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();
            assert_eq!(factor.dimension(), 2);
            assert_eq!(factor.capacity(), 3);

            let mut rhs = [12.0, 17.0];
            factor.solve_in_place(&mut rhs).unwrap();
            assert!((rhs[0] - 1.0).abs() <= 32.0 * <$float>::EPSILON);
            assert!((rhs[1] - 2.0).abs() <= 32.0 * <$float>::EPSILON);

            let mut transposed_rhs = [8.0, 19.0];
            factor
                .solve_transpose_in_place(&mut transposed_rhs)
                .unwrap();
            assert!((transposed_rhs[0] - 1.0).abs() <= 32.0 * <$float>::EPSILON);
            assert!((transposed_rhs[1] - 2.0).abs() <= 32.0 * <$float>::EPSILON);

            factor
                .replace_row(RowIndex(0), &[6.0, 5.0], &mut workspace)
                .unwrap();
            factor
                .replace_column(ColumnIndex(0), &[6.0, 8.0], &mut workspace)
                .unwrap();
            let removal = factor
                .remove(RowIndex(0), ColumnIndex(1), &mut workspace)
                .unwrap();
            assert_eq!(
                removal,
                Removal {
                    moved_row: Some(RowIndex(1)),
                    moved_column: None,
                }
            );
            assert_eq!(factor.dimension(), 1);
            let mut rhs = [16.0];
            factor.solve_in_place(&mut rhs).unwrap();
            assert!((rhs[0] - 2.0).abs() <= 32.0 * <$float>::EPSILON);
        }
    };
}

lifecycle!(supports_the_complete_f32_lifecycle, f32);
lifecycle!(supports_the_complete_f64_lifecycle, f64);

#[test]
fn validates_storage_without_touching_it() {
    let lengths = storage_lengths(2).unwrap();
    let (l_len, u_len) = (lengths.l, lengths.u);
    let mut exact_l = vec![11.0; l_len];
    let mut exact_u = vec![13.0; u_len];
    let l_before = exact_l.clone();
    let u_before = exact_u.clone();
    {
        let factor = LuMod::from_storage(2, 2, &mut exact_l, &mut exact_u).unwrap();
        assert_eq!(factor.dimension(), 2);
    }
    assert_eq!(exact_l, l_before);
    assert_eq!(exact_u, u_before);

    let mut short_l = vec![0.0; l_len - 1];
    let mut u = vec![0.0; u_len];
    assert!(matches!(
        LuMod::from_storage(0, 2, &mut short_l, &mut u),
        Err(StorageError::InsufficientStorage)
    ));

    let mut l = vec![0.0; l_len];
    let mut short_u = vec![0.0; u_len - 1];
    assert!(matches!(
        LuMod::from_storage(0, 2, &mut l, &mut short_u),
        Err(StorageError::InsufficientStorage)
    ));

    let (mut l, mut u) = storage(2, 0.0);
    assert!(matches!(
        LuMod::from_storage(3, 2, &mut l, &mut u),
        Err(StorageError::InvalidDimension)
    ));

    assert!(matches!(
        LuMod::<f64>::from_storage(0, usize::MAX, &mut [], &mut []),
        Err(StorageError::SizeOverflow)
    ));
}

#[test]
fn validates_workspace_shape_and_operation_capacity() {
    let (mut y, mut z, mut short_w) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 1]);
    assert!(matches!(
        Workspace::new(&mut y, &mut z, &mut short_w),
        Err(StorageError::InsufficientStorage)
    ));

    let (mut l, mut u) = storage(2, 0.0);
    let mut factor = LuMod::from_storage(0, 2, &mut l, &mut u).unwrap();
    let (mut y, mut z, mut w) = (vec![0.0; 0], vec![0.0; 0], vec![0.0; 0]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    assert_eq!(
        factor.push(&[], &[], 1.0, &mut workspace),
        Err(UpdateError::InsufficientWorkspace)
    );

    let (mut y, mut z, mut w) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    factor.push(&[], &[], 1.0, &mut workspace).unwrap();
    factor.push(&[0.0], &[0.0], 1.0, &mut workspace).unwrap();
    assert_eq!(
        factor.push(&[0.0, 0.0], &[0.0, 0.0], 1.0, &mut workspace),
        Err(UpdateError::CapacityExceeded)
    );
}

#[test]
fn structural_update_errors_leave_the_factor_unchanged() {
    let (mut l, mut u) = storage(2, 0.0);
    let (mut y, mut z, mut w) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    {
        let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
        let mut factor = LuMod::from_storage(0, 2, &mut l, &mut u).unwrap();
        factor.push(&[], &[], 2.0, &mut workspace).unwrap();
        factor.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();
    }
    let l_before = l.clone();
    let u_before = u.clone();

    {
        let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
        let mut factor = LuMod::from_storage(2, 2, &mut l, &mut u).unwrap();
        assert_eq!(
            factor.replace_row(RowIndex(2), &[1.0, 2.0], &mut workspace),
            Err(UpdateError::RowOutOfBounds)
        );
        assert_eq!(
            factor.replace_column(ColumnIndex(usize::MAX), &[1.0, 2.0], &mut workspace),
            Err(UpdateError::ColumnOutOfBounds)
        );
        assert_eq!(
            factor.replace_row(RowIndex(0), &[1.0], &mut workspace),
            Err(UpdateError::LengthMismatch)
        );
        assert_eq!(
            factor.remove(RowIndex(0), ColumnIndex(2), &mut workspace),
            Err(UpdateError::ColumnOutOfBounds)
        );
        assert_eq!(factor.dimension(), 2);
    }

    assert_eq!(l, l_before);
    assert_eq!(u, u_before);
}

#[test]
fn singular_updates_succeed_and_solve_errors_preserve_rhs() {
    let (mut l, mut u) = storage(1, 0.0);
    let (mut y, mut z, mut w) = (vec![0.0; 1], vec![0.0; 1], vec![0.0; 1]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(0, 1, &mut l, &mut u).unwrap();
    factor.push(&[], &[], 0.0, &mut workspace).unwrap();

    for transpose in [false, true] {
        let mut rhs = [9.0];
        let before = rhs.map(f64::to_bits);
        let error = if transpose {
            factor.solve_transpose_in_place(&mut rhs)
        } else {
            factor.solve_in_place(&mut rhs)
        };
        assert_eq!(error, Err(SolveError::Singular { index: 0 }));
        assert_eq!(rhs.map(f64::to_bits), before);
    }

    factor
        .replace_column(ColumnIndex(0), &[2.0], &mut workspace)
        .unwrap();
    let mut rhs = [6.0];
    factor.solve_in_place(&mut rhs).unwrap();
    assert_eq!(rhs, [3.0]);
}

#[test]
fn solve_rejects_only_invalid_length_and_diagonal() {
    for diagonal in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (mut l, mut u) = storage(1, 0.0);
        l[0] = 1.0;
        u[0] = diagonal;
        let factor = LuMod::from_storage(1, 1, &mut l, &mut u).unwrap();
        for transpose in [false, true] {
            let mut rhs = [4.0];
            let before = rhs.map(f64::to_bits);
            let result = if transpose {
                factor.solve_transpose_in_place(&mut rhs)
            } else {
                factor.solve_in_place(&mut rhs)
            };
            assert_eq!(result, Err(SolveError::NonFiniteDiagonal { index: 0 }));
            assert_eq!(rhs.map(f64::to_bits), before);
        }
    }

    let (mut l, mut u) = storage(1, 0.0);
    l[0] = 1.0;
    u[0] = -0.0;
    let factor = LuMod::from_storage(1, 1, &mut l, &mut u).unwrap();
    assert_eq!(
        factor.solve_in_place(&mut []),
        Err(SolveError::LengthMismatch)
    );
    assert_eq!(
        factor.solve_transpose_in_place(&mut [1.0]),
        Err(SolveError::Singular { index: 0 })
    );

    let (mut l, mut u) = storage(1, 0.0);
    l[0] = 1.0;
    u[0] = f64::MIN_POSITIVE;
    let factor = LuMod::from_storage(1, 1, &mut l, &mut u).unwrap();
    assert!(factor.solve_in_place(&mut [1.0]).is_ok());
}

#[test]
fn zero_dimension_and_removal_metadata_follow_last_swap_semantics() {
    let (mut l, mut u) = storage(1, 0.0);
    let (mut y, mut z, mut w) = (vec![0.0; 1], vec![0.0; 1], vec![0.0; 1]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(0, 1, &mut l, &mut u).unwrap();
    assert!(factor.solve_in_place(&mut []).is_ok());
    assert!(factor.solve_transpose_in_place(&mut []).is_ok());
    assert_eq!(
        factor.remove(RowIndex(0), ColumnIndex(0), &mut workspace),
        Err(UpdateError::RowOutOfBounds)
    );

    factor.push(&[], &[], 3.0, &mut workspace).unwrap();
    assert_eq!(
        factor
            .remove(RowIndex(0), ColumnIndex(0), &mut workspace)
            .unwrap(),
        Removal {
            moved_row: None,
            moved_column: None,
        }
    );
    assert_eq!(factor.dimension(), 0);
}

#[test]
fn removal_keeps_the_last_row_candidate_when_its_deleted_column_entry_is_zero() {
    let (mut l, mut u) = storage(2, 0.0);
    l[0] = 1.0;
    l[3] = 1.0;
    u[0] = 1.0;
    u[2] = 1.0;
    let (mut y, mut z, mut w) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(2, 2, &mut l, &mut u).unwrap();

    assert_eq!(
        factor
            .remove(RowIndex(0), ColumnIndex(0), &mut workspace)
            .unwrap(),
        Removal {
            moved_row: Some(RowIndex(1)),
            moved_column: Some(ColumnIndex(1)),
        }
    );
    let mut rhs = [4.0];
    factor.solve_in_place(&mut rhs).unwrap();
    assert_eq!(rhs, [4.0]);
}

#[test]
fn every_last_swap_removal_keeps_the_reduced_matrix_solvable() {
    let matrix: [[f64; 4]; 4] = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 3.0, 9.0, 27.0],
        [1.0, 4.0, 16.0, 64.0],
    ];
    for removed_row in 0..4 {
        for removed_column in 0..4 {
            let (mut l, mut u) = storage(4, 0.0);
            let (mut y, mut z, mut w) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
            let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
            let mut factor = LuMod::from_storage(0, 4, &mut l, &mut u).unwrap();
            for (dimension, matrix_row) in matrix.iter().enumerate() {
                let row = (0..dimension)
                    .map(|column| matrix_row[column])
                    .collect::<Vec<_>>();
                let column = (0..dimension)
                    .map(|row| matrix[row][dimension])
                    .collect::<Vec<_>>();
                factor
                    .push(&row, &column, matrix_row[dimension], &mut workspace)
                    .unwrap();
            }
            factor
                .remove(
                    RowIndex(removed_row),
                    ColumnIndex(removed_column),
                    &mut workspace,
                )
                .unwrap();

            let rows = [
                if removed_row == 0 { 3 } else { 0 },
                if removed_row == 1 { 3 } else { 1 },
                if removed_row == 2 { 3 } else { 2 },
            ];
            let columns = [
                if removed_column == 0 { 3 } else { 0 },
                if removed_column == 1 { 3 } else { 1 },
                if removed_column == 2 { 3 } else { 2 },
            ];
            let solution = [1.0, 2.0, 3.0];
            let mut rhs = [0.0; 3];
            for row in 0..3 {
                for column in 0..3 {
                    rhs[row] += matrix[rows[row]][columns[column]] * solution[column];
                }
            }
            factor.solve_in_place(&mut rhs).unwrap();
            for index in 0..3 {
                assert!(
                    (rhs[index] - solution[index]).abs() <= 1.0e-10,
                    "row={removed_row}, column={removed_column}, index={index}, actual={}",
                    rhs[index]
                );
            }

            let mut transposed_rhs = [0.0; 3];
            for row in 0..3 {
                for column in 0..3 {
                    transposed_rhs[column] += matrix[rows[row]][columns[column]] * solution[row];
                }
            }
            factor
                .solve_transpose_in_place(&mut transposed_rhs)
                .unwrap();
            for index in 0..3 {
                assert!(
                    (transposed_rhs[index] - solution[index]).abs() <= 1.0e-10,
                    "transpose row={removed_row}, column={removed_column}, index={index}, actual={}",
                    transposed_rhs[index]
                );
            }
        }
    }
}

#[test]
fn non_finite_update_values_are_structurally_valid() {
    let (mut l, mut u) = storage(1, 0.0f32);
    let (mut y, mut z, mut w) = (vec![0.0; 1], vec![0.0; 1], vec![0.0; 1]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(0, 1, &mut l, &mut u).unwrap();
    assert!(factor.push(&[], &[], f32::NAN, &mut workspace).is_ok());
    assert_eq!(
        factor.solve_in_place(&mut [1.0]),
        Err(SolveError::NonFiniteDiagonal { index: 0 })
    );
}

#[test]
fn covers_remaining_validation_and_single_row_replacement_paths() {
    let (mut y, mut z, mut w) = (vec![0.0; 1], vec![], vec![0.0; 1]);
    assert!(matches!(
        Workspace::new(&mut y, &mut z, &mut w),
        Err(StorageError::InsufficientStorage)
    ));

    let (mut l, mut u) = storage(2, 0.0);
    let (mut y, mut z, mut w) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(0, 2, &mut l, &mut u).unwrap();
    assert_eq!(
        factor.push(&[1.0], &[], 2.0, &mut workspace),
        Err(UpdateError::LengthMismatch)
    );
    assert_eq!(
        factor.push(&[], &[1.0], 2.0, &mut workspace),
        Err(UpdateError::LengthMismatch)
    );
    factor.push(&[], &[], 2.0, &mut workspace).unwrap();

    let (mut short_y, mut short_z, mut short_w) = (vec![], vec![], vec![]);
    let mut short_workspace = Workspace::new(&mut short_y, &mut short_z, &mut short_w).unwrap();
    assert_eq!(
        factor.replace_row(RowIndex(0), &[4.0], &mut short_workspace),
        Err(UpdateError::InsufficientWorkspace)
    );
    assert_eq!(
        factor.remove(RowIndex(0), ColumnIndex(0), &mut short_workspace),
        Err(UpdateError::InsufficientWorkspace)
    );
    assert_eq!(factor.dimension(), 1);

    factor
        .replace_row(RowIndex(0), &[4.0], &mut workspace)
        .unwrap();
    let mut rhs = [8.0];
    factor.solve_in_place(&mut rhs).unwrap();
    assert_eq!(rhs, [2.0]);
}

fn exercise_numeric_boundary_paths<T: Real>() {
    let tiny = T::EPSILON * T::UPDATE_TINY;
    let half = T::ONE / (T::ONE + T::ONE);
    for (pivot, coupling) in [(tiny, tiny * half), (tiny * half, tiny)] {
        let (mut l, mut u) = storage(2, T::ZERO);
        let (mut y, mut z, mut w) = ([T::ZERO; 2], [T::ZERO; 2], [T::ZERO; 2]);
        let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
        let mut factor = LuMod::from_storage(0, 2, &mut l, &mut u).unwrap();
        factor.push(&[], &[], pivot, &mut workspace).unwrap();
        factor
            .push(&[coupling], &[T::ZERO], T::ONE, &mut workspace)
            .unwrap();
    }

    let (mut l, mut u) = storage(1, T::ZERO);
    let (mut y, mut z, mut w) = ([T::ZERO; 1], [T::ZERO; 1], [T::ZERO; 1]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(0, 1, &mut l, &mut u).unwrap();
    factor.push(&[], &[], T::ONE, &mut workspace).unwrap();
    assert_eq!(
        factor.replace_column(ColumnIndex(0), &[], &mut workspace),
        Err(UpdateError::LengthMismatch)
    );
    factor
        .replace_row(RowIndex(0), &[T::ONE], &mut workspace)
        .unwrap();

    let (mut l, mut u) = storage(3, T::ZERO);
    let (mut y, mut z, mut w) = ([T::ZERO; 3], [T::ZERO; 3], [T::ZERO; 3]);
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factor = LuMod::from_storage(0, 3, &mut l, &mut u).unwrap();
    factor.push(&[], &[], T::ONE, &mut workspace).unwrap();
    factor
        .push(&[T::ZERO], &[T::ZERO], T::ONE, &mut workspace)
        .unwrap();
    factor
        .push(
            &[T::ZERO, T::ZERO],
            &[T::ZERO, T::ZERO],
            T::ONE,
            &mut workspace,
        )
        .unwrap();
    factor
        .replace_column(ColumnIndex(0), &[T::ZERO, T::ZERO, T::ZERO], &mut workspace)
        .unwrap();
}

#[test]
fn covers_f32_and_f64_numeric_boundary_paths() {
    exercise_numeric_boundary_paths::<f32>();
    exercise_numeric_boundary_paths::<f64>();
}
