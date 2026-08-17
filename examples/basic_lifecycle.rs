// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use rlumod::{ColumnIndex, LuMod, Removal, RowIndex, Workspace, storage_lengths};

fn main() {
    let capacity = 3;
    let lengths = storage_lengths(capacity).unwrap();

    // Vec is convenient when capacity is chosen at run time. LuMod itself does
    // not allocate; examples/static_storage.rs shows the fixed-array form.
    let mut l = vec![0.0_f64; lengths.l];
    let mut u = vec![0.0_f64; lengths.u];
    let (mut y, mut z, mut w) = (
        vec![0.0_f64; capacity],
        vec![0.0_f64; capacity],
        vec![0.0_f64; capacity],
    );

    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factors = LuMod::from_storage(0, capacity, &mut l, &mut u).unwrap();

    factors.push(&[], &[], 2.0, &mut workspace).unwrap();
    factors.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();

    // The represented matrix is [[2, 5], [3, 7]].
    let mut rhs = [12.0, 17.0];
    factors.solve_in_place(&mut rhs).unwrap();
    assert!((rhs[0] - 1.0).abs() < 1.0e-12);
    assert!((rhs[1] - 2.0).abs() < 1.0e-12);

    let mut transposed_rhs = [8.0, 19.0];
    factors
        .solve_transpose_in_place(&mut transposed_rhs)
        .unwrap();
    assert!((transposed_rhs[0] - 1.0).abs() < 1.0e-12);
    assert!((transposed_rhs[1] - 2.0).abs() < 1.0e-12);

    factors
        .replace_row(RowIndex(0), &[6.0, 5.0], &mut workspace)
        .unwrap();
    factors
        .replace_column(ColumnIndex(0), &[6.0, 8.0], &mut workspace)
        .unwrap();

    let removal = factors
        .remove(RowIndex(0), ColumnIndex(1), &mut workspace)
        .unwrap();
    assert_eq!(
        removal,
        Removal {
            moved_row: Some(RowIndex(1)),
            moved_column: None,
        }
    );

    let mut rhs = [16.0];
    factors.solve_in_place(&mut rhs).unwrap();
    assert!((rhs[0] - 2.0).abs() < 1.0e-12);
}
