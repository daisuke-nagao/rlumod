// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use rlumod::{ColumnIndex, LuMod, Removal, RowIndex, Workspace, storage_lengths};

fn main() {
    let capacity = 3;
    let lengths = storage_lengths(capacity).unwrap();
    let mut l = vec![0.0_f64; lengths.l];
    let mut u = vec![0.0_f64; lengths.u];
    let (mut y, mut z, mut w) = (
        vec![0.0_f64; capacity],
        vec![0.0_f64; capacity],
        vec![0.0_f64; capacity],
    );
    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factors = LuMod::from_storage(0, capacity, &mut l, &mut u).unwrap();

    // Build diag(1, 2, 3).
    factors.push(&[], &[], 1.0, &mut workspace).unwrap();
    factors.push(&[0.0], &[0.0], 2.0, &mut workspace).unwrap();
    factors
        .push(&[0.0, 0.0], &[0.0, 0.0], 3.0, &mut workspace)
        .unwrap();

    let mut row_ids = vec!["row-a", "row-b", "row-c"];
    let mut column_ids = vec!["column-a", "column-b", "column-c"];
    let removed_row = RowIndex(0);
    let removed_column = ColumnIndex(1);

    let removal = factors
        .remove(removed_row, removed_column, &mut workspace)
        .unwrap();
    assert_eq!(
        removal,
        Removal {
            moved_row: Some(RowIndex(2)),
            moved_column: Some(ColumnIndex(2)),
        }
    );

    // Keep external identities in the same order as the represented matrix.
    assert_eq!(row_ids.swap_remove(removed_row.0), "row-a");
    assert_eq!(column_ids.swap_remove(removed_column.0), "column-b");
    assert_eq!(row_ids, ["row-c", "row-b"]);
    assert_eq!(column_ids, ["column-a", "column-c"]);
    assert_eq!(factors.dimension(), 2);
}
