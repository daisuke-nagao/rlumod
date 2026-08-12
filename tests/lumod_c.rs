// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![cfg(feature = "lumod-c")]

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::{wasm_bindgen_test as test, wasm_bindgen_test_configure};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
wasm_bindgen_test_configure!(run_in_browser);

use rlumod::lumod_c::{LUback, LUforw, LUmod, Lprod, Usolve};

#[test]
fn exposes_lumod_c_algorithm_at_module_root() {
    let _ = LUmod;
    let _ = Lprod;
    let _ = LUforw;
    let _ = LUback;
    let _ = Usolve;
}

#[test]
fn safe_and_lumod_c_share_the_same_update_and_solve_contract() {
    use rlumod::{ColumnIndex, LuMod, RowIndex, Workspace};

    const CAPACITY: usize = 3;
    let (mut safe_l, mut safe_u) = (vec![0.0; 12], vec![0.0; 6]);
    let (mut safe_y, mut safe_z, mut safe_w) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    let (safe_solution, safe_transpose_solution);
    {
        let mut workspace = Workspace::new(&mut safe_y, &mut safe_z, &mut safe_w).unwrap();
        let mut factor = LuMod::from_storage(0, CAPACITY, &mut safe_l, &mut safe_u).unwrap();
        factor.push(&[], &[], 4.0, &mut workspace).unwrap();
        factor.push(&[1.0], &[2.0], 5.0, &mut workspace).unwrap();
        factor
            .push(&[3.0, 1.0], &[1.0, 2.0], 6.0, &mut workspace)
            .unwrap();
        factor
            .replace_row(RowIndex(1), &[2.0, 7.0, 1.0], &mut workspace)
            .unwrap();
        factor
            .replace_column(ColumnIndex(0), &[8.0, 2.0, 1.0], &mut workspace)
            .unwrap();
        factor
            .remove(RowIndex(0), ColumnIndex(1), &mut workspace)
            .unwrap();

        let mut solution = [9.0, 11.0];
        factor.solve_in_place(&mut solution).unwrap();
        safe_solution = solution;
        let mut transpose_solution = [7.0, 13.0];
        factor
            .solve_transpose_in_place(&mut transpose_solution)
            .unwrap();
        safe_transpose_solution = transpose_solution;
    }

    let (mut c_l, mut c_u) = (vec![0.0; 9], vec![0.0; 6]);
    let (mut c_y, mut c_z, mut c_w) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for (n, row, column) in [
        (1, &[4.0][..], &[4.0][..]),
        (2, &[1.0, 5.0][..], &[2.0, 5.0][..]),
        (3, &[3.0, 1.0, 6.0][..], &[1.0, 2.0, 6.0][..]),
    ] {
        c_y[..n].copy_from_slice(row);
        c_z[..n].copy_from_slice(column);
        LUmod(
            1,
            CAPACITY as i32,
            n as i32,
            -1,
            -1,
            &mut c_l,
            &mut c_u,
            &mut c_y,
            &mut c_z,
            &mut c_w,
        );
    }
    c_y.copy_from_slice(&[2.0, 7.0, 1.0]);
    LUmod(
        3, 3, 3, 1, -1, &mut c_l, &mut c_u, &mut c_y, &mut c_z, &mut c_w,
    );
    c_z.copy_from_slice(&[8.0, 2.0, 1.0]);
    LUmod(
        2, 3, 3, -1, 0, &mut c_l, &mut c_u, &mut c_y, &mut c_z, &mut c_w,
    );
    LUmod(
        4, 3, 3, 0, 1, &mut c_l, &mut c_u, &mut c_y, &mut c_z, &mut c_w,
    );

    assert_eq!(&safe_l[..9], c_l);
    assert_eq!(safe_u, c_u);
    assert_eq!(safe_y, c_y);
    assert_eq!(safe_z, c_z);
    assert_eq!(safe_w, c_w);

    let mut c_solution = [9.0, 11.0];
    let mut c_work = [0.0; 2];
    Lprod(1, 3, 2, &mut c_l, &mut c_solution, &mut c_work);
    c_solution.copy_from_slice(&c_work);
    Usolve(1, 3, 2, &mut c_u, &mut c_solution);
    assert_eq!(safe_solution, c_solution);

    let mut c_transpose_solution = [7.0, 13.0];
    Usolve(2, 3, 2, &mut c_u, &mut c_transpose_solution);
    Lprod(2, 3, 2, &mut c_l, &mut c_transpose_solution, &mut c_work);
    c_transpose_solution.copy_from_slice(&c_work);
    assert_eq!(safe_transpose_solution, c_transpose_solution);
}
