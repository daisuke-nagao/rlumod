// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use rlumod::{LuMod, Workspace};

const CAPACITY: usize = 2;
const L_LEN: usize = CAPACITY * (CAPACITY + 1);
const U_LEN: usize = CAPACITY * (CAPACITY + 1) / 2;

fn main() {
    // This portable example binary uses the standard runtime, but all storage
    // passed to the no_std rlumod library is fixed and stack allocated.
    let mut l = [0.0_f64; L_LEN];
    let mut u = [0.0_f64; U_LEN];
    let mut y = [0.0_f64; CAPACITY];
    let mut z = [0.0_f64; CAPACITY];
    let mut w = [0.0_f64; CAPACITY];

    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factors = LuMod::from_storage(0, CAPACITY, &mut l, &mut u).unwrap();
    factors.push(&[], &[], 2.0, &mut workspace).unwrap();
    factors.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();

    let mut rhs = [12.0, 17.0];
    factors.solve_in_place(&mut rhs).unwrap();
    assert!((rhs[0] - 1.0).abs() < 1.0e-12);
    assert!((rhs[1] - 2.0).abs() < 1.0e-12);
}
