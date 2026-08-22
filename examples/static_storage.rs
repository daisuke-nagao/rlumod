// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use rlumod::{LuMod, Workspace};

const CAPACITY: usize = 2;

fn main() {
    // This portable example binary uses the standard runtime, but the macro
    // creates fixed-size storage without heap allocation. It does not promise
    // a physical stack allocation for these local values.
    let (mut l, mut u, mut y, mut z, mut w) = rlumod::stack_storage!(f64; CAPACITY);

    let mut workspace = Workspace::new(&mut y, &mut z, &mut w).unwrap();
    let mut factors = LuMod::from_storage(0, CAPACITY, &mut l, &mut u).unwrap();
    factors.push(&[], &[], 2.0, &mut workspace).unwrap();
    factors.push(&[3.0], &[5.0], 7.0, &mut workspace).unwrap();

    let mut rhs = [12.0, 17.0];
    factors.solve_in_place(&mut rhs).unwrap();
    assert!((rhs[0] - 1.0).abs() < 1.0e-12);
    assert!((rhs[1] - 2.0).abs() < 1.0e-12);
}
