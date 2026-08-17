// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use super::super::*;
use std::format;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::wasm_bindgen_test as test;

const MAX_DIMENSION: usize = 3;
type ModificationCase<'a> = (i32, i32, i32, i32, usize, &'a [f64]);

fn make_identity(l: &mut [f64], u: &mut [f64], dimension: usize) {
    l.fill(0.0);
    u.fill(0.0);
    for index in 0..dimension {
        l[index * MAX_DIMENSION + index] = 1.0;
        let u_row = index * MAX_DIMENSION - index * index.saturating_sub(1) / 2;
        u[u_row] = 1.0;
    }
}

fn expect_factorization(
    l: &[f64],
    u: &[f64],
    expected_matrix: &[f64],
    dimension: usize,
    context: &str,
) {
    assert_eq!(expected_matrix.len(), dimension * dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            let transformed: f64 = (0..dimension)
                .map(|inner| {
                    l[row * MAX_DIMENSION + inner] * expected_matrix[inner * dimension + column]
                })
                .sum();
            let expected = if row <= column {
                u[u_index(row, column, MAX_DIMENSION)]
            } else {
                0.0
            };
            assert!(
                (transformed - expected).abs() <= 1.0e-12,
                "{context}: (L * A)[{row},{column}]={transformed}, U[{row},{column}]={expected}"
            );
        }
    }
}

#[test]
fn generates_an_elementary_transformation() {
    let (mut x, mut y, mut cs, mut sn) = (4.0, 2.0, 0.0, 0.0);
    elmgen(&mut x, &mut y, MACHINE_PRECISION, &mut cs, &mut sn);
    assert_eq!((x, y, cs, sn), (4.0, 0.0, 0.0, -0.5));
}

#[test]
fn applies_an_elementary_transformation_like_the_c_api() {
    let mut x = [1.0, 2.0, 99.0];
    let mut y = [3.0, 4.0, 88.0];

    elm(7, 2, &mut x, &mut y, -1.0, 0.5);

    assert_eq!(x, [3.0, 4.0, 99.0]);
    assert_eq!(y, [2.5, 4.0, 88.0]);
}

#[test]
fn generates_every_elementary_transformation_shape() {
    let (mut x, mut y, mut cs, mut sn) = (0.0, 0.0, 9.0, 9.0);
    elmgen(&mut x, &mut y, MACHINE_PRECISION, &mut cs, &mut sn);
    assert_eq!((x, y, cs, sn), (0.0, 0.0, 0.0, 0.0));

    x = 0.0;
    y = MACHINE_PRECISION * TINY_NUMBER / 2.0;
    elmgen(&mut x, &mut y, MACHINE_PRECISION, &mut cs, &mut sn);
    assert_eq!((x, y), (0.0, 0.0));

    x = 1.0;
    y = 4.0;
    elmgen(&mut x, &mut y, MACHINE_PRECISION, &mut cs, &mut sn);
    assert_eq!((x, y, cs, sn), (4.0, 0.0, -1.0, -0.25));
}

#[test]
fn multiplies_and_solves_triangular_factors() {
    let mut l = [0.0; MAX_DIMENSION * MAX_DIMENSION];
    let mut u = [0.0; MAX_DIMENSION * (MAX_DIMENSION + 1) / 2];
    make_identity(&mut l, &mut u, 2);
    l[3] = 2.0;
    u[0] = 2.0;
    u[1] = 1.0;
    u[3] = 4.0;
    let mut y = [3.0, 4.0];
    let mut z = [0.0; 2];
    Lprod(1, MAX_DIMENSION as i32, 2, &mut l, &mut y, &mut z);
    assert_eq!((z[0], z[1]), (3.0, 10.0));
    Lprod(2, MAX_DIMENSION as i32, 2, &mut l, &mut y, &mut z);
    assert_eq!((z[0], z[1]), (11.0, 4.0));
    let mut rhs = [5.0, 8.0];
    Usolve(1, MAX_DIMENSION as i32, 2, &mut u, &mut rhs);
    assert!((rhs[0] - 1.5).abs() <= 1.0e-12);
    assert!((rhs[1] - 2.0).abs() <= 1.0e-12);
    let mut transpose_rhs = [4.0, 10.0];
    Usolve(2, MAX_DIMENSION as i32, 2, &mut u, &mut transpose_rhs);
    assert!((transpose_rhs[0] - 2.0).abs() <= 1.0e-12);
    assert!((transpose_rhs[1] - 2.0).abs() <= 1.0e-12);

    let mut singular_u = [0.0; MAX_DIMENSION * (MAX_DIMENSION + 1) / 2];
    let mut singular_rhs = [1.0];
    Usolve(
        1,
        MAX_DIMENSION as i32,
        1,
        &mut singular_u,
        &mut singular_rhs,
    );
    assert!(singular_rhs[0].is_infinite());
}

#[test]
fn sweeps_factors_forward_and_backward() {
    let setup = || {
        let mut l = [0.0; MAX_DIMENSION * MAX_DIMENSION];
        let mut u = [0.0; MAX_DIMENSION * (MAX_DIMENSION + 1) / 2];
        make_identity(&mut l, &mut u, 2);
        (l, u)
    };
    let (mut l, mut u) = setup();
    let mut y = [2.0, 3.0];
    LUforw(0, 1, 2, 2, 3, MACHINE_PRECISION, &mut l, &mut u, &mut y);
    assert_eq!(y, [0.0, -1.5]);
    assert_eq!(&l[..6], &[0.0, 1.0, 0.0, 1.0, -0.5, 0.0]);
    assert_eq!(&u[..4], &[2.0, 3.0, 0.0, -1.5]);

    let (mut l, mut u) = setup();
    let mut y = [0.0, 3.0];
    LUforw(0, 1, 2, 1, 3, MACHINE_PRECISION, &mut l, &mut u, &mut y);
    LUforw(1, 1, 2, 1, 3, MACHINE_PRECISION, &mut l, &mut u, &mut y);

    let (mut l, mut u) = setup();
    let mut y = [0.0; 2];
    let mut z = [1.0, 2.0];
    let mut last = 1;
    LUback(
        0,
        &mut last,
        2,
        2,
        3,
        MACHINE_PRECISION,
        &mut l,
        &mut u,
        &mut y,
        &mut z,
    );
    assert_eq!(last, 1);
    assert_eq!(y, [0.0, 1.0]);
    assert_eq!(z, [0.0, 2.0]);
    assert_eq!(&l[..6], &[1.0, -0.5, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(&u[..4], &[1.0, -0.5, 0.0, 1.0]);

    let (mut l, mut u) = setup();
    let mut y = [0.0; 2];
    let mut z = [0.0, 2.0];
    let mut last = 1;
    LUback(
        0,
        &mut last,
        2,
        1,
        3,
        MACHINE_PRECISION,
        &mut l,
        &mut u,
        &mut y,
        &mut z,
    );
    last = -2;
    LUback(
        0,
        &mut last,
        2,
        1,
        3,
        MACHINE_PRECISION,
        &mut l,
        &mut u,
        &mut y,
        &mut z,
    );
    assert_eq!(last, 1);
}

#[test]
fn exercises_every_factor_modification_mode() {
    let valid_cases: &[ModificationCase<'_>] = &[
        (1, 1, -1, -1, 1, &[2.0]),
        (1, 2, -1, -1, 2, &[1.0, 1.0, 2.0, 3.0]),
        (2, 2, -1, 1, 2, &[1.0, 1.0, 0.0, 2.0]),
        (2, 2, -1, 0, 2, &[1.0, 0.0, 2.0, 1.0]),
        (3, 1, 0, -1, 1, &[2.0]),
        (3, 2, 0, -1, 2, &[2.0, 3.0, 0.0, 1.0]),
        (4, 2, 1, 1, 1, &[1.0]),
        (4, 2, 0, 0, 1, &[1.0]),
    ];
    for &(mode, dimension, row, column, result_dimension, expected) in valid_cases {
        let mut l = [0.0; MAX_DIMENSION * MAX_DIMENSION];
        let mut u = [0.0; MAX_DIMENSION * (MAX_DIMENSION + 1) / 2];
        make_identity(&mut l, &mut u, dimension as usize);
        let mut y = [2.0, 3.0, 4.0];
        let mut z = [1.0, 2.0, 3.0];
        let mut work = [0.0; 3];
        LUmod(
            mode, 3, dimension, row, column, &mut l, &mut u, &mut y, &mut z, &mut work,
        );
        expect_factorization(
            &l,
            &u,
            expected,
            result_dimension,
            &format!("mode {mode}, n {dimension}, row {row}, column {column}"),
        );
    }

    for (mode, dimension, row, column) in [
        (99, 2, -1, -1),
        (1, 0, -1, -1),
        (2, 2, -1, -1),
        (2, 2, -1, 2),
        (3, 2, -1, -1),
        (3, 2, 2, -1),
        (4, 0, 0, 0),
        (4, 2, -1, 0),
        (4, 2, 2, 0),
        (4, 2, 0, -1),
        (4, 2, 0, 2),
    ] {
        let mut l = [0.0; MAX_DIMENSION * MAX_DIMENSION];
        let mut u = [0.0; MAX_DIMENSION * (MAX_DIMENSION + 1) / 2];
        make_identity(&mut l, &mut u, dimension.max(1) as usize);
        let mut y = [2.0, 3.0, 4.0];
        let mut z = [1.0, 2.0, 3.0];
        let mut work = [0.0; 3];
        let before = (l, u, y, z, work);
        LUmod(
            mode, 3, dimension, row, column, &mut l, &mut u, &mut y, &mut z, &mut work,
        );
        assert_eq!(
            (l, u, y, z, work),
            before,
            "mode {mode}, n {dimension}, row {row}, column {column} must be a no-op"
        );
    }
}

#[test]
fn handles_empty_forward_and_backward_sweeps() {
    let mut l = [0.0];
    let mut u = [0.0];
    let mut y = [7.0];
    LUforw(0, 0, 1, 1, 1, MACHINE_PRECISION, &mut l, &mut u, &mut y);
    assert_eq!(u, [7.0]);
    LUforw(0, 0, 0, 0, 1, MACHINE_PRECISION, &mut [], &mut [], &mut []);

    let mut last = 9;
    LUback(
        0,
        &mut last,
        0,
        0,
        1,
        MACHINE_PRECISION,
        &mut [],
        &mut [],
        &mut [],
        &mut [],
    );
    assert_eq!(last, 0);

    let mut first = [1.0];
    let mut second = [2.0];
    apply_pair(&mut first[0], &mut second[0], 1.0, 0.0);
    assert_eq!((first, second), ([1.0], [2.0]));
}
