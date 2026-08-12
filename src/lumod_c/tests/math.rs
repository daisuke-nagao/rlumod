// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![allow(clippy::needless_range_loop)] // Storage formulas are clearest with explicit indices.

use super::super::*;
use crate::math_contract::{self, Factor, Matrix, Vector};
use std::{format, vec, vec::Vec};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::wasm_bindgen_test as test;

const CANARY: f64 = f64::from_bits(0x4272_3456_789a_bc00);

fn guarded(values: &Vector, capacity: usize) -> Vec<f64> {
    let mut result = vec![0.0; capacity + 2];
    result[0] = CANARY;
    result[capacity + 1] = -CANARY;
    result[1..1 + values.len()].copy_from_slice(values);
    result
}

fn logical(values: &[f64]) -> &[f64] {
    &values[1..values.len() - 1]
}

fn logical_mut(values: &mut [f64]) -> &mut [f64] {
    let end = values.len() - 1;
    &mut values[1..end]
}

fn expect_canaries(values: &[f64]) {
    assert_eq!(values.first(), Some(&CANARY));
    assert_eq!(values.last(), Some(&-CANARY));
}

struct DenseFactor {
    maximum_dimension: usize,
    dimension: usize,
    l_storage: Vec<f64>,
    u_storage: Vec<f64>,
}

impl DenseFactor {
    fn new(maximum_dimension: usize) -> Self {
        let l_storage = guarded(&vec![], maximum_dimension * maximum_dimension);
        let u_storage = guarded(&vec![], maximum_dimension * (maximum_dimension + 1) / 2);
        Self {
            maximum_dimension,
            dimension: 0,
            l_storage,
            u_storage,
        }
    }
}

impl Factor for DenseFactor {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn expand(&mut self, new_row: &Vector, new_column: &Vector) {
        let new_dimension = self.dimension + 1;
        assert_eq!(new_row.len(), new_dimension);
        assert_eq!(new_column.len(), new_dimension);
        let mut row = guarded(new_row, self.maximum_dimension);
        let mut column = guarded(new_column, self.maximum_dimension);
        let mut work = guarded(&vec![], self.maximum_dimension);
        LUmod(
            1,
            self.maximum_dimension as i32,
            new_dimension as i32,
            -1,
            -1,
            logical_mut(&mut self.l_storage),
            logical_mut(&mut self.u_storage),
            logical_mut(&mut row),
            logical_mut(&mut column),
            logical_mut(&mut work),
        );
        expect_canaries(&row);
        expect_canaries(&column);
        expect_canaries(&work);
        self.dimension = new_dimension;
    }

    fn replace_column(&mut self, column_index: usize, replacement: &Vector) {
        let mut row = guarded(&vec![], self.maximum_dimension);
        let mut column = guarded(replacement, self.maximum_dimension);
        let original = column.clone();
        let mut work = guarded(&vec![], self.maximum_dimension);
        LUmod(
            2,
            self.maximum_dimension as i32,
            self.dimension as i32,
            -1,
            column_index as i32,
            logical_mut(&mut self.l_storage),
            logical_mut(&mut self.u_storage),
            logical_mut(&mut row),
            logical_mut(&mut column),
            logical_mut(&mut work),
        );
        assert_eq!(
            column, original,
            "mode 2 must not alter the replacement column"
        );
        expect_canaries(&row);
        expect_canaries(&work);
    }

    fn replace_row(&mut self, row_index: usize, replacement: &Vector) {
        let mut row = guarded(replacement, self.maximum_dimension);
        let mut column = guarded(&vec![], self.maximum_dimension);
        let mut work = guarded(&vec![], self.maximum_dimension);
        LUmod(
            3,
            self.maximum_dimension as i32,
            self.dimension as i32,
            row_index as i32,
            -1,
            logical_mut(&mut self.l_storage),
            logical_mut(&mut self.u_storage),
            logical_mut(&mut row),
            logical_mut(&mut column),
            logical_mut(&mut work),
        );
        expect_canaries(&row);
        expect_canaries(&column);
        expect_canaries(&work);
    }

    fn erase(&mut self, row_index: usize, column_index: usize) {
        let mut row = guarded(&vec![], self.maximum_dimension);
        let mut column = guarded(&vec![], self.maximum_dimension);
        let mut work = guarded(&vec![], self.maximum_dimension);
        LUmod(
            4,
            self.maximum_dimension as i32,
            self.dimension as i32,
            row_index as i32,
            column_index as i32,
            logical_mut(&mut self.l_storage),
            logical_mut(&mut self.u_storage),
            logical_mut(&mut row),
            logical_mut(&mut column),
            logical_mut(&mut work),
        );
        expect_canaries(&row);
        expect_canaries(&column);
        expect_canaries(&work);
        self.dimension -= 1;
    }

    fn l(&self) -> Matrix {
        let mut result = vec![vec![0.0; self.dimension]; self.dimension];
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                result[row][column] =
                    logical(&self.l_storage)[row * self.maximum_dimension + column];
            }
        }
        result
    }

    fn u(&self) -> Matrix {
        let mut result = vec![vec![0.0; self.dimension]; self.dimension];
        for row in 0..self.dimension {
            let row_start = row * self.maximum_dimension - row * row.saturating_sub(1) / 2;
            for column in row..self.dimension {
                result[row][column] = logical(&self.u_storage)[row_start + column - row];
            }
        }
        result
    }

    fn solve(&mut self, rhs: &Vector, transpose: bool) -> Vector {
        let mut work = guarded(rhs, self.maximum_dimension);
        let mut result = guarded(&vec![], self.maximum_dimension);
        if !transpose {
            Lprod(
                1,
                self.maximum_dimension as i32,
                self.dimension as i32,
                logical_mut(&mut self.l_storage),
                logical_mut(&mut work),
                logical_mut(&mut result),
            );
            Usolve(
                1,
                self.maximum_dimension as i32,
                self.dimension as i32,
                logical_mut(&mut self.u_storage),
                logical_mut(&mut result),
            );
        } else {
            Usolve(
                2,
                self.maximum_dimension as i32,
                self.dimension as i32,
                logical_mut(&mut self.u_storage),
                logical_mut(&mut work),
            );
            Lprod(
                2,
                self.maximum_dimension as i32,
                self.dimension as i32,
                logical_mut(&mut self.l_storage),
                logical_mut(&mut work),
                logical_mut(&mut result),
            );
        }
        expect_canaries(&work);
        expect_canaries(&result);
        logical(&result)[..self.dimension].to_vec()
    }

    fn storage_is_valid(&self) -> bool {
        self.l_storage.first() == Some(&CANARY)
            && self.l_storage.last() == Some(&-CANARY)
            && self.u_storage.first() == Some(&CANARY)
            && self.u_storage.last() == Some(&-CANARY)
            && self
                .l()
                .iter()
                .flatten()
                .chain(self.u().iter().flatten())
                .all(|value| value.is_finite())
    }
}

#[test]
fn maintains_the_mathematical_contract_across_every_update_mode() {
    math_contract::exercise_complete_update_lifecycle(&mut DenseFactor::new(7));
}

#[test]
fn replaces_the_only_column_and_row() {
    let mut factor = DenseFactor::new(1);
    let mut expected = vec![vec![2.0]];
    factor.expand(&vec![2.0], &vec![91.0]);
    math_contract::expect_mathematically_correct(&mut factor, &expected, "scalar mode 1", true);
    factor.replace_column(0, &vec![-4.0]);
    expected[0][0] = -4.0;
    math_contract::expect_mathematically_correct(&mut factor, &expected, "scalar mode 2", true);
    factor.replace_row(0, &vec![3.0]);
    expected[0][0] = 3.0;
    math_contract::expect_mathematically_correct(&mut factor, &expected, "scalar mode 3", true);
}

#[test]
fn factors_well_conditioned_matrices_across_dimensions_and_scales() {
    for scale in [2.0f64.powi(-40), 1.0, 2.0f64.powi(40), -2.0f64.powi(20)] {
        let mut factor = DenseFactor::new(8);
        let source = math_contract::make_well_conditioned_matrix(8, scale);
        for dimension in 1..=8 {
            let expected = math_contract::leading_principal(&source, dimension);
            factor.expand(
                &math_contract::row(&expected, dimension - 1),
                &math_contract::column(&expected, dimension - 1),
            );
            math_contract::expect_mathematically_correct(
                &mut factor,
                &expected,
                &format!("dimension={dimension}, scale={scale}"),
                true,
            );
        }
    }
}

#[test]
fn represents_singular_matrices_without_inventing_a_nonsingular_factor() {
    let singular = vec![
        vec![1.0, 2.0, 0.0],
        vec![2.0, 4.0, 0.0],
        vec![0.0, 0.0, 3.0],
    ];
    let mut factor = DenseFactor::new(3);
    for dimension in 1..=3 {
        let expected = math_contract::leading_principal(&singular, dimension);
        factor.expand(
            &math_contract::row(&expected, dimension - 1),
            &math_contract::column(&expected, dimension - 1),
        );
    }
    math_contract::expect_mathematically_correct(
        &mut factor,
        &singular,
        "rank-deficient matrix",
        false,
    );
    assert!(
        factor
            .u()
            .iter()
            .enumerate()
            .any(|(index, row)| row[index].abs() <= 1.0e-14)
    );
}

#[test]
fn product_and_triangular_solve_agree_with_independent_dense_arithmetic() {
    const MAXIMUM_DIMENSION: usize = 6;
    const DIMENSION: usize = 4;
    let l = vec![
        vec![1.0, 0.5, 0.0, -0.25],
        vec![0.0, -1.0, 2.0, 0.0],
        vec![0.75, 0.0, 1.0, 0.5],
        vec![0.0, 1.0, 0.0, 1.0],
    ];
    let u = vec![
        vec![4.0, -1.0, 0.0, 2.0],
        vec![0.0, -3.0, 1.5, 0.0],
        vec![0.0, 0.0, 2.0, -0.5],
        vec![0.0, 0.0, 0.0, 5.0],
    ];
    let mut l_storage = guarded(&vec![], MAXIMUM_DIMENSION * MAXIMUM_DIMENSION);
    let mut u_storage = guarded(&vec![], MAXIMUM_DIMENSION * (MAXIMUM_DIMENSION + 1) / 2);
    for row in 0..DIMENSION {
        for column in 0..DIMENSION {
            logical_mut(&mut l_storage)[row * MAXIMUM_DIMENSION + column] = l[row][column];
        }
        let row_start = row * MAXIMUM_DIMENSION - row * row.saturating_sub(1) / 2;
        for column in row..DIMENSION {
            logical_mut(&mut u_storage)[row_start + column - row] = u[row][column];
        }
    }
    let input = vec![1.0, -2.0, 0.5, 3.0];
    for mode in [1, 2] {
        let mut actual_input = guarded(&input, MAXIMUM_DIMENSION);
        let original = actual_input.clone();
        let mut result = guarded(&vec![], MAXIMUM_DIMENSION);
        Lprod(
            mode,
            MAXIMUM_DIMENSION as i32,
            DIMENSION as i32,
            logical_mut(&mut l_storage),
            logical_mut(&mut actual_input),
            logical_mut(&mut result),
        );
        assert_eq!(actual_input, original);
        let expected = math_contract::multiply_vector(&l, &input, mode == 2);
        for index in 0..DIMENSION {
            assert!((logical(&result)[index] - expected[index]).abs() <= 1.0e-13);
        }
    }
    let solution = vec![0.5, -1.0, 2.0, -0.25];
    for mode in [1, 2] {
        let rhs = math_contract::multiply_vector(&u, &solution, mode == 2);
        let mut actual = guarded(&rhs, MAXIMUM_DIMENSION);
        Usolve(
            mode,
            MAXIMUM_DIMENSION as i32,
            DIMENSION as i32,
            logical_mut(&mut u_storage),
            logical_mut(&mut actual),
        );
        for index in 0..DIMENSION {
            assert!((logical(&actual)[index] - solution[index]).abs() <= 1.0e-13);
        }
    }
}

#[test]
fn validates_the_zero_dimension_factor() {
    let mut factor = DenseFactor::new(1);
    crate::math_contract::expect_mathematically_correct(
        &mut factor,
        &vec![],
        "zero dimension",
        true,
    );
}
