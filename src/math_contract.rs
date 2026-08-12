// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![allow(clippy::needless_range_loop)] // Matrix formulas are clearest with explicit row/column indices.

use std::{format, vec, vec::Vec};

pub(crate) type Matrix = Vec<Vec<f64>>;
pub(crate) type Vector = Vec<f64>;

pub(crate) trait Factor {
    fn dimension(&self) -> usize;
    fn expand(&mut self, new_row: &Vector, new_column: &Vector);
    fn replace_column(&mut self, column: usize, replacement: &Vector);
    fn replace_row(&mut self, row: usize, replacement: &Vector);
    fn erase(&mut self, row: usize, column: usize);
    fn l(&self) -> Matrix;
    fn u(&self) -> Matrix;
    fn solve(&mut self, right_hand_side: &Vector, transpose: bool) -> Vector;
    fn storage_is_valid(&self) -> bool;
}

pub(crate) fn leading_principal(matrix: &Matrix, dimension: usize) -> Matrix {
    matrix[..dimension]
        .iter()
        .map(|row| row[..dimension].to_vec())
        .collect()
}

pub(crate) fn make_well_conditioned_matrix(dimension: usize, scale: f64) -> Matrix {
    let mut result = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        let mut off_diagonal_sum = 0.0;
        for column in 0..dimension {
            if row == column || (3 * row + 5 * column + 1) % 4 == 0 {
                continue;
            }
            let numerator = (7 * (row + 1) + 11 * (column + 1)) % 9;
            let numerator = numerator as i32 - 4;
            result[row][column] = scale * f64::from(numerator) / 8.0;
            off_diagonal_sum += result[row][column].abs();
        }
        result[row][row] =
            (off_diagonal_sum + scale.abs() * (2.0 + 0.25 * row as f64)).copysign(scale);
    }
    result
}

pub(crate) fn row(matrix: &Matrix, row: usize) -> Vector {
    matrix[row].clone()
}

pub(crate) fn column(matrix: &Matrix, column: usize) -> Vector {
    matrix.iter().map(|row| row[column]).collect()
}

fn combined_column(matrix: &Matrix, target: usize, source: usize, multiplier: f64) -> Vector {
    let mut result = column(matrix, target);
    let source = column(matrix, source);
    for (value, source) in result.iter_mut().zip(source) {
        *value += multiplier * source;
    }
    result
}

fn combined_row(matrix: &Matrix, target: usize, source: usize, multiplier: f64) -> Vector {
    let mut result = row(matrix, target);
    let source = row(matrix, source);
    for (value, source) in result.iter_mut().zip(source) {
        *value += multiplier * source;
    }
    result
}

fn replace_column(matrix: &mut Matrix, column: usize, replacement: &Vector) {
    for (row, value) in matrix.iter_mut().zip(replacement) {
        row[column] = *value;
    }
}

fn replace_row(matrix: &mut Matrix, row: usize, replacement: &Vector) {
    matrix[row] = replacement.clone();
}

fn delete_by_last_swap(matrix: &Matrix, deleted_row: usize, deleted_column: usize) -> Matrix {
    let old_dimension = matrix.len();
    let new_dimension = old_dimension - 1;
    let mut result = vec![vec![0.0; new_dimension]; new_dimension];
    for row in 0..new_dimension {
        let source_row = if row == deleted_row {
            old_dimension - 1
        } else {
            row
        };
        for column in 0..new_dimension {
            let source_column = if column == deleted_column {
                old_dimension - 1
            } else {
                column
            };
            result[row][column] = matrix[source_row][source_column];
        }
    }
    result
}

fn append_stable_border(matrix: &Matrix) -> Matrix {
    let old_dimension = matrix.len();
    let new_dimension = old_dimension + 1;
    let mut result = vec![vec![0.0; new_dimension]; new_dimension];
    for row in 0..old_dimension {
        result[row][..old_dimension].copy_from_slice(&matrix[row]);
    }
    let mut border_magnitude = 0.0;
    for index in 0..old_dimension {
        let new_column = if index % 2 == 0 {
            0.25 * (index + 1) as f64
        } else {
            0.0
        };
        let new_row = if index % 2 == 0 {
            0.0
        } else {
            -0.2 * (index + 1) as f64
        };
        result[index][old_dimension] = new_column;
        result[old_dimension][index] = new_row;
        border_magnitude += new_column.abs() + new_row.abs();
    }
    result[old_dimension][old_dimension] = border_magnitude + 5.0;
    result
}

pub(crate) fn multiply_matrix(left: &Matrix, right: &Matrix) -> Matrix {
    let dimension = left.len();
    let mut result = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            result[row][column] = (0..dimension)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum();
        }
    }
    result
}

pub(crate) fn multiply_vector(matrix: &Matrix, vector: &Vector, transpose: bool) -> Vector {
    (0..matrix.len())
        .map(|row| {
            (0..matrix.len())
                .map(|column| {
                    let coefficient = if transpose {
                        matrix[column][row]
                    } else {
                        matrix[row][column]
                    };
                    coefficient * vector[column]
                })
                .sum()
        })
        .collect()
}

fn matrix_infinity_norm(matrix: &Matrix) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn vector_infinity_norm(vector: &Vector) -> f64 {
    vector.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn matrix_maximum_difference(left: &Matrix, right: &Matrix) -> f64 {
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn vector_maximum_difference(left: &Vector, right: &Vector) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn known_solution(dimension: usize) -> Vector {
    (0..dimension)
        .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 } * (index + 1) as f64 / 3.0)
        .collect()
}

pub(crate) fn expect_mathematically_correct<F: Factor>(
    factor: &mut F,
    expected_matrix: &Matrix,
    stage: &str,
    test_solves: bool,
) {
    let dimension = expected_matrix.len();
    assert_eq!(factor.dimension(), dimension, "{stage}: dimension");
    assert!(factor.storage_is_valid(), "{stage}: invalid factor storage");
    let l = factor.l();
    let u = factor.u();
    assert_eq!(l.len(), dimension, "{stage}: L dimension");
    assert_eq!(u.len(), dimension, "{stage}: U dimension");
    for row in 0..dimension {
        for column in 0..dimension {
            assert!(
                l[row][column].is_finite(),
                "{stage}: non-finite L[{row}][{column}]"
            );
            assert!(
                u[row][column].is_finite(),
                "{stage}: non-finite U[{row}][{column}]"
            );
            if column < row {
                assert_eq!(u[row][column], 0.0, "{stage}: U is not upper triangular");
            }
        }
    }
    let product = multiply_matrix(&l, expected_matrix);
    let factor_scale =
        matrix_infinity_norm(&l) * matrix_infinity_norm(expected_matrix) + matrix_infinity_norm(&u);
    let factor_error = matrix_maximum_difference(&product, &u);
    let normalized_error = if factor_scale == 0.0 {
        factor_error
    } else {
        factor_error / factor_scale
    };
    let tolerance = 8192.0 * dimension as f64 * f64::EPSILON;
    assert!(
        normalized_error <= tolerance,
        "{stage}: factor error={factor_error}, scale={factor_scale}, normalized={normalized_error}"
    );

    if !test_solves {
        return;
    }
    let expected_solution = known_solution(dimension);
    for transpose in [false, true] {
        let right_hand_side = multiply_vector(expected_matrix, &expected_solution, transpose);
        let actual = factor.solve(&right_hand_side, transpose);
        assert_eq!(
            actual.len(),
            expected_solution.len(),
            "{stage}: solve dimension"
        );
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{stage}: non-finite solution"
        );
        let forward_error = vector_maximum_difference(&actual, &expected_solution);
        assert!(
            forward_error <= 1.0e-10 * vector_infinity_norm(&expected_solution).max(1.0),
            "{stage}: forward error={forward_error}, transpose={transpose}"
        );
        let reconstructed = multiply_vector(expected_matrix, &actual, transpose);
        let residual = vector_maximum_difference(&reconstructed, &right_hand_side);
        let scale = matrix_infinity_norm(expected_matrix) * vector_infinity_norm(&actual)
            + vector_infinity_norm(&right_hand_side);
        let normalized = if scale == 0.0 {
            residual
        } else {
            residual / scale
        };
        assert!(
            normalized <= tolerance,
            "{stage}: residual={residual}, scale={scale}, transpose={transpose}"
        );
    }
}

pub(crate) fn exercise_complete_update_lifecycle<F: Factor>(factor: &mut F) {
    let source = make_well_conditioned_matrix(6, 1.0);
    let mut expected = Matrix::new();
    for dimension in 1..=6 {
        let next = leading_principal(&source, dimension);
        factor.expand(&row(&next, dimension - 1), &column(&next, dimension - 1));
        expected = next;
        expect_mathematically_correct(factor, &expected, &format!("mode 1, n={dimension}"), true);
    }
    for (target, source, multiplier) in [(0, 1, 0.25), (2, 4, -0.5), (5, 0, 0.125), (2, 1, 0.2)] {
        let replacement = combined_column(&expected, target, source, multiplier);
        factor.replace_column(target, &replacement);
        replace_column(&mut expected, target, &replacement);
        expect_mathematically_correct(factor, &expected, &format!("mode 2, column={target}"), true);
    }
    for (target, source, multiplier) in [(0, 1, -0.25), (3, 5, 0.375), (5, 2, -0.125), (3, 1, 0.2)]
    {
        let replacement = combined_row(&expected, target, source, multiplier);
        factor.replace_row(target, &replacement);
        replace_row(&mut expected, target, &replacement);
        expect_mathematically_correct(factor, &expected, &format!("mode 3, row={target}"), true);
    }
    factor.erase(1, 3);
    expected = delete_by_last_swap(&expected, 1, 3);
    expect_mathematically_correct(
        factor,
        &expected,
        "mode 4, unequal interior row and column",
        true,
    );
    let regrown = append_stable_border(&expected);
    factor.expand(
        &row(&regrown, factor.dimension()),
        &column(&regrown, factor.dimension()),
    );
    expected = regrown;
    expect_mathematically_correct(factor, &expected, "mode 1 after shrink", true);
    for (row, column) in [(5, 0), (0, 4), (3, 3)] {
        factor.erase(row, column);
        expected = delete_by_last_swap(&expected, row, column);
        expect_mathematically_correct(
            factor,
            &expected,
            &format!("mode 4, row={row}, column={column}"),
            true,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn matrix_oracle_uses_zero_based_rows_and_columns() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        assert_eq!(row(&matrix, 0), vec![1.0, 2.0, 3.0]);
        assert_eq!(column(&matrix, 0), vec![1.0, 4.0, 7.0]);
        assert_eq!(combined_column(&matrix, 0, 2, 10.0), vec![31.0, 64.0, 97.0]);
        assert_eq!(combined_row(&matrix, 0, 2, 0.5), vec![4.5, 6.0, 7.5]);
        assert_eq!(
            delete_by_last_swap(&matrix, 0, 1),
            vec![vec![7.0, 9.0], vec![4.0, 6.0]]
        );
    }
}
