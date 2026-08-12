// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use super::*;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use wasm_bindgen_test::wasm_bindgen_test as test;

struct TestMatrix([[f64; 2]; 2]);

impl MatrixRead<f64> for TestMatrix {
    fn get(&self, row: usize, column: usize) -> f64 {
        self.0[row][column]
    }
}

impl Matrix<f64> for TestMatrix {
    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.0[row][column] = value;
    }
}

#[test]
fn default_row_transform_updates_the_requested_range() {
    let mut matrix = TestMatrix([[1.0, 2.0], [3.0, 4.0]]);
    matrix.transform_rows(
        0,
        1,
        0,
        2,
        Transform {
            swap: true,
            multiplier: 0.5,
        },
    );
    assert_eq!(matrix.0, [[3.0, 4.0], [2.5, 4.0]]);
}

fn exercise_sweep_boundaries<T: Real>() {
    let numerics = Numerics {
        epsilon: T::EPSILON,
        update_tiny: T::UPDATE_TINY,
    };
    let mut l_storage = [T::ONE, T::ZERO, T::ZERO, T::ONE];
    let mut u_storage = [T::ONE, T::ZERO, T::ONE];
    let mut l = DenseL::new(&mut l_storage, 2);
    let mut u = PackedU::new(&mut u_storage, 2);
    assert_eq!(
        backward(
            0,
            0,
            false,
            0,
            0,
            &mut l,
            &mut u,
            &mut [],
            &mut [],
            numerics,
        ),
        0
    );

    let mut y = [T::ZERO; 2];
    let mut z = [T::ONE, T::ZERO];
    assert_eq!(
        backward(0, 1, true, 2, 2, &mut l, &mut u, &mut y, &mut z, numerics),
        0
    );

    forward(0, 1, 1, 1, &mut l, &mut u, &mut [T::ZERO], numerics);
    replace_row(
        0,
        1,
        &mut l,
        &mut u,
        &mut [T::ONE],
        &mut [T::ZERO],
        &mut [T::ZERO],
        numerics,
    );
}

#[test]
fn sweep_boundaries_handle_zero_dimension_trailing_zeros_and_the_terminal_row() {
    exercise_sweep_boundaries::<f32>();
    exercise_sweep_boundaries::<f64>();
}

fn exercise_zero_transforms<T: Real>() {
    let numerics = Numerics {
        epsilon: T::ONE,
        update_tiny: T::ONE,
    };
    let (mut x, mut y) = (T::ZERO, T::ZERO);
    let transform = elementary(&mut x, &mut y, numerics);
    assert!(!transform.swap);
    assert!(transform.multiplier == T::ZERO);

    y = T::ONE;
    let transform = elementary(&mut x, &mut y, numerics);
    assert!(!transform.swap);
    assert!(transform.multiplier == T::ZERO);

    let (mut x, mut y) = (T::ONE, T::ONE);
    apply_pair(
        &mut x,
        &mut y,
        Transform {
            swap: false,
            multiplier: T::ZERO,
        },
    );
    assert!(x == T::ONE && y == T::ONE);
}

#[test]
fn zero_elementary_transforms_and_zero_multiplier_are_no_ops() {
    exercise_zero_transforms::<f32>();
    exercise_zero_transforms::<f64>();
}
