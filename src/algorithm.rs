// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use core::mem;

use crate::api::Real;

#[derive(Clone, Copy)]
pub(crate) struct Numerics<T> {
    pub(crate) epsilon: T,
    pub(crate) update_tiny: T,
}

#[derive(Clone, Copy)]
pub(crate) struct Transform<T> {
    pub(crate) swap: bool,
    pub(crate) multiplier: T,
}

pub(crate) trait MatrixRead<T: Real> {
    fn get(&self, row: usize, column: usize) -> T;

    #[allow(clippy::needless_range_loop)] // Dense compatibility panics at the original index.
    fn copy_row_to_dense(&self, row: usize, dense: &mut [T], first: usize, end: usize) {
        for column in first..end {
            dense[column] = self.get(row, column);
        }
    }
}

pub(crate) trait Matrix<T: Real>: MatrixRead<T> {
    fn set(&mut self, row: usize, column: usize, value: T);

    fn clear_row(&mut self, row: usize, first: usize, end: usize) {
        for column in first..end {
            self.set(row, column, T::ZERO);
        }
    }

    #[allow(clippy::needless_range_loop)] // Dense compatibility panics at the original index.
    fn copy_dense_to_row(&mut self, row: usize, dense: &[T], first: usize, end: usize) {
        for column in first..end {
            self.set(row, column, dense[column]);
        }
    }

    fn transform_rows(
        &mut self,
        x_row: usize,
        y_row: usize,
        first: usize,
        end: usize,
        transform: Transform<T>,
    ) {
        for column in first..end {
            let (mut x, mut y) = (self.get(x_row, column), self.get(y_row, column));
            apply_pair(&mut x, &mut y, transform);
            self.set(x_row, column, x);
            self.set(y_row, column, y);
        }
    }

    fn transform_row_with_dense(
        &mut self,
        row: usize,
        dense: &mut [T],
        first: usize,
        end: usize,
        matrix_is_x: bool,
        transform: Transform<T>,
    ) {
        #[allow(clippy::needless_range_loop)]
        for index in first..end {
            let matrix_value = self.get(row, index);
            if matrix_is_x {
                let (mut x, mut y) = (matrix_value, dense[index]);
                apply_pair(&mut x, &mut y, transform);
                self.set(row, index, x);
                dense[index] = y;
            } else {
                let (mut x, mut y) = (dense[index], matrix_value);
                apply_pair(&mut x, &mut y, transform);
                dense[index] = x;
                self.set(row, index, y);
            }
        }
    }

    fn remove_l_column(&mut self, removed: usize, last: usize, rows: usize, work: &mut [T]) {
        for (row, work_value) in work.iter_mut().enumerate().take(rows) {
            *work_value = self.get(row, removed);
            if removed < last {
                self.set(row, removed, self.get(row, last));
            }
        }
    }
}

pub(crate) struct DenseL<S> {
    storage: S,
    stride: usize,
}

impl<S> DenseL<S> {
    pub(crate) fn new(storage: S, stride: usize) -> Self {
        Self { storage, stride }
    }
}

impl<T: Real, S: AsRef<[T]>> MatrixRead<T> for DenseL<S> {
    fn get(&self, row: usize, column: usize) -> T {
        self.storage.as_ref()[row * self.stride + column]
    }
}

impl<T: Real, S: AsRef<[T]> + AsMut<[T]>> Matrix<T> for DenseL<S> {
    fn set(&mut self, row: usize, column: usize, value: T) {
        self.storage.as_mut()[row * self.stride + column] = value;
    }

    fn transform_rows(
        &mut self,
        x_row: usize,
        y_row: usize,
        first: usize,
        end: usize,
        transform: Transform<T>,
    ) {
        let matrix = self.storage.as_mut();
        for column in first..end {
            let x = x_row * self.stride + column;
            let y = y_row * self.stride + column;
            if transform.swap {
                matrix.swap(x, y);
            }
            if transform.multiplier != T::ZERO {
                matrix[y] += transform.multiplier * matrix[x];
            }
        }
    }
}

pub(crate) struct PackedU<S> {
    storage: S,
    stride: usize,
}

impl<S> PackedU<S> {
    pub(crate) fn new(storage: S, stride: usize) -> Self {
        Self { storage, stride }
    }
}

impl<T: Real, S: AsRef<[T]>> MatrixRead<T> for PackedU<S> {
    fn get(&self, row: usize, column: usize) -> T {
        self.storage.as_ref()[u_index(row, column, self.stride)]
    }
}

impl<T: Real, S: AsRef<[T]> + AsMut<[T]>> Matrix<T> for PackedU<S> {
    fn set(&mut self, row: usize, column: usize, value: T) {
        self.storage.as_mut()[u_index(row, column, self.stride)] = value;
    }

    fn clear_row(&mut self, row: usize, first: usize, end: usize) {
        for column in first.max(row)..end {
            self.set(row, column, T::ZERO);
        }
    }
}

pub(crate) fn push<T: Real, L: Matrix<T>, U: Matrix<T>>(
    n: usize,
    l: &mut L,
    u: &mut U,
    y: &mut [T],
    z: &[T],
    w: &mut [T],
    numerics: Numerics<T>,
) {
    let last = n - 1;
    l.clear_row(last, 0, n);
    u.clear_row(last, 0, n);
    l.set(last, last, T::ONE);
    if n == 1 {
        u.set(0, 0, y[0]);
        return;
    }
    l_product(false, last, l, z, w);
    for (row, &value) in w.iter().enumerate().take(last) {
        u.set(row, last, value);
        l.set(row, last, T::ZERO);
    }
    forward(0, last, n, n, l, u, y, numerics);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replace_column<T: Real, L: Matrix<T>, U: Matrix<T>>(
    column: usize,
    n: usize,
    l: &mut L,
    u: &mut U,
    y: &mut [T],
    z: &[T],
    w: &mut [T],
    numerics: Numerics<T>,
) {
    l_product(false, n, l, z, w);
    for (row, &value) in w.iter().enumerate().take(column + 1) {
        u.set(row, column, value);
    }
    if column + 1 < n {
        let last = backward(column + 1, n - 1, true, n, n, l, u, y, w, numerics);
        y[column] = w[last];
        forward(column, last, n, n, l, u, y, numerics);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn replace_row<T: Real, L: Matrix<T>, U: Matrix<T>>(
    row: usize,
    n: usize,
    l: &mut L,
    u: &mut U,
    y: &mut [T],
    z: &mut [T],
    w: &mut [T],
    numerics: Numerics<T>,
) {
    if n == 1 {
        l.set(0, 0, T::ONE);
        u.set(0, 0, y[0]);
        return;
    }
    for (index, work) in w.iter_mut().enumerate().take(n) {
        *work = l.get(index, row);
        l.set(index, row, T::ZERO);
    }
    let last = backward(0, n - 1, true, n, n, l, u, z, w, numerics);
    l.clear_row(last, 0, n);
    l.set(last, row, T::ONE);
    forward(0, last, n, n, l, u, y, numerics);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn remove<T: Real, L: Matrix<T>, U: Matrix<T>>(
    row: usize,
    column: usize,
    n: usize,
    l: &mut L,
    u: &mut U,
    y: &mut [T],
    z: &mut [T],
    w: &mut [T],
    numerics: Numerics<T>,
) {
    let reduced = n - 1;
    if column < reduced {
        for (index, work) in w.iter_mut().enumerate().take(n) {
            *work = u.get(index, reduced);
        }
        for (index, &value) in w.iter().enumerate().take(column + 1) {
            u.set(index, column, value);
        }
        let last = backward(column + 1, reduced, true, n, reduced, l, u, y, w, numerics);
        y[column] = w[last];
        forward(column, last, n, reduced, l, u, y, numerics);
    }
    l.remove_l_column(row, reduced, n, w);
    backward(0, reduced, false, n, reduced, l, u, z, w, numerics);
}

#[allow(clippy::needless_range_loop)] // Preserves partial writes before short-output panics.
pub(crate) fn l_product<T: Real, L: MatrixRead<T>>(
    transpose: bool,
    n: usize,
    l: &L,
    input: &[T],
    output: &mut [T],
) {
    let result_len = n.min(output.len());
    output[..result_len].fill(T::ZERO);
    for index in 0..n {
        output[index] = l_product_value(transpose, n, l, input, index);
    }
}

pub(crate) fn l_product_value<T: Real, L: MatrixRead<T>>(
    transpose: bool,
    n: usize,
    l: &L,
    input: &[T],
    index: usize,
) -> T {
    let mut value = T::ZERO;
    for (other, &input_value) in input.iter().enumerate().take(n) {
        value += if transpose {
            l.get(other, index)
        } else {
            l.get(index, other)
        } * input_value;
    }
    value
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forward<T: Real, L: Matrix<T>, U: Matrix<T>>(
    first: usize,
    last: usize,
    n: usize,
    columns: usize,
    l: &mut L,
    u: &mut U,
    y: &mut [T],
    numerics: Numerics<T>,
) {
    for row in first..last.min(n) {
        let mut pivot = u.get(row, row);
        if y[row].abs() > numerics.epsilon * pivot.abs() {
            let transform = elementary(&mut pivot, &mut y[row], numerics);
            u.set(row, row, pivot);
            u.transform_row_with_dense(row, y, row + 1, columns, true, transform);
            l.transform_rows(row, last, 0, n, transform);
        }
    }
    if last < n {
        u.copy_dense_to_row(last, y, last, columns);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn backward<T: Real, L: Matrix<T>, U: Matrix<T>>(
    first: usize,
    mut selected: usize,
    trim_trailing_zeros: bool,
    n: usize,
    columns: usize,
    l: &mut L,
    u: &mut U,
    y: &mut [T],
    z: &mut [T],
    numerics: Numerics<T>,
) -> usize {
    if n == 0 {
        return 0;
    }
    selected = selected.min(n - 1);
    while trim_trailing_zeros
        && selected > first
        && z.get(selected).copied().unwrap_or(T::ZERO).abs() <= numerics.epsilon
    {
        selected -= 1;
    }
    let mut zlast = z[selected];
    u.copy_row_to_dense(selected, y, selected, columns);
    for row in (first..selected).rev() {
        y[row] = T::ZERO;
        if z[row].abs() <= numerics.epsilon * zlast.abs() {
            continue;
        }
        let mut eliminated = z[row];
        let transform = elementary(&mut zlast, &mut eliminated, numerics);
        z[row] = eliminated;
        u.transform_row_with_dense(row, y, row, columns, false, transform);
        l.transform_rows(selected, row, 0, n, transform);
    }
    z[selected] = zlast;
    selected
}

pub(crate) fn elementary<T: Real>(x: &mut T, y: &mut T, numerics: Numerics<T>) -> Transform<T> {
    let tiny = numerics.epsilon * numerics.update_tiny;
    let transform = if x.abs() >= y.abs() {
        if x.abs() <= tiny {
            *x = T::ZERO;
            Transform {
                swap: false,
                multiplier: T::ZERO,
            }
        } else {
            Transform {
                swap: false,
                multiplier: -*y / *x,
            }
        }
    } else if y.abs() <= tiny {
        *x = T::ZERO;
        Transform {
            swap: false,
            multiplier: T::ZERO,
        }
    } else {
        let multiplier = -*x / *y;
        *x = *y;
        Transform {
            swap: true,
            multiplier,
        }
    };
    *y = T::ZERO;
    transform
}

pub(crate) fn u_solve<T: Real, U: MatrixRead<T>>(transpose: bool, n: usize, u: &U, rhs: &mut [T]) {
    if !transpose {
        for row in (0..n).rev() {
            let mut sum = T::ZERO;
            for (column, &rhs_value) in rhs.iter().enumerate().take(n).skip(row + 1) {
                sum += u.get(row, column) * rhs_value;
            }
            rhs[row] = (rhs[row] - sum) / u.get(row, row);
        }
    } else {
        for row in 0..n {
            let mut sum = T::ZERO;
            for (column, &rhs_value) in rhs.iter().enumerate().take(row) {
                sum += u.get(column, row) * rhs_value;
            }
            rhs[row] = (rhs[row] - sum) / u.get(row, row);
        }
    }
}

pub(crate) fn apply_pair<T: Real>(x: &mut T, y: &mut T, transform: Transform<T>) {
    if transform.swap {
        mem::swap(x, y);
    }
    if transform.multiplier != T::ZERO {
        *y += transform.multiplier * *x;
    }
}

pub(crate) fn u_index(row: usize, column: usize, stride: usize) -> usize {
    row * stride - row * row.saturating_sub(1) / 2 + column - row
}

#[cfg(test)]
#[path = "../tests/internal/algorithm.rs"]
mod tests;
