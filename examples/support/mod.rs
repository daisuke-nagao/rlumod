// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![allow(dead_code)] // Each example binary uses one shared entry point.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)] // Mirrors the C example.

#[cfg(feature = "lumod-c")]
use rlumod::lumod_c;
use rlumod::{ColumnIndex, LuMod, RowIndex, Workspace};
use std::fmt::Write;
use std::time::Instant;

const DEFAULT_MAXIMUM_DIMENSION: usize = 75;
const BIG_NUMBER: f64 = 1.0e30;
const TINY_NUMBER: f64 = 1.0e-4;
const MATH_PRECISION: f64 = 1.0e-16;
const ERROR_LIMIT: f64 = 1.0e-6;

trait Factors {
    fn reset_for_test(&mut self);
    fn update(
        &mut self,
        mode: i32,
        dimension: usize,
        row: i32,
        column: i32,
        y: &mut [f64],
        z: &mut [f64],
        work: &mut [f64],
    );
    fn multiply_l(&mut self, mode: i32, dimension: usize, y: &mut [f64], z: &mut [f64]);
    fn solve_u(&mut self, mode: i32, dimension: usize, y: &mut [f64]);
    fn solve(&mut self, transpose: bool, dimension: usize, rhs: &mut [f64]) {
        let mut result = vec![0.0; dimension];
        if transpose {
            self.solve_u(2, dimension, rhs);
            self.multiply_l(2, dimension, rhs, &mut result);
        } else {
            self.multiply_l(1, dimension, rhs, &mut result);
            self.solve_u(1, dimension, &mut result);
        }
        rhs[..dimension].copy_from_slice(&result);
    }
    fn l_maximum(&self, row: usize, dimension: usize) -> f64;
    fn u_maximum(&self, row: usize, dimension: usize) -> f64;
    fn u_diagonal(&self, row: usize) -> f64;
    fn set_u_diagonal(&mut self, row: usize, value: f64);
    fn u_item(&self, row: usize, column: usize) -> f64;
}

struct SafeFactors {
    maximum_dimension: usize,
    dimension: usize,
    l: Vec<f64>,
    u: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    work: Vec<f64>,
}

impl SafeFactors {
    fn new(maximum_dimension: usize) -> Self {
        Self {
            maximum_dimension,
            dimension: 0,
            l: vec![0.0; maximum_dimension * maximum_dimension + maximum_dimension],
            u: vec![0.0; maximum_dimension * (maximum_dimension + 1) / 2],
            y: vec![0.0; maximum_dimension],
            z: vec![0.0; maximum_dimension],
            work: vec![0.0; maximum_dimension],
        }
    }

    fn u_index(&self, row: usize, column: usize) -> usize {
        row * self.maximum_dimension - row * row.saturating_sub(1) / 2 + column - row
    }
}

impl Factors for SafeFactors {
    fn reset_for_test(&mut self) {
        self.dimension = 0;
        self.l.fill(0.0);
        self.u.fill(0.0);
    }

    fn update(
        &mut self,
        mode: i32,
        dimension: usize,
        row: i32,
        column: i32,
        y: &mut [f64],
        z: &mut [f64],
        _work: &mut [f64],
    ) {
        let mut factor = LuMod::from_storage(
            self.dimension,
            self.maximum_dimension,
            &mut self.l,
            &mut self.u,
        )
        .unwrap();
        let mut workspace = Workspace::new(&mut self.y, &mut self.z, &mut self.work).unwrap();
        match mode {
            1 => factor
                .push(
                    &y[..dimension - 1],
                    &z[..dimension - 1],
                    y[dimension - 1],
                    &mut workspace,
                )
                .unwrap(),
            2 => factor
                .replace_column(
                    ColumnIndex(column as usize),
                    &z[..dimension],
                    &mut workspace,
                )
                .unwrap(),
            3 => factor
                .replace_row(RowIndex(row as usize), &y[..dimension], &mut workspace)
                .unwrap(),
            4 => {
                factor
                    .remove(
                        RowIndex(row as usize),
                        ColumnIndex(column as usize),
                        &mut workspace,
                    )
                    .unwrap();
            }
            _ => unreachable!(),
        }
        self.dimension = factor.dimension();
    }

    fn multiply_l(&mut self, mode: i32, dimension: usize, y: &mut [f64], z: &mut [f64]) {
        z[..dimension].fill(0.0);
        if mode == 1 {
            for row in 0..dimension {
                z[row] = (0..dimension)
                    .map(|column| self.l[row * self.maximum_dimension + column] * y[column])
                    .sum();
            }
        } else {
            for row in 0..dimension {
                for column in 0..dimension {
                    z[column] += self.l[row * self.maximum_dimension + column] * y[row];
                }
            }
        }
    }

    fn solve_u(&mut self, mode: i32, dimension: usize, y: &mut [f64]) {
        if mode == 1 {
            for row in (0..dimension).rev() {
                let sum: f64 = (row + 1..dimension)
                    .map(|column| self.u[self.u_index(row, column)] * y[column])
                    .sum();
                y[row] = (y[row] - sum) / self.u[self.u_index(row, row)];
            }
        } else {
            for row in 0..dimension {
                let sum: f64 = (0..row)
                    .map(|column| self.u[self.u_index(column, row)] * y[column])
                    .sum();
                y[row] = (y[row] - sum) / self.u[self.u_index(row, row)];
            }
        }
    }

    fn solve(&mut self, transpose: bool, dimension: usize, rhs: &mut [f64]) {
        let factor = LuMod::from_storage(
            self.dimension,
            self.maximum_dimension,
            &mut self.l,
            &mut self.u,
        )
        .unwrap();
        if transpose {
            factor
                .solve_transpose_in_place(&mut rhs[..dimension])
                .unwrap();
        } else {
            factor.solve_in_place(&mut rhs[..dimension]).unwrap();
        }
    }

    fn l_maximum(&self, row: usize, dimension: usize) -> f64 {
        let start = row * self.maximum_dimension;
        let relative = self.l[start..start + dimension]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map_or(0, |entry| entry.0);
        self.l[relative].abs()
    }

    fn u_maximum(&self, row: usize, dimension: usize) -> f64 {
        let start = self.u_index(row, row);
        let relative = self.u[start..start + dimension - row]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map_or(0, |entry| entry.0);
        self.u[relative].abs()
    }

    fn u_diagonal(&self, row: usize) -> f64 {
        self.u[self.u_index(row, row)]
    }

    fn set_u_diagonal(&mut self, row: usize, value: f64) {
        let index = self.u_index(row, row);
        self.u[index] = value;
    }

    fn u_item(&self, row: usize, column: usize) -> f64 {
        self.u[self.u_index(row, column)]
    }
}

#[cfg(feature = "lumod-c")]
struct LumodCFactors {
    maximum_dimension: usize,
    l: Vec<f64>,
    u: Vec<f64>,
}

#[cfg(feature = "lumod-c")]
impl LumodCFactors {
    fn new(maximum_dimension: usize) -> Self {
        Self {
            maximum_dimension,
            l: vec![0.0; maximum_dimension * maximum_dimension],
            u: vec![0.0; maximum_dimension * (maximum_dimension + 1) / 2],
        }
    }

    fn u_index(&self, row: usize, column: usize) -> usize {
        row * self.maximum_dimension - row * row.saturating_sub(1) / 2 + column - row
    }
}

#[cfg(feature = "lumod-c")]
impl Factors for LumodCFactors {
    fn reset_for_test(&mut self) {}

    fn update(
        &mut self,
        mode: i32,
        dimension: usize,
        row: i32,
        column: i32,
        y: &mut [f64],
        z: &mut [f64],
        work: &mut [f64],
    ) {
        lumod_c::LUmod(
            mode,
            self.maximum_dimension as i32,
            dimension as i32,
            row,
            column,
            &mut self.l,
            &mut self.u,
            y,
            z,
            work,
        );
    }

    fn multiply_l(&mut self, mode: i32, dimension: usize, y: &mut [f64], z: &mut [f64]) {
        lumod_c::Lprod(
            mode,
            self.maximum_dimension as i32,
            dimension as i32,
            &mut self.l,
            y,
            z,
        );
    }

    fn solve_u(&mut self, mode: i32, dimension: usize, y: &mut [f64]) {
        lumod_c::Usolve(
            mode,
            self.maximum_dimension as i32,
            dimension as i32,
            &mut self.u,
            y,
        );
    }

    fn l_maximum(&self, row: usize, dimension: usize) -> f64 {
        let start = row * self.maximum_dimension;
        let relative = self.l[start..start + dimension]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map_or(0, |entry| entry.0);
        self.l[relative].abs()
    }

    fn u_maximum(&self, row: usize, dimension: usize) -> f64 {
        let start = self.u_index(row, row);
        let relative = self.u[start..start + dimension - row]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
            .map_or(0, |entry| entry.0);
        self.u[relative].abs()
    }

    fn u_diagonal(&self, row: usize) -> f64 {
        self.u[self.u_index(row, row)]
    }

    fn set_u_diagonal(&mut self, row: usize, value: f64) {
        let index = self.u_index(row, row);
        self.u[index] = value;
    }

    fn u_item(&self, row: usize, column: usize) -> f64 {
        self.u[self.u_index(row, column)]
    }
}

pub fn run_safe(arguments: &[String]) -> String {
    let Some(options) = Options::parse(arguments) else {
        return help_text().to_owned();
    };
    run_tests(options, SafeFactors::new(options.maximum_dimension))
}

#[cfg(feature = "lumod-c")]
pub fn run_lumod_c(arguments: &[String]) -> String {
    let Some(options) = Options::parse(arguments) else {
        return help_text().to_owned();
    };
    run_tests(options, LumodCFactors::new(options.maximum_dimension))
}

#[derive(Clone, Copy)]
struct Options {
    maximum_dimension: usize,
    test_count: usize,
    density: f64,
    verbose: bool,
}

impl Options {
    fn parse(arguments: &[String]) -> Option<Self> {
        let mut options = Self {
            maximum_dimension: DEFAULT_MAXIMUM_DIMENSION,
            test_count: 1,
            density: 0.25,
            verbose: false,
        };
        let mut resized = false;
        for argument in arguments {
            match argument.as_str() {
                "-h" => return None,
                "-v" => options.verbose = true,
                "-fd" => options.density = 1.0,
                "-hd" => options.density = 0.75,
                "-md" => options.density = 0.5,
                "-dd" => options.density = 0.25,
                "-ld" => options.density = 0.1,
                "-td" => options.density = 0.03,
                value => {
                    let parsed = value.parse::<usize>().unwrap_or(0);
                    if resized {
                        options.test_count = parsed;
                    } else {
                        options.maximum_dimension = parsed;
                        resized = true;
                    }
                }
            }
        }
        Some(options)
    }
}

fn help_text() -> &'static str {
    "rlumod: a Rust adaptation of LUMOD v2.0\n\
Original LUMOD by Michael Saunders, Systems Optimization Laboratory (SOL)\n\
Dense-matrix C adaptation by Kjell Eikland\n\
Usage: <rlumod|lumod-c> [-h] [-v] [-fd|-hd|-md|-dd|-ld|-td] [size] [count]\n"
}

fn run_tests<F: Factors>(options: Options, mut factors: F) -> String {
    let m = options.maximum_dimension;
    let n = m;
    let nrowb = 2 * m;
    let mut b = vec![0.0; nrowb * 2 * m];
    let mut c = vec![0.0; m * m];
    let mut diagonal = vec![0.0; m];
    let mut x = vec![0.0; m];
    let mut v = vec![0.0; m];
    let mut work = vec![0.0; nrowb];
    let mut y = vec![0.0; m];
    let mut z = vec![0.0; m];
    let mut seeds = [123456, 234567, 345678];
    let mut stats = [0.0; 7];
    stats[3] = BIG_NUMBER;
    let mut output = String::new();
    let mut elapsed = 0.0;
    for test in 1..=options.test_count {
        if options.verbose {
            let _ = write!(
                output,
                "\n\n------------------------------------------\n LUTEST {test:4}.      m = {m:6}\n------------------------------------------\n"
            );
        } else {
            let _ = write!(output, "\nTest {test} ({n}x{m})");
        }
        set_matrix(
            &mut b,
            0,
            0,
            m,
            n,
            nrowb,
            &mut diagonal,
            &mut v,
            &mut work,
            options.density,
            &mut seeds,
        );
        set_matrix(
            &mut b,
            0,
            n,
            m,
            n,
            nrowb,
            &mut diagonal,
            &mut v,
            &mut work,
            options.density,
            &mut seeds,
        );
        set_matrix(
            &mut b,
            m,
            0,
            m,
            n,
            nrowb,
            &mut diagonal,
            &mut v,
            &mut work,
            options.density,
            &mut seeds,
        );
        factors.reset_for_test();
        let started = Instant::now();

        if options.verbose {
            output.push_str("\nLUmod (mode 1):\n");
        } else {
            output.push_str("\n1");
        }
        for dimension in 1..=m {
            for index in 0..dimension {
                y[index] = b[matrix_index(nrowb, dimension - 1, index)];
                z[index] = b[matrix_index(nrowb, index, dimension - 1)];
                c[matrix_index(n, dimension - 1, index)] = y[index];
                c[matrix_index(n, index, dimension - 1)] = z[index];
            }
            factors.update(1, dimension, -1, -1, &mut y, &mut z, &mut work);
            if options.verbose {
                let _ = writeln!(output, " Row and column {dimension} added");
            } else {
                output.push('.');
            }
            check_factors(
                m,
                n,
                dimension,
                &c,
                &mut factors,
                &mut stats,
                &mut v,
                &mut work,
                &mut x,
                &mut y,
                options.verbose,
                &mut output,
            );
        }

        if options.verbose {
            output.push_str("\nLUmod (mode 2):\n");
        } else {
            output.push_str("\n2");
        }
        for column in 0..n {
            for row in 0..m {
                z[row] = b[matrix_index(nrowb, row, n + column)];
                c[matrix_index(n, row, column)] = z[row];
            }
            factors.update(2, m, -1, column as i32, &mut y, &mut z, &mut work);
            if options.verbose {
                let _ = writeln!(
                    output,
                    "Column {} replaced by column {}",
                    column + 1,
                    n + column + 1
                );
            } else {
                output.push('.');
            }
            check_factors(
                m,
                n,
                m,
                &c,
                &mut factors,
                &mut stats,
                &mut v,
                &mut work,
                &mut x,
                &mut y,
                options.verbose,
                &mut output,
            );
        }

        if options.verbose {
            output.push_str("\nLUmod (mode 3):\n");
        } else {
            output.push_str("\n3");
        }
        for row in 0..m {
            for column in 0..n {
                y[column] = b[matrix_index(nrowb, m + row, column)];
                c[matrix_index(n, row, column)] = y[column];
            }
            factors.update(3, m, row as i32, -1, &mut y, &mut z, &mut work);
            if options.verbose {
                let _ = writeln!(output, "Row {} replaced by row {}", row + 1, m + row + 1);
            } else {
                output.push('.');
            }
            check_factors(
                m,
                n,
                m,
                &c,
                &mut factors,
                &mut stats,
                &mut v,
                &mut work,
                &mut x,
                &mut y,
                options.verbose,
                &mut output,
            );
        }

        if options.verbose {
            output.push_str("\nLUmod (mode 4):\n");
        } else {
            output.push_str("\n4");
        }
        let mut dimension = m;
        for _ in 1..m {
            let target = (dimension / 2).max(1) - 1;
            for row in 0..dimension {
                c[matrix_index(n, row, target)] = c[matrix_index(n, row, dimension - 1)];
            }
            for column in 0..dimension - 1 {
                c[matrix_index(n, target, column)] = c[matrix_index(n, dimension - 1, column)];
            }
            factors.update(
                4,
                dimension,
                target as i32,
                target as i32,
                &mut y,
                &mut z,
                &mut work,
            );
            dimension -= 1;
            if options.verbose {
                let _ = writeln!(
                    output,
                    "Row {} and column {} deleted",
                    target + 1,
                    target + 1
                );
            } else {
                output.push('.');
            }
            check_factors(
                m,
                n,
                dimension,
                &c,
                &mut factors,
                &mut stats,
                &mut v,
                &mut work,
                &mut x,
                &mut y,
                options.verbose,
                &mut output,
            );
        }
        elapsed = started.elapsed().as_secs_f64();
    }

    let last_error = stats[4].max(stats[5]).max(stats[6]);
    output.push_str("\n\nData range---------------------------Density ");
    let _ = writeln!(
        output,
        "{:>5.1}%",
        options.density * options.density * 100.0
    );
    output.push_str("    max|Cij|     max|Lij|     max|Uij|     min|Ujj|\n");
    for value in &stats[..4] {
        let _ = write!(output, "{} ", format_general(*value, 12));
    }
    output.push('\n');
    output.push_str("Solver accuracy------------------------------------\n");
    output.push_str("   max|LC-U|    max|Cx-b|   max|C'y-b|       STATUS\n");
    for value in &stats[4..] {
        let _ = write!(output, "{} ", format_general(*value, 12));
    }
    let _ = writeln!(
        output,
        "{:>12}",
        if last_error <= ERROR_LIMIT {
            "OK!"
        } else {
            "Bug?"
        }
    );
    output.push_str("---------------------------------------------------\n");
    let _ = writeln!(output, "Time in seconds: {elapsed:.3}");
    output
}

fn set_matrix(
    matrix: &mut [f64],
    row_offset: usize,
    column_offset: usize,
    rows: usize,
    columns: usize,
    stride: usize,
    diagonal: &mut [f64],
    v: &mut [f64],
    w: &mut [f64],
    density: f64,
    seeds: &mut [i32; 3],
) {
    randomdens(columns, diagonal, 21.0, 22.0, 1.0, seeds);
    randomdens(columns, v, -10.0, 10.0, density, seeds);
    randomdens(columns, w, -10.0, 10.0, density, seeds);
    for column in 0..columns {
        for row in 0..rows {
            matrix[matrix_index(stride, row_offset + row, column_offset + column)] =
                v[row] * w[column];
        }
        matrix[matrix_index(stride, row_offset + column, column_offset + column)] +=
            diagonal[column];
    }
}

fn randomdens(n: usize, x: &mut [f64], low: f64, high: f64, density: f64, seeds: &mut [i32; 3]) {
    let mut mask = vec![0.0; n];
    ddrand(n, x, seeds);
    ddrand(n, &mut mask, seeds);
    for index in 0..n.min(x.len()) {
        x[index] = if mask[index] < density {
            low + (high - low) * x[index]
        } else {
            0.0
        };
    }
}

fn ddrand(n: usize, x: &mut [f64], seeds: &mut [i32; 3]) {
    for value in x.iter_mut().take(n) {
        let mut s1 = 171 * (seeds[0] % 177) - 2 * (seeds[0] / 177);
        let mut s2 = 172 * (seeds[1] % 176) - 35 * (seeds[1] / 176);
        let mut s3 = 170 * (seeds[2] % 178) - 63 * (seeds[2] / 178);
        if s1 < 0 {
            s1 += 30269;
        }
        if s2 < 0 {
            s2 += 30307;
        }
        if s3 < 0 {
            s3 += 30323;
        }
        seeds.copy_from_slice(&[s1, s2, s3]);
        let sum = s1 as f64 / 30269.0 + s2 as f64 / 30307.0 + s3 as f64 / 30323.0;
        *value = sum.fract().abs();
    }
}

#[allow(clippy::too_many_arguments)]
fn check_factors<F: Factors>(
    _maximum_dimension: usize,
    matrix_stride: usize,
    dimension: usize,
    matrix: &[f64],
    factors: &mut F,
    stats: &mut [f64; 7],
    v: &mut [f64],
    w: &mut [f64],
    x: &mut [f64],
    y: &mut [f64],
    verbose: bool,
    output: &mut String,
) -> f64 {
    let mut c_max = 0.0_f64;
    let mut l_max = 0.0_f64;
    let mut u_max = 0.0_f64;
    let mut u_min = BIG_NUMBER;
    for index in 0..dimension {
        c_max = c_max.max(
            (0..dimension)
                .map(|row| matrix[matrix_index(matrix_stride, row, index)].abs())
                .fold(0.0, f64::max),
        );
        l_max = l_max.max(factors.l_maximum(index, dimension));
        u_max = u_max.max(factors.u_maximum(index, dimension));
        let diagonal = factors.u_diagonal(index);
        u_min = u_min.min(diagonal.abs());
        if diagonal == 0.0 {
            factors.set_u_diagonal(index, MATH_PRECISION);
        }
    }

    let mut factor_error = 0.0_f64;
    for column in 0..dimension {
        let mut source = vec![0.0; dimension];
        for row in 0..dimension {
            source[row] = matrix[matrix_index(matrix_stride, row, column)];
        }
        factors.multiply_l(1, dimension, &mut source, v);
        w[..dimension].fill(0.0);
        for row in 0..=column {
            w[row] = factors.u_item(row, column);
        }
        for row in 0..dimension {
            factor_error = factor_error.max((w[row] - v[row]).abs());
        }
    }

    x[0] = 1.0;
    for index in 1..dimension {
        x[index] = -x[index - 1] / 2.0;
    }
    v[..dimension].fill(0.0);
    for column in 0..dimension {
        for row in 0..dimension {
            v[row] += x[column] * matrix[matrix_index(matrix_stride, row, column)];
        }
    }
    y[..dimension].copy_from_slice(&v[..dimension]);
    x[..dimension].copy_from_slice(&v[..dimension]);
    factors.solve(false, dimension, x);
    for column in 0..dimension {
        for row in 0..dimension {
            y[row] -= x[column] * matrix[matrix_index(matrix_stride, row, column)];
        }
    }
    let solve_error = y[..dimension]
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));

    y[0] = 1.0;
    for index in 1..dimension {
        y[index] = -y[index - 1] / 2.0;
    }
    for column in 0..dimension {
        w[column] = (0..dimension)
            .map(|row| matrix[matrix_index(matrix_stride, row, column)] * y[row])
            .sum();
    }
    x[..dimension].copy_from_slice(&w[..dimension]);
    factors.solve(true, dimension, w);
    y[..dimension].copy_from_slice(&w[..dimension]);
    for column in 0..dimension {
        x[column] -= (0..dimension)
            .map(|row| matrix[matrix_index(matrix_stride, row, column)] * y[row])
            .sum::<f64>();
    }
    let transpose_error = x[..dimension]
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));

    if verbose {
        let mark = if solve_error.max(transpose_error) > TINY_NUMBER {
            "*"
        } else {
            " "
        };
        output.push_str("   n   max|Lij|   max|Uij|   min|Ujj|  max|LC-U|  max|Cx-b| max|C'y-b|\n");
        let _ = write!(output, "{dimension:4}");
        for value in [
            l_max,
            u_max,
            u_min,
            factor_error,
            solve_error,
            transpose_error,
        ] {
            let _ = write!(output, " {}", format_general(value, 10));
        }
        let _ = writeln!(output, "{mark}");
    }
    stats[0] = stats[0].max(c_max);
    stats[1] = stats[1].max(l_max);
    stats[2] = stats[2].max(u_max);
    stats[3] = stats[3].min(u_min);
    stats[4] = stats[4].max(factor_error);
    stats[5] = stats[5].max(solve_error);
    stats[6] = stats[6].max(transpose_error);
    stats[4].max(stats[5]).max(stats[6])
}

fn matrix_index(stride: usize, row: usize, column: usize) -> usize {
    column * stride + row
}

fn format_general(value: f64, width: usize) -> String {
    let text = if value == 0.0 {
        "0".to_owned()
    } else {
        let exponent = value.abs().log10().floor() as i32;
        if !(-4..6).contains(&exponent) {
            let scientific = format!("{value:.5e}");
            let (mantissa, exponent) = scientific.split_once('e').unwrap();
            format!(
                "{}e{}",
                trim_decimal(mantissa),
                exponent.replace("+0", "+").replace("-0", "-")
            )
        } else {
            let decimal_places = (5 - exponent).max(0) as usize;
            trim_decimal(&format!("{value:.decimal_places$}"))
        }
    };
    format!("{text:>width$}")
}

fn trim_decimal(value: &str) -> String {
    if !value.contains('.') {
        return value.to_owned();
    }
    value.trim_end_matches('0').trim_end_matches('.').to_owned()
}

#[cfg(all(test, feature = "lumod-c"))]
mod tests {
    use super::*;

    const COMMON_PREFIX: &str = "\nTest 1 (3x3)\n1...\n2...\n3...\n4";
    const COMMON_SUFFIX: &str = "\n\nData range---------------------------Density   6.2%\n    max|Cij|     max|Lij|     max|Uij|     min|Ujj|\n     52.5681            1      52.5681      21.3502 \nSolver accuracy------------------------------------\n   max|LC-U|    max|Cx-b|   max|C'y-b|       STATUS\n           0  1.77636e-15  7.10543e-15          OK!\n---------------------------------------------------\nTime in seconds: 0.000\n";

    fn without_time(output: &str) -> &str {
        output.split_once("Time in seconds:").unwrap().0
    }

    #[test]
    fn help_distinguishes_rlumod_from_the_original_implementations() {
        let help = help_text();

        assert!(help.starts_with("rlumod: a Rust adaptation of LUMOD v2.0"));
        assert!(
            help.contains(
                "Original LUMOD by Michael Saunders, Systems Optimization Laboratory (SOL)"
            )
        );
        assert!(help.contains("Dense-matrix C adaptation by Kjell Eikland"));
        assert!(!help.starts_with("LUMOD v2.0 by"));
    }

    #[test]
    fn lumod_c_stdout_matches_the_c_example() {
        let output = run_lumod_c(&["3".into(), "1".into()]);
        assert_eq!(
            without_time(&output),
            without_time(&format!("{COMMON_PREFIX}..{COMMON_SUFFIX}"))
        );
    }

    #[test]
    fn safe_and_lumod_c_stdout_only_differ_in_elapsed_time() {
        let arguments = ["3".into(), "1".into()];
        assert_eq!(
            without_time(&run_safe(&arguments)),
            without_time(&run_lumod_c(&arguments))
        );
    }
}
