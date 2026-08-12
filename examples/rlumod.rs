// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

mod support;

fn main() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    print!("{}", support::run_safe(&arguments));
}
