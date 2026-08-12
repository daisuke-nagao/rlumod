// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#[path = "support/mod.rs"]
mod support;

fn main() {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    print!("{}", support::run_lumod_c(&arguments));
}
