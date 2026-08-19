// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#![no_std]

extern crate rlumod;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

// Stock core may retain this reference even when the final crate aborts on panic.
#[cfg(not(target_env = "msvc"))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    loop {
        core::hint::spin_loop();
    }
}
