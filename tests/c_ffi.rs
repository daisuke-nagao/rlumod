#![cfg(feature = "c-ffi-one-based")]

#[test]
fn exposes_c_ffi_from_rlumod() {
    let (mut x, mut y, mut cs, mut sn) = (4.0, 2.0, 9.0, 9.0);

    unsafe { rlumod::ffi::elmgen(&mut x, &mut y, 2.22e-16, &mut cs, &mut sn) };

    assert_eq!((x, y, cs, sn), (4.0, 0.0, 0.0, -0.5));
}
