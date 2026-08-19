use super::*;
use crate::lumod_c as rust;

const CANARY: f64 = 123_456.0;

fn logical(storage: &[f64]) -> &[f64] {
    &storage[1..storage.len() - 1]
}

fn run_rust_lifecycle(l: &mut [f64], u: &mut [f64], y: &mut [f64], z: &mut [f64], w: &mut [f64]) {
    for (n, row, column) in [
        (1, &[4.0][..], &[4.0][..]),
        (2, &[1.0, 5.0][..], &[2.0, 5.0][..]),
        (3, &[3.0, 1.0, 6.0][..], &[1.0, 2.0, 6.0][..]),
    ] {
        y[..n].copy_from_slice(row);
        z[..n].copy_from_slice(column);
        rust::LUmod(1, 3, n as i32, -1, -1, l, u, y, z, w);
    }
    y.copy_from_slice(&[2.0, 7.0, 1.0]);
    rust::LUmod(3, 3, 3, 1, -1, l, u, y, z, w);
    z.copy_from_slice(&[8.0, 2.0, 1.0]);
    rust::LUmod(2, 3, 3, -1, 0, l, u, y, z, w);
    rust::LUmod(4, 3, 3, 0, 1, l, u, y, z, w);
}

#[test]
fn exports_one_based_lifecycle_and_solve_abi() {
    let (mut expected_l, mut expected_u) = ([0.0; 9], [0.0; 6]);
    let (mut expected_y, mut expected_z, mut expected_w) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    run_rust_lifecycle(
        &mut expected_l,
        &mut expected_u,
        &mut expected_y,
        &mut expected_z,
        &mut expected_w,
    );

    let (mut l, mut u) = ([CANARY; 11], [CANARY; 8]);
    let (mut y, mut z, mut w) = ([CANARY; 5], [CANARY; 5], [CANARY; 5]);
    unsafe {
        for (n, row, column) in [
            (1, &[4.0][..], &[4.0][..]),
            (2, &[1.0, 5.0][..], &[2.0, 5.0][..]),
            (3, &[3.0, 1.0, 6.0][..], &[1.0, 2.0, 6.0][..]),
        ] {
            y[1..=n].copy_from_slice(row);
            z[1..=n].copy_from_slice(column);
            LUmod(
                1,
                3,
                n as i32,
                0,
                0,
                l.as_mut_ptr(),
                u.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
                w.as_mut_ptr(),
            );
        }
        y[1..4].copy_from_slice(&[2.0, 7.0, 1.0]);
        LUmod(
            3,
            3,
            3,
            2,
            0,
            l.as_mut_ptr(),
            u.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
            w.as_mut_ptr(),
        );
        z[1..4].copy_from_slice(&[8.0, 2.0, 1.0]);
        LUmod(
            2,
            3,
            3,
            0,
            1,
            l.as_mut_ptr(),
            u.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
            w.as_mut_ptr(),
        );
        LUmod(
            4,
            3,
            3,
            1,
            2,
            l.as_mut_ptr(),
            u.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
            w.as_mut_ptr(),
        );
    }

    assert_eq!(logical(&l), expected_l);
    assert_eq!(logical(&u), expected_u);
    assert_eq!(logical(&y), expected_y);
    assert_eq!(logical(&z), expected_z);
    assert_eq!(logical(&w), expected_w);
    assert_eq!((l[0], l[10], u[0], u[7]), (CANARY, CANARY, CANARY, CANARY));
    assert_eq!(
        (y[0], y[4], z[0], z[4], w[0], w[4]),
        (CANARY, CANARY, CANARY, CANARY, CANARY, CANARY),
    );

    let mut expected_solution = [9.0, 11.0];
    let mut expected_work = [0.0; 2];
    rust::Lprod(
        1,
        3,
        2,
        &mut expected_l,
        &mut expected_solution,
        &mut expected_work,
    );
    expected_solution.copy_from_slice(&expected_work);
    rust::Usolve(1, 3, 2, &mut expected_u, &mut expected_solution);

    let mut solution = [CANARY, 9.0, 11.0, CANARY];
    let mut work = [CANARY, 0.0, 0.0, CANARY];
    unsafe {
        Lprod(
            1,
            3,
            2,
            l.as_mut_ptr(),
            solution.as_mut_ptr(),
            work.as_mut_ptr(),
        );
        solution[1..3].copy_from_slice(&work[1..3]);
        Usolve(1, 3, 2, u.as_mut_ptr(), solution.as_mut_ptr());
    }
    assert_eq!(&solution[1..3], expected_solution);
    assert_eq!(
        (solution[0], solution[3], work[0], work[3]),
        (CANARY, CANARY, CANARY, CANARY)
    );

    let mut expected_transpose = [7.0, 13.0];
    rust::Usolve(2, 3, 2, &mut expected_u, &mut expected_transpose);
    rust::Lprod(
        2,
        3,
        2,
        &mut expected_l,
        &mut expected_transpose,
        &mut expected_work,
    );
    expected_transpose.copy_from_slice(&expected_work);

    let mut transpose = [CANARY, 7.0, 13.0, CANARY];
    unsafe {
        Usolve(2, 3, 2, u.as_mut_ptr(), transpose.as_mut_ptr());
        Lprod(
            2,
            3,
            2,
            l.as_mut_ptr(),
            transpose.as_mut_ptr(),
            work.as_mut_ptr(),
        );
        transpose[1..3].copy_from_slice(&work[1..3]);
    }
    assert_eq!(&transpose[1..3], expected_transpose);
}

fn identity_storage() -> ([f64; 9], [f64; 6]) {
    let mut l = [0.0; 9];
    let mut u = [0.0; 6];
    l[0] = 1.0;
    l[4] = 1.0;
    u[0] = 1.0;
    u[3] = 1.0;
    (l, u)
}

#[test]
fn translates_forward_and_backward_indices() {
    let (mut expected_l, mut expected_u) = identity_storage();
    let mut expected_y = [2.0, 3.0, 0.0];
    rust::LUforw(
        0,
        1,
        2,
        2,
        3,
        2.22e-16,
        &mut expected_l,
        &mut expected_u,
        &mut expected_y,
    );

    let (base_l, base_u) = identity_storage();
    let (mut l, mut u) = ([CANARY; 11], [CANARY; 8]);
    l[1..10].copy_from_slice(&base_l);
    u[1..7].copy_from_slice(&base_u);
    let mut y = [CANARY, 2.0, 3.0, 0.0, CANARY];
    unsafe {
        LUforw(
            1,
            2,
            2,
            2,
            3,
            2.22e-16,
            l.as_mut_ptr(),
            u.as_mut_ptr(),
            y.as_mut_ptr(),
        )
    };
    assert_eq!(logical(&l), expected_l);
    assert_eq!(logical(&u), expected_u);
    assert_eq!(logical(&y), expected_y);
    assert_eq!(
        (l[0], l[10], u[0], u[7], y[0], y[4]),
        (CANARY, CANARY, CANARY, CANARY, CANARY, CANARY)
    );

    for initial_last in [2, -2] {
        let (mut expected_l, mut expected_u) = identity_storage();
        let (mut expected_y, mut expected_z) = ([0.0; 3], [2.0, 0.0, 0.0]);
        let mut expected_last = if initial_last > 0 {
            initial_last - 1
        } else {
            initial_last
        };
        rust::LUback(
            0,
            &mut expected_last,
            2,
            2,
            3,
            2.22e-16,
            &mut expected_l,
            &mut expected_u,
            &mut expected_y,
            &mut expected_z,
        );

        let (base_l, base_u) = identity_storage();
        let (mut l, mut u) = ([CANARY; 11], [CANARY; 8]);
        l[1..10].copy_from_slice(&base_l);
        u[1..7].copy_from_slice(&base_u);
        let (mut y, mut z) = (
            [CANARY, 0.0, 0.0, 0.0, CANARY],
            [CANARY, 2.0, 0.0, 0.0, CANARY],
        );
        let mut actual_last = initial_last;
        unsafe {
            LUback(
                1,
                &mut actual_last,
                2,
                2,
                3,
                2.22e-16,
                l.as_mut_ptr(),
                u.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
            )
        };
        assert_eq!(actual_last, if initial_last > 0 { 1 } else { 2 });
        assert_eq!(actual_last, expected_last + 1);
        assert_eq!(logical(&l), expected_l);
        assert_eq!(logical(&u), expected_u);
        assert_eq!(logical(&y), expected_y);
        assert_eq!(logical(&z), expected_z);
        assert_eq!(
            (l[0], l[10], u[0], u[7], y[0], y[4], z[0], z[4]),
            (
                CANARY, CANARY, CANARY, CANARY, CANARY, CANARY, CANARY, CANARY
            )
        );
    }
}

#[test]
fn exports_low_level_transform_abi() {
    let mut x = [CANARY, 1.0, 2.0, CANARY];
    let mut y = [CANARY, 3.0, 4.0, CANARY];
    unsafe { elm(7, 2, x.as_mut_ptr(), y.as_mut_ptr(), -1.0, 0.5) };
    assert_eq!(x, [CANARY, 3.0, 4.0, CANARY]);
    assert_eq!(y, [CANARY, 2.5, 4.0, CANARY]);

    let (mut scalar_x, mut scalar_y, mut cs, mut sn) = (4.0, 2.0, 9.0, 9.0);
    unsafe { elmgen(&mut scalar_x, &mut scalar_y, 2.22e-16, &mut cs, &mut sn) };
    assert_eq!((scalar_x, scalar_y, cs, sn), (4.0, 0.0, 0.0, -0.5));
}

#[test]
fn rejects_invalid_scalars_before_touching_pointers() {
    let mut last = 2;
    unsafe {
        LUmod(
            99,
            3,
            3,
            0,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        LUmod(
            1,
            3,
            4,
            0,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        Lprod(
            1,
            3,
            0,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        Usolve(1, 3, 0, core::ptr::null_mut(), core::ptr::null_mut());
        LUforw(
            2,
            1,
            2,
            2,
            3,
            2.22e-16,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        LUforw(
            1,
            2,
            2,
            4,
            3,
            2.22e-16,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        LUback(
            1,
            core::ptr::null_mut(),
            2,
            2,
            3,
            2.22e-16,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        LUback(
            2,
            &mut last,
            1,
            1,
            3,
            2.22e-16,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        elm(1, 0, core::ptr::null_mut(), core::ptr::null_mut(), 0.0, 0.0);
        elmgen(
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            2.22e-16,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
    }
    assert_eq!(last, 2);
}

#[test]
fn rejects_lengths_that_cannot_form_rust_slices() {
    assert_eq!(storage_lengths(1), Some((1, 1)));
    assert_eq!(storage_lengths(c_int::MAX), None);
}
