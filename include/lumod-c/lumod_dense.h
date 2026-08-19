// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#ifndef UUID_9A76DC0D_EB08_4049_BB6E_2D5A8239FDCE
#define UUID_9A76DC0D_EB08_4049_BB6E_2D5A8239FDCE

/*
 * rlumod dense LUmod compatibility API.
 *
 * Factor and vector storage is one-based: element zero is an unused dummy.
 * L needs maxmod * maxmod + 1 doubles and U needs
 * maxmod * (maxmod + 1) / 2 + 1 doubles. Vector buffers need one dummy
 * followed by every element required by the operation.
 *
 * Every pointer must be non-null, aligned, writable for its required length,
 * and non-overlapping with every other mutable region used by the call.
 */

#ifdef __cplusplus
extern "C" {
#endif

void LUmod(int mode, int maxmod, int n, int krow, int kcol,
           double *L, double *U, double *y, double *z, double *w);
void Lprod(int mode, int maxmod, int n,
           double *L, double *y, double *z);
void Usolve(int mode, int maxmod, int n, double *U, double *y);

#ifdef __cplusplus
}
#endif

#endif /* UUID_9A76DC0D_EB08_4049_BB6E_2D5A8239FDCE */
