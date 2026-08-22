// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#ifndef UUID_D1748435_8FFF_4A42_8B1B_3508D14C8466
#define UUID_D1748435_8FFF_4A42_8B1B_3508D14C8466

#include <stddef.h>
#include <stdint.h>

/*
 * rlumod zero-based checked C ABI.
 *
 * This interface reports detectable argument errors, but remains an unsafe
 * raw-pointer ABI. The caller must provide live, initialized allocations of
 * the declared lengths for the duration of each call. Except for immutable
 * row and column inputs, participating memory regions must not overlap.
 *
 * Factor and workspace descriptor fields may be read after initialization,
 * but must not be modified by the caller. Push and remove update dimension.
 * Calls using the same factor or buffers must be externally synchronized;
 * solves modify private scratch storage inside L.
 *
 * A buffer pointer may be null only when its length is zero. Optional removal
 * and solve-error outputs may always be null.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t rlumod_status;

#define RLUMOD_STATUS_OK ((rlumod_status)0)
#define RLUMOD_STATUS_NULL_POINTER ((rlumod_status)1)
#define RLUMOD_STATUS_MISALIGNED_POINTER ((rlumod_status)2)
#define RLUMOD_STATUS_SIZE_OVERFLOW ((rlumod_status)3)
#define RLUMOD_STATUS_OVERLAPPING_BUFFERS ((rlumod_status)4)
#define RLUMOD_STATUS_INVALID_DIMENSION ((rlumod_status)5)
#define RLUMOD_STATUS_INSUFFICIENT_STORAGE ((rlumod_status)6)
#define RLUMOD_STATUS_CAPACITY_EXCEEDED ((rlumod_status)7)
#define RLUMOD_STATUS_ROW_OUT_OF_BOUNDS ((rlumod_status)8)
#define RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS ((rlumod_status)9)
#define RLUMOD_STATUS_LENGTH_MISMATCH ((rlumod_status)10)
#define RLUMOD_STATUS_INSUFFICIENT_WORKSPACE ((rlumod_status)11)
#define RLUMOD_STATUS_SINGULAR ((rlumod_status)12)
#define RLUMOD_STATUS_NON_FINITE_DIAGONAL ((rlumod_status)13)

typedef struct rlumod_f32_factor {
  size_t dimension;
  size_t capacity;
  float *l;
  size_t l_len;
  float *u;
  size_t u_len;
} rlumod_f32_factor;

typedef struct rlumod_f64_factor {
  size_t dimension;
  size_t capacity;
  double *l;
  size_t l_len;
  double *u;
  size_t u_len;
} rlumod_f64_factor;

typedef struct rlumod_f32_workspace {
  float *y;
  size_t y_len;
  float *z;
  size_t z_len;
  float *w;
  size_t w_len;
} rlumod_f32_workspace;

typedef struct rlumod_f64_workspace {
  double *y;
  size_t y_len;
  double *z;
  size_t z_len;
  double *w;
  size_t w_len;
} rlumod_f64_workspace;

typedef struct rlumod_removal {
  uint8_t has_moved_row;
  size_t moved_row;
  uint8_t has_moved_column;
  size_t moved_column;
} rlumod_removal;

rlumod_status rlumod_storage_lengths(size_t capacity, size_t *l_len,
                                     size_t *u_len);

rlumod_status rlumod_f32_factor_init(
    rlumod_f32_factor *factor, size_t dimension, size_t capacity, float *l,
    size_t l_len, float *u, size_t u_len);
rlumod_status rlumod_f32_workspace_init(
    rlumod_f32_workspace *workspace, float *y, size_t y_len, float *z,
    size_t z_len, float *w, size_t w_len);
rlumod_status rlumod_f32_push(rlumod_f32_factor *factor, const float *row,
                              size_t row_len, const float *column,
                              size_t column_len, float diagonal,
                              const rlumod_f32_workspace *workspace);
rlumod_status rlumod_f32_replace_row(
    rlumod_f32_factor *factor, size_t row, const float *values,
    size_t values_len, const rlumod_f32_workspace *workspace);
rlumod_status rlumod_f32_replace_column(
    rlumod_f32_factor *factor, size_t column, const float *values,
    size_t values_len, const rlumod_f32_workspace *workspace);
rlumod_status rlumod_f32_remove(
    rlumod_f32_factor *factor, size_t row, size_t column,
    const rlumod_f32_workspace *workspace, rlumod_removal *removal);
rlumod_status rlumod_f32_solve_in_place(const rlumod_f32_factor *factor,
                                        float *rhs, size_t rhs_len,
                                        size_t *error_index);
rlumod_status rlumod_f32_solve_transpose_in_place(
    const rlumod_f32_factor *factor, float *rhs, size_t rhs_len,
    size_t *error_index);

rlumod_status rlumod_f64_factor_init(
    rlumod_f64_factor *factor, size_t dimension, size_t capacity, double *l,
    size_t l_len, double *u, size_t u_len);
rlumod_status rlumod_f64_workspace_init(
    rlumod_f64_workspace *workspace, double *y, size_t y_len, double *z,
    size_t z_len, double *w, size_t w_len);
rlumod_status rlumod_f64_push(rlumod_f64_factor *factor, const double *row,
                              size_t row_len, const double *column,
                              size_t column_len, double diagonal,
                              const rlumod_f64_workspace *workspace);
rlumod_status rlumod_f64_replace_row(
    rlumod_f64_factor *factor, size_t row, const double *values,
    size_t values_len, const rlumod_f64_workspace *workspace);
rlumod_status rlumod_f64_replace_column(
    rlumod_f64_factor *factor, size_t column, const double *values,
    size_t values_len, const rlumod_f64_workspace *workspace);
rlumod_status rlumod_f64_remove(
    rlumod_f64_factor *factor, size_t row, size_t column,
    const rlumod_f64_workspace *workspace, rlumod_removal *removal);
rlumod_status rlumod_f64_solve_in_place(const rlumod_f64_factor *factor,
                                        double *rhs, size_t rhs_len,
                                        size_t *error_index);
rlumod_status rlumod_f64_solve_transpose_in_place(
    const rlumod_f64_factor *factor, double *rhs, size_t rhs_len,
    size_t *error_index);

#ifdef __cplusplus
}
#endif

#endif /* UUID_D1748435_8FFF_4A42_8B1B_3508D14C8466 */
