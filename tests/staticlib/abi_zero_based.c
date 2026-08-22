// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

#include <rlumod/rlumod.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#define RLUMOD_ALIGNAS(type) alignas(type)
#define RLUMOD_ALIGNOF(type) alignof(type)
#define RLUMOD_STATIC_ASSERT(condition) static_assert((condition), #condition)
#else
#define RLUMOD_ALIGNAS(type) _Alignas(type)
#define RLUMOD_ALIGNOF(type) _Alignof(type)
#define RLUMOD_STATIC_ASSERT(condition) _Static_assert((condition), #condition)
#endif

RLUMOD_STATIC_ASSERT(sizeof(rlumod_status) == sizeof(int32_t));
RLUMOD_STATIC_ASSERT(RLUMOD_ALIGNOF(size_t) > 1);

RLUMOD_STATIC_ASSERT(sizeof(rlumod_f32_factor) == sizeof(rlumod_f64_factor));
RLUMOD_STATIC_ASSERT(RLUMOD_ALIGNOF(rlumod_f32_factor) ==
                     RLUMOD_ALIGNOF(rlumod_f64_factor));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_factor, dimension) ==
                     offsetof(rlumod_f64_factor, dimension));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_factor, capacity) ==
                     offsetof(rlumod_f64_factor, capacity));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_factor, l) ==
                     offsetof(rlumod_f64_factor, l));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_factor, l_len) ==
                     offsetof(rlumod_f64_factor, l_len));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_factor, u) ==
                     offsetof(rlumod_f64_factor, u));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_factor, u_len) ==
                     offsetof(rlumod_f64_factor, u_len));

RLUMOD_STATIC_ASSERT(sizeof(rlumod_f32_workspace) ==
                     sizeof(rlumod_f64_workspace));
RLUMOD_STATIC_ASSERT(RLUMOD_ALIGNOF(rlumod_f32_workspace) ==
                     RLUMOD_ALIGNOF(rlumod_f64_workspace));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_workspace, y) ==
                     offsetof(rlumod_f64_workspace, y));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_workspace, y_len) ==
                     offsetof(rlumod_f64_workspace, y_len));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_workspace, z) ==
                     offsetof(rlumod_f64_workspace, z));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_workspace, z_len) ==
                     offsetof(rlumod_f64_workspace, z_len));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_workspace, w) ==
                     offsetof(rlumod_f64_workspace, w));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_f32_workspace, w_len) ==
                     offsetof(rlumod_f64_workspace, w_len));

RLUMOD_STATIC_ASSERT(offsetof(rlumod_removal, has_moved_row) == 0);
RLUMOD_STATIC_ASSERT(offsetof(rlumod_removal, has_moved_row) <
                     offsetof(rlumod_removal, moved_row));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_removal, moved_row) <
                     offsetof(rlumod_removal, has_moved_column));
RLUMOD_STATIC_ASSERT(offsetof(rlumod_removal, has_moved_column) <
                     offsetof(rlumod_removal, moved_column));
RLUMOD_STATIC_ASSERT(sizeof(rlumod_removal) >=
                     offsetof(rlumod_removal, moved_column) + sizeof(size_t));

typedef rlumod_status (*storage_lengths_fn)(size_t, size_t *, size_t *);
typedef rlumod_status (*f32_factor_init_fn)(
    rlumod_f32_factor *, size_t, size_t, float *, size_t, float *, size_t);
typedef rlumod_status (*f64_factor_init_fn)(
    rlumod_f64_factor *, size_t, size_t, double *, size_t, double *, size_t);
typedef rlumod_status (*f32_workspace_init_fn)(
    rlumod_f32_workspace *, float *, size_t, float *, size_t, float *, size_t);
typedef rlumod_status (*f64_workspace_init_fn)(
    rlumod_f64_workspace *, double *, size_t, double *, size_t, double *,
    size_t);
typedef rlumod_status (*f32_push_fn)(
    rlumod_f32_factor *, const float *, size_t, const float *, size_t, float,
    const rlumod_f32_workspace *);
typedef rlumod_status (*f64_push_fn)(
    rlumod_f64_factor *, const double *, size_t, const double *, size_t, double,
    const rlumod_f64_workspace *);
typedef rlumod_status (*f32_replace_fn)(rlumod_f32_factor *, size_t,
                                        const float *, size_t,
                                        const rlumod_f32_workspace *);
typedef rlumod_status (*f64_replace_fn)(rlumod_f64_factor *, size_t,
                                        const double *, size_t,
                                        const rlumod_f64_workspace *);
typedef rlumod_status (*f32_remove_fn)(rlumod_f32_factor *, size_t, size_t,
                                       const rlumod_f32_workspace *,
                                       rlumod_removal *);
typedef rlumod_status (*f64_remove_fn)(rlumod_f64_factor *, size_t, size_t,
                                       const rlumod_f64_workspace *,
                                       rlumod_removal *);
typedef rlumod_status (*f32_solve_fn)(const rlumod_f32_factor *, float *,
                                      size_t, size_t *);
typedef rlumod_status (*f64_solve_fn)(const rlumod_f64_factor *, double *,
                                      size_t, size_t *);

static void check_all_prototypes(void) {
  storage_lengths_fn volatile storage_lengths = rlumod_storage_lengths;
  f32_factor_init_fn volatile f32_factor_init = rlumod_f32_factor_init;
  f64_factor_init_fn volatile f64_factor_init = rlumod_f64_factor_init;
  f32_workspace_init_fn volatile f32_workspace_init =
      rlumod_f32_workspace_init;
  f64_workspace_init_fn volatile f64_workspace_init =
      rlumod_f64_workspace_init;
  f32_push_fn volatile f32_push = rlumod_f32_push;
  f64_push_fn volatile f64_push = rlumod_f64_push;
  f32_replace_fn volatile f32_replace_row = rlumod_f32_replace_row;
  f32_replace_fn volatile f32_replace_column = rlumod_f32_replace_column;
  f64_replace_fn volatile f64_replace_row = rlumod_f64_replace_row;
  f64_replace_fn volatile f64_replace_column = rlumod_f64_replace_column;
  f32_remove_fn volatile f32_remove = rlumod_f32_remove;
  f64_remove_fn volatile f64_remove = rlumod_f64_remove;
  f32_solve_fn volatile f32_solve = rlumod_f32_solve_in_place;
  f32_solve_fn volatile f32_solve_transpose =
      rlumod_f32_solve_transpose_in_place;
  f64_solve_fn volatile f64_solve = rlumod_f64_solve_in_place;
  f64_solve_fn volatile f64_solve_transpose =
      rlumod_f64_solve_transpose_in_place;

  (void)storage_lengths;
  (void)f32_factor_init;
  (void)f64_factor_init;
  (void)f32_workspace_init;
  (void)f64_workspace_init;
  (void)f32_push;
  (void)f64_push;
  (void)f32_replace_row;
  (void)f32_replace_column;
  (void)f64_replace_row;
  (void)f64_replace_column;
  (void)f32_remove;
  (void)f64_remove;
  (void)f32_solve;
  (void)f32_solve_transpose;
  (void)f64_solve;
  (void)f64_solve_transpose;
}

typedef struct f32_fixture {
  float l[6];
  float u[3];
  float y[2];
  float z[2];
  float w[2];
  rlumod_f32_factor factor;
  rlumod_f32_workspace workspace;
} f32_fixture;

typedef struct f64_fixture {
  double l[6];
  double u[3];
  double y[2];
  double z[2];
  double w[2];
  rlumod_f64_factor factor;
  rlumod_f64_workspace workspace;
} f64_fixture;

static int prepare_f32(f32_fixture *fixture, size_t capacity) {
  memset(fixture, 0, sizeof(*fixture));
  if (rlumod_f32_factor_init(&fixture->factor, 0, capacity, fixture->l, 6,
                             fixture->u, 3) != RLUMOD_STATUS_OK) {
    return 0;
  }
  return rlumod_f32_workspace_init(&fixture->workspace, fixture->y, 2,
                                   fixture->z, 2, fixture->w,
                                   2) == RLUMOD_STATUS_OK;
}

static int prepare_f64(f64_fixture *fixture, size_t capacity) {
  memset(fixture, 0, sizeof(*fixture));
  if (rlumod_f64_factor_init(&fixture->factor, 0, capacity, fixture->l, 6,
                             fixture->u, 3) != RLUMOD_STATUS_OK) {
    return 0;
  }
  return rlumod_f64_workspace_init(&fixture->workspace, fixture->y, 2,
                                   fixture->z, 2, fixture->w,
                                   2) == RLUMOD_STATUS_OK;
}

static int check_descriptor_and_removal_layouts(void) {
  f32_fixture f32;
  f64_fixture f64;
  const float f32_zero[1] = {0.0F};
  const double f64_zero[1] = {0.0};
  rlumod_removal f32_removal = {9, 9, 9, 9};
  rlumod_removal f64_removal = {9, 9, 9, 9};

  if (!prepare_f32(&f32, 2) || f32.factor.dimension != 0 ||
      f32.factor.capacity != 2 || f32.factor.l != f32.l ||
      f32.factor.l_len != 6 || f32.factor.u != f32.u ||
      f32.factor.u_len != 3 || f32.workspace.y != f32.y ||
      f32.workspace.y_len != 2 || f32.workspace.z != f32.z ||
      f32.workspace.z_len != 2 || f32.workspace.w != f32.w ||
      f32.workspace.w_len != 2) {
    return 21;
  }
  if (rlumod_f32_push(&f32.factor, NULL, 0, NULL, 0, 1.0F,
                      &f32.workspace) != RLUMOD_STATUS_OK ||
      rlumod_f32_push(&f32.factor, f32_zero, 1, f32_zero, 1, 1.0F,
                      &f32.workspace) != RLUMOD_STATUS_OK ||
      rlumod_f32_remove(&f32.factor, 1, 0, &f32.workspace, &f32_removal) !=
          RLUMOD_STATUS_OK ||
      f32_removal.has_moved_row != 0 || f32_removal.moved_row != 0 ||
      f32_removal.has_moved_column != 1 || f32_removal.moved_column != 1) {
    return 22;
  }

  if (!prepare_f64(&f64, 2) || f64.factor.dimension != 0 ||
      f64.factor.capacity != 2 || f64.factor.l != f64.l ||
      f64.factor.l_len != 6 || f64.factor.u != f64.u ||
      f64.factor.u_len != 3 || f64.workspace.y != f64.y ||
      f64.workspace.y_len != 2 || f64.workspace.z != f64.z ||
      f64.workspace.z_len != 2 || f64.workspace.w != f64.w ||
      f64.workspace.w_len != 2) {
    return 23;
  }
  if (rlumod_f64_push(&f64.factor, NULL, 0, NULL, 0, 1.0,
                      &f64.workspace) != RLUMOD_STATUS_OK ||
      rlumod_f64_push(&f64.factor, f64_zero, 1, f64_zero, 1, 1.0,
                      &f64.workspace) != RLUMOD_STATUS_OK ||
      rlumod_f64_remove(&f64.factor, 0, 1, &f64.workspace, &f64_removal) !=
          RLUMOD_STATUS_OK ||
      f64_removal.has_moved_row != 1 || f64_removal.moved_row != 1 ||
      f64_removal.has_moved_column != 0 || f64_removal.moved_column != 0) {
    return 24;
  }
  return 0;
}

static int check_status_results(void) {
  static const rlumod_status documented[14] = {
      RLUMOD_STATUS_OK,
      RLUMOD_STATUS_NULL_POINTER,
      RLUMOD_STATUS_MISALIGNED_POINTER,
      RLUMOD_STATUS_SIZE_OVERFLOW,
      RLUMOD_STATUS_OVERLAPPING_BUFFERS,
      RLUMOD_STATUS_INVALID_DIMENSION,
      RLUMOD_STATUS_INSUFFICIENT_STORAGE,
      RLUMOD_STATUS_CAPACITY_EXCEEDED,
      RLUMOD_STATUS_ROW_OUT_OF_BOUNDS,
      RLUMOD_STATUS_COLUMN_OUT_OF_BOUNDS,
      RLUMOD_STATUS_LENGTH_MISMATCH,
      RLUMOD_STATUS_INSUFFICIENT_WORKSPACE,
      RLUMOD_STATUS_SINGULAR,
      RLUMOD_STATUS_NON_FINITE_DIAGONAL,
  };
  size_t l_len = 91;
  size_t u_len = 92;
  size_t index;
  size_t i;
  double l_before[6];
  double u_before[3];
  RLUMOD_ALIGNAS(size_t) unsigned char misaligned_storage[sizeof(size_t) + 1] = {
      0};
  /*
   * This target-specific ABI probe uses the implementation-defined
   * integer-to-pointer conversion supported by every CI target. Rust rejects
   * the address before either language dereferences it.
   */
  size_t *misaligned =
      (size_t *)((uintptr_t)(void *)misaligned_storage + (uintptr_t)1);
  double short_l[1] = {0.0};
  double short_u[1] = {0.0};
  rlumod_f64_factor invalid_factor = {77, 88, NULL, 99, NULL, 100};
  f64_fixture full;
  f64_fixture singular;
  f64_fixture non_finite;
  rlumod_f64_workspace empty_workspace = {NULL, 0, NULL, 0, NULL, 0};
  const double value[1] = {1.0};
  double rhs[1] = {1.0};

  for (i = 0; i < sizeof(documented) / sizeof(documented[0]); ++i) {
    if (documented[i] != (rlumod_status)i) {
      return 50;
    }
  }

  if (rlumod_storage_lengths(1, &l_len, &u_len) != documented[0] ||
      l_len != 2 || u_len != 1) {
    return 100;
  }
  u_len = 92;
  if (rlumod_storage_lengths(1, NULL, &u_len) != documented[1] ||
      u_len != 92) {
    return 101;
  }
  if (rlumod_storage_lengths(1, misaligned, &u_len) != documented[2]) {
    return 102;
  }
  l_len = 91;
  u_len = 92;
  if (rlumod_storage_lengths(SIZE_MAX, &l_len, &u_len) != documented[3] ||
      l_len != 91 || u_len != 92) {
    return 103;
  }
  if (rlumod_storage_lengths(1, &l_len, &l_len) != documented[4] ||
      l_len != 91) {
    return 104;
  }
  if (rlumod_f64_factor_init(&invalid_factor, 2, 1, short_l, 1, short_u, 1) !=
          documented[5] ||
      invalid_factor.dimension != 77 || invalid_factor.capacity != 88 ||
      invalid_factor.l != NULL || invalid_factor.l_len != 99 ||
      invalid_factor.u != NULL || invalid_factor.u_len != 100) {
    return 105;
  }
  if (rlumod_f64_factor_init(&invalid_factor, 0, 1, short_l, 1, short_u, 1) !=
          documented[6] ||
      invalid_factor.dimension != 77 || invalid_factor.capacity != 88 ||
      invalid_factor.l != NULL || invalid_factor.l_len != 99 ||
      invalid_factor.u != NULL || invalid_factor.u_len != 100) {
    return 106;
  }

  if (!prepare_f64(&full, 0) ||
      rlumod_f64_push(&full.factor, NULL, 0, NULL, 0, 1.0,
                      &full.workspace) != documented[7]) {
    return 107;
  }
  if (!prepare_f64(&full, 2) ||
      rlumod_f64_push(&full.factor, NULL, 0, NULL, 0, 1.0,
                      &full.workspace) != documented[0]) {
    return 108;
  }
  memcpy(l_before, full.l, sizeof(l_before));
  memcpy(u_before, full.u, sizeof(u_before));
  if (rlumod_f64_replace_row(&full.factor, 1, value, 1, &full.workspace) !=
      documented[8]) {
    return 109;
  }
  if (rlumod_f64_replace_column(&full.factor, 1, value, 1, &full.workspace) !=
      documented[9]) {
    return 110;
  }
  if (rlumod_f64_solve_in_place(&full.factor, rhs, 0, NULL) !=
      documented[10]) {
    return 111;
  }
  if (rlumod_f64_workspace_init(&empty_workspace, NULL, 0, NULL, 0, NULL, 0) !=
          documented[0] ||
      rlumod_f64_replace_row(&full.factor, 0, value, 1, &empty_workspace) !=
          documented[11]) {
    return 112;
  }
  if (full.factor.dimension != 1 ||
      memcmp(full.l, l_before, sizeof(l_before)) != 0 ||
      memcmp(full.u, u_before, sizeof(u_before)) != 0) {
    return 117;
  }

  if (!prepare_f64(&singular, 1) ||
      rlumod_f64_push(&singular.factor, NULL, 0, NULL, 0, 0.0,
                      &singular.workspace) != documented[0]) {
    return 113;
  }
  index = SIZE_MAX;
  rhs[0] = 1.0;
  if (rlumod_f64_solve_in_place(&singular.factor, rhs, 1, &index) !=
          documented[12] ||
      index != 0 || rhs[0] != 1.0) {
    return 114;
  }

  if (!prepare_f64(&non_finite, 1) ||
      rlumod_f64_push(&non_finite.factor, NULL, 0, NULL, 0, NAN,
                      &non_finite.workspace) != documented[0]) {
    return 115;
  }
  index = SIZE_MAX;
  rhs[0] = 1.0;
  if (rlumod_f64_solve_transpose_in_place(&non_finite.factor, rhs, 1,
                                          &index) != documented[13] ||
      index != 0 || rhs[0] != 1.0) {
    return 116;
  }
  return 0;
}

int main(void) {
  int result;

  check_all_prototypes();
  result = check_descriptor_and_removal_layouts();
  if (result != 0) {
    return result;
  }
  return check_status_results();
}
