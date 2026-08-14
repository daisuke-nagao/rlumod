// SPDX-FileCopyrightText: 2026 Daisuke Nagao
// SPDX-License-Identifier: MIT

use core::{
    cell::Cell,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

use crate::algorithm::{self, DenseL, Numerics, PackedU};

mod sealed {
    pub trait Sealed {}
}

/// A floating-point type supported by [`LuMod`].
///
/// This trait is sealed; the initial API supports only `f32` and `f64`.
pub trait Real:
    sealed::Sealed
    + Add<Output = Self>
    + AddAssign
    + Copy
    + Div<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + PartialOrd
    + Sub<Output = Self>
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Difference between `1.0` and the next representable value.
    const EPSILON: Self;
    /// Small magnitude used to stabilize elementary transformations.
    const UPDATE_TINY: Self;

    /// Returns the absolute value of `self`.
    fn abs(self) -> Self;
    /// Returns whether `self` is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

macro_rules! real {
    ($type:ty) => {
        impl sealed::Sealed for $type {}

        impl Real for $type {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const EPSILON: Self = Self::EPSILON;
            const UPDATE_TINY: Self = 1.0e-4;

            fn abs(self) -> Self {
                self.abs()
            }

            fn is_finite(self) -> bool {
                self.is_finite()
            }
        }
    };
}

real!(f32);
real!(f64);

fn numerics<T: Real>() -> Numerics<T> {
    Numerics {
        epsilon: T::EPSILON,
        update_tiny: T::UPDATE_TINY,
    }
}

/// A zero-based row index in the represented matrix.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowIndex(
    /// The zero-based index.
    pub usize,
);

/// A zero-based column index in the represented matrix.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnIndex(
    /// The zero-based index.
    pub usize,
);

/// Describes which last row and column were moved by [`LuMod::remove`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Removal {
    /// Original index of the last row, when it was moved into the removed row.
    pub moved_row: Option<RowIndex>,
    /// Original index of the last column, when it was moved into the removed column.
    pub moved_column: Option<ColumnIndex>,
}

/// An error in caller-owned storage or its requested dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StorageError {
    /// The logical dimension exceeds the storage capacity.
    InvalidDimension,
    /// A buffer is too short or workspace buffers have different lengths.
    InsufficientStorage,
    /// Computing the storage length overflowed [`usize`].
    SizeOverflow,
}

/// Required lengths of the caller-owned factor buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageLengths {
    /// Length of the L-factor buffer, including internal solve workspace.
    pub l: usize,
    /// Length of the packed U-factor buffer.
    pub u: usize,
}

/// Computes the factor-buffer lengths required for `capacity`.
///
/// The L buffer contains `capacity * capacity` factor elements followed
/// by `capacity` solve-work elements. The U buffer contains
/// `capacity * (capacity + 1) / 2` packed upper-triangular elements.
///
/// # Errors
///
/// Returns [`StorageError::SizeOverflow`] if either length cannot be represented
/// by [`usize`].
///
/// ```
/// let lengths = rlumod::storage_lengths(4).unwrap();
/// let mut l = vec![0.0; lengths.l];
/// let mut u = vec![0.0; lengths.u];
/// let factor = rlumod::LuMod::from_storage(0, 4, &mut l, &mut u).unwrap();
/// assert_eq!(factor.capacity(), 4);
/// ```
pub fn storage_lengths(capacity: usize) -> Result<StorageLengths, StorageError> {
    let next = capacity.checked_add(1).ok_or(StorageError::SizeOverflow)?;
    let l = capacity
        .checked_mul(next)
        .ok_or(StorageError::SizeOverflow)?;
    let u = if capacity % 2 == 0 {
        (capacity / 2) * next
    } else {
        capacity * (next / 2)
    };
    Ok(StorageLengths { l, u })
}

/// An error that prevents a structural factor update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UpdateError {
    /// A push was requested when the dimension already equals the capacity.
    CapacityExceeded,
    /// A row index is outside `0..dimension`.
    RowOutOfBounds,
    /// A column index is outside `0..dimension`.
    ColumnOutOfBounds,
    /// An input row or column does not have exactly `dimension` elements.
    LengthMismatch,
    /// Each workspace buffer is shorter than the operation requires.
    InsufficientWorkspace,
}

/// An error that prevents solving with the current factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveError {
    /// The right-hand side does not have exactly `dimension` elements.
    LengthMismatch,
    /// The U factor has a zero diagonal entry.
    Singular {
        /// Zero-based index of the first zero diagonal entry.
        index: usize,
    },
    /// The U factor has an infinite or NaN diagonal entry.
    NonFiniteDiagonal {
        /// Zero-based index of the first non-finite diagonal entry.
        index: usize,
    },
}

/// Caller-owned work buffers used by factor updates.
///
/// All three buffers always have the same length. An operation may require
/// more capacity than a successfully constructed workspace currently has.
pub struct Workspace<'a, T> {
    y: &'a mut [T],
    z: &'a mut [T],
    w: &'a mut [T],
}

impl<'a, T> Workspace<'a, T> {
    /// Creates reusable update workspace from three equally sized buffers.
    ///
    /// The buffers are scratch storage and their contents are not preserved by
    /// successful updates.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InsufficientStorage`] if the lengths differ.
    pub fn new(y: &'a mut [T], z: &'a mut [T], w: &'a mut [T]) -> Result<Self, StorageError> {
        if y.len() != z.len() || y.len() != w.len() {
            return Err(StorageError::InsufficientStorage);
        }
        Ok(Self { y, z, w })
    }

    fn has_capacity(&self, required: usize) -> bool {
        self.y.len() >= required
    }
}

/// An updatable square-matrix representation backed by caller-owned factor storage.
///
/// The logical dimension never exceeds the capacity. The borrowed buffers
/// contain the factors for the current square matrix and all update and solve
/// operations allocate no heap memory. A represented matrix may be singular or
/// contain non-finite values; solve operations report invalid U diagonals.
pub struct LuMod<'a, T> {
    n: usize,
    capacity: usize,
    l: &'a mut [T],
    u: &'a mut [T],
    solve_work: &'a [Cell<T>],
}

impl<'a, T: Real> LuMod<'a, T> {
    /// Adopts existing factor storage without changing its contents.
    ///
    /// `l` needs `capacity * capacity` factor elements followed by
    /// `capacity` solve-work elements. `u` uses packed upper-triangular
    /// storage and needs `capacity * (capacity + 1) / 2` elements. Extra
    /// elements are ignored.
    ///
    /// # Preconditions
    ///
    /// Unless `dimension` is zero, the required prefixes of `l` and `u` must
    /// already encode factors produced for the same `capacity`, such as buffers
    /// retained from an earlier `LuMod`. This mathematical condition is not
    /// validated.
    ///
    /// # Postconditions
    ///
    /// The buffer contents are unchanged and the returned value has the given
    /// dimension and capacity.
    ///
    /// # Errors
    ///
    /// Returns [`StorageError::InvalidDimension`] when `dimension > capacity`,
    /// [`StorageError::SizeOverflow`] when buffer-length arithmetic overflows,
    /// or [`StorageError::InsufficientStorage`] when either buffer is too short.
    pub fn from_storage(
        dimension: usize,
        capacity: usize,
        l: &'a mut [T],
        u: &'a mut [T],
    ) -> Result<Self, StorageError> {
        if dimension > capacity {
            return Err(StorageError::InvalidDimension);
        }
        let lengths = storage_lengths(capacity)?;
        let factor_len = lengths.l - capacity;
        if l.len() < lengths.l || u.len() < lengths.u {
            return Err(StorageError::InsufficientStorage);
        }

        let (l, remainder) = l.split_at_mut(factor_len);
        let solve_work = Cell::from_mut(&mut remainder[..capacity]).as_slice_of_cells();
        Ok(Self {
            n: dimension,
            capacity,
            l,
            u: &mut u[..lengths.u],
            solve_work,
        })
    }

    /// Returns the current matrix dimension.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Returns the maximum matrix dimension supported by the borrowed storage.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Appends a last row and column to the represented matrix.
    ///
    /// If the current matrix is `A`, success represents
    /// `[[A, column], [row, diagonal]]` and increases the dimension by one.
    /// `row` and `column` must each contain the current dimension, and the
    /// workspace must contain at least the new dimension. Singular and
    /// non-finite values are accepted.
    ///
    /// # Errors
    ///
    /// Returns [`UpdateError::CapacityExceeded`], [`UpdateError::LengthMismatch`],
    /// or [`UpdateError::InsufficientWorkspace`] when the corresponding
    /// structural requirement is not met. On error, the factors and dimension
    /// are unchanged.
    pub fn push(
        &mut self,
        row: &[T],
        column: &[T],
        diagonal: T,
        workspace: &mut Workspace<'_, T>,
    ) -> Result<(), UpdateError> {
        if self.n == self.capacity {
            return Err(UpdateError::CapacityExceeded);
        }
        if row.len() != self.n || column.len() != self.n {
            return Err(UpdateError::LengthMismatch);
        }
        let new_dimension = self.n + 1;
        if !workspace.has_capacity(new_dimension) {
            return Err(UpdateError::InsufficientWorkspace);
        }

        workspace.y[..self.n].copy_from_slice(row);
        workspace.z[..self.n].copy_from_slice(column);
        workspace.y[self.n] = diagonal;
        workspace.z[self.n] = diagonal;
        let mut l = DenseL::new(&mut *self.l, self.capacity);
        let mut u = PackedU::new(&mut *self.u, self.capacity);
        algorithm::push(
            new_dimension,
            &mut l,
            &mut u,
            workspace.y,
            workspace.z,
            workspace.w,
            numerics(),
        );
        self.n = new_dimension;
        Ok(())
    }

    /// Replaces one row of the represented matrix.
    ///
    /// On success, `values` becomes the complete row at the zero-based `row`
    /// index. The dimension is unchanged. Singular and non-finite values are
    /// accepted.
    ///
    /// # Errors
    ///
    /// Returns [`UpdateError::RowOutOfBounds`], [`UpdateError::LengthMismatch`],
    /// or [`UpdateError::InsufficientWorkspace`]. On error, the factors and
    /// dimension are unchanged.
    pub fn replace_row(
        &mut self,
        row: RowIndex,
        values: &[T],
        workspace: &mut Workspace<'_, T>,
    ) -> Result<(), UpdateError> {
        if row.0 >= self.n {
            return Err(UpdateError::RowOutOfBounds);
        }
        self.validate_replacement(values, workspace)?;
        workspace.y[..self.n].copy_from_slice(values);
        let mut l = DenseL::new(&mut *self.l, self.capacity);
        let mut u = PackedU::new(&mut *self.u, self.capacity);
        algorithm::replace_row(
            row.0,
            self.n,
            &mut l,
            &mut u,
            workspace.y,
            workspace.z,
            workspace.w,
            numerics(),
        );
        Ok(())
    }

    /// Replaces one column of the represented matrix.
    ///
    /// On success, `values` becomes the complete column at the zero-based
    /// `column` index. The dimension is unchanged. Singular and non-finite
    /// values are accepted.
    ///
    /// # Errors
    ///
    /// Returns [`UpdateError::ColumnOutOfBounds`],
    /// [`UpdateError::LengthMismatch`], or
    /// [`UpdateError::InsufficientWorkspace`]. On error, the factors and
    /// dimension are unchanged.
    pub fn replace_column(
        &mut self,
        column: ColumnIndex,
        values: &[T],
        workspace: &mut Workspace<'_, T>,
    ) -> Result<(), UpdateError> {
        if column.0 >= self.n {
            return Err(UpdateError::ColumnOutOfBounds);
        }
        self.validate_replacement(values, workspace)?;
        workspace.z[..self.n].copy_from_slice(values);
        let mut l = DenseL::new(&mut *self.l, self.capacity);
        let mut u = PackedU::new(&mut *self.u, self.capacity);
        algorithm::replace_column(
            column.0,
            self.n,
            &mut l,
            &mut u,
            workspace.y,
            workspace.z,
            workspace.w,
            numerics(),
        );
        Ok(())
    }

    /// Removes a row and column, reducing the dimension by one.
    ///
    /// Before removal, the last row is moved into `row` and the last column is
    /// moved into `column` when either requested index is not already last.
    /// The returned [`Removal`] identifies the original last indices that were
    /// moved, allowing callers to keep external row and column identities in
    /// sync. The workspace must contain at least the current dimension.
    ///
    /// # Errors
    ///
    /// Returns [`UpdateError::RowOutOfBounds`],
    /// [`UpdateError::ColumnOutOfBounds`], or
    /// [`UpdateError::InsufficientWorkspace`]. On error, the factors and
    /// dimension are unchanged.
    pub fn remove(
        &mut self,
        row: RowIndex,
        column: ColumnIndex,
        workspace: &mut Workspace<'_, T>,
    ) -> Result<Removal, UpdateError> {
        if row.0 >= self.n {
            return Err(UpdateError::RowOutOfBounds);
        }
        if column.0 >= self.n {
            return Err(UpdateError::ColumnOutOfBounds);
        }
        if !workspace.has_capacity(self.n) {
            return Err(UpdateError::InsufficientWorkspace);
        }

        let last = self.n - 1;
        let removal = Removal {
            moved_row: (row.0 != last).then_some(RowIndex(last)),
            moved_column: (column.0 != last).then_some(ColumnIndex(last)),
        };
        let mut l = DenseL::new(&mut *self.l, self.capacity);
        let mut u = PackedU::new(&mut *self.u, self.capacity);
        algorithm::remove(
            row.0,
            column.0,
            self.n,
            &mut l,
            &mut u,
            workspace.y,
            workspace.z,
            workspace.w,
            numerics(),
        );
        self.n = last;
        Ok(removal)
    }

    /// Solves `A * x = rhs` and overwrites `rhs` with `x`.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::LengthMismatch`] unless `rhs.len() == dimension`,
    /// [`SolveError::Singular`] for a zero U-factor diagonal, or
    /// [`SolveError::NonFiniteDiagonal`] for an infinite or NaN diagonal. On
    /// error, `rhs` is unchanged.
    pub fn solve_in_place(&self, rhs: &mut [T]) -> Result<(), SolveError> {
        self.validate_solve(rhs)?;
        let l = DenseL::new(&*self.l, self.capacity);
        for row in 0..self.n {
            self.solve_work[row].set(algorithm::l_product_value(false, self.n, &l, rhs, row));
        }
        for (value, work) in rhs.iter_mut().zip(self.solve_work) {
            *value = work.get();
        }
        let u = PackedU::new(&*self.u, self.capacity);
        algorithm::u_solve(false, self.n, &u, rhs);
        Ok(())
    }

    /// Solves `Aᵀ * x = rhs` and overwrites `rhs` with `x`.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::LengthMismatch`] unless `rhs.len() == dimension`,
    /// [`SolveError::Singular`] for a zero U-factor diagonal, or
    /// [`SolveError::NonFiniteDiagonal`] for an infinite or NaN diagonal. On
    /// error, `rhs` is unchanged.
    pub fn solve_transpose_in_place(&self, rhs: &mut [T]) -> Result<(), SolveError> {
        self.validate_solve(rhs)?;
        let u = PackedU::new(&*self.u, self.capacity);
        algorithm::u_solve(true, self.n, &u, rhs);
        let l = DenseL::new(&*self.l, self.capacity);
        for column in 0..self.n {
            self.solve_work[column].set(algorithm::l_product_value(true, self.n, &l, rhs, column));
        }
        for (value, work) in rhs.iter_mut().zip(self.solve_work) {
            *value = work.get();
        }
        Ok(())
    }

    fn validate_replacement(
        &self,
        values: &[T],
        workspace: &Workspace<'_, T>,
    ) -> Result<(), UpdateError> {
        if values.len() != self.n {
            return Err(UpdateError::LengthMismatch);
        }
        if !workspace.has_capacity(self.n) {
            return Err(UpdateError::InsufficientWorkspace);
        }
        Ok(())
    }

    fn validate_solve(&self, rhs: &[T]) -> Result<(), SolveError> {
        if rhs.len() != self.n {
            return Err(SolveError::LengthMismatch);
        }
        for index in 0..self.n {
            let diagonal = self.u[algorithm::u_index(index, index, self.capacity)];
            if diagonal == T::ZERO {
                return Err(SolveError::Singular { index });
            }
            if !diagonal.is_finite() {
                return Err(SolveError::NonFiniteDiagonal { index });
            }
        }
        Ok(())
    }
}
