# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the independent `c-ffi-zero-based` feature with stable checked C ABI
  descriptors and status values for the complete `f32` and `f64` safe-API
  lifecycle.
- Added the public `stack_storage!` macro for compile-time, fixed-size,
  zero-initialized factor and workspace arrays.

## [0.2.0] - 2026-08-20

### Added

- Integrated the user-facing `LUmod`, `Lprod`, and `Usolve` C entry points into
  the optional `c-ffi-one-based` feature with their original one-based storage
  convention. The feature enables `lumod-c`; the crate remains an `rlib`, while
  downstream integration owns the panic handler and final `staticlib` crate
  type.

### Changed

- Restricted the `lumod-c` Rust API to `LUmod`, `Lprod`, and `Usolve`, removing
  `LUforw`, `LUback`, `elm`, and `elmgen`, and made the C FFI module link-only.
  Removing the previously released `LUforw` and `LUback` functions is a
  breaking change from 0.1.2.

## [0.1.2] - 2026-08-18

### Added

- Added user-oriented lifecycle, fixed-storage, and removal examples, plus
  crate-level guidance for storage, numerical behavior, and migration from
  dense C LUmod.

### Changed

- Clarified the crate's scope and configured docs.rs to include the optional
  compatibility API.

## [0.1.1] - 2026-08-15

### Changed

- Corrected the public documentation and package metadata to describe the
  implemented matrix updates and linear solves without implying a conventional
  LU decomposition or sparse support.

## [0.1.0] - 2026-08-14

### Added

- A safe, `no_std` Rust API for incremental square-matrix updates using caller-owned storage.
- Row and column insertion, replacement, and removal, plus in-place solves for `f32` and `f64`.
- An optional `lumod-c` feature providing a lower-level compatibility API.
- Automated Keep a Changelog validation for commits that modify this file.

[Unreleased]: https://github.com/daisuke-nagao/rlumod/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/daisuke-nagao/rlumod/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/daisuke-nagao/rlumod/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/daisuke-nagao/rlumod/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/daisuke-nagao/rlumod/releases/tag/v0.1.0
