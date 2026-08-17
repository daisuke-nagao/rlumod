# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a separate `no_std` static library exposing the complete seven-function
  dense `lumod-c` ABI with its original one-based storage convention.
- Exposed the low-level `elm` and `elmgen` operations through the optional
  Rust `lumod-c` slice API.

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

[Unreleased]: https://github.com/daisuke-nagao/rlumod/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/daisuke-nagao/rlumod/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/daisuke-nagao/rlumod/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/daisuke-nagao/rlumod/releases/tag/v0.1.0
