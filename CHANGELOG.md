# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A safe, `no_std` Rust API for updating dense LU factorizations using caller-owned storage.
- Row and column insertion, replacement, and removal, plus in-place solves for `f32` and `f64`.
- An optional `lumod-c` feature providing a lower-level compatibility API.
- Automated Keep a Changelog validation for commits that modify this file.

[Unreleased]: https://github.com/daisuke-nagao/rlumod/commits/HEAD
