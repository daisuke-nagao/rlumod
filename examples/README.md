# Examples

## User-oriented examples

`basic_lifecycle.rs` demonstrates the complete safe-API lifecycle: building a
matrix, solving normal and transposed systems, replacing a row and column, and
removing a row and column.

```console
cargo run --example basic_lifecycle
```

`static_storage.rs` performs a build and solve with fixed arrays created by
`rlumod::stack_storage!`. The example binary has the normal Rust standard
runtime, while all storage supplied to the `no_std` library is fixed and
allocation-free. The macro does not guarantee physical stack placement, and a
valid large capacity can still overflow the stack when used as a local value.

```console
cargo run --example static_storage
```

`remove_with_ids.rs` keeps external row and column identifiers synchronized
with `LuMod::remove` using `swap_remove` semantics.

```console
cargo run --example remove_with_ids
```

## Validation drivers

`rlumod.rs` is a Rust adaptation of the original LUmod validation program. It
generates matrices, exercises every update mode, and reports factor and solve
accuracy. It is useful for regression testing rather than as an introduction
to the API.

```console
cargo run --example rlumod -- -h
```

`lumod-c.rs` runs the same validation driver through the low-level
compatibility API:

```console
cargo run --features lumod-c --example lumod-c -- -h
```

Both validation drivers accept `-v`, one of `-fd`, `-hd`, `-md`, `-dd`, `-ld`,
or `-td` to select density, followed by an optional maximum dimension and test
count.
