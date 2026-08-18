# block 0.1.6 compatibility patch

This directory contains the source of Steven Sheldon's MIT-licensed
[`block` 0.1.6](https://github.com/SSheldon/rust-block) crate. Vecgra changes
two mechanical FFI compatibility details in `src/lib.rs`:

- `_NSConcreteStackBlock` is declared as `c_void`, then its address is cast to
  the existing opaque `Class` pointer type.
- The crate's implicit C ABIs are written explicitly as `extern "C"`.

The crates.io source declares the extern static itself as the uninhabited
`Class` type. Rust warns that this will become a hard error. The patched
declaration preserves the external symbol address and Objective-C block ABI
while avoiding an impossible Rust static value.

Remove this patch when GPUI no longer resolves `block` 0.1.6.
