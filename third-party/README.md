# Third-party compatibility code

`noop-ztracing` and `noop-ztracing-macro` independently implement the small
surface GPUI uses when Zed tracing is disabled. Vecgra never enables the
`ZTRACING` configuration, so the upstream implementation is already a no-op at
runtime. The local crates prevent disabled GPL-licensed instrumentation code
from becoming a dependency of the Apache-2.0 Studio binary.

These compatibility crates are original Vecgra code licensed under Apache-2.0.
They deliberately do not implement Zed's optional tracing backend. Cargo will
fail at compile time if a future GPUI revision starts requiring a larger API.

`block` is the MIT-licensed `block` 0.1.6 crate with a narrowly scoped FFI
compatibility fix: `_NSConcreteStackBlock` is declared as `c_void` and its
address is cast to the opaque class pointer. This removes Rust's
`uninhabited_static` future-incompatibility warning without changing its ABI or
runtime behavior. See that directory's license and README for provenance.
