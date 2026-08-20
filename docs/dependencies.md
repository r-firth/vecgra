# Dependency and licence policy

Vecgra locks every dependency in `Cargo.lock` and checks advisories, licences,
and source origins with `cargo-deny`. Git dependencies are limited to the
declared GPUI, Bezel, gpui-component, egraph-rs, proptest, font-kit, and xim-rs
repositories in `deny.toml`.

## Deliberate compatibility patches

The desktop graph contains two narrow local patches under `third-party/`:

- `noop-ztracing` and `noop-ztracing-macro` implement the no-op API that GPUI
  uses when Zed tracing is disabled. This keeps unused GPL tracing code out of
  the Apache-2.0 Studio binary. A larger future API requirement fails at build
  time instead of silently becoming a partial implementation.
- `block` contains the MIT-licensed `block` 0.1.6 source with explicit C ABIs
  and a corrected opaque extern-static declaration. It removes a Rust
  future-incompatibility warning without changing the Objective-C Blocks ABI.

Each patch has a local README, licence, provenance, and removal condition. They
are source patches, not forks of product behavior.

## Advisory posture

`deny.toml` records the remaining unmaintained transitive advisories and why
they are present. They currently enter through the pinned desktop stack (or a
development-only benchmark), not the storage engine's default dependency
graph. New vulnerabilities, unknown registries, copyleft dependencies, and
unexplained advisories fail CI.

`cargo audit` also reports `rustls-pemfile` from Zed's locked reqwest
source. Cargo does not resolve that package into any of Vecgra's declared
target graphs, so `cargo-deny` correctly treats it as outside the build rather
than adding a non-matching exception.

Run the policy locally with:

```sh
cargo deny check advisories licenses sources
```

The headless `vecgra`, `vecgra-cli`, and `vecgra-embedding` crates remain
independent of GPUI and its native graphics dependency graph.
