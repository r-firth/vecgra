# Contributing to Vecgra

Vecgra is an experimental vector-native graph database. Contributions are
welcome, especially when they improve correctness, explainability, portable
performance, or the quality of hybrid graph/vector execution.

## Development setup

The repository pins its Rust toolchain in `rust-toolchain.toml`. Build and test
the headless engine and CLI with:

```sh
cargo build --locked
cargo test --workspace --all-targets --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo deny check advisories licenses sources
```

Vecgra Studio has only been verified on Apple Silicon macOS. The database and
CLI do not depend on the desktop stack.

## Change expectations

- Add a regression test for bug fixes and observable behavior changes.
- Keep file-format reads fallible. Never trust offsets, counts, alignment, or
  checksums merely because Vecgra wrote the file.
- Document the invariant immediately above every `unsafe` block.
- Preserve portable scalar implementations for SIMD paths and test them
  against one another.
- Measure performance changes in release mode and report recall beside ANN
  latency. A faster approximate result with unreported quality is a regression.
- Keep refactors behavior-preserving and separate from feature changes where
  practical.
- Do not commit API keys, benchmark corpora, generated databases, or local
  provider caches.

Database parsing also has a libFuzzer target. With nightly Rust and
`cargo-fuzz` installed, run `cargo +nightly fuzz run database_file`; turn every
fixed crash into a stable regression test. See [`fuzz/README.md`](fuzz/README.md).

When a change affects the on-disk container, update `docs/architecture.md` and
add reopen, corruption, or torn-write coverage as appropriate. When it affects
a published benchmark, include the exact corpus, command, hardware, build
profile, latency distribution, and quality measurement.

## Pull requests

Keep pull requests focused and explain:

1. the behavior or invariant being changed;
2. how it was verified;
3. any performance, memory, compatibility, or file-format effect.

By submitting a contribution, you agree that it is licensed under the
repository's Apache License 2.0.
