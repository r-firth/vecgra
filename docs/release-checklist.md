# Public release checklist

This is the publication gate for the pre-1.0 source release. It does not imply
a stable API, storage format, binary package, or production support promise.

## Source tree

- Confirm package paths, benchmark namespaces, assets, crate directories, and
  documentation use the Vecgra name consistently.
- Confirm no generated `.vg` files, provider caches, fuzz corpora, build
  output, credentials, or local brand notes are tracked.
- Review `git diff --check`, the final staged diff, and the commit contents
  before publishing. The rename should land as one coherent commit.
- Keep `Cargo.lock`, `fuzz/Cargo.lock`, all licences, and third-party provenance
  files in the source release.

## Verification

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --all-targets --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
cargo deny check advisories licenses sources
cargo build --release -p vecgra-cli
cargo build --release -p vecgra-studio
scripts/ci-smoke.sh target/release/vecgra
```

Also run `cargo machete --with-metadata`, validate the workflow with
`actionlint`, and give `database_file` a sanitizer-backed fuzz run. Reproduce
headline benchmarks only when their corpus and hardware are available; do not
replace measured values with estimates.

## Repository settings

- Enable private vulnerability reporting, secret scanning, Dependabot alerts,
  and dependency updates.
- Protect the default branch with the three CI jobs required.
- Check that issue security links, contribution policy, code of conduct, and
  Apache-2.0 licence render correctly.
- Revoke any temporary provider credentials used during development before the
  repository becomes public.

## Presentation

- Add the demo video at the marked position near the top of `README.md`.
- Demonstrate one semantic relationship result, its animated connected focus,
  an exact evidence path, relationship inspection, and a deep zoom from the
  complete overview.
- State that hash embeddings exercise mechanics only; use one embedding model
  consistently for a semantic-quality demo.
- Keep benchmark caveats visible and describe Studio as Apple Silicon macOS
  verified until other platforms have native launch evidence.
