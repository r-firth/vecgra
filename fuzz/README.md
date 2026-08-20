# Database file fuzzing

The `database_file` target exercises both arbitrary whole files and arbitrary
transaction tails appended to a valid Vecgra header. Successful opens also run
the full integrity verifier.

Install nightly Rust and cargo-fuzz, then run:

```sh
cargo install cargo-fuzz --locked
cargo +nightly fuzz run database_file
```

Keep generated fuzz inputs and crash artifacts local. A minimized crashing input
should become a deterministic regression test in `crates/vecgra/tests` before
the fix is merged.
