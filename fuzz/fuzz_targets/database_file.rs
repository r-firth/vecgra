#![no_main]

use libfuzzer_sys::fuzz_target;
use std::fs::OpenOptions;
use std::io::Write;
use vecgra::{Database, DatabaseOptions, Similarity, VectorEncoding};

const MAX_INPUT_BYTES: usize = 16 * 1024 * 1024;

fuzz_target!(|input: &[u8]| {
    if input.len() > MAX_INPUT_BYTES {
        return;
    }

    let directory = tempfile::tempdir().expect("create fuzz directory");
    let path = directory.path().join("input.vg");
    if input.first().is_some_and(|selector| selector & 1 == 1) {
        let database = Database::create(
            &path,
            DatabaseOptions {
                vector_dimension: 4,
                similarity: Similarity::Cosine,
                vector_encoding: VectorEncoding::F16,
                sync_on_commit: false,
            },
        )
        .expect("create valid fuzz header");
        drop(database);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("reopen fuzz database")
            .write_all(&input[1..])
            .expect("append fuzz log tail");
    } else {
        std::fs::write(&path, input.get(1..).unwrap_or_default()).expect("write fuzz database");
    }

    if let Ok(database) = Database::open_read_only(&path) {
        let _ = database.read().verify_integrity();
    }
});
