use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use vecgra::{Database, DatabaseOptions, ElementRef, Result, Value, VectorEncoding, VectorTarget};

fn main() -> Result<()> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("vecgra-quickstart-{nonce}.vg"));
    let database = Database::create(
        &path,
        DatabaseOptions {
            vector_dimension: 3,
            vector_encoding: VectorEncoding::F16,
            ..DatabaseOptions::new(3)
        },
    )?;

    let mut transaction = database.transaction();
    let rust = transaction.create_node(
        "Language",
        [("name", Value::String(Arc::from("Rust")))],
        &[vec![1.0, 0.0, 0.0]],
    );
    let vecgra = transaction.create_node(
        "Project",
        [("name", Value::String(Arc::from("Vecgra")))],
        &[vec![0.9, 0.1, 0.0]],
    );
    transaction.create_edge(
        vecgra,
        rust,
        "WRITTEN_IN",
        std::iter::empty::<(&str, Value)>(),
        &[vec![0.8, 0.2, 0.0]],
    );
    transaction.commit()?;

    {
        let graph = database.read();
        let hits = graph.vector_search(&[1.0, 0.0, 0.0], VectorTarget::Both, 3, None)?;
        assert_eq!(
            hits.first().map(|hit| hit.element),
            Some(ElementRef::Node(rust))
        );

        let written_in = graph
            .label_id("WRITTEN_IN")
            .expect("interned relationship label");
        let relationships = graph.neighbors(
            vecgra,
            vecgra::Direction::Outgoing,
            vecgra::EdgeFilter {
                label: Some(written_in),
            },
        )?;
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].target, rust);
    }

    drop(database);
    std::fs::remove_file(path)?;
    println!("created, searched, traversed, and removed a Vecgra database");
    Ok(())
}
