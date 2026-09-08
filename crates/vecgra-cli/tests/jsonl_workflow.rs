use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};
use vecgra::{Database, DatabaseOptions, Value};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "vecgra-cli-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let fixture = Self(directory);
        let database =
            Database::create(fixture.0.join("graph.vg"), DatabaseOptions::new(4)).unwrap();
        let mut transaction = database.transaction();
        transaction.create_node(
            "Existing",
            [("name", Value::String("original".into()))],
            &[],
        );
        transaction.commit().unwrap();
        fixture.write("nodes.jsonl", "");
        fixture.write("edges.jsonl", "");
        fixture
    }

    fn write(&self, name: &str, contents: &str) {
        fs::write(self.0.join(name), contents).unwrap();
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_vecgra"))
            .current_dir(&self.0)
            .args(args)
            .output()
            .unwrap()
    }

    fn append(&self) -> Output {
        self.run(&["append-jsonl", "graph.vg", "nodes.jsonl", "edges.jsonl"])
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn append_connects_new_nodes_to_existing_nodes_and_accepts_edge_only_batches() {
    let fixture = Fixture::new();
    // Integer 0 is a batch ID; {"node":0} is an existing database ID.
    fixture.write(
        "nodes.jsonl",
        r#"{"id":0,"label":"New","vectors":[[0,1,0,0]]}"#,
    );
    fixture.write(
        "edges.jsonl",
        r#"{"source":0,"target":{"node":0},"label":"REMEMBERS","vectors":[[1,0,0,0],[0,0,1,0]]}"#,
    );
    let output = fixture.append();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "nodes\t1\nedges\t1\nvectors\t3\n"
    );
    fixture.write("nodes.jsonl", "");
    fixture.write(
        "edges.jsonl",
        r#"{"source":{"node":0},"target":{"node":1},"label":"LINKS"}"#,
    );
    let output = fixture.append();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let database = Database::open_read_only(fixture.0.join("graph.vg")).unwrap();
    let read = database.read();
    assert_eq!(
        (
            read.stats().nodes,
            read.stats().edges,
            read.stats().indexed_vectors
        ),
        (2, 2, 3)
    );
    let edge = read.edge(0).unwrap();
    assert_eq!((edge.source, edge.target, edge.vector_count), (1, 0, 2));
    assert_eq!(
        read.node_vector_owned(1, 0).unwrap().unwrap(),
        vec![0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        read.edge_vector_owned(0, 1).unwrap().unwrap(),
        vec![0.0, 0.0, 1.0, 0.0]
    );
    let reverse = read.edge(1).unwrap();
    assert_eq!((reverse.source, reverse.target), (0, 1));
    read.verify_integrity().unwrap();
}

#[test]
fn invalid_append_never_changes_database_bytes() {
    let fixture = Fixture::new();
    let before = fs::read(fixture.0.join("graph.vg")).unwrap();
    for (nodes, edges) in [
        (
            "{\"id\":\"a\",\"label\":\"N\"}\n{\"id\":\"a\",\"label\":\"N\"}",
            "",
        ),
        (
            r#"{"id":"a","label":"N"}"#,
            r#"{"source":"a","target":"missing","label":"E"}"#,
        ),
        (
            r#"{"id":"a","label":"N"}"#,
            r#"{"source":"a","target":{"node":99},"label":"E"}"#,
        ),
        (
            r#"{"id":"a","label":"N","vectors":[[1,0,0],[0,1,0,0,0]]}"#,
            "",
        ),
        ("{\"id\":\"a\",\"label\":\"N\"}\ninvalid json", ""),
        (
            r#"{"id":"a","label":"N"}"#,
            r#"{"source":"a","target":"a","label":"E","vectors":[[0,0,0,0]]}"#,
        ),
    ] {
        fixture.write("nodes.jsonl", nodes);
        fixture.write("edges.jsonl", edges);
        assert!(
            !fixture.append().status.success(),
            "accepted invalid batch: {nodes} {edges}"
        );
        assert!(
            fs::read(fixture.0.join("graph.vg")).unwrap() == before,
            "database bytes changed"
        );
    }
}

#[test]
fn extra_arguments_are_rejected_before_import_or_append() {
    let fixture = Fixture::new();
    fixture.write("nodes.jsonl", r#"{"id":"a","label":"N"}"#);
    let before = fs::read(fixture.0.join("graph.vg")).unwrap();
    assert!(
        !fixture
            .run(&[
                "append-jsonl",
                "graph.vg",
                "nodes.jsonl",
                "edges.jsonl",
                "typo"
            ])
            .status
            .success()
    );
    assert!(
        fs::read(fixture.0.join("graph.vg")).unwrap() == before,
        "database bytes changed"
    );
    assert!(
        !fixture
            .run(&[
                "import-jsonl",
                "nodes.jsonl",
                "edges.jsonl",
                "new.vg",
                "4",
                "f16",
                "typo"
            ])
            .status
            .success()
    );
    assert!(!fixture.0.join("new.vg").exists());
}

#[test]
fn cli_reads_do_not_repair_a_torn_tail() {
    use std::io::Write;
    let fixture = Fixture::new();
    let path = fixture.0.join("graph.vg");
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"torn-tail")
        .unwrap();
    let before = fs::read(&path).unwrap();
    for command in ["stats", "check"] {
        let output = fixture.run(&[command, "graph.vg"]);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            fs::read(&path).unwrap() == before,
            "read command changed database bytes"
        );
    }
}

#[test]
fn append_in_another_process_cannot_open_an_active_writer() {
    let fixture = Fixture::new();
    let database = Database::open(fixture.0.join("graph.vg")).unwrap();
    fixture.write("nodes.jsonl", r#"{"id":"a","label":"N"}"#);
    let before = fs::read(database.path()).unwrap();
    assert!(!fixture.append().status.success());
    assert_eq!(fs::read(database.path()).unwrap(), before);
    #[cfg(unix)]
    assert!(fixture.run(&["stats", "graph.vg"]).status.success());
    drop(database);
    assert!(fixture.append().status.success());
}
