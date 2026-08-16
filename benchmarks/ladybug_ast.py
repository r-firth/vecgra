#!/usr/bin/env python3
"""Load and benchmark the ripgrep AST export in LadybugDB.

This is intentionally outside the product crates. It compares ordinary graph
pattern execution on exactly the topology exported by `vg export-ladybug`.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import ladybug as lb


QUERIES = {
    "file_to_root_100": """
        MATCH (f:File)-[r:HAS_SYNTAX]->(s:Syntax)
        RETURN f.id, r.edge_id, s.id LIMIT 100
    """,
    "ast_child_100": """
        MATCH (a:Syntax)-[r:AST_CHILD]->(b:Syntax)
        RETURN a.id, r.edge_id, b.id LIMIT 100
    """,
    "node_degree": """
        MATCH (a:Syntax {id: 199639})-[r:AST_CHILD]-(b:Syntax)
        RETURN a.id, r.edge_id, b.id
    """,
}


def load(database_path: Path, csv_directory: Path) -> None:
    if database_path.exists():
        raise SystemExit(f"refusing to overwrite existing database: {database_path}")
    started = time.perf_counter()
    db = lb.Database(str(database_path))
    connection = lb.Connection(db)
    connection.execute("CREATE NODE TABLE File(id INT64 PRIMARY KEY, path STRING)")
    connection.execute(
        "CREATE NODE TABLE Syntax(id INT64 PRIMARY KEY, kind STRING, detail STRING)"
    )
    connection.execute(
        "CREATE REL TABLE HAS_SYNTAX(FROM File TO Syntax, edge_id INT64)"
    )
    connection.execute(
        "CREATE REL TABLE AST_CHILD(FROM Syntax TO Syntax, edge_id INT64)"
    )
    for table, filename in [
        ("File", "files.csv"),
        ("Syntax", "syntax.csv"),
        ("HAS_SYNTAX", "has_syntax.csv"),
        ("AST_CHILD", "ast_child.csv"),
    ]:
        path = (csv_directory / filename).resolve()
        connection.execute(f'COPY {table} FROM "{path}" (HEADER=true)')
    connection.execute("CHECKPOINT")
    elapsed = time.perf_counter() - started
    print(f"load_ms\t{elapsed * 1000:.3f}")
    print(f"syntax_nodes\t{scalar(connection, 'MATCH (n:Syntax) RETURN count(*)')}")
    print(
        f"ast_child_edges\t{scalar(connection, 'MATCH ()-[r:AST_CHILD]->() RETURN count(*)')}"
    )


def benchmark(database_path: Path, iterations: int) -> None:
    opened = time.perf_counter()
    db = lb.Database(str(database_path))
    connection = lb.Connection(db)
    print(f"open_ms\t{(time.perf_counter() - opened) * 1000:.3f}")
    for name, query in QUERIES.items():
        for _ in range(5):
            connection.execute(query).get_all()
        samples = []
        row_count = 0
        for _ in range(iterations):
            started = time.perf_counter_ns()
            rows = connection.execute(query).get_all()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            row_count = len(rows)
        samples.sort()
        print(f"query\t{name}")
        print(f"rows\t{row_count}")
        print(f"min_ms\t{samples[0]:.3f}")
        print(f"p50_ms\t{statistics.median(samples):.3f}")
        print(f"p95_ms\t{samples[int((len(samples) - 1) * 0.95)]:.3f}")
        print(f"max_ms\t{samples[-1]:.3f}")


def scalar(connection: lb.Connection, query: str) -> int:
    return int(connection.execute(query).get_next()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    load_parser = subparsers.add_parser("load")
    load_parser.add_argument("database", type=Path)
    load_parser.add_argument("csv_directory", type=Path)
    bench_parser = subparsers.add_parser("bench")
    bench_parser.add_argument("database", type=Path)
    bench_parser.add_argument("--iterations", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.command == "load":
        load(arguments.database, arguments.csv_directory)
    else:
        benchmark(arguments.database, arguments.iterations)


if __name__ == "__main__":
    main()
