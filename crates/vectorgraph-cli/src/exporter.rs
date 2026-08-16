use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use vectorgraph::{Database, ReadGuard, Value};

pub(crate) fn export_ladybug_csv(
    database_path: &Path,
    directory: &Path,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(directory)?;
    let database = Database::open(database_path)?;
    let read = database.read();
    let mut files = csv(directory.join("files.csv"), "id,path\n")?;
    let mut syntax = csv(directory.join("syntax.csv"), "id,kind,detail\n")?;
    let mut ast_edges = csv(directory.join("ast_child.csv"), "from,to,edge_id\n")?;
    let mut syntax_edges = csv(directory.join("has_syntax.csv"), "from,to,edge_id\n")?;
    let mut file_count = 0;
    let mut syntax_count = 0;
    for id in read.node_ids() {
        let node = read.node(id).ok_or("node disappeared during export")?;
        match read.symbol(node.label) {
            Some("File") => {
                let path = string_property(&read, &node.properties, "path").unwrap_or("");
                writeln!(files, "{},{}", node.id, csv_field(path))?;
                file_count += 1;
            }
            Some("Syntax") => {
                let kind = string_property(&read, &node.properties, "kind").unwrap_or("");
                let detail = string_property(&read, &node.properties, "name")
                    .or_else(|| string_property(&read, &node.properties, "text"))
                    .unwrap_or("");
                writeln!(
                    syntax,
                    "{},{},{}",
                    node.id,
                    csv_field(kind),
                    csv_field(detail)
                )?;
                syntax_count += 1;
            }
            _ => {}
        }
    }
    let mut ast_edge_count = 0;
    let mut syntax_edge_count = 0;
    for id in read.edge_ids() {
        let edge = read.edge(id).ok_or("edge disappeared during export")?;
        match read.symbol(edge.label) {
            Some("AST_CHILD") => {
                writeln!(ast_edges, "{},{},{}", edge.source, edge.target, edge.id)?;
                ast_edge_count += 1;
            }
            Some("HAS_SYNTAX") => {
                writeln!(syntax_edges, "{},{},{}", edge.source, edge.target, edge.id)?;
                syntax_edge_count += 1;
            }
            _ => {}
        }
    }
    files.flush()?;
    syntax.flush()?;
    ast_edges.flush()?;
    syntax_edges.flush()?;
    println!("files\t{file_count}");
    println!("syntax_nodes\t{syntax_count}");
    println!("ast_child_edges\t{ast_edge_count}");
    println!("has_syntax_edges\t{syntax_edge_count}");
    Ok(())
}

fn csv(path: impl AsRef<Path>, header: &str) -> Result<BufWriter<File>, Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(header.as_bytes())?;
    Ok(writer)
}

fn string_property<'a>(
    read: &ReadGuard<'_>,
    properties: &'a [vectorgraph::Property],
    key: &str,
) -> Option<&'a str> {
    match read.property(properties, key)? {
        Value::String(value) => Some(value),
        _ => None,
    }
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_quoting_is_rfc4180_compatible() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("a\"b"), "\"a\"\"b\"");
    }
}
