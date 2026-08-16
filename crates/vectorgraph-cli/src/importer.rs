use crate::embedder::{Embedder, EmbeddingCache};
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tree_sitter::{Node as SyntaxNode, Parser};
use vectorgraph::{BulkLoader, DatabaseOptions, NodeId, Similarity, Value, VectorEncoding};

#[derive(Debug)]
struct AstRecord {
    parent: Option<usize>,
    child_index: u32,
    field: Option<Arc<str>>,
    kind: Arc<str>,
    name: Option<Arc<str>>,
    text: Option<Arc<str>>,
    context: Option<Arc<str>>,
    start_byte: u32,
    end_byte: u32,
    start_line: u32,
    end_line: u32,
    has_error: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct ImportCounts {
    files: usize,
    syntax_nodes: usize,
    edges: usize,
    bytes: usize,
}

#[derive(Debug)]
struct ParsedFile {
    relative_path: Arc<str>,
    bytes: usize,
    ast: Vec<AstRecord>,
}

pub(crate) fn import_rust_repository(
    repository: &Path,
    database_path: &Path,
    embedder: Box<dyn Embedder>,
) -> Result<(), Box<dyn Error>> {
    if !repository.is_dir() {
        return Err(format!("{} is not a directory", repository.display()).into());
    }
    if database_path.try_exists()? {
        return Err(format!(
            "bulk-load destination already exists: {}",
            database_path.display()
        )
        .into());
    }
    let repository = repository.canonicalize()?;
    let mut embeddings = EmbeddingCache::new(embedder);
    let started = Instant::now();
    let repository_name = repository
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repository");
    let repository_text = format!("Code repository named {repository_name}");

    let mut files = Vec::new();
    collect_rust_files(&repository, &mut files)?;
    files.sort_unstable();
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_rust::LANGUAGE.into())?;
    let mut counts = ImportCounts::default();
    let mut parsed_files = Vec::with_capacity(files.len());
    let mut unique_embedding_texts = HashSet::new();
    unique_embedding_texts.insert(repository_text.clone());
    unique_embedding_texts.insert("Code repository contains source file".to_owned());
    unique_embedding_texts.insert("File has a parsed Rust syntax tree".to_owned());

    for path in files {
        let source = fs::read(&path)?;
        if source.len() > 16 * 1024 * 1024 {
            eprintln!("skipping {}: larger than 16 MiB", path.display());
            continue;
        }
        let Some(tree) = parser.parse(&source, None) else {
            eprintln!("skipping {}: parser returned no tree", path.display());
            continue;
        };
        let relative = path.strip_prefix(&repository).unwrap_or(&path);
        let relative_text: Arc<str> = Arc::from(relative.to_string_lossy().as_ref());
        let ast = collect_ast(tree.root_node(), &source)?;
        unique_embedding_texts.insert(format!("Rust source file at {relative_text}"));
        for record in &ast {
            unique_embedding_texts.insert(syntax_embedding_text(record));
            unique_embedding_texts.insert(context_embedding_text(record, &relative_text));
            if let Some(parent) = record.parent {
                unique_embedding_texts.insert(ast_edge_embedding_text(&ast[parent], record));
            }
        }
        counts.files += 1;
        counts.syntax_nodes += ast.len();
        counts.edges += ast.len() + 1;
        counts.bytes += source.len();
        parsed_files.push(ParsedFile {
            relative_path: relative_text,
            bytes: source.len(),
            ast,
        });
        if counts.files % 50 == 0 {
            eprintln!(
                "parsed {} files, {} syntax nodes, {} unique embedding texts",
                counts.files,
                counts.syntax_nodes,
                unique_embedding_texts.len()
            );
        }
    }

    eprintln!(
        "embedding {} unique payloads for {} graph vectors with {}",
        unique_embedding_texts.len(),
        1 + 2 * counts.files + 4 * counts.syntax_nodes,
        embeddings.name()
    );
    embeddings.ensure(unique_embedding_texts.iter().map(String::as_str))?;

    let mut database = BulkLoader::new(
        database_path,
        DatabaseOptions {
            vector_dimension: embeddings.dimension(),
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F16,
            sync_on_commit: true,
        },
    )?;
    let repository_vector = embeddings.vector(&repository_text)?;
    let repository_id = database.create_node(
        "Repository",
        [
            ("name", Value::String(Arc::from(repository_name))),
            (
                "path",
                Value::String(Arc::from(repository.to_string_lossy().as_ref())),
            ),
        ],
        &[repository_vector],
    )?;

    let mut stored_vectors = 1usize;
    for (file_index, parsed) in parsed_files.iter().enumerate() {
        let file_embedding_text = format!("Rust source file at {}", parsed.relative_path);
        let root_edge_embedding_text = "File has a parsed Rust syntax tree";

        let file_vector = embeddings.vector(&file_embedding_text)?;
        let file_id = database.create_node(
            "File",
            [
                ("path", Value::String(parsed.relative_path.clone())),
                ("bytes", Value::Int(parsed.bytes as i64)),
                ("language", Value::String(Arc::from("rust"))),
            ],
            &[file_vector],
        )?;
        let contains_vector = embeddings.vector("Code repository contains source file")?;
        database.create_edge(
            repository_id,
            file_id,
            "CONTAINS",
            std::iter::empty::<(&str, Value)>(),
            &[contains_vector],
        )?;

        let mut syntax_ids: Vec<NodeId> = Vec::with_capacity(parsed.ast.len());
        for record in &parsed.ast {
            let mut properties = vec![
                ("kind", Value::String(record.kind.clone())),
                ("start_byte", Value::Int(record.start_byte as i64)),
                ("end_byte", Value::Int(record.end_byte as i64)),
                ("start_line", Value::Int(record.start_line as i64)),
                ("end_line", Value::Int(record.end_line as i64)),
                ("has_error", Value::Bool(record.has_error)),
            ];
            if let Some(name) = &record.name {
                properties.push(("name", Value::String(name.clone())));
            }
            if let Some(text) = &record.text {
                properties.push(("text", Value::String(text.clone())));
            }
            if let Some(context) = &record.context {
                properties.push(("context", Value::String(context.clone())));
            }
            let structural_vector = embeddings.vector(&syntax_embedding_text(record))?;
            let context_vector =
                embeddings.vector(&context_embedding_text(record, &parsed.relative_path))?;
            let node_id =
                database.create_node("Syntax", properties, &[structural_vector, context_vector])?;
            syntax_ids.push(node_id);
        }

        if let Some(root_id) = syntax_ids.first().copied() {
            let edge_vector = embeddings.vector(root_edge_embedding_text)?;
            let context_vector = embeddings.vector(&context_embedding_text(
                &parsed.ast[0],
                &parsed.relative_path,
            ))?;
            database.create_edge(
                file_id,
                root_id,
                "HAS_SYNTAX",
                std::iter::empty::<(&str, Value)>(),
                &[edge_vector, context_vector],
            )?;
        }
        for (index, record) in parsed.ast.iter().enumerate() {
            let Some(parent) = record.parent else {
                continue;
            };
            let mut properties = vec![("child_index", Value::Int(record.child_index as i64))];
            if let Some(field) = &record.field {
                properties.push(("field", Value::String(field.clone())));
            }
            let edge_text = ast_edge_embedding_text(&parsed.ast[parent], record);
            let structural_vector = embeddings.vector(&edge_text)?;
            let context_vector =
                embeddings.vector(&context_embedding_text(record, &parsed.relative_path))?;
            database.create_edge(
                syntax_ids[parent],
                syntax_ids[index],
                "AST_CHILD",
                properties,
                &[structural_vector, context_vector],
            )?;
        }
        stored_vectors += 2 + 4 * parsed.ast.len();
        let imported = file_index + 1;
        if imported % 50 == 0 {
            eprintln!(
                "stored {imported}/{} files and {} vectors",
                counts.files, stored_vectors
            );
        }
    }

    let stats = database.finish()?;
    println!("database\t{}", database_path.display());
    println!("files\t{}", counts.files);
    println!("source_bytes\t{}", counts.bytes);
    println!("syntax_nodes\t{}", counts.syntax_nodes);
    println!("nodes\t{}", stats.nodes);
    println!("edges\t{}", stats.edges);
    println!("vectors\t{}", stats.indexed_vectors);
    println!("embedder\t{}", embeddings.name());
    println!("unique_embedding_texts\t{}", embeddings.embedded_texts());
    println!("elapsed_ms\t{}", started.elapsed().as_millis());
    Ok(())
}

fn syntax_embedding_text(record: &AstRecord) -> String {
    let detail = record
        .name
        .as_deref()
        .or(record.text.as_deref())
        .unwrap_or("");
    if detail.is_empty() {
        format!("Rust {} syntax", record.kind)
    } else {
        format!("Rust {} syntax for {detail}", record.kind)
    }
}

fn context_embedding_text(record: &AstRecord, relative_path: &str) -> String {
    match record.context.as_deref() {
        Some(context) => format!("Rust source context {context} in {relative_path}"),
        None => format!("Rust file scope in {relative_path}"),
    }
}

fn ast_edge_embedding_text(parent: &AstRecord, child: &AstRecord) -> String {
    match child.field.as_deref() {
        Some(field) => format!(
            "Rust AST relationship: {} has {field} child {}",
            parent.kind, child.kind,
        ),
        None => format!(
            "Rust AST relationship: {} contains child {}",
            parent.kind, child.kind,
        ),
    }
}

fn collect_rust_files(directory: &Path, result: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_dir() {
            let name = entry.file_name();
            if matches!(
                name.to_str(),
                Some(".git" | "target" | "vendor" | "node_modules")
            ) {
                continue;
            }
            collect_rust_files(&path, result)?;
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            result.push(path);
        }
    }
    Ok(())
}

fn collect_ast(root: SyntaxNode<'_>, source: &[u8]) -> Result<Vec<AstRecord>, Box<dyn Error>> {
    let mut result = Vec::new();
    collect_node(root, None, 0, None, None, source, &mut result)?;
    Ok(result)
}

fn collect_node(
    node: SyntaxNode<'_>,
    parent: Option<usize>,
    child_index: u32,
    field: Option<&str>,
    inherited_context: Option<Arc<str>>,
    source: &[u8],
    result: &mut Vec<AstRecord>,
) -> Result<(), Box<dyn Error>> {
    let index = result.len();
    let kind: Arc<str> = Arc::from(node.kind());
    let name = node
        .child_by_field_name("name")
        .and_then(|name| short_text(name, source))
        .or_else(|| {
            (node.kind() == "impl_item")
                .then(|| node.child_by_field_name("type"))
                .flatten()
                .and_then(|name| short_text(name, source))
        });
    let text = if name.is_none() && node.named_child_count() == 0 {
        short_text(node, source)
    } else {
        None
    };
    let start = node.start_position();
    let end = node.end_position();
    let context = scope_context(node.kind(), name.as_deref(), inherited_context.as_ref());
    result.push(AstRecord {
        parent,
        child_index,
        field: field.map(Arc::from),
        kind,
        name,
        text,
        context: context.clone(),
        start_byte: node.start_byte().try_into()?,
        end_byte: node.end_byte().try_into()?,
        start_line: start.row.try_into()?,
        end_line: end.row.try_into()?,
        has_error: node.has_error(),
    });

    let mut named_index = 0;
    for raw_index in 0..node.child_count() {
        let raw_index: u32 = raw_index.try_into()?;
        let Some(child) = node.child(raw_index) else {
            continue;
        };
        if !child.is_named() {
            continue;
        }
        let field = node.field_name_for_child(raw_index);
        collect_node(
            child,
            Some(index),
            named_index,
            field,
            context.clone(),
            source,
            result,
        )?;
        named_index += 1;
    }
    Ok(())
}

fn scope_context(kind: &str, name: Option<&str>, inherited: Option<&Arc<str>>) -> Option<Arc<str>> {
    let is_scope = matches!(
        kind,
        "function_item"
            | "impl_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "mod_item"
            | "macro_definition"
            | "const_item"
            | "static_item"
            | "type_item"
    );
    if !is_scope {
        return inherited.cloned();
    }
    let component = name.map_or_else(|| kind.to_owned(), |name| format!("{kind} {name}"));
    Some(Arc::from(match inherited {
        Some(parent) => format!("{parent} / {component}"),
        None => component,
    }))
}

fn short_text(node: SyntaxNode<'_>, source: &[u8]) -> Option<Arc<str>> {
    let bytes = source.get(node.byte_range())?;
    if bytes.len() > 160 || bytes.iter().any(|byte| *byte == b'\n' || *byte == b'\r') {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?.trim();
    (!text.is_empty()).then(|| Arc::from(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_ast_contains_named_hierarchy() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        let source = b"fn answer() -> u64 { 42 }";
        let tree = parser.parse(source, None).unwrap();
        let records = collect_ast(tree.root_node(), source).unwrap();
        assert!(
            records
                .iter()
                .any(|record| record.kind.as_ref() == "function_item")
        );
        assert!(
            records
                .iter()
                .any(|record| record.name.as_deref() == Some("answer"))
        );
        assert!(records.iter().skip(1).all(|record| record.parent.is_some()));
    }

    #[test]
    fn structural_edge_text_is_stable_and_semantic() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        let source = b"fn answer() -> u64 { 42 }";
        let tree = parser.parse(source, None).unwrap();
        let records = collect_ast(tree.root_node(), source).unwrap();
        let body = records
            .iter()
            .find(|record| record.field.as_deref() == Some("body"))
            .unwrap();
        let parent = &records[body.parent.unwrap()];
        assert!(ast_edge_embedding_text(parent, body).contains("has body child"));
        assert!(body.context.as_deref().unwrap().contains("answer"));
        assert!(context_embedding_text(body, "src/lib.rs").contains("answer"));
    }

    #[test]
    fn structural_and_context_payloads_have_different_granularity() {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .unwrap();
        let source = b"fn answer() -> u64 { 42 }";
        let tree = parser.parse(source, None).unwrap();
        let records = collect_ast(tree.root_node(), source).unwrap();
        let literal = records
            .iter()
            .find(|record| record.kind.as_ref() == "integer_literal")
            .unwrap();
        assert!(!syntax_embedding_text(literal).contains("src/lib.rs"));
        assert!(context_embedding_text(literal, "src/lib.rs").contains("src/lib.rs"));
    }
}
