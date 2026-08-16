use regex::Regex;
use std::error::Error;
use std::sync::OnceLock;
use vectorgraph::{
    Direction, NodeFilter, OneHopQuery, PatternMatch, ReadGuard, SemanticOneHopQuery,
    SemanticPatternMatch,
};

#[derive(Debug, PartialEq, Eq)]
struct PatternSpec {
    start_label: Option<String>,
    edge_label: Option<String>,
    end_label: Option<String>,
    direction: Direction,
    limit: usize,
}

pub(crate) fn execute(
    read: &ReadGuard<'_>,
    statement: &str,
) -> Result<Vec<PatternMatch>, Box<dyn Error>> {
    let spec = parse(statement)?;
    let Some(query) = resolve_pattern(read, &spec) else {
        return Ok(Vec::new());
    };
    Ok(read.match_one_hop(&query))
}

pub(crate) fn execute_semantic(
    read: &ReadGuard<'_>,
    statement: &str,
    vector: &[f32],
) -> Result<Vec<SemanticPatternMatch>, Box<dyn Error>> {
    let spec = parse(statement)?;
    let Some(pattern) = resolve_pattern(read, &spec) else {
        return Ok(Vec::new());
    };
    Ok(read.match_semantic_one_hop(
        vector,
        &SemanticOneHopQuery {
            seed_count: pattern.limit.saturating_mul(8).clamp(64, 4_096),
            pattern,
            ..SemanticOneHopQuery::default()
        },
    )?)
}

fn resolve_pattern(read: &ReadGuard<'_>, spec: &PatternSpec) -> Option<OneHopQuery> {
    let start_label = resolve_label(read, spec.start_label.as_deref())?;
    let edge_label = resolve_label(read, spec.edge_label.as_deref())?;
    let end_label = resolve_label(read, spec.end_label.as_deref())?;
    Some(OneHopQuery {
        start: NodeFilter {
            label: start_label,
            properties: Vec::new(),
        },
        edge_label,
        end: NodeFilter {
            label: end_label,
            properties: Vec::new(),
        },
        direction: spec.direction,
        limit: spec.limit,
    })
}

fn resolve_label(read: &ReadGuard<'_>, label: Option<&str>) -> Option<Option<u32>> {
    match label {
        Some(label) => read.label_id(label).map(Some),
        None => Some(None),
    }
}

fn parse(statement: &str) -> Result<PatternSpec, Box<dyn Error>> {
    static STATEMENT: OnceLock<Regex> = OnceLock::new();
    let regex = STATEMENT.get_or_init(|| {
        Regex::new(
            r"(?ix)^\s*MATCH\s*
              \(\s*[A-Z_][A-Z0-9_]*\s*(?::\s*(?<START>[A-Z_][A-Z0-9_]*))?\s*\)\s*
              (?<LEFT><-|- )?\s*\[\s*[A-Z_][A-Z0-9_]*\s*(?::\s*(?<EDGE>[A-Z_][A-Z0-9_]*))?\s*\]\s*
              (?<RIGHT>->|-)\s*
              \(\s*[A-Z_][A-Z0-9_]*\s*(?::\s*(?<END>[A-Z_][A-Z0-9_]*))?\s*\)\s*
              RETURN\s+.+?
              (?:\s+LIMIT\s+(?<LIMIT>[0-9]+))?\s*;?\s*$",
        )
        .expect("valid Cypher subset regex")
    });
    let captures = regex.captures(statement).ok_or(
        "unsupported query; expected MATCH (a:Label)-[e:TYPE]->(b:Label) RETURN ... [LIMIT n]",
    )?;
    let left = captures.name("LEFT").map_or("-", |value| value.as_str());
    let right = captures.name("RIGHT").map_or("-", |value| value.as_str());
    let direction = match (left.trim(), right.trim()) {
        ("-", "->") => Direction::Outgoing,
        ("<-", "-") => Direction::Incoming,
        ("-", "-") => Direction::Both,
        _ => return Err("invalid relationship direction".into()),
    };
    let limit = captures
        .name("LIMIT")
        .map(|value| value.as_str().parse())
        .transpose()?
        .unwrap_or(100);
    Ok(PatternSpec {
        start_label: capture(&captures, "START"),
        edge_label: capture(&captures, "EDGE"),
        end_label: capture(&captures, "END"),
        direction,
        limit,
    })
}

fn capture(captures: &regex::Captures<'_>, name: &str) -> Option<String> {
    captures.name(name).map(|value| value.as_str().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_one_hop_cypher_surface() {
        let query =
            parse("MATCH (file:File)-[syntax:HAS_SYNTAX]->(root:Syntax) RETURN file, root LIMIT 7")
                .unwrap();
        assert_eq!(query.start_label.as_deref(), Some("File"));
        assert_eq!(query.edge_label.as_deref(), Some("HAS_SYNTAX"));
        assert_eq!(query.end_label.as_deref(), Some("Syntax"));
        assert_eq!(query.direction, Direction::Outgoing);
        assert_eq!(query.limit, 7);
    }

    #[test]
    fn parses_incoming_and_undirected_relationships() {
        assert_eq!(
            parse("MATCH (a)<-[e:REL]-(b) RETURN a").unwrap().direction,
            Direction::Incoming
        );
        assert_eq!(
            parse("MATCH (a)-[e:REL]-(b) RETURN a").unwrap().direction,
            Direction::Both
        );
    }
}
