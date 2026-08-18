//! A vector-native embedded labelled property graph database.
//!
//! The storage kernel keeps its dependency surface deliberately small.
//! Importers and language-specific integrations belong in adjacent crates.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use vecgra::{Database, DatabaseOptions, ElementRef, Value, VectorTarget};
//!
//! # fn main() -> vecgra::Result<()> {
//! let database = Database::create("knowledge.vg", DatabaseOptions::new(3))?;
//! let mut transaction = database.transaction();
//! let rust = transaction.create_node(
//!     "Language",
//!     [("name", Value::String(Arc::from("Rust")))],
//!     &[vec![1.0, 0.0, 0.0]],
//! );
//! let vecgra = transaction.create_node(
//!     "Project",
//!     [("name", Value::String(Arc::from("Vecgra")))],
//!     &[vec![0.9, 0.1, 0.0]],
//! );
//! transaction.create_edge(
//!     vecgra,
//!     rust,
//!     "WRITTEN_IN",
//!     std::iter::empty::<(&str, Value)>(),
//!     &[vec![0.8, 0.2, 0.0]],
//! );
//! transaction.commit()?;
//!
//! let graph = database.read();
//! let hits = graph.vector_search(&[1.0, 0.0, 0.0], VectorTarget::Both, 5, None)?;
//! assert_eq!(hits.first().map(|hit| hit.element), Some(ElementRef::Node(rust)));
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod ann;
mod bulk;
mod codec;
mod database;
mod error;
mod graph;
mod model;
mod simd;
mod vector;

pub use bulk::BulkLoader;
pub use database::{Database, DatabaseOptions, IntegrityReport, ReadGuard, Transaction};
pub use error::{Error, Result};
pub use graph::{
    Direction, EdgeFilter, ElementFilter, ElementFilterPlan, ElementFilterStrategy,
    FilteredVectorSearchResult, GraphRangeSearchOptions, GraphRangeSearchResult, GraphStats,
    NodeFilter, NumericRangeFilter, NumericRangePlan, NumericRangeStrategy, NumericValue,
    OneHopPlan, OneHopQuery, OneHopStrategy, PatternMatch, SemanticOneHopQuery, SemanticPathHit,
    SemanticPathOptions, SemanticPatternMatch, ShortestPath, ShortestPathOptions,
    ShortestPathResult, ShortestPathStrategy, ShortestPathTermination,
};
pub use model::{
    Edge, EdgeId, ElementRef, ElementSet, LabelId, Node, NodeId, Property, PropertyKeyId, Value,
};
pub use vector::{
    LateInteractionHit, Similarity, VectorEncoding, VectorHit, VectorSearchPlan,
    VectorSearchStrategy, VectorTarget,
};
