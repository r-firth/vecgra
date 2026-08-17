//! A vector-native embedded labelled property graph database.
//!
//! The storage kernel keeps its dependency surface deliberately small.
//! Importers and language-specific integrations belong in adjacent crates.

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
    SemanticPathOptions, SemanticPatternMatch,
};
pub use model::{
    Edge, EdgeId, ElementRef, ElementSet, LabelId, Node, NodeId, Property, PropertyKeyId, Value,
};
pub use vector::{
    LateInteractionHit, Similarity, VectorEncoding, VectorHit, VectorSearchPlan,
    VectorSearchStrategy, VectorTarget,
};
