//! UI-independent graph snapshots, layout, camera math, LOD, and hit testing
//! for Vecgra Studio.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use petgraph::Graph;
use petgraph::graph::NodeIndex;
use petgraph::prelude::Undirected;
use petgraph_drawing::DrawingEuclidean2d;
use petgraph_layout_omega::Omega;
use petgraph_layout_sgd::{Scheduler as _, SchedulerExponential};
use petgraph_linalg_rdmds::RdMds;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vecgra::{
    Database, Direction, EdgeFilter, EdgeId, ElementRef, GraphStats, NodeId, Property, ReadGuard,
    ShortestPathOptions, ShortestPathStrategy, ShortestPathTermination, Value, VectorTarget,
};

pub use vecgra::{
    Direction as PathDirection, ShortestPathStrategy as EvidencePathStrategy,
    ShortestPathTermination as EvidencePathTermination, Value as PropertyValue,
};

mod camera;
mod layout;
mod scene;
mod search;

pub use camera::*;
pub use layout::*;
pub use scene::*;
pub use search::*;

use layout::{
    initialize_by_label, layout_orbits, owned_properties, recenter, run_force_layout,
    run_resistance_structure_layout, stable_unit, symbol_or_unknown, try_layout_clustered_forest,
    try_layout_radial_forest,
};

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
