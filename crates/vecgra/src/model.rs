use roaring::RoaringTreemap;
use std::sync::Arc;

/// Stable identifier for a node within one database.
pub type NodeId = u64;
/// Stable identifier for a relationship within one database.
pub type EdgeId = u64;
/// Interned identifier for a node or relationship label.
pub type LabelId = u32;
/// Interned identifier for a property key.
pub type PropertyKeyId = u32;

/// A typed reference to either kind of graph element.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementRef {
    /// A node identifier.
    Node(NodeId),
    /// A relationship identifier.
    Edge(EdgeId),
}

/// A compressed, typed set of graph elements used to fuse structural and
/// vector execution. Node and edge IDs occupy separate namespaces, so the set
/// preserves their type while supporting fast set algebra and membership.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementSet {
    nodes: RoaringTreemap,
    edges: RoaringTreemap,
}

impl ElementSet {
    /// Creates an empty element set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an element and returns whether the set changed.
    pub fn insert(&mut self, element: ElementRef) -> bool {
        match element {
            ElementRef::Node(id) => self.nodes.insert(id),
            ElementRef::Edge(id) => self.edges.insert(id),
        }
    }

    /// Removes an element and returns whether it was present.
    pub fn remove(&mut self, element: ElementRef) -> bool {
        match element {
            ElementRef::Node(id) => self.nodes.remove(id),
            ElementRef::Edge(id) => self.edges.remove(id),
        }
    }

    /// Returns whether the set contains the typed element.
    pub fn contains(&self, element: ElementRef) -> bool {
        match element {
            ElementRef::Node(id) => self.nodes.contains(id),
            ElementRef::Edge(id) => self.edges.contains(id),
        }
    }

    /// Returns `true` when the set contains no nodes or relationships.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    /// Returns the combined number of nodes and relationships.
    pub fn len(&self) -> u64 {
        self.nodes.len().saturating_add(self.edges.len())
    }

    /// Returns the number of nodes.
    pub fn node_len(&self) -> u64 {
        self.nodes.len()
    }

    /// Returns the number of relationships.
    pub fn edge_len(&self) -> u64 {
        self.edges.len()
    }

    /// Iterates over node identifiers in ascending order.
    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = NodeId> + '_ {
        self.nodes.iter()
    }

    /// Iterates over relationship identifiers in ascending order.
    pub fn edge_ids(&self) -> impl DoubleEndedIterator<Item = EdgeId> + '_ {
        self.edges.iter()
    }

    /// Returns the union of two typed sets.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.nodes |= &other.nodes;
        result.edges |= &other.edges;
        result
    }

    /// Returns the intersection of two typed sets.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.nodes &= &other.nodes;
        result.edges &= &other.edges;
        result
    }

    /// Returns the elements present in `self` but not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.nodes -= &other.nodes;
        result.edges -= &other.edges;
        result
    }

    pub(crate) fn insert_node(&mut self, id: NodeId) {
        self.nodes.insert(id);
    }

    pub(crate) fn insert_edge(&mut self, id: EdgeId) {
        self.edges.insert(id);
    }

    pub(crate) fn clone_nodes_from(&mut self, ids: &RoaringTreemap) {
        self.nodes |= ids;
    }

    pub(crate) fn clone_edges_from(&mut self, ids: &RoaringTreemap) {
        self.edges |= ids;
    }
}

impl FromIterator<ElementRef> for ElementSet {
    fn from_iter<T: IntoIterator<Item = ElementRef>>(iter: T) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}

impl Extend<ElementRef> for ElementSet {
    fn extend<T: IntoIterator<Item = ElementRef>>(&mut self, iter: T) {
        for element in iter {
            self.insert(element);
        }
    }
}

/// A property value stored directly in the graph.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// An explicit null value.
    Null,
    /// A Boolean value.
    Bool(bool),
    /// A signed 64-bit integer.
    Int(i64),
    /// A 64-bit floating-point value.
    Float(f64),
    /// UTF-8 text.
    String(Arc<str>),
    /// Arbitrary binary data.
    Bytes(Arc<[u8]>),
    /// A reference to a node in this database.
    Node(NodeId),
    /// A reference to a relationship in this database.
    Edge(EdgeId),
}

/// One interned key and value attached to a graph element.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    /// Interned property-key identifier.
    pub key: PropertyKeyId,
    /// Stored property value.
    pub value: Value,
}

/// An owned view of a node record.
#[derive(Clone, Debug)]
pub struct Node {
    /// Stable node identifier.
    pub id: NodeId,
    /// Interned node label.
    pub label: LabelId,
    /// Properties ordered by their interned keys.
    pub properties: Arc<[Property]>,
    /// Number of vector facets owned by the node.
    pub vector_count: u32,
    /// Monotonic record generation, incremented by replacement.
    pub generation: u64,
    pub(crate) pending_vectors: Arc<[f32]>,
}

/// An owned view of a directed relationship record.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Stable relationship identifier.
    pub id: EdgeId,
    /// Source node identifier.
    pub source: NodeId,
    /// Target node identifier.
    pub target: NodeId,
    /// Interned relationship label.
    pub label: LabelId,
    /// Properties ordered by their interned keys.
    pub properties: Arc<[Property]>,
    /// Number of vector facets owned by the relationship.
    pub vector_count: u32,
    /// Monotonic record generation, incremented by replacement.
    pub generation: u64,
    pub(crate) vector_offset: usize,
    pub(crate) pending_vectors: Arc<[f32]>,
}
