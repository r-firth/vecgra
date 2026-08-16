use roaring::RoaringTreemap;
use std::sync::Arc;

pub type NodeId = u64;
pub type EdgeId = u64;
pub type LabelId = u32;
pub type PropertyKeyId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementRef {
    Node(NodeId),
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, element: ElementRef) -> bool {
        match element {
            ElementRef::Node(id) => self.nodes.insert(id),
            ElementRef::Edge(id) => self.edges.insert(id),
        }
    }

    pub fn remove(&mut self, element: ElementRef) -> bool {
        match element {
            ElementRef::Node(id) => self.nodes.remove(id),
            ElementRef::Edge(id) => self.edges.remove(id),
        }
    }

    pub fn contains(&self, element: ElementRef) -> bool {
        match element {
            ElementRef::Node(id) => self.nodes.contains(id),
            ElementRef::Edge(id) => self.edges.contains(id),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.edges.is_empty()
    }

    pub fn len(&self) -> u64 {
        self.nodes.len().saturating_add(self.edges.len())
    }

    pub fn node_len(&self) -> u64 {
        self.nodes.len()
    }

    pub fn edge_len(&self) -> u64 {
        self.edges.len()
    }

    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = NodeId> + '_ {
        self.nodes.iter()
    }

    pub fn edge_ids(&self) -> impl DoubleEndedIterator<Item = EdgeId> + '_ {
        self.edges.iter()
    }

    pub fn union(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.nodes |= &other.nodes;
        result.edges |= &other.edges;
        result
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.nodes &= &other.nodes;
        result.edges &= &other.edges;
        result
    }

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

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<str>),
    Bytes(Arc<[u8]>),
    Node(NodeId),
    Edge(EdgeId),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub key: PropertyKeyId,
    pub value: Value,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub label: LabelId,
    pub properties: Arc<[Property]>,
    pub vector_count: u32,
    pub generation: u64,
    pub(crate) pending_vectors: Arc<[f32]>,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub label: LabelId,
    pub properties: Arc<[Property]>,
    pub vector_count: u32,
    pub generation: u64,
    pub(crate) vector_offset: usize,
    pub(crate) pending_vectors: Arc<[f32]>,
}
