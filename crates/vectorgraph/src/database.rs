use crate::codec::{self, Header};
use crate::error::{Error, Result};
use crate::graph::{
    Direction, EdgeFilter, ElementFilter, FilteredVectorSearchResult, Graph,
    GraphRangeSearchOptions, GraphRangeSearchResult, GraphStats, Mutation, NumericRangeFilter,
    NumericRangePlan, OneHopPlan, OneHopQuery, PatternMatch, SemanticOneHopQuery, SemanticPathHit,
    SemanticPathOptions, SemanticPatternMatch, ShortestPathOptions, ShortestPathResult,
};
use crate::model::{Edge, EdgeId, ElementSet, LabelId, Node, NodeId, Value};
use crate::vector::{
    LateInteractionHit, Similarity, VectorEncoding, VectorHit, VectorSearchPlan, VectorTarget,
};
use memmap2::MmapOptions;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard};

#[derive(Clone, Copy, Debug)]
pub struct DatabaseOptions {
    pub vector_dimension: usize,
    pub similarity: Similarity,
    pub vector_encoding: VectorEncoding,
    pub sync_on_commit: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegrityReport {
    pub nodes: usize,
    pub edges: usize,
    pub indexed_vectors: usize,
    pub transactions: u64,
    pub vector_bytes_verified: usize,
    pub vector_checksum_blocks_verified: usize,
}

impl DatabaseOptions {
    pub fn new(vector_dimension: usize) -> Self {
        Self {
            vector_dimension,
            similarity: Similarity::Cosine,
            vector_encoding: VectorEncoding::F16,
            sync_on_commit: true,
        }
    }
}

#[derive(Debug)]
struct Inner {
    path: PathBuf,
    file: Mutex<File>,
    graph: RwLock<Graph>,
    next_node_id: AtomicU64,
    next_edge_id: AtomicU64,
    next_transaction_id: AtomicU64,
    sync_on_commit: bool,
    read_only: bool,
    vector_encoding: VectorEncoding,
}

#[derive(Clone, Debug)]
pub struct Database {
    inner: Arc<Inner>,
}

impl Database {
    pub fn create(path: impl AsRef<Path>, options: DatabaseOptions) -> Result<Self> {
        if options.vector_dimension == 0 {
            return Err(Error::InvalidArgument(
                "vector dimension must be greater than zero".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        codec::write_header(
            &mut file,
            Header {
                dimension: options.vector_dimension,
                similarity: options.similarity,
                vector_encoding: options.vector_encoding,
                snapshot_metadata_len: 0,
                snapshot_vector_offset: 0,
                snapshot_vector_len: 0,
            },
        )?;
        file.sync_all()?;
        Ok(Self {
            inner: Arc::new(Inner {
                path,
                file: Mutex::new(file),
                graph: RwLock::new(Graph::new(options.vector_dimension, options.similarity)),
                next_node_id: AtomicU64::new(0),
                next_edge_id: AtomicU64::new(0),
                next_transaction_id: AtomicU64::new(1),
                sync_on_commit: options.sync_on_commit,
                read_only: false,
                vector_encoding: options.vector_encoding,
            }),
        })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_internal(path.as_ref(), false)
    }

    /// Opens a database without requesting write permission or repairing a
    /// torn log tail. Valid committed frames remain readable and transactions
    /// that contain mutations are rejected at commit.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_internal(path.as_ref(), true)
    }

    fn open_internal(path: &Path, read_only: bool) -> Result<Self> {
        let path = path.to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .open(&path)?;
        let header = codec::read_header(&mut file)?;
        let mut graph = Graph::new(header.dimension, header.similarity);
        let mut next_transaction_id = 1;
        if header.has_snapshot() {
            // SAFETY: the mapping is read-only, owns its OS mapping, and all
            // section ranges are checked before they are exposed.
            let map = Arc::new(unsafe { MmapOptions::new().map(&file)? });
            graph.begin_snapshot_load(map.clone());
            let snapshot = codec::read_snapshot(&map, header, |operation| {
                graph.apply_snapshot_operation(operation)
            })?;
            graph.finish_snapshot_load(snapshot.csr)?;
            if let Some(records) = snapshot.records {
                graph.install_mapped_records(map.clone(), records)?;
            }
            if let Some(property_index) = snapshot.property_index {
                graph.install_mapped_property_index(map.clone(), property_index)?;
            }
            if let Some(numeric_property_index) = snapshot.numeric_property_index {
                graph.install_mapped_numeric_property_index(map.clone(), numeric_property_index)?;
            }
            graph.install_mapped_vectors(
                map.clone(),
                snapshot.vectors.byte_offset,
                snapshot.vectors.byte_len,
                header.vector_encoding,
                snapshot.vectors.checksum,
                snapshot.vectors.block_checksums,
            )?;
            if let Some(sketches) = snapshot.sketches {
                graph.install_mapped_sketches(map.clone(), sketches)?;
            }
            graph.mark_transaction_applied();
        }
        let valid_end = codec::replay_frames(
            &mut file,
            header.dimension,
            header.vector_encoding,
            header.log_offset(),
            |transaction_id, operations| {
                graph.apply(operations)?;
                graph.mark_transaction_applied();
                let following = transaction_id
                    .checked_add(1)
                    .ok_or_else(|| Error::Corrupt("transaction id space is exhausted".into()))?;
                next_transaction_id = next_transaction_id.max(following);
                Ok(())
            },
        )?;
        if !read_only && file.metadata()?.len() != valid_end {
            file.set_len(valid_end)?;
            file.sync_data()?;
        }
        let next_node_id = graph.next_node_id();
        let next_edge_id = graph.next_edge_id();
        Ok(Self {
            inner: Arc::new(Inner {
                path,
                file: Mutex::new(file),
                graph: RwLock::new(graph),
                next_node_id: AtomicU64::new(next_node_id),
                next_edge_id: AtomicU64::new(next_edge_id),
                next_transaction_id: AtomicU64::new(next_transaction_id),
                sync_on_commit: !read_only,
                read_only,
                vector_encoding: header.vector_encoding,
            }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub fn vector_dimension(&self) -> usize {
        self.inner.graph.read().unwrap().dimension()
    }

    pub fn similarity(&self) -> Similarity {
        self.inner.graph.read().unwrap().similarity()
    }

    pub fn vector_encoding(&self) -> VectorEncoding {
        self.inner.vector_encoding
    }

    pub fn is_read_only(&self) -> bool {
        self.inner.read_only
    }

    pub fn transaction(&self) -> Transaction<'_> {
        Transaction {
            database: self,
            mutations: Vec::new(),
            committed: false,
        }
    }

    pub fn read(&self) -> ReadGuard<'_> {
        ReadGuard {
            graph: self.inner.graph.read().unwrap(),
        }
    }

    pub fn compact_to(
        &self,
        destination: impl AsRef<Path>,
        vector_encoding: VectorEncoding,
    ) -> Result<GraphStats> {
        let destination = destination.as_ref();
        if destination == self.path() {
            return Err(Error::InvalidArgument(
                "compact_to destination must differ from the open database".into(),
            ));
        }
        let mut output = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(destination)?;
        let graph = self.inner.graph.read().unwrap();
        let stats = graph.stats();
        let result = (|| -> Result<()> {
            #[cfg(unix)]
            graph.advise_checkpoint(memmap2::Advice::Sequential);
            let mut snapshot = codec::SnapshotBuilder::new(graph.dimension(), vector_encoding)?;
            let symbols = graph.snapshot_symbols();
            snapshot.append(&symbols)?;

            let mut cursor = 0;
            loop {
                if graph.append_snapshot_node_batch(&mut cursor, 4_096, &mut snapshot)? == 0 {
                    break;
                }
            }
            cursor = 0;
            loop {
                if graph.append_snapshot_edge_batch(&mut cursor, 4_096, &mut snapshot)? == 0 {
                    break;
                }
            }
            let lengths = snapshot.finish_to(&mut output)?;
            #[cfg(unix)]
            graph.advise_checkpoint(memmap2::Advice::Normal);
            codec::write_header(
                &mut output,
                Header {
                    dimension: graph.dimension(),
                    similarity: graph.similarity(),
                    vector_encoding,
                    snapshot_metadata_len: lengths.metadata_len,
                    snapshot_vector_offset: codec::HEADER_LEN + lengths.metadata_len,
                    snapshot_vector_len: lengths.vector_len,
                },
            )?;
            output.flush()?;
            output.sync_all()?;
            Ok(())
        })();
        drop(graph);
        drop(output);
        if let Err(error) = result {
            let _ = std::fs::remove_file(destination);
            return Err(error);
        }
        Ok(stats)
    }
}

pub struct Transaction<'a> {
    database: &'a Database,
    mutations: Vec<Mutation>,
    committed: bool,
}

impl Transaction<'_> {
    pub fn create_node<I, K>(
        &mut self,
        label: impl Into<Arc<str>>,
        properties: I,
        vectors: &[Vec<f32>],
    ) -> NodeId
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        let id = self
            .database
            .inner
            .next_node_id
            .fetch_add(1, Ordering::Relaxed);
        self.mutations.push(Mutation::PutNode {
            id,
            label: label.into(),
            properties: properties
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
            vectors: flatten_vectors(vectors),
            vector_count: vectors.len() as u32,
        });
        id
    }

    pub fn create_edge<I, K>(
        &mut self,
        source: NodeId,
        target: NodeId,
        label: impl Into<Arc<str>>,
        properties: I,
        vectors: &[Vec<f32>],
    ) -> EdgeId
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        let id = self
            .database
            .inner
            .next_edge_id
            .fetch_add(1, Ordering::Relaxed);
        self.mutations.push(Mutation::PutEdge {
            id,
            source,
            target,
            label: label.into(),
            properties: properties
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
            vectors: flatten_vectors(vectors),
            vector_count: vectors.len() as u32,
        });
        id
    }

    pub fn update_node<I, K>(
        &mut self,
        id: NodeId,
        label: impl Into<Arc<str>>,
        properties: I,
        vectors: &[Vec<f32>],
    ) -> Result<()>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        if self.database.inner.graph.read().unwrap().node(id).is_none() {
            return Err(Error::NotFound("node", id));
        }
        self.mutations.push(Mutation::PutNode {
            id,
            label: label.into(),
            properties: properties
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
            vectors: flatten_vectors(vectors),
            vector_count: vectors.len() as u32,
        });
        Ok(())
    }

    pub fn update_edge<I, K>(
        &mut self,
        id: EdgeId,
        source: NodeId,
        target: NodeId,
        label: impl Into<Arc<str>>,
        properties: I,
        vectors: &[Vec<f32>],
    ) -> Result<()>
    where
        I: IntoIterator<Item = (K, Value)>,
        K: Into<Arc<str>>,
    {
        if self.database.inner.graph.read().unwrap().edge(id).is_none() {
            return Err(Error::NotFound("edge", id));
        }
        self.mutations.push(Mutation::PutEdge {
            id,
            source,
            target,
            label: label.into(),
            properties: properties
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
            vectors: flatten_vectors(vectors),
            vector_count: vectors.len() as u32,
        });
        Ok(())
    }

    pub fn delete_node(&mut self, id: NodeId, detach: bool) {
        self.mutations.push(Mutation::DeleteNode { id, detach });
    }

    pub fn delete_edge(&mut self, id: EdgeId) {
        self.mutations.push(Mutation::DeleteEdge { id });
    }

    pub fn commit(mut self) -> Result<()> {
        if self.committed {
            return Err(Error::Conflict("transaction was already committed".into()));
        }
        if self.mutations.is_empty() {
            self.committed = true;
            return Ok(());
        }
        if self.database.inner.read_only {
            return Err(Error::InvalidArgument(
                "cannot commit mutations to a read-only database".into(),
            ));
        }
        let mut graph = self.database.inner.graph.write().unwrap();
        let operations = graph.prepare(&self.mutations)?;
        let transaction_id = self
            .database
            .inner
            .next_transaction_id
            .fetch_add(1, Ordering::Relaxed);
        let frame = codec::encode_frame(
            transaction_id,
            &operations,
            self.database.inner.vector_encoding,
        )?;
        {
            let mut file = self.database.inner.file.lock().unwrap();
            codec::append_frame(&mut *file, &frame)?;
            file.flush()?;
            if self.database.inner.sync_on_commit {
                file.sync_data()?;
            }
        }
        graph.apply(&operations)?;
        graph.mark_transaction_applied();
        self.committed = true;
        Ok(())
    }
}

pub struct ReadGuard<'a> {
    graph: RwLockReadGuard<'a, Graph>,
}

impl ReadGuard<'_> {
    pub fn node(&self, id: NodeId) -> Option<Node> {
        self.graph.node(id)
    }

    pub fn edge(&self, id: EdgeId) -> Option<Edge> {
        self.graph.edge(id)
    }

    pub fn node_ids(&self) -> Vec<NodeId> {
        self.graph.node_ids()
    }

    pub fn edge_ids(&self) -> Vec<EdgeId> {
        self.graph.edge_ids()
    }

    pub fn node_vector(&self, id: NodeId, index: usize) -> Result<Option<&[f32]>> {
        self.graph.node_vector(id, index)
    }

    pub fn edge_vector(&self, id: EdgeId, index: usize) -> Result<Option<&[f32]>> {
        self.graph.edge_vector(id, index)
    }

    /// Copies one vector without hydrating the complete mapped vector column.
    pub fn node_vector_owned(&self, id: NodeId, index: usize) -> Result<Option<Vec<f32>>> {
        self.graph.node_vector_owned(id, index)
    }

    /// Copies one vector without hydrating the complete mapped vector column.
    pub fn edge_vector_owned(&self, id: EdgeId, index: usize) -> Result<Option<Vec<f32>>> {
        self.graph.edge_vector_owned(id, index)
    }

    pub fn label_id(&self, label: &str) -> Option<LabelId> {
        self.graph.symbol_id(label)
    }

    pub fn symbol(&self, id: u32) -> Option<&str> {
        self.graph.symbol(id)
    }

    pub fn nodes_with_label(&self, label: LabelId) -> Vec<Node> {
        self.graph.nodes_with_label(label)
    }

    /// Returns a compressed typed candidate set without hydrating records.
    pub fn elements_with_label(&self, label: LabelId, target: VectorTarget) -> ElementSet {
        self.graph.elements_with_label(label, target)
    }

    /// Materializes an exact label/property predicate as a compressed set that
    /// can be combined with traversal results or passed to vector search.
    pub fn elements_matching(&self, target: VectorTarget, filter: &ElementFilter) -> ElementSet {
        self.graph.elements_matching(target, filter)
    }

    /// Explains the exact access path selected for `elements_matching`.
    pub fn element_filter_plan(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
    ) -> crate::ElementFilterPlan {
        self.graph.element_filter_plan(target, filter)
    }

    /// Materializes an exact same-typed numeric range as a compressed set.
    /// Current checkpoints use an ordered mapped posting table, with an exact
    /// scan fallback for older compatible files.
    pub fn elements_matching_numeric_range(
        &self,
        target: VectorTarget,
        filter: &NumericRangeFilter,
    ) -> Result<ElementSet> {
        self.graph.elements_matching_numeric_range(target, filter)
    }

    /// Explains the access path selected for a numeric range predicate.
    pub fn numeric_range_plan(
        &self,
        target: VectorTarget,
        filter: &NumericRangeFilter,
    ) -> Result<NumericRangePlan> {
        self.graph.numeric_range_plan(target, filter)
    }

    /// Fuses a conjunctive equality predicate and zero or more typed numeric
    /// ranges before adaptive vector execution. Results include every selected
    /// scalar and vector access plan.
    pub fn vector_search_filtered_adaptive(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        equality: Option<&ElementFilter>,
        numeric_ranges: &[NumericRangeFilter],
    ) -> Result<FilteredVectorSearchResult> {
        self.graph
            .vector_search_filtered_adaptive(query, target, limit, equality, numeric_ranges)
    }

    /// Expands all nodes in `seeds` by one hop and returns both reached nodes
    /// and the traversed edges as a compressed set.
    pub fn expand_element_set(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        filter: EdgeFilter,
    ) -> Result<ElementSet> {
        self.graph.expand_element_set(seeds, direction, filter)
    }

    /// Returns the exact neighborhood reachable within `max_hops`. Seed nodes
    /// are excluded; reached nodes and traversed edges share one typed,
    /// compressed result set. Cycles are expanded only at shortest depth.
    pub fn expand_element_set_hops(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        filter: EdgeFilter,
        max_hops: usize,
    ) -> Result<ElementSet> {
        self.graph
            .expand_element_set_hops(seeds, direction, filter, max_hops)
    }

    /// Returns only the nodes reachable within `max_hops`, optionally keeping
    /// the seed nodes and applying an ordinary node label/property predicate.
    /// Traversed relationship IDs are deliberately not materialized.
    pub fn nodes_within_hops(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        edge_filter: EdgeFilter,
        max_hops: usize,
        include_seeds: bool,
        node_filter: Option<&ElementFilter>,
    ) -> Result<ElementSet> {
        self.graph.nodes_within_hops(
            seeds,
            direction,
            edge_filter,
            max_hops,
            include_seeds,
            node_filter,
        )
    }

    /// Finds an exact unweighted path with an inspectable adaptive traversal
    /// strategy while keeping hop and frontier-expansion bounds visible in the
    /// result. `ExpansionLimit` is reported distinctly from a conclusive
    /// absence within `max_hops`.
    pub fn shortest_path(
        &self,
        start: crate::NodeId,
        end: crate::NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        self.graph.shortest_path(start, end, options)
    }

    /// Adaptive nearest-neighbor search constrained to a bounded graph range.
    /// Reachability and optional node predicates are evaluated before exact or
    /// approximate vector candidate selection.
    pub fn vector_search_graph_range_adaptive(
        &self,
        query: &[f32],
        seeds: &ElementSet,
        options: &GraphRangeSearchOptions,
    ) -> Result<GraphRangeSearchResult> {
        self.graph
            .vector_search_graph_range_adaptive(query, seeds, options)
    }

    pub fn neighbors(
        &self,
        node: NodeId,
        direction: Direction,
        filter: EdgeFilter,
    ) -> Result<Vec<Edge>> {
        self.graph.neighbors(node, direction, filter)
    }

    /// Streams adjacent `(node_id, edge_id)` pairs. This avoids cloning edge
    /// records and is the preferred primitive for graph algorithms.
    pub fn visit_neighbors(
        &self,
        node: NodeId,
        direction: Direction,
        filter: EdgeFilter,
        visitor: impl FnMut(NodeId, crate::EdgeId),
    ) -> Result<()> {
        self.graph.visit_neighbors(node, direction, filter, visitor)
    }

    pub fn match_one_hop(&self, query: &OneHopQuery) -> Vec<PatternMatch> {
        self.graph.match_one_hop(query)
    }

    /// Explains whether a one-hop pattern will scan a relationship posting or
    /// expand adjacency from its selective start/end node predicate.
    pub fn one_hop_plan(&self, query: &OneHopQuery) -> OneHopPlan {
        self.graph.one_hop_plan(query)
    }

    pub fn match_semantic_one_hop(
        &self,
        vector_query: &[f32],
        query: &SemanticOneHopQuery,
    ) -> Result<Vec<SemanticPatternMatch>> {
        self.graph.match_semantic_one_hop(vector_query, query)
    }

    pub fn vector_search(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<VectorHit>> {
        self.graph.vector_search(query, target, limit, label)
    }

    /// Exact prefiltered search over graph-derived candidates. Constraints are
    /// applied before vector scoring rather than to a global top-k afterward.
    pub fn vector_search_within(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        self.graph.vector_search_within(query, allowed, limit)
    }

    /// Binary-sketch selection restricted to `allowed`, followed by exact
    /// reranking. The budget is spent only on eligible graph elements.
    pub fn vector_search_within_approximate(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
        candidate_elements: usize,
    ) -> Result<Vec<VectorHit>> {
        self.graph
            .vector_search_within_approximate(query, allowed, limit, candidate_elements)
    }

    pub fn vector_search_within_adaptive(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        self.graph
            .vector_search_within_adaptive(query, allowed, limit)
    }

    pub fn vector_search_within_plan(&self, allowed: &ElementSet) -> VectorSearchPlan {
        self.graph.vector_search_within_plan(allowed)
    }

    /// Searches the immutable checkpoint through a compact binary coarse
    /// index, exactly reranks `candidate_vectors`, and always scans WAL vectors
    /// exhaustively. Cosine databases without a checkpoint transparently use
    /// exact search.
    pub fn vector_search_approximate(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        candidate_vectors: usize,
    ) -> Result<Vec<VectorHit>> {
        self.graph
            .vector_search_approximate(query, target, limit, label, candidate_vectors)
    }

    /// Chooses an exact prefiltered scan or the persisted sketch-and-rerank
    /// tier from the current cardinality and storage layout.
    pub fn vector_search_adaptive(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<VectorHit>> {
        self.graph
            .vector_search_adaptive(query, target, limit, label)
    }

    pub fn vector_search_plan(
        &self,
        target: VectorTarget,
        label: Option<LabelId>,
    ) -> VectorSearchPlan {
        self.graph.vector_search_plan(target, label)
    }

    /// Weighted MaxSim over multiple query vectors and each element's native
    /// vector facets. `weights == None` assigns equal weight to every query.
    pub fn late_interaction_search(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<LateInteractionHit>> {
        self.graph
            .late_interaction_search(queries, weights, target, limit, label)
    }

    /// Weighted MaxSim restricted to a graph-derived candidate set.
    pub fn late_interaction_search_within(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        self.graph
            .late_interaction_search_within(queries, weights, allowed, limit)
    }

    pub fn late_interaction_search_within_approximate(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
        candidate_elements: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        self.graph.late_interaction_search_within_approximate(
            queries,
            weights,
            allowed,
            limit,
            candidate_elements,
        )
    }

    pub fn late_interaction_search_within_adaptive(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        self.graph
            .late_interaction_search_within_adaptive(queries, weights, allowed, limit)
    }

    /// Uses whole-element binary-sketch MaxSim to select a bounded candidate
    /// set, then applies exact weighted late interaction. The budget is in
    /// graph elements, not individual stored vectors.
    pub fn late_interaction_search_approximate(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        candidate_elements: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        self.graph.late_interaction_search_approximate(
            queries,
            weights,
            target,
            limit,
            label,
            candidate_elements,
        )
    }

    pub fn late_interaction_search_adaptive(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<LateInteractionHit>> {
        self.graph
            .late_interaction_search_adaptive(queries, weights, target, limit, label)
    }

    pub fn semantic_paths(
        &self,
        query: &[f32],
        options: &SemanticPathOptions,
    ) -> Result<Vec<SemanticPathHit>> {
        self.graph.semantic_paths(query, options)
    }

    pub fn semantic_expand(
        &self,
        query: &[f32],
        starts: &[NodeId],
        options: &SemanticPathOptions,
    ) -> Result<Vec<SemanticPathHit>> {
        self.graph.semantic_expand(query, starts, options)
    }

    pub fn stats(&self) -> GraphStats {
        self.graph.stats()
    }

    /// Explicitly decodes the immutable F16 vector base into an F32 cache.
    ///
    /// This can improve repeated exact scans at the cost of roughly doubling
    /// the base vector bytes in owned memory. Search never performs this large
    /// allocation implicitly. The returned value is the cache size in bytes.
    pub fn warm_vector_cache(&self) -> Result<usize> {
        self.graph.warm_vector_cache()
    }

    /// Number of owned bytes currently used by the optional full F32 cache.
    pub fn vector_cache_bytes(&self) -> usize {
        self.graph.vector_cache_bytes()
    }

    /// Eagerly verifies every lazy vector checksum. Header, metadata, fixed
    /// records, CSR, indexes, and complete WAL frames were already validated
    /// while opening the database; this closes the remaining lazy-data gap
    /// without decoding or allocating the vector column.
    pub fn verify_integrity(&self) -> Result<IntegrityReport> {
        let stats = self.graph.stats();
        let (vector_bytes_verified, vector_checksum_blocks_verified) =
            self.graph.verify_integrity()?;
        Ok(IntegrityReport {
            nodes: stats.nodes,
            edges: stats.edges,
            indexed_vectors: stats.indexed_vectors,
            transactions: stats.transactions,
            vector_bytes_verified,
            vector_checksum_blocks_verified,
        })
    }

    pub fn property<'a>(&self, properties: &'a [crate::Property], key: &str) -> Option<&'a Value> {
        let key = self.graph.symbol_id(key)?;
        properties
            .binary_search_by_key(&key, |property| property.key)
            .ok()
            .map(|index| &properties[index].value)
    }
}

fn flatten_vectors(vectors: &[Vec<f32>]) -> Vec<f32> {
    let length = vectors.iter().map(Vec::len).sum();
    let mut flattened = Vec::with_capacity(length);
    for vector in vectors {
        flattened.extend_from_slice(vector);
    }
    flattened
}
