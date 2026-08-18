use super::*;

impl Graph {
    pub(crate) fn new(dimension: usize, similarity: Similarity) -> Self {
        Self {
            dimension,
            similarity,
            symbols: Vec::new(),
            symbol_ids: HashMap::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            mapped_nodes: None,
            mapped_edges: None,
            mapped_property_index: None,
            mapped_numeric_property_index: None,
            node_overlays: HashMap::new(),
            edge_overlays: HashMap::new(),
            outgoing: MutableAdjacency::dense(),
            incoming: MutableAdjacency::dense(),
            base_outgoing: None,
            base_incoming: None,
            loading_snapshot: false,
            snapshot_map: None,
            owned_properties: Vec::new(),
            nodes_by_label: HashMap::new(),
            edges_by_label: HashMap::new(),
            vector_data: VectorData::Owned(Vec::new()),
            sketch_index: OnceLock::new(),
            indexed_vectors: 0,
            indexed_node_vectors: 0,
            indexed_edge_vectors: 0,
            node_count: 0,
            edge_count: 0,
            transactions: 0,
        }
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn similarity(&self) -> Similarity {
        self.similarity
    }

    pub(crate) fn mark_transaction_applied(&mut self) {
        self.transactions += 1;
    }

    pub(crate) fn next_node_id(&self) -> NodeId {
        self.nodes
            .len()
            .max(
                self.mapped_nodes
                    .as_ref()
                    .map_or(0, |records| records.slots),
            )
            .max(
                self.node_overlays
                    .keys()
                    .max()
                    .and_then(|id| usize::try_from(*id).ok())
                    .map_or(0, |id| id.saturating_add(1)),
            ) as NodeId
    }

    pub(crate) fn next_edge_id(&self) -> EdgeId {
        self.edges
            .len()
            .max(
                self.mapped_edges
                    .as_ref()
                    .map_or(0, |records| records.slots),
            )
            .max(
                self.edge_overlays
                    .keys()
                    .max()
                    .and_then(|id| usize::try_from(*id).ok())
                    .map_or(0, |id| id.saturating_add(1)),
            ) as EdgeId
    }

    pub(super) fn node_record(&self, id: NodeId) -> Option<StoredNode> {
        if !self.node_overlays.is_empty()
            && let Some(record) = self.node_overlays.get(&id)
        {
            return *record;
        }
        if !self.nodes.is_empty()
            && let Some(record) = self.nodes.get(id as usize).and_then(|record| *record)
        {
            return Some(record);
        }
        self.mapped_nodes.as_ref()?.get(id)
    }

    pub(super) fn edge_record(&self, id: EdgeId) -> Option<StoredEdge> {
        if !self.edge_overlays.is_empty()
            && let Some(record) = self.edge_overlays.get(&id)
        {
            return *record;
        }
        if !self.edges.is_empty()
            && let Some(record) = self.edges.get(id as usize).and_then(|record| *record)
        {
            return Some(record);
        }
        self.mapped_edges.as_ref()?.get(id)
    }

    pub(super) fn node_records(&self) -> Box<dyn Iterator<Item = StoredNode> + '_> {
        if let Some(mapped) = &self.mapped_nodes {
            if self.node_overlays.is_empty() {
                Box::new(mapped.iter())
            } else {
                Box::new(
                    mapped
                        .iter()
                        .filter(|record| !self.node_overlays.contains_key(&record.id))
                        .chain(self.node_overlays.values().filter_map(|record| *record)),
                )
            }
        } else {
            Box::new(self.nodes.iter().filter_map(|record| *record))
        }
    }

    pub(super) fn edge_records(&self) -> Box<dyn Iterator<Item = StoredEdge> + '_> {
        if let Some(mapped) = &self.mapped_edges {
            if self.edge_overlays.is_empty() {
                Box::new(mapped.iter())
            } else {
                Box::new(
                    mapped
                        .iter()
                        .filter(|record| !self.edge_overlays.contains_key(&record.id))
                        .chain(self.edge_overlays.values().filter_map(|record| *record)),
                )
            }
        } else {
            Box::new(self.edges.iter().filter_map(|record| *record))
        }
    }

    /// Streams the vector-relevant node columns without decoding properties or
    /// generations. Checkpoint-only databases take a monomorphic mapped path;
    /// mutable/legacy graphs retain the unified record semantics.
    #[inline]
    pub(super) fn visit_node_vector_fields(
        &self,
        mut visitor: impl FnMut(NodeId, LabelId, usize, u32) -> Result<()>,
    ) -> Result<()> {
        if self.nodes.is_empty()
            && self.node_overlays.is_empty()
            && let Some(records) = &self.mapped_nodes
        {
            for index in 0..records.count {
                let (id, label, offset, count) = records.vector_fields_at(index);
                visitor(id, label, offset, count)?;
            }
        } else {
            for node in self.node_records() {
                visitor(node.id, node.label, node.vector_offset, node.vector_count)?;
            }
        }
        Ok(())
    }

    #[inline]
    pub(super) fn visit_edge_vector_fields(
        &self,
        mut visitor: impl FnMut(EdgeId, LabelId, usize, u32) -> Result<()>,
    ) -> Result<()> {
        if self.edges.is_empty()
            && self.edge_overlays.is_empty()
            && let Some(records) = &self.mapped_edges
        {
            for index in 0..records.count {
                let (id, label, offset, count) = records.vector_fields_at(index);
                visitor(id, label, offset, count)?;
            }
        } else {
            for edge in self.edge_records() {
                visitor(edge.id, edge.label, edge.vector_offset, edge.vector_count)?;
            }
        }
        Ok(())
    }

    pub(crate) fn install_mapped_records(
        &mut self,
        map: Arc<Mmap>,
        sections: crate::codec::SnapshotRecordSections,
    ) -> Result<()> {
        self.mapped_nodes = Some(MappedNodeRecords {
            map: map.clone(),
            byte_offset: sections.node_byte_offset,
            count: sections.node_count,
            slots: sections.node_slots,
            property_byte_offset: sections.property_byte_offset,
        });
        self.mapped_edges = Some(MappedEdgeRecords {
            map,
            byte_offset: sections.edge_byte_offset,
            count: sections.edge_count,
            slots: sections.edge_slots,
            property_byte_offset: sections.property_byte_offset,
        });
        self.nodes.clear();
        self.nodes.shrink_to_fit();
        self.edges.clear();
        self.edges.shrink_to_fit();
        self.node_count = sections.node_count;
        self.edge_count = sections.edge_count;
        self.indexed_vectors = 0;
        self.indexed_node_vectors = 0;
        self.indexed_edge_vectors = 0;
        self.nodes_by_label.clear();
        self.edges_by_label.clear();
        for node in self.mapped_nodes.as_ref().unwrap().iter() {
            if node.label as usize >= self.symbols.len() {
                return Err(Error::Corrupt(
                    "mapped node label exceeds dictionary".into(),
                ));
            }
            self.indexed_vectors = self
                .indexed_vectors
                .checked_add(node.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped vector count overflow".into()))?;
            self.indexed_node_vectors = self
                .indexed_node_vectors
                .checked_add(node.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped node vector count overflow".into()))?;
            self.nodes_by_label
                .entry(node.label)
                .or_default()
                .insert(node.id);
        }
        for edge in self.mapped_edges.as_ref().unwrap().iter() {
            if edge.label as usize >= self.symbols.len() {
                return Err(Error::Corrupt(
                    "mapped edge label exceeds dictionary".into(),
                ));
            }
            self.indexed_vectors = self
                .indexed_vectors
                .checked_add(edge.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped vector count overflow".into()))?;
            self.indexed_edge_vectors = self
                .indexed_edge_vectors
                .checked_add(edge.vector_count as usize)
                .ok_or_else(|| Error::Corrupt("mapped edge vector count overflow".into()))?;
            self.edges_by_label
                .entry(edge.label)
                .or_default()
                .insert(edge.id);
        }
        Ok(())
    }

    pub(crate) fn install_mapped_property_index(
        &mut self,
        map: Arc<Mmap>,
        section: crate::codec::SnapshotPropertyIndexSection,
    ) -> Result<()> {
        self.mapped_property_index = Some(MappedPropertyIndex::new(map, section)?);
        Ok(())
    }

    pub(crate) fn install_mapped_numeric_property_index(
        &mut self,
        map: Arc<Mmap>,
        section: crate::codec::SnapshotNumericPropertyIndexSection,
    ) -> Result<()> {
        self.mapped_numeric_property_index = Some(MappedNumericPropertyIndex::new(map, section)?);
        Ok(())
    }

    pub(crate) fn node(&self, id: NodeId) -> Option<Node> {
        let node = self.node_record(id)?;
        Some(Node {
            id: node.id,
            label: node.label,
            properties: node
                .properties
                .get(self.snapshot_map.as_deref(), &self.owned_properties),
            vector_count: node.vector_count,
            generation: node.generation.get(),
            pending_vectors: Arc::from([]),
        })
    }

    pub(crate) fn edge(&self, id: EdgeId) -> Option<Edge> {
        let edge = self.edge_record(id)?;
        Some(Edge {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label,
            properties: edge
                .properties
                .get(self.snapshot_map.as_deref(), &self.owned_properties),
            vector_count: edge.vector_count,
            generation: edge.generation.get(),
            vector_offset: edge.vector_offset,
            pending_vectors: Arc::from([]),
        })
    }

    #[inline]
    pub(super) fn has_node(&self, id: NodeId) -> bool {
        if let Some(record) = self.node_overlays.get(&id) {
            return record.is_some();
        }
        if self.nodes.get(id as usize).is_some_and(Option::is_some) {
            return true;
        }
        self.mapped_nodes
            .as_ref()
            .is_some_and(|records| records.contains(id))
    }

    #[inline]
    pub(super) fn stored_edge(&self, id: EdgeId) -> Option<StoredEdge> {
        self.edge_record(id)
    }

    pub(crate) fn node_ids(&self) -> Vec<NodeId> {
        self.node_records().map(|node| node.id).collect()
    }

    pub(crate) fn edge_ids(&self) -> Vec<EdgeId> {
        self.edge_records().map(|edge| edge.id).collect()
    }

    pub(crate) fn symbol_id(&self, value: &str) -> Option<u32> {
        self.symbol_ids.get(value).copied()
    }

    pub(crate) fn symbol(&self, id: u32) -> Option<&str> {
        self.symbols.get(id as usize).map(AsRef::as_ref)
    }

    pub(crate) fn stats(&self) -> GraphStats {
        GraphStats {
            nodes: self.node_count,
            edges: self.edge_count,
            labels: self.symbols.len(),
            indexed_vectors: self.indexed_vectors,
            transactions: self.transactions,
        }
    }

    pub(crate) fn warm_vector_cache(&self) -> Result<usize> {
        self.vector_data.warm_f32_cache()
    }

    pub(crate) fn vector_cache_bytes(&self) -> usize {
        self.vector_data.f32_cache_bytes()
    }

    pub(crate) fn verify_integrity(&self) -> Result<(usize, usize)> {
        self.vector_data.verify_all_bytes()
    }

    pub(crate) fn snapshot_symbols(&self) -> Vec<Operation> {
        self.symbols
            .iter()
            .enumerate()
            .map(|(id, value)| Operation::InternSymbol {
                id: id as u32,
                value: value.clone(),
            })
            .collect()
    }

    #[cfg(unix)]
    pub(crate) fn advise_checkpoint(&self, advice: memmap2::Advice) {
        if let Some(map) = &self.snapshot_map {
            let _ = map.advise(advice);
        }
    }

    pub(crate) fn apply_snapshot_operation(&mut self, operation: SnapshotOperation) -> Result<()> {
        match operation {
            SnapshotOperation::InternSymbol { id, value } => {
                if id as usize != self.symbols.len() {
                    return Err(Error::Corrupt(format!(
                        "snapshot symbol id {id} is not the next dictionary id {}",
                        self.symbols.len()
                    )));
                }
                self.symbols.push(value.clone());
                self.symbol_ids.insert(value, id);
            }
            SnapshotOperation::PutNode {
                id,
                generation,
                label,
                property_byte_offset,
                property_byte_len,
                vector_count,
                vector_offset,
            } => self.install_snapshot_node(StoredNode {
                id,
                label,
                properties: StoredProperties::mapped(property_byte_offset, property_byte_len),
                vector_count,
                generation: nonzero_generation(generation)?,
                vector_offset,
            })?,
            SnapshotOperation::PutEdge {
                id,
                generation,
                source,
                target,
                label,
                property_byte_offset,
                property_byte_len,
                vector_count,
                vector_offset,
            } => self.install_snapshot_edge(StoredEdge {
                id,
                source,
                target,
                label,
                properties: StoredProperties::mapped(property_byte_offset, property_byte_len),
                vector_count,
                generation: nonzero_generation(generation)?,
                vector_offset,
            })?,
        }
        Ok(())
    }

    pub(crate) fn begin_snapshot_load(&mut self, map: Arc<Mmap>) {
        self.loading_snapshot = true;
        self.snapshot_map = Some(map);
        self.outgoing = MutableAdjacency::sparse();
        self.incoming = MutableAdjacency::sparse();
    }

    pub(crate) fn finish_snapshot_load(
        &mut self,
        csr: Option<crate::codec::SnapshotCsrSections>,
    ) -> Result<()> {
        if let Some(csr) = csr {
            let map = self.snapshot_map.as_ref().unwrap().clone();
            self.base_outgoing = Some(CsrAdjacency::mapped(
                map.clone(),
                csr.out_offsets,
                csr.out_ids,
            )?);
            self.base_incoming = Some(CsrAdjacency::mapped(map, csr.in_offsets, csr.in_ids)?);
        } else {
            self.base_outgoing = Some(CsrAdjacency::from_edges(
                &self.edges,
                self.nodes.len(),
                true,
            )?);
            self.base_incoming = Some(CsrAdjacency::from_edges(
                &self.edges,
                self.nodes.len(),
                false,
            )?);
        }
        self.loading_snapshot = false;
        Ok(())
    }

    pub(crate) fn install_mapped_vectors(
        &mut self,
        map: Arc<Mmap>,
        byte_offset: usize,
        byte_len: usize,
        encoding: VectorEncoding,
        checksum: u32,
        block_checksums: Option<Arc<[u32]>>,
    ) -> Result<()> {
        let bytes_per_vector_value = match encoding {
            VectorEncoding::F32 => 4,
            VectorEncoding::F16 => 2,
        };
        if !byte_len.is_multiple_of(bytes_per_vector_value) {
            return Err(Error::Corrupt(
                "checkpoint vector section has a partial value".into(),
            ));
        }
        let float_count = byte_len / bytes_per_vector_value;
        let required = self
            .node_records()
            .map(|node| node.vector_offset + node.vector_count as usize * self.dimension)
            .chain(
                self.edge_records()
                    .map(|edge| edge.vector_offset + edge.vector_count as usize * self.dimension),
            )
            .max()
            .unwrap_or(0);
        if required > float_count {
            return Err(Error::Corrupt(format!(
                "checkpoint metadata references {required} floats but vector section has {float_count}"
            )));
        }
        let verified_blocks = (0..block_checksums
            .as_ref()
            .map_or(0, |checksums| checksums.len()))
            .map(|_| OnceLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        self.vector_data = VectorData::Mapped {
            map,
            byte_offset,
            byte_len,
            encoding,
            checksum,
            block_checksums,
            verified: OnceLock::new(),
            verified_blocks,
            decoded: OnceLock::new(),
            delta: Vec::new(),
        };
        Ok(())
    }

    pub(crate) fn install_mapped_sketches(
        &mut self,
        map: Arc<Mmap>,
        section: crate::codec::SnapshotSketchSection,
    ) -> Result<()> {
        let word_count = section
            .entry_count
            .checked_mul(section.words_per_signature)
            .ok_or_else(|| Error::Corrupt("mapped sketch word count overflow".into()))?;
        #[cfg(target_endian = "little")]
        if let Some(columns) = section.owner_columns {
            let index = BinarySketchIndex::mapped_columns(
                self.dimension,
                map,
                columns.owner_byte_offset,
                columns.owner_kind_byte_offset,
                columns.label_byte_offset,
                section.byte_offset,
                section.entry_count,
                section.words_per_signature,
                word_count,
            );
            self.sketch_index.set(Ok(index)).map_err(|_| {
                Error::Corrupt("checkpoint installs sketches more than once".into())
            })?;
            return Ok(());
        }
        let mut owners = vec![0u64; section.entry_count];
        let mut owner_kinds = vec![0u8; section.entry_count];
        let mut labels = vec![0u32; section.entry_count];
        let mut populated = vec![false; section.entry_count];
        for node in self.node_records() {
            for vector_index in 0..node.vector_count {
                let float_offset = node.vector_offset + vector_index as usize * self.dimension;
                let ordinal = float_offset / self.dimension;
                if !float_offset.is_multiple_of(self.dimension)
                    || ordinal >= owners.len()
                    || populated[ordinal]
                {
                    return Err(Error::Corrupt(
                        "indexed checkpoint node sketch ordinal is invalid".into(),
                    ));
                }
                owners[ordinal] = node.id;
                owner_kinds[ordinal] = 0;
                labels[ordinal] = node.label;
                populated[ordinal] = true;
            }
        }
        for edge in self.edge_records() {
            for vector_index in 0..edge.vector_count {
                let float_offset = edge.vector_offset + vector_index as usize * self.dimension;
                let ordinal = float_offset / self.dimension;
                if !float_offset.is_multiple_of(self.dimension)
                    || ordinal >= owners.len()
                    || populated[ordinal]
                {
                    return Err(Error::Corrupt(
                        "indexed checkpoint edge sketch ordinal is invalid".into(),
                    ));
                }
                owners[ordinal] = edge.id;
                owner_kinds[ordinal] = 1;
                labels[ordinal] = edge.label;
                populated[ordinal] = true;
            }
        }
        if populated.iter().any(|populated| !populated) {
            return Err(Error::Corrupt(
                "sketch entries do not densely cover checkpoint vectors".into(),
            ));
        }
        let index = BinarySketchIndex::mapped(
            self.dimension,
            owners,
            owner_kinds,
            labels,
            map,
            section.byte_offset,
            section.words_per_signature,
            word_count,
        );
        self.sketch_index
            .set(Ok(index))
            .map_err(|_| Error::Corrupt("checkpoint installs sketches more than once".into()))?;
        Ok(())
    }

    pub(super) fn install_snapshot_node(&mut self, node: StoredNode) -> Result<()> {
        let index = node.id as usize;
        grow_slots(&mut self.nodes, index);
        self.outgoing.grow(index);
        self.incoming.grow(index);
        if self.nodes[index].is_some() {
            return Err(Error::Corrupt(format!(
                "checkpoint contains duplicate node {}",
                node.id
            )));
        }
        self.node_count += 1;
        self.indexed_vectors += node.vector_count as usize;
        self.indexed_node_vectors += node.vector_count as usize;
        self.nodes_by_label
            .entry(node.label)
            .or_default()
            .insert(node.id);
        self.nodes[index] = Some(node);
        Ok(())
    }

    pub(super) fn install_snapshot_edge(&mut self, edge: StoredEdge) -> Result<()> {
        if !self.has_node(edge.source) || !self.has_node(edge.target) {
            return Err(Error::Corrupt(format!(
                "checkpoint edge {} refers to a missing endpoint",
                edge.id
            )));
        }
        let index = edge.id as usize;
        grow_slots(&mut self.edges, index);
        if self.edges[index].is_some() {
            return Err(Error::Corrupt(format!(
                "checkpoint contains duplicate edge {}",
                edge.id
            )));
        }
        self.edge_count += 1;
        self.indexed_vectors += edge.vector_count as usize;
        self.indexed_edge_vectors += edge.vector_count as usize;
        self.edges_by_label
            .entry(edge.label)
            .or_default()
            .insert(edge.id);
        if !self.loading_snapshot {
            self.outgoing.push(edge.source, edge.id);
            self.incoming.push(edge.target, edge.id);
        }
        self.edges[index] = Some(edge);
        Ok(())
    }

    pub(crate) fn append_snapshot_node_batch(
        &self,
        cursor: &mut usize,
        batch_size: usize,
        snapshot: &mut crate::codec::SnapshotBuilder,
    ) -> Result<usize> {
        let mut appended = 0;
        let slots = self.next_node_id() as usize;
        while *cursor < slots && appended < batch_size {
            if let Some(node) = self.node_record(*cursor as NodeId) {
                let start = node.vector_offset;
                let float_count = node.vector_count as usize * self.dimension;
                let properties = node
                    .properties
                    .encoded(self.snapshot_map.as_deref(), &self.owned_properties)?;
                match self.vector_data.snapshot_range(
                    start,
                    float_count,
                    snapshot.vector_encoding(),
                )? {
                    SnapshotVectorRange::Encoded(vectors) => snapshot.append_node_encoded(
                        node.id,
                        node.generation.get(),
                        node.label,
                        &properties,
                        node.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::F32(vectors) => snapshot.append_node_f32(
                        node.id,
                        node.generation.get(),
                        node.label,
                        &properties,
                        node.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::OwnedF32(vectors) => snapshot.append_node_f32(
                        node.id,
                        node.generation.get(),
                        node.label,
                        &properties,
                        node.vector_count,
                        &vectors,
                    )?,
                }
                appended += 1;
            }
            *cursor += 1;
        }
        Ok(appended)
    }

    pub(crate) fn append_snapshot_edge_batch(
        &self,
        cursor: &mut usize,
        batch_size: usize,
        snapshot: &mut crate::codec::SnapshotBuilder,
    ) -> Result<usize> {
        let mut appended = 0;
        let slots = self.next_edge_id() as usize;
        while *cursor < slots && appended < batch_size {
            if let Some(edge) = self.edge_record(*cursor as EdgeId) {
                let start = edge.vector_offset;
                let float_count = edge.vector_count as usize * self.dimension;
                let properties = edge
                    .properties
                    .encoded(self.snapshot_map.as_deref(), &self.owned_properties)?;
                match self.vector_data.snapshot_range(
                    start,
                    float_count,
                    snapshot.vector_encoding(),
                )? {
                    SnapshotVectorRange::Encoded(vectors) => snapshot.append_edge_encoded(
                        edge.id,
                        edge.generation.get(),
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::F32(vectors) => snapshot.append_edge_f32(
                        edge.id,
                        edge.generation.get(),
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        vectors,
                    )?,
                    SnapshotVectorRange::OwnedF32(vectors) => snapshot.append_edge_f32(
                        edge.id,
                        edge.generation.get(),
                        edge.source,
                        edge.target,
                        edge.label,
                        &properties,
                        edge.vector_count,
                        &vectors,
                    )?,
                }
                appended += 1;
            }
            *cursor += 1;
        }
        Ok(appended)
    }

    pub(crate) fn prepare(&self, mutations: &[Mutation]) -> Result<Vec<Operation>> {
        let mut operations = Vec::with_capacity(mutations.len() * 2);
        let mut pending_symbols: HashMap<Arc<str>, u32> = HashMap::new();
        let mut next_symbol = self.symbols.len() as u32;
        let mut touched_nodes = HashSet::new();
        let mut touched_edges = HashSet::new();
        let mut staged_nodes = HashSet::new();
        let mut deleted_edges = HashSet::new();

        for mutation in mutations {
            match mutation {
                Mutation::PutNode { id, .. } => {
                    if !touched_nodes.insert(*id) {
                        return Err(Error::Conflict(format!(
                            "node {id} is mutated more than once"
                        )));
                    }
                    staged_nodes.insert(*id);
                }
                Mutation::PutEdge { id, .. } | Mutation::DeleteEdge { id } => {
                    if !touched_edges.insert(*id) {
                        return Err(Error::Conflict(format!(
                            "edge {id} is mutated more than once"
                        )));
                    }
                    if matches!(mutation, Mutation::DeleteEdge { .. }) {
                        deleted_edges.insert(*id);
                    }
                }
                Mutation::DeleteNode { id, .. } => {
                    if !touched_nodes.insert(*id) {
                        return Err(Error::Conflict(format!(
                            "node {id} is mutated more than once"
                        )));
                    }
                }
            }
        }

        for mutation in mutations {
            match mutation {
                Mutation::PutNode {
                    id,
                    label,
                    properties,
                    vectors,
                    vector_count,
                } => {
                    validate_vectors(self.dimension, self.similarity, vectors, *vector_count)?;
                    let label = self.resolve_symbol(
                        label,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let properties = self.resolve_properties(
                        properties,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let generation = match self.node_record(*id) {
                        Some(existing) => {
                            existing.generation.get().checked_add(1).ok_or_else(|| {
                                Error::Conflict(format!("node {id} generation is exhausted"))
                            })?
                        }
                        None => 1,
                    };
                    operations.push(Operation::PutNode(Node {
                        id: *id,
                        label,
                        properties: properties.into(),
                        vector_count: *vector_count,
                        generation,
                        pending_vectors: vectors.clone().into(),
                    }));
                }
                Mutation::PutEdge {
                    id,
                    source,
                    target,
                    label,
                    properties,
                    vectors,
                    vector_count,
                } => {
                    for endpoint in [source, target] {
                        if !self.has_node(*endpoint) && !staged_nodes.contains(endpoint) {
                            return Err(Error::NotFound("endpoint node", *endpoint));
                        }
                    }
                    validate_vectors(self.dimension, self.similarity, vectors, *vector_count)?;
                    let label = self.resolve_symbol(
                        label,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let properties = self.resolve_properties(
                        properties,
                        &mut pending_symbols,
                        &mut next_symbol,
                        &mut operations,
                    );
                    let generation = match self.stored_edge(*id) {
                        Some(existing) => {
                            existing.generation.get().checked_add(1).ok_or_else(|| {
                                Error::Conflict(format!("edge {id} generation is exhausted"))
                            })?
                        }
                        None => 1,
                    };
                    operations.push(Operation::PutEdge(Edge {
                        id: *id,
                        source: *source,
                        target: *target,
                        label,
                        properties: properties.into(),
                        vector_count: *vector_count,
                        generation,
                        vector_offset: 0,
                        pending_vectors: vectors.clone().into(),
                    }));
                }
                Mutation::DeleteEdge { id } => {
                    if self.stored_edge(*id).is_none() {
                        return Err(Error::NotFound("edge", *id));
                    }
                    operations.push(Operation::DeleteEdge(*id));
                }
                Mutation::DeleteNode { id, detach } => {
                    if !self.has_node(*id) {
                        return Err(Error::NotFound("node", *id));
                    }
                    let incident = self.incident_edge_ids(*id);
                    if !*detach && incident.iter().any(|edge| !deleted_edges.contains(edge)) {
                        return Err(Error::Conflict(format!(
                            "node {id} still has incident edges; use detach deletion"
                        )));
                    }
                    if *detach {
                        for edge in incident {
                            if touched_edges.insert(edge) {
                                operations.push(Operation::DeleteEdge(edge));
                            }
                        }
                    }
                    operations.push(Operation::DeleteNode(*id));
                }
            }
        }
        // Persistence and replay must never depend on the order in which the
        // caller happened to stage mutations. Dictionary entries precede all
        // users, nodes precede their edges, and incident edges disappear
        // before their nodes.
        operations.sort_by_key(operation_order);
        Ok(operations)
    }

    pub(super) fn incident_edge_ids(&self, node: NodeId) -> HashSet<EdgeId> {
        let mut edges = HashSet::new();
        self.collect_incident_ids(node, &mut edges);
        edges
    }

    pub(super) fn collect_incident_ids(&self, node: NodeId, result: &mut HashSet<EdgeId>) {
        let outgoing_base = self
            .base_outgoing
            .as_ref()
            .map(|base| base.get(node))
            .unwrap_or_default();
        let incoming_base = self
            .base_incoming
            .as_ref()
            .map(|base| base.get(node))
            .unwrap_or_default();
        for &id in outgoing_base
            .iter()
            .chain(self.outgoing.get(node))
            .chain(incoming_base)
            .chain(self.incoming.get(node))
        {
            if self
                .stored_edge(id)
                .is_some_and(|edge| edge.source == node || edge.target == node)
            {
                result.insert(id);
            }
        }
    }

    pub(super) fn resolve_symbol(
        &self,
        value: &Arc<str>,
        pending: &mut HashMap<Arc<str>, u32>,
        next: &mut u32,
        operations: &mut Vec<Operation>,
    ) -> u32 {
        if let Some(id) = self.symbol_ids.get(value).copied() {
            return id;
        }
        if let Some(id) = pending.get(value).copied() {
            return id;
        }
        let id = *next;
        *next += 1;
        pending.insert(value.clone(), id);
        operations.push(Operation::InternSymbol {
            id,
            value: value.clone(),
        });
        id
    }

    pub(super) fn resolve_properties(
        &self,
        properties: &[(Arc<str>, Value)],
        pending: &mut HashMap<Arc<str>, u32>,
        next: &mut u32,
        operations: &mut Vec<Operation>,
    ) -> Vec<Property> {
        let mut result: Vec<_> = properties
            .iter()
            .map(|(key, value)| Property {
                key: self.resolve_symbol(key, pending, next, operations),
                value: value.clone(),
            })
            .collect();
        result.sort_unstable_by_key(|property| property.key);
        result.dedup_by_key(|property| property.key);
        result
    }

    pub(crate) fn apply(&mut self, operations: &[Operation]) -> Result<()> {
        for operation in operations {
            match operation {
                Operation::InternSymbol { id, value } => {
                    if *id as usize != self.symbols.len() {
                        return Err(Error::Corrupt(format!(
                            "symbol id {id} is not the next dictionary id {}",
                            self.symbols.len()
                        )));
                    }
                    self.symbols.push(value.clone());
                    self.symbol_ids.insert(value.clone(), *id);
                }
                Operation::PutNode(node) => self.apply_node(node.clone())?,
                Operation::PutEdge(edge) => self.apply_edge(edge.clone())?,
                Operation::DeleteNode(id) => {
                    if let Some(node) = self.node_record(*id) {
                        self.node_count -= 1;
                        self.indexed_vectors -= node.vector_count as usize;
                        self.indexed_node_vectors -= node.vector_count as usize;
                        if let Some(ids) = self.nodes_by_label.get_mut(&node.label) {
                            ids.remove(node.id);
                        }
                        if self.mapped_nodes.is_some() {
                            self.node_overlays.insert(*id, None);
                        } else if let Some(slot) = self.nodes.get_mut(*id as usize) {
                            *slot = None;
                        }
                    }
                }
                Operation::DeleteEdge(id) => {
                    if let Some(edge) = self.edge_record(*id) {
                        self.edge_count -= 1;
                        self.indexed_vectors -= edge.vector_count as usize;
                        self.indexed_edge_vectors -= edge.vector_count as usize;
                        if let Some(ids) = self.edges_by_label.get_mut(&edge.label) {
                            ids.remove(edge.id);
                        }
                        self.remove_from_adjacency(edge.source, edge.target, edge.id);
                        if self.mapped_edges.is_some() {
                            self.edge_overlays.insert(*id, None);
                        } else if let Some(slot) = self.edges.get_mut(*id as usize) {
                            *slot = None;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn apply_node(&mut self, node: Node) -> Result<()> {
        validate_vectors(
            self.dimension,
            self.similarity,
            &node.pending_vectors,
            node.vector_count,
        )?;
        let index = node.id as usize;
        if self.mapped_nodes.is_none() {
            grow_slots(&mut self.nodes, index);
            self.outgoing.grow(index);
            self.incoming.grow(index);
        }
        if let Some(existing) = self.node_record(node.id) {
            self.indexed_vectors -= existing.vector_count as usize;
            self.indexed_node_vectors -= existing.vector_count as usize;
            if let Some(ids) = self.nodes_by_label.get_mut(&existing.label) {
                ids.remove(existing.id);
            }
        } else {
            self.node_count += 1;
        }
        let vector_offset = self.append_vectors(&node.pending_vectors)?;
        self.indexed_vectors += node.vector_count as usize;
        self.indexed_node_vectors += node.vector_count as usize;
        self.nodes_by_label
            .entry(node.label)
            .or_default()
            .insert(node.id);
        let properties = self.store_owned_properties(node.properties)?;
        let stored = StoredNode {
            id: node.id,
            label: node.label,
            properties,
            vector_count: node.vector_count,
            generation: nonzero_generation(node.generation)?,
            vector_offset,
        };
        if self.mapped_nodes.is_some() {
            self.node_overlays.insert(node.id, Some(stored));
        } else {
            self.nodes[index] = Some(stored);
        }
        Ok(())
    }

    pub(super) fn apply_edge(&mut self, edge: Edge) -> Result<()> {
        validate_vectors(
            self.dimension,
            self.similarity,
            &edge.pending_vectors,
            edge.vector_count,
        )?;
        if !self.has_node(edge.source) || !self.has_node(edge.target) {
            return Err(Error::Corrupt(format!(
                "edge {} refers to a missing endpoint",
                edge.id
            )));
        }
        let index = edge.id as usize;
        if self.mapped_edges.is_none() {
            grow_slots(&mut self.edges, index);
        }
        if let Some(existing) = self.edge_record(edge.id) {
            self.indexed_vectors -= existing.vector_count as usize;
            self.indexed_edge_vectors -= existing.vector_count as usize;
            if let Some(ids) = self.edges_by_label.get_mut(&existing.label) {
                ids.remove(existing.id);
            }
            let (source, target, id) = (existing.source, existing.target, existing.id);
            self.remove_from_adjacency(source, target, id);
        } else {
            self.edge_count += 1;
        }
        let vector_offset = self.append_vectors(&edge.pending_vectors)?;
        self.indexed_vectors += edge.vector_count as usize;
        self.indexed_edge_vectors += edge.vector_count as usize;
        self.edges_by_label
            .entry(edge.label)
            .or_default()
            .insert(edge.id);
        self.outgoing.push(edge.source, edge.id);
        self.incoming.push(edge.target, edge.id);
        let properties = self.store_owned_properties(edge.properties)?;
        let stored = StoredEdge {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            label: edge.label,
            properties,
            vector_count: edge.vector_count,
            generation: nonzero_generation(edge.generation)?,
            vector_offset,
        };
        if self.mapped_edges.is_some() {
            self.edge_overlays.insert(edge.id, Some(stored));
        } else {
            self.edges[index] = Some(stored);
        }
        Ok(())
    }

    pub(super) fn remove_from_adjacency(&mut self, source: NodeId, target: NodeId, edge: EdgeId) {
        self.outgoing.remove(source, edge);
        self.incoming.remove(target, edge);
    }

    pub(super) fn store_owned_properties(
        &mut self,
        properties: Arc<[Property]>,
    ) -> Result<StoredProperties> {
        let reference = StoredProperties::owned(self.owned_properties.len())?;
        self.owned_properties.push(properties);
        Ok(reference)
    }

    pub(super) fn append_vectors(&mut self, vectors: &[f32]) -> Result<usize> {
        let dimension = self.dimension;
        let similarity = self.similarity;
        let (vector_data, offset) = match &mut self.vector_data {
            VectorData::Owned(vector_data) => {
                let offset = vector_data.len();
                (vector_data, offset)
            }
            VectorData::Mapped {
                byte_len,
                encoding,
                delta,
                ..
            } => {
                let base_float_count = *byte_len
                    / match encoding {
                        VectorEncoding::F32 => 4,
                        VectorEncoding::F16 => 2,
                    };
                let offset = base_float_count + delta.len();
                (delta, offset)
            }
        };
        if similarity == Similarity::Dot {
            vector_data.extend_from_slice(vectors);
            return Ok(offset);
        }
        vector_data.reserve(vectors.len());
        for vector in vectors.chunks_exact(dimension) {
            let start = vector_data.len();
            vector_data.extend_from_slice(vector);
            vector::normalize(&mut vector_data[start..])?;
        }
        Ok(offset)
    }
}
