use super::*;

impl Graph {
    pub(crate) fn vector_search(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<VectorHit>> {
        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(true)?;
        let mut top = TopK::new(limit);
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(node) = self.node_record(id) else {
                        continue;
                    };
                    if node.label != label {
                        continue;
                    }
                    self.score_element(
                        &query,
                        ElementRef::Node(node.id),
                        node.vector_offset,
                        node.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            } else {
                self.visit_node_vector_fields(|id, _label, vector_offset, vector_count| {
                    self.score_element(
                        &query,
                        ElementRef::Node(id),
                        vector_offset,
                        vector_count,
                        &scorer,
                        &mut top,
                    )
                })?;
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = label {
                for id in self.edges_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(edge) = self.edge_record(id) else {
                        continue;
                    };
                    if edge.label != label {
                        continue;
                    }
                    self.score_element(
                        &query,
                        ElementRef::Edge(edge.id),
                        edge.vector_offset,
                        edge.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            } else {
                self.visit_edge_vector_fields(|id, _label, vector_offset, vector_count| {
                    self.score_element(
                        &query,
                        ElementRef::Edge(id),
                        vector_offset,
                        vector_count,
                        &scorer,
                        &mut top,
                    )
                })?;
            }
        }
        Ok(top.finish())
    }

    /// Exact vector search over a compressed graph-derived candidate set. The
    /// set is applied before scoring, so graph constraints do not suffer the
    /// recall loss of post-filtering a global top-k result.
    pub(crate) fn vector_search_within(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        for id in allowed.node_ids() {
            let element = ElementRef::Node(id);
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            self.score_element(
                &query,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        for id in allowed.edge_ids() {
            let element = ElementRef::Edge(id);
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            self.score_element(
                &query,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        Ok(top.finish())
    }

    pub(crate) fn vector_search_within_approximate(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
        candidate_elements: usize,
    ) -> Result<Vec<VectorHit>> {
        if candidate_elements == 0 {
            return Err(Error::InvalidArgument(
                "approximate candidate budget must be greater than zero".into(),
            ));
        }
        let allowed_elements = usize::try_from(allowed.len()).unwrap_or(usize::MAX);
        let base_float_count = self.vector_data.base_float_count();
        if self.similarity != Similarity::Cosine
            || base_float_count == 0
            || candidate_elements >= allowed_elements
        {
            return self.vector_search_within(query, allowed, limit);
        }

        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        let candidates = index.candidate_entries(
            &query,
            VectorTarget::Both,
            None,
            Some(allowed),
            candidate_elements.max(limit),
        );
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            let Some((label, vector_offset, vector_count)) =
                self.element_vector_fields(candidate.element)
            else {
                continue;
            };
            if label != candidate.label
                || candidate.float_offset < vector_offset
                || candidate.float_offset >= vector_offset + vector_count as usize * self.dimension
                || !scored.insert(candidate.element)
            {
                continue;
            }
            self.score_element(
                &query,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }

        // The persisted sketch covers the immutable base. Search only allowed
        // WAL elements exhaustively, preserving read-your-writes semantics.
        for element in allowed
            .node_ids()
            .map(ElementRef::Node)
            .chain(allowed.edge_ids().map(ElementRef::Edge))
        {
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            if vector_offset < base_float_count || !scored.insert(element) {
                continue;
            }
            self.score_element(
                &query,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        Ok(top.finish())
    }

    pub(crate) fn vector_search_within_plan(&self, allowed: &ElementSet) -> VectorSearchPlan {
        let estimated_vectors = self.estimated_set_vector_count(allowed);
        let estimated_floats = estimated_vectors.saturating_mul(self.dimension);
        // Candidate-set scans gather non-contiguous records. Measured
        // crossovers differ substantially by vector width: reranking 5k of
        // 20k 768-D MoReVec rows wins, while exact still beats reranking 20k
        // of 100k 200-D VIBE rows. Keep this policy separate from contiguous
        // whole-column search and make the candidate fraction explicit.
        let candidate_vectors =
            adaptive_candidate_budget_for_set(estimated_vectors, self.dimension);
        let strategy = if self.similarity == Similarity::Cosine
            && self.vector_data.base_float_count() != 0
            && candidate_vectors < estimated_vectors
        {
            VectorSearchStrategy::BinarySketchRerank
        } else {
            VectorSearchStrategy::Exact
        };
        VectorSearchPlan {
            strategy,
            estimated_vectors,
            estimated_floats,
            candidate_vectors: if strategy == VectorSearchStrategy::Exact {
                estimated_vectors
            } else {
                candidate_vectors
            },
        }
    }

    pub(crate) fn vector_search_within_adaptive(
        &self,
        query: &[f32],
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        let plan = self.vector_search_within_plan(allowed);
        match plan.strategy {
            VectorSearchStrategy::Exact => self.vector_search_within(query, allowed, limit),
            VectorSearchStrategy::BinarySketchRerank => {
                self.vector_search_within_approximate(query, allowed, limit, plan.candidate_vectors)
            }
        }
    }

    /// Scores whole graph elements with weighted late interaction: each query
    /// vector takes its best matching vector facet on an element, then those
    /// per-query maxima are averaged by weight. This naturally supports token,
    /// chunk, structural/context, and multimodal facets in one embedding space.
    pub(crate) fn late_interaction_search(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<LateInteractionHit>> {
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(true)?;
        let mut top = TopK::new(limit);
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(node) = self.node_record(id) else {
                        continue;
                    };
                    if node.label == label {
                        self.score_late_interaction_element(
                            &queries,
                            &weights,
                            ElementRef::Node(node.id),
                            node.vector_offset,
                            node.vector_count,
                            &scorer,
                            &mut top,
                        )?;
                    }
                }
            } else {
                for node in self.node_records() {
                    self.score_late_interaction_element(
                        &queries,
                        &weights,
                        ElementRef::Node(node.id),
                        node.vector_offset,
                        node.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = label {
                for id in self.edges_by_label.get(&label).into_iter().flatten().rev() {
                    let Some(edge) = self.edge_record(id) else {
                        continue;
                    };
                    if edge.label == label {
                        self.score_late_interaction_element(
                            &queries,
                            &weights,
                            ElementRef::Edge(edge.id),
                            edge.vector_offset,
                            edge.vector_count,
                            &scorer,
                            &mut top,
                        )?;
                    }
                }
            } else {
                for edge in self.edge_records() {
                    self.score_late_interaction_element(
                        &queries,
                        &weights,
                        ElementRef::Edge(edge.id),
                        edge.vector_offset,
                        edge.vector_count,
                        &scorer,
                        &mut top,
                    )?;
                }
            }
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_within(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        for element in allowed
            .node_ids()
            .map(ElementRef::Node)
            .chain(allowed.edge_ids().map(ElementRef::Edge))
        {
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            self.score_late_interaction_element(
                &queries,
                &weights,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_within_approximate(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
        candidate_elements: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        if candidate_elements == 0 {
            return Err(Error::InvalidArgument(
                "late-interaction candidate budget must be greater than zero".into(),
            ));
        }
        let allowed_elements = usize::try_from(allowed.len()).unwrap_or(usize::MAX);
        let base_float_count = self.vector_data.base_float_count();
        if self.similarity != Similarity::Cosine
            || base_float_count == 0
            || candidate_elements >= allowed_elements
        {
            return self.late_interaction_search_within(queries, weights, allowed, limit);
        }
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        let candidates = index.candidate_elements_multivector(
            &queries,
            &weights,
            VectorTarget::Both,
            None,
            Some(allowed),
            candidate_elements.max(limit),
        );
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            if !self.sketch_entry_is_current(candidate) || !scored.insert(candidate.element) {
                continue;
            }
            let Some((vector_offset, vector_count)) = self.element_vector_span(candidate.element)
            else {
                continue;
            };
            self.score_late_interaction_element(
                &queries,
                &weights,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        for element in allowed
            .node_ids()
            .map(ElementRef::Node)
            .chain(allowed.edge_ids().map(ElementRef::Edge))
        {
            let Some((_label, vector_offset, vector_count)) = self.element_vector_fields(element)
            else {
                continue;
            };
            if vector_offset < base_float_count || !scored.insert(element) {
                continue;
            }
            self.score_late_interaction_element(
                &queries,
                &weights,
                element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_within_adaptive(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        allowed: &ElementSet,
        limit: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        let plan = self.vector_search_within_plan(allowed);
        match plan.strategy {
            VectorSearchStrategy::Exact => {
                self.late_interaction_search_within(queries, weights, allowed, limit)
            }
            VectorSearchStrategy::BinarySketchRerank => self
                .late_interaction_search_within_approximate(
                    queries,
                    weights,
                    allowed,
                    limit,
                    plan.candidate_vectors,
                ),
        }
    }

    pub(crate) fn late_interaction_search_approximate(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        candidate_elements: usize,
    ) -> Result<Vec<LateInteractionHit>> {
        if candidate_elements == 0 {
            return Err(Error::InvalidArgument(
                "late-interaction candidate budget must be greater than zero".into(),
            ));
        }
        let eligible_elements = self.eligible_element_upper_bound(target, label);
        if self.similarity != Similarity::Cosine
            || self.vector_data.base_float_count() == 0
            || candidate_elements >= eligible_elements
        {
            return self.late_interaction_search(queries, weights, target, limit, label);
        }
        let (queries, weights) =
            prepare_late_interaction_queries(queries, weights, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        let candidates = index.candidate_elements_multivector(
            &queries,
            &weights,
            target,
            label,
            None,
            candidate_elements.max(limit),
        );
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            if !self.sketch_entry_is_current(candidate) || !scored.insert(candidate.element) {
                continue;
            }
            let Some((vector_offset, vector_count)) = self.element_vector_span(candidate.element)
            else {
                continue;
            };
            self.score_late_interaction_element(
                &queries,
                &weights,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }

        // As with single-vector ANN, the immutable checkpoint index is paired
        // with an exhaustive mutable delta so new and replaced elements are
        // immediately visible.
        let base_float_count = self.vector_data.base_float_count();
        if target.accepts(ElementRef::Node(0))
            && (!self.node_overlays.is_empty() || !self.nodes.is_empty())
        {
            for node in self.node_records() {
                let element = ElementRef::Node(node.id);
                if node.vector_offset < base_float_count
                    || label.is_some_and(|label| node.label != label)
                    || !scored.insert(element)
                {
                    continue;
                }
                self.score_late_interaction_element(
                    &queries,
                    &weights,
                    element,
                    node.vector_offset,
                    node.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        if target.accepts(ElementRef::Edge(0))
            && (!self.edge_overlays.is_empty() || !self.edges.is_empty())
        {
            for edge in self.edge_records() {
                let element = ElementRef::Edge(edge.id);
                if edge.vector_offset < base_float_count
                    || label.is_some_and(|label| edge.label != label)
                    || !scored.insert(element)
                {
                    continue;
                }
                self.score_late_interaction_element(
                    &queries,
                    &weights,
                    element,
                    edge.vector_offset,
                    edge.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        self.finish_late_interaction_hits(&queries, &weights, &scorer, top.finish())
    }

    pub(crate) fn late_interaction_search_adaptive(
        &self,
        queries: &[Vec<f32>],
        weights: Option<&[f32]>,
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<LateInteractionHit>> {
        let plan = self.vector_search_plan(target, label);
        match plan.strategy {
            VectorSearchStrategy::Exact => {
                self.late_interaction_search(queries, weights, target, limit, label)
            }
            VectorSearchStrategy::BinarySketchRerank => self.late_interaction_search_approximate(
                queries,
                weights,
                target,
                limit,
                label,
                plan.candidate_vectors,
            ),
        }
    }

    pub(crate) fn vector_search_approximate(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        candidate_vectors: usize,
    ) -> Result<Vec<VectorHit>> {
        if candidate_vectors == 0 {
            return Err(Error::InvalidArgument(
                "approximate search candidate budget must be greater than zero".into(),
            ));
        }
        let base_float_count = self.vector_data.base_float_count();
        if self.similarity != Similarity::Cosine
            || base_float_count == 0
            || candidate_vectors >= self.eligible_element_upper_bound(target, label)
        {
            return self.vector_search(query, target, limit, label);
        }

        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let index = self
            .sketch_index
            .get_or_init(|| self.build_sketch_index().map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|message| Error::Corrupt(message.clone()))?;
        if candidate_vectors >= index.element_count() {
            return self.vector_search(&query, target, limit, label);
        }
        let candidates =
            index.candidate_entries(&query, target, label, None, candidate_vectors.max(limit));
        self.rerank_approximate_candidates(
            &query,
            target,
            limit,
            label,
            base_float_count,
            candidates,
        )
    }

    fn rerank_approximate_candidates(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
        base_float_count: usize,
        candidates: Vec<SketchEntry>,
    ) -> Result<Vec<VectorHit>> {
        let scorer = self.vector_data.scorer(false)?;
        let mut top = TopK::new(limit);
        let mut scored = HashSet::with_capacity(candidates.len());
        for candidate in candidates {
            let Some((current_label, vector_offset, vector_count)) =
                self.element_vector_fields(candidate.element)
            else {
                continue;
            };
            if current_label != candidate.label
                || candidate.float_offset < vector_offset
                || candidate.float_offset >= vector_offset + vector_count as usize * self.dimension
                || !scored.insert(candidate.element)
            {
                continue;
            }
            self.score_element(
                query,
                candidate.element,
                vector_offset,
                vector_count,
                &scorer,
                &mut top,
            )?;
        }

        // The checkpoint sketch is immutable. WAL vectors are deliberately
        // searched exhaustively, mirroring an LSM delta and guaranteeing that
        // fresh writes are immediately visible without rebuilding the base.
        if target.accepts(ElementRef::Node(0))
            && (!self.node_overlays.is_empty() || !self.nodes.is_empty())
        {
            for node in self.node_records() {
                if node.vector_offset < base_float_count
                    || label.is_some_and(|label| node.label != label)
                    || !scored.insert(ElementRef::Node(node.id))
                {
                    continue;
                }
                self.score_element(
                    query,
                    ElementRef::Node(node.id),
                    node.vector_offset,
                    node.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        if target.accepts(ElementRef::Edge(0))
            && (!self.edge_overlays.is_empty() || !self.edges.is_empty())
        {
            for edge in self.edge_records() {
                if edge.vector_offset < base_float_count
                    || label.is_some_and(|label| edge.label != label)
                    || !scored.insert(ElementRef::Edge(edge.id))
                {
                    continue;
                }
                self.score_element(
                    query,
                    ElementRef::Edge(edge.id),
                    edge.vector_offset,
                    edge.vector_count,
                    &scorer,
                    &mut top,
                )?;
            }
        }
        Ok(top.finish())
    }

    pub(crate) fn vector_search_plan(
        &self,
        target: VectorTarget,
        label: Option<LabelId>,
    ) -> VectorSearchPlan {
        let estimated_vectors = self.eligible_vector_count(target, label);
        let estimated_floats = estimated_vectors.saturating_mul(self.dimension);
        let target_covers_all_indexed = (self.indexed_node_vectors == 0
            || target.accepts(ElementRef::Node(0)))
            && (self.indexed_edge_vectors == 0 || target.accepts(ElementRef::Edge(0)));
        let high_fidelity_sketch = label.is_none()
            && target_covers_all_indexed
            && estimated_vectors == self.eligible_element_upper_bound(target, label);
        let candidate_vectors =
            adaptive_candidate_budget(estimated_vectors, self.dimension, high_fidelity_sketch);
        let strategy = if self.similarity == Similarity::Cosine
            && self.vector_data.base_float_count() != 0
            && candidate_vectors < estimated_vectors
        {
            VectorSearchStrategy::BinarySketchRerank
        } else {
            VectorSearchStrategy::Exact
        };
        VectorSearchPlan {
            strategy,
            estimated_vectors,
            estimated_floats,
            candidate_vectors: if strategy == VectorSearchStrategy::Exact {
                estimated_vectors
            } else {
                candidate_vectors
            },
        }
    }

    pub(crate) fn vector_search_adaptive(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        label: Option<LabelId>,
    ) -> Result<Vec<VectorHit>> {
        let plan = self.vector_search_plan(target, label);
        match plan.strategy {
            VectorSearchStrategy::Exact => self.vector_search(query, target, limit, label),
            VectorSearchStrategy::BinarySketchRerank => {
                self.vector_search_approximate(query, target, limit, label, plan.candidate_vectors)
            }
        }
    }

    fn eligible_vector_count(&self, target: VectorTarget, label: Option<LabelId>) -> usize {
        if label.is_none() {
            let mut count = 0usize;
            if target.accepts(ElementRef::Node(0)) {
                count = count.saturating_add(self.indexed_node_vectors);
            }
            if target.accepts(ElementRef::Edge(0)) {
                count = count.saturating_add(self.indexed_edge_vectors);
            }
            return count;
        }
        let mut count = 0usize;
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten().rev() {
                    if let Some(node) = self.node_record(id)
                        && node.label == label
                    {
                        count = count.saturating_add(node.vector_count as usize);
                    }
                }
            } else {
                count = count.saturating_add(
                    self.node_records()
                        .map(|node| node.vector_count as usize)
                        .sum(),
                );
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = label {
                for id in self.edges_by_label.get(&label).into_iter().flatten().rev() {
                    if let Some(edge) = self.edge_record(id)
                        && edge.label == label
                    {
                        count = count.saturating_add(edge.vector_count as usize);
                    }
                }
            } else {
                count = count.saturating_add(
                    self.edge_records()
                        .map(|edge| edge.vector_count as usize)
                        .sum(),
                );
            }
        }
        count
    }

    fn estimated_set_vector_count(&self, allowed: &ElementSet) -> usize {
        let node_average = if self.node_count == 0 {
            0
        } else {
            self.indexed_node_vectors.div_ceil(self.node_count)
        };
        let edge_average = if self.edge_count == 0 {
            0
        } else {
            self.indexed_edge_vectors.div_ceil(self.edge_count)
        };
        usize::try_from(allowed.node_len())
            .unwrap_or(usize::MAX)
            .saturating_mul(node_average)
            .saturating_add(
                usize::try_from(allowed.edge_len())
                    .unwrap_or(usize::MAX)
                    .saturating_mul(edge_average),
            )
    }

    pub(super) fn eligible_element_upper_bound(
        &self,
        target: VectorTarget,
        label: Option<LabelId>,
    ) -> usize {
        let mut count = 0usize;
        if target.accepts(ElementRef::Node(0)) {
            count = count.saturating_add(label.map_or(self.node_count, |label| {
                self.nodes_by_label
                    .get(&label)
                    .map_or(0, |ids| usize::try_from(ids.len()).unwrap_or(usize::MAX))
            }));
        }
        if target.accepts(ElementRef::Edge(0)) {
            count = count.saturating_add(label.map_or(self.edge_count, |label| {
                self.edges_by_label
                    .get(&label)
                    .map_or(0, |ids| usize::try_from(ids.len()).unwrap_or(usize::MAX))
            }));
        }
        count
    }

    fn build_sketch_index(&self) -> Result<BinarySketchIndex> {
        let base_float_count = self.vector_data.base_float_count();
        let base_vectors = base_float_count / self.dimension;
        let mut index = BinarySketchIndex::new(self.dimension, base_vectors);
        let mut vector = vec![0.0; self.dimension];
        let mut workspace = Vec::new();
        for node in self.node_records() {
            for vector_index in 0..node.vector_count {
                let float_offset = node.vector_offset + vector_index as usize * self.dimension;
                if float_offset + self.dimension > base_float_count {
                    continue;
                }
                self.vector_data.copy_vector(float_offset, &mut vector)?;
                index.push(
                    SketchEntry {
                        element: ElementRef::Node(node.id),
                        label: node.label,
                        float_offset,
                    },
                    &vector,
                    &mut workspace,
                );
            }
        }
        for edge in self.edge_records() {
            for vector_index in 0..edge.vector_count {
                let float_offset = edge.vector_offset + vector_index as usize * self.dimension;
                if float_offset + self.dimension > base_float_count {
                    continue;
                }
                self.vector_data.copy_vector(float_offset, &mut vector)?;
                index.push(
                    SketchEntry {
                        element: ElementRef::Edge(edge.id),
                        label: edge.label,
                        float_offset,
                    },
                    &vector,
                    &mut workspace,
                );
            }
        }
        Ok(index)
    }

    fn sketch_entry_is_current(&self, entry: SketchEntry) -> bool {
        self.element_vector_fields(entry.element).is_some_and(
            |(label, vector_offset, vector_count)| {
                label == entry.label
                    && entry.float_offset >= vector_offset
                    && entry.float_offset < vector_offset + vector_count as usize * self.dimension
            },
        )
    }

    pub(crate) fn semantic_paths(
        &self,
        query: &[f32],
        options: &SemanticPathOptions,
    ) -> Result<Vec<SemanticPathHit>> {
        let seeds = self.vector_search_adaptive(
            query,
            VectorTarget::Nodes,
            options.seed_count,
            options.seed_label,
        )?;
        let starts: Vec<_> = seeds
            .into_iter()
            .filter_map(|hit| match hit.element {
                ElementRef::Node(id) => Some(id),
                ElementRef::Edge(_) => None,
            })
            .collect();
        self.semantic_expand(query, &starts, options)
    }

    pub(crate) fn semantic_expand(
        &self,
        query: &[f32],
        starts: &[NodeId],
        options: &SemanticPathOptions,
    ) -> Result<Vec<SemanticPathHit>> {
        validate_semantic_options(options)?;
        let query = vector::prepare_query(query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut frontier = BinaryHeap::new();
        let mut best_scores = HashMap::new();
        let mut results: HashMap<NodeId, SemanticPathHit> = HashMap::new();

        for &node in starts {
            let Some(node_record) = self.node_record(node) else {
                return Err(Error::NotFound("start node", node));
            };
            let Some((seed_score, _)) = self.element_score(
                &query,
                node_record.vector_offset,
                node_record.vector_count,
                &scorer,
            )?
            else {
                continue;
            };
            if best_scores
                .get(&node)
                .is_none_or(|score| seed_score > *score)
            {
                best_scores.insert(node, seed_score);
                frontier.push(PathState {
                    seed: node,
                    node,
                    score: seed_score,
                    seed_score,
                    path: Vec::new(),
                });
            }
        }

        let mut expansions = 0;
        while let Some(state) = frontier.pop() {
            if best_scores
                .get(&state.node)
                .is_some_and(|score| state.score < *score)
            {
                continue;
            }
            if options.include_seeds || !state.path.is_empty() {
                results.insert(
                    state.node,
                    SemanticPathHit {
                        seed: state.seed,
                        node: state.node,
                        score: state.score,
                        seed_score: state.seed_score,
                        path: state.path.clone(),
                    },
                );
            }
            if state.path.len() >= options.max_hops || expansions >= options.max_expansions {
                continue;
            }
            expansions += 1;
            for edge in self.neighbors(
                state.node,
                options.direction,
                EdgeFilter {
                    label: options.edge_label,
                },
            )? {
                let next = if edge.source == state.node {
                    edge.target
                } else {
                    edge.source
                };
                if state.path.contains(&edge.id) {
                    continue;
                }
                let Some(next_node) = self.node_record(next) else {
                    continue;
                };
                let Some((edge_score, _)) =
                    self.element_score(&query, edge.vector_offset, edge.vector_count, &scorer)?
                else {
                    continue;
                };
                let Some((node_score, _)) = self.element_score(
                    &query,
                    next_node.vector_offset,
                    next_node.vector_count,
                    &scorer,
                )?
                else {
                    continue;
                };
                let semantic_score = (options.node_weight * node_score
                    + options.edge_weight * edge_score)
                    / (options.node_weight + options.edge_weight);
                let degree_penalty = options.degree_penalty * (self.degree(next) as f32).ln_1p();
                let score = (options.path_decay * state.score
                    + (1.0 - options.path_decay) * semantic_score)
                    * options.hop_penalty
                    - degree_penalty;
                if best_scores.get(&next).is_some_and(|best| score <= *best) {
                    continue;
                }
                best_scores.insert(next, score);
                let mut path = state.path.clone();
                path.push(edge.id);
                frontier.push(PathState {
                    seed: state.seed,
                    node: next,
                    score,
                    seed_score: state.seed_score,
                    path,
                });
            }
        }

        let mut results: Vec<_> = results.into_values().collect();
        results.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.node.cmp(&right.node))
        });
        results.truncate(options.limit);
        Ok(results)
    }

    fn score_element(
        &self,
        query: &[f32],
        element: ElementRef,
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
        top: &mut TopK,
    ) -> Result<()> {
        let best = self.element_score(query, vector_offset, vector_count, scorer)?;
        if let Some((score, vector_index)) = best {
            top.push(VectorHit {
                element,
                score,
                vector_index,
            });
        }
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "hot-loop scoring keeps query and mapped-vector state borrowed separately"
    )]
    fn score_late_interaction_element(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        element: ElementRef,
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
        top: &mut TopK,
    ) -> Result<()> {
        if let Some(score) =
            self.late_interaction_score(queries, weights, vector_offset, vector_count, scorer)?
        {
            top.push(VectorHit {
                element,
                score,
                vector_index: 0,
            });
        }
        Ok(())
    }

    fn late_interaction_score(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
    ) -> Result<Option<f32>> {
        if vector_count == 0 {
            return Ok(None);
        }
        let mut score = 0.0;
        for (query, weight) in queries.iter().zip(weights) {
            let Some((best, _)) = self.element_score(query, vector_offset, vector_count, scorer)?
            else {
                return Ok(None);
            };
            score += best * weight;
        }
        Ok(Some(score))
    }

    fn finish_late_interaction_hits(
        &self,
        queries: &[Vec<f32>],
        weights: &[f32],
        scorer: &VectorScorer<'_>,
        hits: Vec<VectorHit>,
    ) -> Result<Vec<LateInteractionHit>> {
        let mut result = Vec::with_capacity(hits.len());
        for hit in hits {
            let Some((vector_offset, vector_count)) = self.element_vector_span(hit.element) else {
                continue;
            };
            let mut matched_vector_indices = Vec::with_capacity(queries.len());
            let mut score = 0.0;
            for (query, weight) in queries.iter().zip(weights) {
                let Some((best, vector_index)) =
                    self.element_score(query, vector_offset, vector_count, scorer)?
                else {
                    continue;
                };
                score += best * weight;
                matched_vector_indices.push(vector_index);
            }
            result.push(LateInteractionHit {
                element: hit.element,
                score,
                matched_vector_indices,
            });
        }
        Ok(result)
    }

    fn element_vector_span(&self, element: ElementRef) -> Option<(usize, u32)> {
        self.element_vector_fields(element)
            .map(|(_label, offset, count)| (offset, count))
    }

    #[inline]
    fn element_vector_fields(&self, element: ElementRef) -> Option<(LabelId, usize, u32)> {
        match element {
            ElementRef::Node(id) => {
                if let Some(record) = self.node_overlays.get(&id) {
                    return record
                        .map(|record| (record.label, record.vector_offset, record.vector_count));
                }
                if let Some(record) = self.nodes.get(id as usize).and_then(|record| *record) {
                    return Some((record.label, record.vector_offset, record.vector_count));
                }
                self.mapped_nodes.as_ref()?.vector_fields(id)
            }
            ElementRef::Edge(id) => {
                if let Some(record) = self.edge_overlays.get(&id) {
                    return record
                        .map(|record| (record.label, record.vector_offset, record.vector_count));
                }
                if let Some(record) = self.edges.get(id as usize).and_then(|record| *record) {
                    return Some((record.label, record.vector_offset, record.vector_count));
                }
                self.mapped_edges.as_ref()?.vector_fields(id)
            }
        }
    }

    pub(super) fn element_score(
        &self,
        query: &[f32],
        vector_offset: usize,
        vector_count: u32,
        scorer: &VectorScorer<'_>,
    ) -> Result<Option<(f32, u32)>> {
        let mut best = None;
        for vector_index in 0..vector_count {
            let start = vector_offset + vector_index as usize * self.dimension;
            let score = scorer.score(query, start)?;
            if best.is_none_or(|(current, _)| score > current) {
                best = Some((score, vector_index));
            }
        }
        Ok(best)
    }

    fn degree(&self, node: NodeId) -> usize {
        let mut edges = HashSet::new();
        self.collect_incident_ids(node, &mut edges);
        edges.len()
    }

    pub(crate) fn element_vector(
        &self,
        offset: usize,
        vector_count: u32,
        index: usize,
    ) -> Result<Option<&[f32]>> {
        if index >= vector_count as usize {
            return Ok(None);
        }
        let Some(start) = index
            .checked_mul(self.dimension)
            .and_then(|index| offset.checked_add(index))
        else {
            return Ok(None);
        };
        self.vector_data.f32_range(start, self.dimension)
    }

    pub(crate) fn node_vector(&self, id: NodeId, index: usize) -> Result<Option<&[f32]>> {
        let Some(node) = self.node_record(id) else {
            return Ok(None);
        };
        self.element_vector(node.vector_offset, node.vector_count, index)
    }

    pub(crate) fn edge_vector(&self, id: EdgeId, index: usize) -> Result<Option<&[f32]>> {
        let Some(edge) = self.edge_record(id) else {
            return Ok(None);
        };
        self.element_vector(edge.vector_offset, edge.vector_count, index)
    }

    fn element_vector_owned(
        &self,
        offset: usize,
        vector_count: u32,
        index: usize,
    ) -> Result<Option<Vec<f32>>> {
        if index >= vector_count as usize {
            return Ok(None);
        }
        let Some(start) = index
            .checked_mul(self.dimension)
            .and_then(|index| offset.checked_add(index))
        else {
            return Ok(None);
        };
        let mut vector = vec![0.0; self.dimension];
        self.vector_data.copy_vector(start, &mut vector)?;
        Ok(Some(vector))
    }

    pub(crate) fn node_vector_owned(&self, id: NodeId, index: usize) -> Result<Option<Vec<f32>>> {
        let Some(node) = self.node_record(id) else {
            return Ok(None);
        };
        self.element_vector_owned(node.vector_offset, node.vector_count, index)
    }

    pub(crate) fn edge_vector_owned(&self, id: EdgeId, index: usize) -> Result<Option<Vec<f32>>> {
        let Some(edge) = self.edge_record(id) else {
            return Ok(None);
        };
        self.element_vector_owned(edge.vector_offset, edge.vector_count, index)
    }
}
