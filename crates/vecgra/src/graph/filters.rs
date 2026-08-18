use super::*;

impl Graph {
    pub(crate) fn nodes_with_label(&self, label: LabelId) -> Vec<Node> {
        let Some(ids) = self.nodes_by_label.get(&label) else {
            return Vec::new();
        };
        ids.iter()
            .rev()
            .filter_map(|id| {
                let node = self.node(id)?;
                (node.label == label).then_some(node)
            })
            .collect()
    }

    pub(crate) fn elements_with_label(&self, label: LabelId, target: VectorTarget) -> ElementSet {
        let mut result = ElementSet::new();
        if target.accepts(ElementRef::Node(0))
            && let Some(ids) = self.nodes_by_label.get(&label)
        {
            result.clone_nodes_from(ids);
        }
        if target.accepts(ElementRef::Edge(0))
            && let Some(ids) = self.edges_by_label.get(&label)
        {
            result.clone_edges_from(ids);
        }
        result
    }

    pub(crate) fn element_filter_plan(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
    ) -> ElementFilterPlan {
        let full_count = if target.accepts(ElementRef::Node(0)) {
            self.node_count
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_count
        } else {
            0
        });
        let label_count = filter.label.map(|label| {
            let mut count = 0u64;
            if target.accepts(ElementRef::Node(0)) {
                count = count.saturating_add(
                    self.nodes_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            if target.accepts(ElementRef::Edge(0)) {
                count = count.saturating_add(
                    self.edges_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            usize::try_from(count).unwrap_or(usize::MAX)
        });
        let overlay_count = if target.accepts(ElementRef::Node(0)) {
            self.node_overlays.len()
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_overlays.len()
        } else {
            0
        });
        let property = self.mapped_property_index.as_ref().and_then(|index| {
            filter
                .properties
                .iter()
                .enumerate()
                .map(|(predicate, (key, value))| {
                    let range_len = crate::codec::numeric_value_index_key(value)
                        .and_then(|(tag, sortable)| {
                            self.mapped_numeric_property_index.as_ref().map(|index| {
                                index
                                    .range(
                                        *key,
                                        tag,
                                        Bound::Included(sortable),
                                        Bound::Included(sortable),
                                    )
                                    .len()
                            })
                        })
                        .unwrap_or_else(|| {
                            index
                                .range(*key, crate::codec::property_value_fingerprint(value))
                                .len()
                        });
                    (predicate, range_len.saturating_add(overlay_count))
                })
                .min_by_key(|(_, count)| *count)
        });

        match (label_count, property) {
            (Some(label), Some((predicate, property))) if property < label => ElementFilterPlan {
                strategy: ElementFilterStrategy::PropertyPosting,
                candidate_upper_bound: property,
                property_predicate: Some(predicate),
            },
            (Some(label), _) => ElementFilterPlan {
                strategy: ElementFilterStrategy::LabelPosting,
                candidate_upper_bound: label,
                property_predicate: None,
            },
            (None, Some((predicate, property))) => ElementFilterPlan {
                strategy: ElementFilterStrategy::PropertyPosting,
                candidate_upper_bound: property,
                property_predicate: Some(predicate),
            },
            (None, None) => ElementFilterPlan {
                strategy: ElementFilterStrategy::FullScan,
                candidate_upper_bound: full_count,
                property_predicate: None,
            },
        }
    }

    /// Evaluates an exact label/property predicate into the same compressed
    /// candidate representation consumed by traversal and vector operators.
    pub(crate) fn elements_matching(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
    ) -> ElementSet {
        let plan = self.element_filter_plan(target, filter);
        if plan.strategy == ElementFilterStrategy::PropertyPosting {
            let predicate = plan
                .property_predicate
                .expect("property posting plans identify their predicate");
            let (key, value) = &filter.properties[predicate];
            if let Some((tag, sortable)) = crate::codec::numeric_value_index_key(value)
                && let Some(index) = self.mapped_numeric_property_index.as_ref()
            {
                let range = index.range(
                    *key,
                    tag,
                    Bound::Included(sortable),
                    Bound::Included(sortable),
                );
                return self.elements_matching_numeric_property_range(target, filter, index, range);
            }
            let index = self
                .mapped_property_index
                .as_ref()
                .expect("property posting plans require a mapped index");
            let range = index.range(*key, crate::codec::property_value_fingerprint(value));
            return self.elements_matching_property_range(target, filter, index, range);
        }

        let mut result = ElementSet::new();
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = filter.label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten() {
                    if let Some(node) = self.node_record(id)
                        && stored_element_matches(
                            node.label,
                            node.properties,
                            filter,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_node(id);
                    }
                }
            } else {
                for node in self.node_records() {
                    if stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_node(node.id);
                    }
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = filter.label {
                for id in self.edges_by_label.get(&label).into_iter().flatten() {
                    if let Some(edge) = self.edge_record(id)
                        && stored_element_matches(
                            edge.label,
                            edge.properties,
                            filter,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_edge(id);
                    }
                }
            } else {
                for edge in self.edge_records() {
                    if stored_element_matches(
                        edge.label,
                        edge.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_edge(edge.id);
                    }
                }
            }
        }
        result
    }

    fn elements_matching_property_range(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
        index: &MappedPropertyIndex,
        range: std::ops::Range<usize>,
    ) -> ElementSet {
        let mut result = ElementSet::new();
        for ordinal in range {
            let entry = index.entry_at(ordinal);
            if entry.kind == 0 && target.accepts(ElementRef::Node(entry.id)) {
                if let Some(node) = self.node_record(entry.id)
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(entry.id);
                }
            } else if entry.kind == 1
                && target.accepts(ElementRef::Edge(entry.id))
                && let Some(edge) = self.edge_record(entry.id)
                && stored_element_matches(
                    edge.label,
                    edge.properties,
                    filter,
                    self.snapshot_map.as_deref(),
                    &self.owned_properties,
                )
            {
                result.insert_edge(entry.id);
            }
        }

        // The mapped posting table describes the immutable checkpoint. WAL
        // overlays are small and scanned exactly so property changes become
        // visible immediately without synchronous index maintenance.
        if target.accepts(ElementRef::Node(0)) {
            for (&id, record) in &self.node_overlays {
                if let Some(node) = record
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(id);
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            for (&id, record) in &self.edge_overlays {
                if let Some(edge) = record
                    && stored_element_matches(
                        edge.label,
                        edge.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_edge(id);
                }
            }
        }
        result
    }

    fn elements_matching_numeric_property_range(
        &self,
        target: VectorTarget,
        filter: &ElementFilter,
        index: &MappedNumericPropertyIndex,
        range: std::ops::Range<usize>,
    ) -> ElementSet {
        let mut result = ElementSet::new();
        for ordinal in range {
            let entry = index.entry_at(ordinal);
            if entry.kind == 0 && target.accepts(ElementRef::Node(entry.id)) {
                if let Some(node) = self.node_record(entry.id)
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(entry.id);
                }
            } else if entry.kind == 1
                && target.accepts(ElementRef::Edge(entry.id))
                && let Some(edge) = self.edge_record(entry.id)
                && stored_element_matches(
                    edge.label,
                    edge.properties,
                    filter,
                    self.snapshot_map.as_deref(),
                    &self.owned_properties,
                )
            {
                result.insert_edge(entry.id);
            }
        }
        if target.accepts(ElementRef::Node(0)) {
            for (&id, record) in &self.node_overlays {
                if let Some(node) = record
                    && stored_element_matches(
                        node.label,
                        node.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_node(id);
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            for (&id, record) in &self.edge_overlays {
                if let Some(edge) = record
                    && stored_element_matches(
                        edge.label,
                        edge.properties,
                        filter,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    )
                {
                    result.insert_edge(id);
                }
            }
        }
        result
    }

    pub(crate) fn numeric_range_plan(
        &self,
        target: VectorTarget,
        filter: &NumericRangeFilter,
    ) -> Result<NumericRangePlan> {
        let prepared = prepare_numeric_range(filter)?;
        let full_count = if target.accepts(ElementRef::Node(0)) {
            self.node_count
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_count
        } else {
            0
        });
        let label_count = filter.label.map(|label| {
            let mut count = 0u64;
            if target.accepts(ElementRef::Node(0)) {
                count = count.saturating_add(
                    self.nodes_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            if target.accepts(ElementRef::Edge(0)) {
                count = count.saturating_add(
                    self.edges_by_label
                        .get(&label)
                        .map_or(0, RoaringTreemap::len),
                );
            }
            usize::try_from(count).unwrap_or(usize::MAX)
        });
        let overlay_count = if target.accepts(ElementRef::Node(0)) {
            self.node_overlays.len()
        } else {
            0
        }
        .saturating_add(if target.accepts(ElementRef::Edge(0)) {
            self.edge_overlays.len()
        } else {
            0
        });
        let numeric_count = self.mapped_numeric_property_index.as_ref().map(|index| {
            index
                .range(filter.key, prepared.tag, prepared.lower, prepared.upper)
                .len()
                .saturating_add(overlay_count)
        });
        Ok(match (label_count, numeric_count) {
            (Some(label), Some(numeric)) if numeric < label => NumericRangePlan {
                strategy: NumericRangeStrategy::NumericPosting,
                candidate_upper_bound: numeric,
            },
            (Some(label), _) => NumericRangePlan {
                strategy: NumericRangeStrategy::LabelPosting,
                candidate_upper_bound: label,
            },
            (None, Some(numeric)) => NumericRangePlan {
                strategy: NumericRangeStrategy::NumericPosting,
                candidate_upper_bound: numeric,
            },
            (None, None) => NumericRangePlan {
                strategy: NumericRangeStrategy::FullScan,
                candidate_upper_bound: full_count,
            },
        })
    }

    /// Evaluates a same-typed integer or floating-point range without
    /// hydrating mapped records. Checkpoint postings are reconciled against WAL
    /// overlays so inserts and property changes are immediately visible.
    pub(crate) fn elements_matching_numeric_range(
        &self,
        target: VectorTarget,
        filter: &NumericRangeFilter,
    ) -> Result<ElementSet> {
        let prepared = prepare_numeric_range(filter)?;
        let plan = self.numeric_range_plan(target, filter)?;
        if plan.strategy == NumericRangeStrategy::NumericPosting {
            let index = self
                .mapped_numeric_property_index
                .as_ref()
                .expect("numeric posting plans require a mapped index");
            let range = index.range(filter.key, prepared.tag, prepared.lower, prepared.upper);
            let mut result = ElementSet::new();
            for ordinal in range {
                let entry = index.entry_at(ordinal);
                if entry.kind == 0 && target.accepts(ElementRef::Node(entry.id)) {
                    if let Some(overlay) = self.node_overlays.get(&entry.id) {
                        if let Some(node) = overlay
                            && stored_element_matches_numeric_range(
                                node.label,
                                node.properties,
                                filter,
                                prepared,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            )
                        {
                            result.insert_node(entry.id);
                        }
                    } else if self
                        .node_record(entry.id)
                        .is_some_and(|node| filter.label.is_none_or(|label| node.label == label))
                    {
                        // Unlike equality fingerprints, numeric sort keys are
                        // exact and collision-free. Metadata CRC + open-time
                        // index validation lets immutable rows skip reparsing
                        // their property blobs here.
                        result.insert_node(entry.id);
                    }
                } else if entry.kind == 1 && target.accepts(ElementRef::Edge(entry.id)) {
                    if let Some(overlay) = self.edge_overlays.get(&entry.id) {
                        if let Some(edge) = overlay
                            && stored_element_matches_numeric_range(
                                edge.label,
                                edge.properties,
                                filter,
                                prepared,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            )
                        {
                            result.insert_edge(entry.id);
                        }
                    } else if self
                        .edge_record(entry.id)
                        .is_some_and(|edge| filter.label.is_none_or(|label| edge.label == label))
                    {
                        result.insert_edge(entry.id);
                    }
                }
            }
            if target.accepts(ElementRef::Node(0)) {
                for (&id, record) in &self.node_overlays {
                    if let Some(node) = record
                        && stored_element_matches_numeric_range(
                            node.label,
                            node.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_node(id);
                    }
                }
            }
            if target.accepts(ElementRef::Edge(0)) {
                for (&id, record) in &self.edge_overlays {
                    if let Some(edge) = record
                        && stored_element_matches_numeric_range(
                            edge.label,
                            edge.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_edge(id);
                    }
                }
            }
            return Ok(result);
        }

        let mut result = ElementSet::new();
        if target.accepts(ElementRef::Node(0)) {
            if let Some(label) = filter.label {
                for id in self.nodes_by_label.get(&label).into_iter().flatten() {
                    if let Some(node) = self.node_record(id)
                        && stored_element_matches_numeric_range(
                            node.label,
                            node.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_node(id);
                    }
                }
            } else {
                for node in self.node_records() {
                    if stored_element_matches_numeric_range(
                        node.label,
                        node.properties,
                        filter,
                        prepared,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_node(node.id);
                    }
                }
            }
        }
        if target.accepts(ElementRef::Edge(0)) {
            if let Some(label) = filter.label {
                for id in self.edges_by_label.get(&label).into_iter().flatten() {
                    if let Some(edge) = self.edge_record(id)
                        && stored_element_matches_numeric_range(
                            edge.label,
                            edge.properties,
                            filter,
                            prepared,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        )
                    {
                        result.insert_edge(id);
                    }
                }
            } else {
                for edge in self.edge_records() {
                    if stored_element_matches_numeric_range(
                        edge.label,
                        edge.properties,
                        filter,
                        prepared,
                        self.snapshot_map.as_deref(),
                        &self.owned_properties,
                    ) {
                        result.insert_edge(edge.id);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Executes scalar predicates into compressed sets, intersects them, and
    /// only then chooses exact or sketch/rerank vector execution. Predicate
    /// plans are computed first and evaluated from the lowest conservative
    /// candidate bound to minimize the live intermediate set.
    pub(crate) fn vector_search_filtered_adaptive(
        &self,
        query: &[f32],
        target: VectorTarget,
        limit: usize,
        equality: Option<&ElementFilter>,
        numeric_ranges: &[NumericRangeFilter],
    ) -> Result<FilteredVectorSearchResult> {
        #[derive(Clone, Copy)]
        enum Constraint {
            Equality,
            Numeric(usize),
        }

        let equality_plan = equality.map(|filter| self.element_filter_plan(target, filter));
        let numeric_range_plans = numeric_ranges
            .iter()
            .map(|filter| self.numeric_range_plan(target, filter))
            .collect::<Result<Vec<_>>>()?;
        let mut execution =
            Vec::with_capacity(numeric_ranges.len() + usize::from(equality.is_some()));
        if let Some(plan) = equality_plan {
            execution.push((plan.candidate_upper_bound, Constraint::Equality));
        }
        execution.extend(
            numeric_range_plans
                .iter()
                .enumerate()
                .map(|(index, plan)| (plan.candidate_upper_bound, Constraint::Numeric(index))),
        );
        execution.sort_unstable_by_key(|(candidate_upper_bound, _)| *candidate_upper_bound);

        let mut candidates: Option<ElementSet> = None;
        for (_, constraint) in execution {
            let current = match constraint {
                Constraint::Equality => {
                    self.elements_matching(target, equality.expect("planned equality filter"))
                }
                Constraint::Numeric(index) => {
                    self.elements_matching_numeric_range(target, &numeric_ranges[index])?
                }
            };
            candidates = Some(match candidates {
                Some(previous) => previous.intersection(&current),
                None => current,
            });
            if candidates.as_ref().is_some_and(ElementSet::is_empty) {
                break;
            }
        }

        if let Some(candidates) = candidates {
            let vector_plan = self.vector_search_within_plan(&candidates);
            let hits = match vector_plan.strategy {
                VectorSearchStrategy::Exact => {
                    self.vector_search_within(query, &candidates, limit)?
                }
                VectorSearchStrategy::BinarySketchRerank => self.vector_search_within_approximate(
                    query,
                    &candidates,
                    limit,
                    vector_plan.candidate_vectors,
                )?,
            };
            Ok(FilteredVectorSearchResult {
                hits,
                candidate_elements: candidates.len(),
                equality_plan,
                numeric_range_plans,
                vector_plan,
            })
        } else {
            let vector_plan = self.vector_search_plan(target, None);
            let hits = match vector_plan.strategy {
                VectorSearchStrategy::Exact => self.vector_search(query, target, limit, None)?,
                VectorSearchStrategy::BinarySketchRerank => self.vector_search_approximate(
                    query,
                    target,
                    limit,
                    None,
                    vector_plan.candidate_vectors,
                )?,
            };
            Ok(FilteredVectorSearchResult {
                hits,
                candidate_elements: self.eligible_element_upper_bound(target, None) as u64,
                equality_plan,
                numeric_range_plans,
                vector_plan,
            })
        }
    }
}
