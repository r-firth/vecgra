use super::*;

impl Graph {
    pub(crate) fn expand_element_set(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        filter: EdgeFilter,
    ) -> Result<ElementSet> {
        let mut result = ElementSet::new();
        for node in seeds.node_ids() {
            self.visit_neighbors(node, direction, filter, |neighbor, edge| {
                result.insert_node(neighbor);
                result.insert_edge(edge);
            })?;
        }
        Ok(result)
    }

    /// Computes the exact bounded neighborhood of a typed candidate set.
    ///
    /// Seed nodes are not included in the result. Every newly reached node is
    /// expanded at most once (at its shortest hop depth), while every matching
    /// edge encountered from a frontier is retained. This gives cyclic and
    /// parallel-edge graphs deterministic set semantics without application-side
    /// frontier materialization.
    pub(crate) fn expand_element_set_hops(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        filter: EdgeFilter,
        max_hops: usize,
    ) -> Result<ElementSet> {
        let mut result = ElementSet::new();
        if max_hops == 0 || seeds.node_len() == 0 {
            return Ok(result);
        }

        let mut visited = ElementSet::new();
        let mut frontier = ElementSet::new();
        for node in seeds.node_ids() {
            visited.insert_node(node);
            frontier.insert_node(node);
        }

        for _ in 0..max_hops {
            let mut next = ElementSet::new();
            for node in frontier.node_ids() {
                self.visit_neighbors(node, direction, filter, |neighbor, edge| {
                    result.insert_edge(edge);
                    if !visited.contains(ElementRef::Node(neighbor)) {
                        visited.insert_node(neighbor);
                        next.insert_node(neighbor);
                        result.insert_node(neighbor);
                    }
                })?;
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        Ok(result)
    }

    /// Computes a node-only bounded neighborhood for graph-range retrieval.
    ///
    /// Unlike `expand_element_set_hops`, this does not retain every traversed
    /// edge. That distinction is material for broad ranges where edge IDs can
    /// outnumber reachable nodes several times over.
    pub(crate) fn nodes_within_hops(
        &self,
        seeds: &ElementSet,
        direction: Direction,
        edge_filter: EdgeFilter,
        max_hops: usize,
        include_seeds: bool,
        node_filter: Option<&ElementFilter>,
    ) -> Result<ElementSet> {
        let mut visited = ElementSet::new();
        for node in seeds.node_ids() {
            if !self.has_node(node) {
                return Err(Error::NotFound("node", node));
            }
            visited.insert_node(node);
        }
        let mut frontier = visited.clone();
        let mut result = if include_seeds {
            visited.clone()
        } else {
            ElementSet::new()
        };

        for _ in 0..max_hops {
            let mut next = ElementSet::new();
            for node in frontier.node_ids() {
                self.visit_neighbors(node, direction, edge_filter, |neighbor, _edge| {
                    if !visited.contains(ElementRef::Node(neighbor)) {
                        visited.insert_node(neighbor);
                        next.insert_node(neighbor);
                        result.insert_node(neighbor);
                    }
                })?;
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        if let Some(filter) = node_filter {
            let matching = self.elements_matching(VectorTarget::Nodes, filter);
            result = result.intersection(&matching);
        }
        Ok(result)
    }

    /// Finds an exact unweighted shortest path within explicit hop and work
    /// bounds. Frontier and adjacency order are normalized by stable IDs so a
    /// graph with multiple shortest paths returns the same evidence chain
    /// before and after checkpoint compaction.
    pub(crate) fn shortest_path(
        &self,
        start: NodeId,
        end: NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        if !self.has_node(start) {
            return Err(Error::NotFound("node", start));
        }
        if !self.has_node(end) {
            return Err(Error::NotFound("node", end));
        }
        if start == end {
            return Ok(ShortestPathResult {
                path: Some(ShortestPath {
                    nodes: vec![start],
                    edges: Vec::new(),
                }),
                strategy: ShortestPathStrategy::BreadthFirst,
                termination: ShortestPathTermination::Found,
                visited_nodes: 1,
                start_expanded_nodes: 0,
                end_expanded_nodes: 0,
                expanded_nodes: 0,
                examined_relationships: 0,
            });
        }
        if options.max_hops == 0 {
            return Ok(ShortestPathResult {
                path: None,
                strategy: ShortestPathStrategy::BreadthFirst,
                termination: ShortestPathTermination::NotFoundWithinHops,
                visited_nodes: 1,
                start_expanded_nodes: 0,
                end_expanded_nodes: 0,
                expanded_nodes: 0,
                examined_relationships: 0,
            });
        }

        if options.max_hops == 1 {
            self.shortest_path_breadth_first(start, end, options)
        } else {
            self.shortest_path_bidirectional(start, end, options)
        }
    }

    fn shortest_path_breadth_first(
        &self,
        start: NodeId,
        end: NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        let mut visited = HashSet::from([start]);
        let mut parents: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut frontier = vec![start];
        let mut expanded_nodes = 0;
        let mut examined_relationships = 0;

        for _depth in 0..options.max_hops {
            frontier.sort_unstable();
            let mut next = Vec::new();
            for node in frontier {
                if expanded_nodes == options.max_expansions {
                    return Ok(ShortestPathResult {
                        path: None,
                        strategy: ShortestPathStrategy::BreadthFirst,
                        termination: ShortestPathTermination::ExpansionLimit,
                        visited_nodes: visited.len(),
                        start_expanded_nodes: expanded_nodes,
                        end_expanded_nodes: 0,
                        expanded_nodes,
                        examined_relationships,
                    });
                }
                expanded_nodes += 1;
                let mut adjacent = Vec::new();
                self.visit_neighbors(
                    node,
                    options.direction,
                    options.edge_filter,
                    |neighbor, edge| adjacent.push((neighbor, edge)),
                )?;
                adjacent.sort_unstable();
                examined_relationships = examined_relationships.saturating_add(adjacent.len());
                for (neighbor, edge) in adjacent {
                    if !visited.insert(neighbor) {
                        continue;
                    }
                    parents.insert(neighbor, (node, edge));
                    if neighbor == end {
                        let path = reconstruct_shortest_path(start, end, &parents);
                        return Ok(ShortestPathResult {
                            path: Some(path),
                            strategy: ShortestPathStrategy::BreadthFirst,
                            termination: ShortestPathTermination::Found,
                            visited_nodes: visited.len(),
                            start_expanded_nodes: expanded_nodes,
                            end_expanded_nodes: 0,
                            expanded_nodes,
                            examined_relationships,
                        });
                    }
                    next.push(neighbor);
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }

        Ok(ShortestPathResult {
            path: None,
            strategy: ShortestPathStrategy::BreadthFirst,
            termination: ShortestPathTermination::NotFoundWithinHops,
            visited_nodes: visited.len(),
            start_expanded_nodes: expanded_nodes,
            end_expanded_nodes: 0,
            expanded_nodes,
            examined_relationships,
        })
    }

    fn shortest_path_bidirectional(
        &self,
        start: NodeId,
        end: NodeId,
        options: &ShortestPathOptions,
    ) -> Result<ShortestPathResult> {
        let mut forward_depths = HashMap::from([(start, 0usize)]);
        let mut reverse_depths = HashMap::from([(end, 0usize)]);
        let mut forward_parents: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut reverse_next: HashMap<NodeId, (NodeId, EdgeId)> = HashMap::new();
        let mut forward_frontier = vec![start];
        let mut reverse_frontier = vec![end];
        let mut forward_depth = 0usize;
        let mut reverse_depth = 0usize;
        let mut best_length = None;
        let mut expanded_nodes = 0usize;
        let mut start_expanded_nodes = 0usize;
        let mut end_expanded_nodes = 0usize;
        let mut examined_relationships = 0usize;

        loop {
            let proven_length =
                best_length.filter(|&length| forward_depth.saturating_add(reverse_depth) >= length);
            if let Some(length) = proven_length {
                return Ok(ShortestPathResult {
                    path: best_bidirectional_path(
                        start,
                        end,
                        length,
                        &forward_depths,
                        &reverse_depths,
                        &forward_parents,
                        &reverse_next,
                    ),
                    strategy: ShortestPathStrategy::BidirectionalBreadthFirst,
                    termination: ShortestPathTermination::Found,
                    visited_nodes: visited_union_len(&forward_depths, &reverse_depths),
                    start_expanded_nodes,
                    end_expanded_nodes,
                    expanded_nodes,
                    examined_relationships,
                });
            }
            if forward_depth.saturating_add(reverse_depth) >= options.max_hops
                || forward_frontier.is_empty()
                || reverse_frontier.is_empty()
            {
                let path = best_length.and_then(|length| {
                    best_bidirectional_path(
                        start,
                        end,
                        length,
                        &forward_depths,
                        &reverse_depths,
                        &forward_parents,
                        &reverse_next,
                    )
                });
                return Ok(ShortestPathResult {
                    termination: if path.is_some() {
                        ShortestPathTermination::Found
                    } else {
                        ShortestPathTermination::NotFoundWithinHops
                    },
                    path,
                    strategy: ShortestPathStrategy::BidirectionalBreadthFirst,
                    visited_nodes: visited_union_len(&forward_depths, &reverse_depths),
                    start_expanded_nodes,
                    end_expanded_nodes,
                    expanded_nodes,
                    examined_relationships,
                });
            }

            // A bidirectional proof advances only after a complete layer. If
            // exactly one frontier fits in the remaining expansion budget,
            // prefer it even when its adjacency estimate is more expensive;
            // partially expanding the cheaper layer could otherwise exhaust
            // the budget without proving a path already visible from the
            // other side. When both (or neither) fit, score the complete next
            // layer by node expansions plus a cheap upper bound on adjacency
            // reads, then retain forward order as the deterministic tie-break.
            // The estimate is conservative across mapped CSR and WAL overlays;
            // filters are applied during the actual expansion.
            let reverse_search_direction = reverse_direction(options.direction);
            let remaining_expansions = options.max_expansions.saturating_sub(expanded_nodes);
            let forward_fits = forward_frontier.len() <= remaining_expansions;
            let reverse_fits = reverse_frontier.len() <= remaining_expansions;
            let expand_forward = match (forward_fits, reverse_fits) {
                (true, false) => true,
                (false, true) => false,
                (true, true) | (false, false) => {
                    let forward_work =
                        self.frontier_work_upper_bound(&forward_frontier, options.direction);
                    let reverse_work =
                        self.frontier_work_upper_bound(&reverse_frontier, reverse_search_direction);
                    forward_work < reverse_work
                        || (forward_work == reverse_work
                            && forward_frontier.len() <= reverse_frontier.len())
                }
            };
            let (frontier, own_depths, other_depths, parents, direction) = if expand_forward {
                (
                    &mut forward_frontier,
                    &mut forward_depths,
                    &reverse_depths,
                    &mut forward_parents,
                    options.direction,
                )
            } else {
                (
                    &mut reverse_frontier,
                    &mut reverse_depths,
                    &forward_depths,
                    &mut reverse_next,
                    reverse_search_direction,
                )
            };
            frontier.sort_unstable();
            let mut next = Vec::new();
            for node in std::mem::take(frontier) {
                if expanded_nodes == options.max_expansions {
                    return Ok(ShortestPathResult {
                        path: None,
                        strategy: ShortestPathStrategy::BidirectionalBreadthFirst,
                        termination: ShortestPathTermination::ExpansionLimit,
                        visited_nodes: visited_union_len(&forward_depths, &reverse_depths),
                        start_expanded_nodes,
                        end_expanded_nodes,
                        expanded_nodes,
                        examined_relationships,
                    });
                }
                expanded_nodes += 1;
                if expand_forward {
                    start_expanded_nodes += 1;
                } else {
                    end_expanded_nodes += 1;
                }
                let mut adjacent = Vec::new();
                self.visit_neighbors(node, direction, options.edge_filter, |neighbor, edge| {
                    adjacent.push((neighbor, edge));
                })?;
                adjacent.sort_unstable();
                examined_relationships = examined_relationships.saturating_add(adjacent.len());
                let next_depth = own_depths[&node] + 1;
                for (neighbor, edge) in adjacent {
                    if own_depths.contains_key(&neighbor) {
                        continue;
                    }
                    own_depths.insert(neighbor, next_depth);
                    parents.insert(neighbor, (node, edge));
                    if let Some(&other_depth) = other_depths.get(&neighbor) {
                        let length = next_depth.saturating_add(other_depth);
                        if length <= options.max_hops {
                            best_length = Some(best_length.map_or(length, |best| best.min(length)));
                        }
                    }
                    next.push(neighbor);
                }
            }
            *frontier = next;
            if expand_forward {
                forward_depth += 1;
            } else {
                reverse_depth += 1;
            }
        }
    }

    fn frontier_work_upper_bound(&self, frontier: &[NodeId], direction: Direction) -> usize {
        frontier.iter().fold(0usize, |work, &node| {
            work.saturating_add(1)
                .saturating_add(self.adjacency_len_upper_bound(node, direction))
        })
    }

    fn adjacency_len_upper_bound(&self, node: NodeId, direction: Direction) -> usize {
        let mut count = 0usize;
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            count = count.saturating_add(
                self.base_outgoing
                    .as_ref()
                    .map_or(0, |adjacency| adjacency.get(node).len()),
            );
            count = count.saturating_add(self.outgoing.get(node).len());
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            count = count.saturating_add(
                self.base_incoming
                    .as_ref()
                    .map_or(0, |adjacency| adjacency.get(node).len()),
            );
            count = count.saturating_add(self.incoming.get(node).len());
        }
        count
    }

    pub(crate) fn vector_search_graph_range_adaptive(
        &self,
        query: &[f32],
        seeds: &ElementSet,
        options: &GraphRangeSearchOptions,
    ) -> Result<GraphRangeSearchResult> {
        let candidates = self.nodes_within_hops(
            seeds,
            options.direction,
            options.edge_filter,
            options.max_hops,
            options.include_seeds,
            options.node_filter.as_ref(),
        )?;
        let plan = self.vector_search_within_plan(&candidates);
        let hits = match plan.strategy {
            VectorSearchStrategy::Exact => {
                self.vector_search_within(query, &candidates, options.limit)?
            }
            VectorSearchStrategy::BinarySketchRerank => self.vector_search_within_approximate(
                query,
                &candidates,
                options.limit,
                plan.candidate_vectors,
            )?,
        };
        Ok(GraphRangeSearchResult {
            hits,
            candidate_nodes: candidates.node_len(),
            plan,
        })
    }

    pub(crate) fn neighbors(
        &self,
        node: NodeId,
        direction: Direction,
        filter: EdgeFilter,
    ) -> Result<Vec<Edge>> {
        if self.node_record(node).is_none() {
            return Err(Error::NotFound("node", node));
        }
        let mut result = Vec::new();
        let mut seen = HashSet::new();
        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some(base) = &self.base_outgoing {
                self.collect_edges(base.get(node), node, true, filter, &mut seen, &mut result);
            }
            self.collect_edges(
                self.outgoing.get(node),
                node,
                true,
                filter,
                &mut seen,
                &mut result,
            );
        }
        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some(base) = &self.base_incoming {
                self.collect_edges(base.get(node), node, false, filter, &mut seen, &mut result);
            }
            self.collect_edges(
                self.incoming.get(node),
                node,
                false,
                filter,
                &mut seen,
                &mut result,
            );
        }
        Ok(result)
    }

    /// Visits adjacent node/edge IDs without materializing edge records or a
    /// result vector. Immutable CSR-only nodes take a zero-allocation path;
    /// nodes touched by the WAL use a small dedup set to reconcile base and
    /// delta adjacency while preserving parallel edges.
    pub(crate) fn visit_neighbors(
        &self,
        node: NodeId,
        direction: Direction,
        filter: EdgeFilter,
        mut visitor: impl FnMut(NodeId, EdgeId),
    ) -> Result<()> {
        if !self.has_node(node) {
            return Err(Error::NotFound("node", node));
        }
        let out_delta_empty = self.outgoing.is_empty();
        let in_delta_empty = self.incoming.is_empty();
        let has_out_delta = !out_delta_empty && !self.outgoing.get(node).is_empty();
        let has_in_delta = !in_delta_empty && !self.incoming.get(node).is_empty();
        let needs_dedup = (matches!(direction, Direction::Outgoing | Direction::Both)
            && has_out_delta)
            || (matches!(direction, Direction::Incoming | Direction::Both) && has_in_delta);
        let mut seen = needs_dedup.then(HashSet::new);

        if matches!(direction, Direction::Outgoing | Direction::Both) {
            if let Some(base) = &self.base_outgoing {
                self.visit_neighbor_slice(
                    base.get(node),
                    node,
                    AdjacencySide::Outgoing,
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
            if !out_delta_empty {
                self.visit_neighbor_slice(
                    self.outgoing.get(node),
                    node,
                    AdjacencySide::Outgoing,
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
        }
        // In an immutable bidirectional CSR, the only edge present in both
        // slices for one node is a self-loop. Outgoing already emitted it.
        let skip_incoming_self = direction == Direction::Both && seen.is_none();
        if matches!(direction, Direction::Incoming | Direction::Both) {
            if let Some(base) = &self.base_incoming {
                self.visit_neighbor_slice(
                    base.get(node),
                    node,
                    AdjacencySide::Incoming {
                        skip_self: skip_incoming_self,
                    },
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
            if !in_delta_empty {
                self.visit_neighbor_slice(
                    self.incoming.get(node),
                    node,
                    AdjacencySide::Incoming {
                        skip_self: skip_incoming_self,
                    },
                    filter,
                    seen.as_mut(),
                    &mut visitor,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn one_hop_plan(&self, query: &OneHopQuery) -> OneHopPlan {
        let edge_candidate_upper_bound = query.edge_label.map_or(self.edge_count, |label| {
            self.edges_by_label
                .get(&label)
                .map_or(0, |ids| usize::try_from(ids.len()).unwrap_or(usize::MAX))
        });
        let start_candidate_upper_bound = self
            .element_filter_plan(VectorTarget::Nodes, &query.start)
            .candidate_upper_bound;
        let end_candidate_upper_bound = self
            .element_filter_plan(VectorTarget::Nodes, &query.end)
            .candidate_upper_bound;
        let average_directional_degree = if self.node_count == 0 {
            0
        } else {
            self.edge_count.div_ceil(self.node_count)
        };
        let direction_factor = if query.direction == Direction::Both {
            2
        } else {
            1
        };
        let start_edge_visits = start_candidate_upper_bound
            .saturating_mul(average_directional_degree)
            .saturating_mul(direction_factor);
        let end_edge_visits = end_candidate_upper_bound
            .saturating_mul(average_directional_degree)
            .saturating_mul(direction_factor);
        let (estimated_edge_visits, _, strategy) = [
            (edge_candidate_upper_bound, 0_u8, OneHopStrategy::EdgeScan),
            (start_edge_visits, 1, OneHopStrategy::StartAdjacency),
            (end_edge_visits, 2, OneHopStrategy::EndAdjacency),
        ]
        .into_iter()
        .min_by_key(|&(cost, priority, _)| (cost, priority))
        .expect("one-hop planner always has physical alternatives");
        OneHopPlan {
            strategy,
            estimated_edge_visits,
            edge_candidate_upper_bound,
            start_candidate_upper_bound,
            end_candidate_upper_bound,
        }
    }

    pub(crate) fn match_one_hop(&self, query: &OneHopQuery) -> Vec<PatternMatch> {
        if query.limit == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(query.limit.min(1024));
        match self.one_hop_plan(query).strategy {
            OneHopStrategy::EdgeScan => {
                let candidates: Box<dyn Iterator<Item = EdgeId> + '_> = match query.edge_label {
                    Some(label) => Box::new(self.edges_by_label.get(&label).into_iter().flatten()),
                    None => Box::new(self.edge_records().map(|edge| edge.id)),
                };
                for edge_id in candidates {
                    let Some(edge) = self.edge_record(edge_id) else {
                        continue;
                    };
                    if query.edge_label.is_some_and(|label| edge.label != label) {
                        continue;
                    }
                    let (orientations, orientation_count) = match query.direction {
                        Direction::Outgoing => ([(edge.source, edge.target); 2], 1),
                        Direction::Incoming => ([(edge.target, edge.source); 2], 1),
                        Direction::Both if edge.source != edge.target => {
                            ([(edge.source, edge.target), (edge.target, edge.source)], 2)
                        }
                        Direction::Both => ([(edge.source, edge.target); 2], 1),
                    };
                    for &(start, end) in &orientations[..orientation_count] {
                        let Some(start_node) = self.node_record(start) else {
                            continue;
                        };
                        let Some(end_node) = self.node_record(end) else {
                            continue;
                        };
                        if stored_node_matches(
                            &start_node,
                            &query.start,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        ) && stored_node_matches(
                            &end_node,
                            &query.end,
                            self.snapshot_map.as_deref(),
                            &self.owned_properties,
                        ) {
                            result.push(PatternMatch {
                                start,
                                edge: edge.id,
                                end,
                            });
                            if result.len() == query.limit {
                                return result;
                            }
                        }
                    }
                }
            }
            OneHopStrategy::StartAdjacency => {
                let starts = self.elements_matching(VectorTarget::Nodes, &query.start);
                for start in starts.node_ids() {
                    self.visit_neighbors(
                        start,
                        query.direction,
                        EdgeFilter {
                            label: query.edge_label,
                        },
                        |end, edge| {
                            if result.len() == query.limit {
                                return;
                            }
                            let Some(end_node) = self.node_record(end) else {
                                return;
                            };
                            if stored_node_matches(
                                &end_node,
                                &query.end,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            ) {
                                result.push(PatternMatch { start, edge, end });
                            }
                        },
                    )
                    .expect("planned start candidates are existing nodes");
                    if result.len() == query.limit {
                        break;
                    }
                }
            }
            OneHopStrategy::EndAdjacency => {
                let ends = self.elements_matching(VectorTarget::Nodes, &query.end);
                let reverse_direction = match query.direction {
                    Direction::Outgoing => Direction::Incoming,
                    Direction::Incoming => Direction::Outgoing,
                    Direction::Both => Direction::Both,
                };
                for end in ends.node_ids() {
                    self.visit_neighbors(
                        end,
                        reverse_direction,
                        EdgeFilter {
                            label: query.edge_label,
                        },
                        |start, edge| {
                            if result.len() == query.limit {
                                return;
                            }
                            let Some(start_node) = self.node_record(start) else {
                                return;
                            };
                            if stored_node_matches(
                                &start_node,
                                &query.start,
                                self.snapshot_map.as_deref(),
                                &self.owned_properties,
                            ) {
                                result.push(PatternMatch { start, edge, end });
                            }
                        },
                    )
                    .expect("planned end candidates are existing nodes");
                    if result.len() == query.limit {
                        break;
                    }
                }
            }
        }
        result
    }

    pub(crate) fn match_semantic_one_hop(
        &self,
        vector_query: &[f32],
        query: &SemanticOneHopQuery,
    ) -> Result<Vec<SemanticPatternMatch>> {
        validate_semantic_one_hop(query)?;
        if query.pattern.limit == 0 {
            return Ok(Vec::new());
        }
        let seeds = self.vector_search_adaptive(
            vector_query,
            VectorTarget::Nodes,
            query.seed_count,
            query.pattern.start.label,
        )?;
        let prepared = vector::prepare_query(vector_query, self.dimension, self.similarity)?;
        let scorer = self.vector_data.scorer(false)?;
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for seed in seeds {
            let ElementRef::Node(start) = seed.element else {
                continue;
            };
            let Some(start_node) = self.node_record(start) else {
                continue;
            };
            if !stored_node_matches(
                &start_node,
                &query.pattern.start,
                self.snapshot_map.as_deref(),
                &self.owned_properties,
            ) {
                continue;
            }
            for edge in self.neighbors(
                start,
                query.pattern.direction,
                EdgeFilter {
                    label: query.pattern.edge_label,
                },
            )? {
                let end = match query.pattern.direction {
                    Direction::Outgoing if edge.source == start => edge.target,
                    Direction::Incoming if edge.target == start => edge.source,
                    Direction::Both if edge.source == start => edge.target,
                    Direction::Both if edge.target == start => edge.source,
                    _ => continue,
                };
                let Some(end_node) = self.node_record(end) else {
                    continue;
                };
                if !stored_node_matches(
                    &end_node,
                    &query.pattern.end,
                    self.snapshot_map.as_deref(),
                    &self.owned_properties,
                ) {
                    continue;
                }
                let pattern = PatternMatch {
                    start,
                    edge: edge.id,
                    end,
                };
                if !seen.insert((pattern.start, pattern.edge, pattern.end)) {
                    continue;
                }
                let stored_edge = self.edge_record(edge.id).unwrap();
                let edge_score = self
                    .element_score(
                        &prepared,
                        stored_edge.vector_offset,
                        stored_edge.vector_count,
                        &scorer,
                    )?
                    .map(|score| score.0);
                let end_score = self
                    .element_score(
                        &prepared,
                        end_node.vector_offset,
                        end_node.vector_count,
                        &scorer,
                    )?
                    .map(|score| score.0);
                let mut weighted = seed.score * query.start_weight;
                let mut total_weight = query.start_weight;
                if let Some(score) = edge_score {
                    weighted += score * query.edge_weight;
                    total_weight += query.edge_weight;
                }
                if let Some(score) = end_score {
                    weighted += score * query.end_weight;
                    total_weight += query.end_weight;
                }
                result.push(SemanticPatternMatch {
                    pattern,
                    score: weighted / total_weight,
                    start_score: seed.score,
                    edge_score,
                    end_score,
                });
            }
        }
        result.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.pattern.start.cmp(&right.pattern.start))
                .then_with(|| left.pattern.edge.cmp(&right.pattern.edge))
                .then_with(|| left.pattern.end.cmp(&right.pattern.end))
        });
        result.truncate(query.pattern.limit);
        Ok(result)
    }

    fn collect_edges(
        &self,
        ids: &[EdgeId],
        node: NodeId,
        outgoing: bool,
        filter: EdgeFilter,
        seen: &mut HashSet<EdgeId>,
        result: &mut Vec<Edge>,
    ) {
        for id in ids {
            if !seen.insert(*id) {
                continue;
            }
            let Some(edge) = self.edge(*id) else {
                continue;
            };
            if (outgoing && edge.source != node) || (!outgoing && edge.target != node) {
                continue;
            }
            if filter.label.is_none_or(|label| edge.label == label) {
                result.push(edge);
            }
        }
    }

    fn visit_neighbor_slice(
        &self,
        ids: &[EdgeId],
        node: NodeId,
        side: AdjacencySide,
        filter: EdgeFilter,
        mut seen: Option<&mut HashSet<EdgeId>>,
        visitor: &mut impl FnMut(NodeId, EdgeId),
    ) {
        let outgoing = matches!(side, AdjacencySide::Outgoing);
        let skip_self = matches!(side, AdjacencySide::Incoming { skip_self: true });
        if self.edges.is_empty()
            && self.edge_overlays.is_empty()
            && let Some(records) = &self.mapped_edges
        {
            for id in ids {
                if seen.as_mut().is_some_and(|seen| !seen.insert(*id)) {
                    continue;
                }
                if let Some(neighbor) = records.neighbor(*id, node, outgoing, filter.label) {
                    if skip_self && neighbor == node {
                        continue;
                    }
                    visitor(neighbor, *id);
                }
            }
            return;
        }
        for id in ids {
            if seen.as_mut().is_some_and(|seen| !seen.insert(*id)) {
                continue;
            }
            let Some(edge) = self.edge_record(*id) else {
                continue;
            };
            if (outgoing && edge.source != node)
                || (!outgoing && edge.target != node)
                || filter.label.is_some_and(|label| edge.label != label)
            {
                continue;
            }
            let neighbor = if outgoing { edge.target } else { edge.source };
            if !skip_self || neighbor != node {
                visitor(neighbor, edge.id);
            }
        }
    }
}
