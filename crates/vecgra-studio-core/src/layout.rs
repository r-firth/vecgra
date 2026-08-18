use super::*;

pub(super) fn run_force_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
    pinned: Option<&[bool]>,
) {
    // The cell-centroid approximation below is excellent for small, direct
    // manipulation layouts, but its square neighborhoods become visible as a
    // lattice once thousands of nodes are present. Large general graphs use
    // sampled t-distribution forces instead: work and auxiliary memory remain
    // linear in edges, while repulsion has no screen-aligned spatial grid.
    //
    // This follows the objective introduced by SNAP-tFDP (Wang et al., IEEE
    // VIS 2026), with a deterministic RNG and portable scalar Rust. The hot
    // loop is deliberately contiguous and branch-light so it can later gain
    // platform SIMD or a GPU backend without changing layout semantics.
    const SAMPLED_T_FORCE_NODE_THRESHOLD: usize = 256;
    if positions.len() >= SAMPLED_T_FORCE_NODE_THRESHOLD && !edges.ids.is_empty() {
        run_sampled_t_force_layout(positions, edges, options, pinned);
    } else {
        run_spatial_force_layout(positions, edges, options, pinned);
    }
}

fn run_spatial_force_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
    pinned: Option<&[bool]>,
) {
    let count = positions.len();
    let mut forces = vec![Vec2::ZERO; count];
    let cell_size = (options.edge_length * 1.8).max(1.0);
    let iterations = options.iterations.min(match count {
        0..=2_000 => options.iterations,
        2_001..=12_000 => 32,
        _ => 12,
    });

    for iteration in 0..iterations {
        forces.fill(Vec2::ZERO);
        let mut cells: HashMap<(i32, i32), (u32, Vec2)> = HashMap::new();
        for position in positions.iter().copied() {
            let key = cell_key(position, cell_size);
            let entry = cells.entry(key).or_insert((0, Vec2::ZERO));
            entry.0 += 1;
            entry.1 += position;
        }

        for (index, position) in positions.iter().copied().enumerate() {
            let own_cell = cell_key(position, cell_size);
            let mut force = position * -options.gravity;
            for cell_y in (own_cell.1 - 2)..=(own_cell.1 + 2) {
                for cell_x in (own_cell.0 - 2)..=(own_cell.0 + 2) {
                    let Some(&(cell_count, sum)) = cells.get(&(cell_x, cell_y)) else {
                        continue;
                    };
                    let center = sum / cell_count as f32;
                    let mut delta = position - center;
                    let mut distance_squared = delta.length_squared();
                    if distance_squared < 0.01 {
                        let angle =
                            stable_unit(index as u64 ^ iteration as u64) * std::f32::consts::TAU;
                        delta = Vec2::new(angle.cos(), angle.sin());
                        distance_squared = 1.0;
                    }
                    let magnitude = options.repulsion * cell_count as f32
                        / (distance_squared + options.edge_length);
                    force += delta * (magnitude / distance_squared.sqrt());
                }
            }
            forces[index] = force;
        }

        for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
            let source = source as usize;
            let target = target as usize;
            let delta = positions[target] - positions[source];
            let distance = delta.length().max(0.01);
            let magnitude = (distance - options.edge_length) * options.attraction;
            let spring = delta * (magnitude / distance);
            forces[source] += spring;
            forces[target] += spring * -1.0;
        }

        let progress = iteration as f32 / iterations.max(1) as f32;
        let temperature = options.edge_length * (1.0 - progress).mul_add(0.22, 0.025);
        for (index, (position, force)) in positions.iter_mut().zip(&forces).enumerate() {
            if pinned.is_some_and(|pins| pins[index]) {
                continue;
            }
            let length = force.length();
            if length > 0.0 && length.is_finite() {
                *position += *force * (temperature.min(length) / length);
            }
        }
    }
}

fn run_sampled_t_force_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
    pinned: Option<&[bool]>,
) {
    let node_count = positions.len();
    if node_count < 2 {
        return;
    }

    let original_frame = Rect::from_points(positions);
    let original_center = original_frame.center();
    let original_extent = original_frame.width().max(original_frame.height());
    if original_extent > 0.000_1 && original_extent.is_finite() {
        let scale = 20.0 / original_extent;
        for position in positions.iter_mut() {
            *position = (*position - original_center) * scale;
        }
    } else {
        // A deterministic phyllotaxis seed avoids coincident gradients when a
        // caller supplies a collapsed layout.
        const GOLDEN_ANGLE: f32 = 2.399_963_1;
        for (index, position) in positions.iter_mut().enumerate() {
            let radius = 0.12 * (index as f32 + 1.0).sqrt();
            let angle = index as f32 * GOLDEN_ANGLE + stable_unit(index as u64);
            *position = Vec2::new(angle.cos(), angle.sin()) * radius;
        }
    }

    // SNAP-tFDP treats the graph as undirected and stores both orientations.
    // Keeping that representation here also gives each endpoint equal chances
    // to anchor negative samples, regardless of the relationship direction.
    let mut directed_edges = Vec::with_capacity(edges.ids.len().saturating_mul(2));
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let source = source as usize;
        let target = target as usize;
        if source < node_count && target < node_count && source != target {
            directed_edges.push((source, target));
            directed_edges.push((target, source));
        }
    }
    if directed_edges.is_empty() {
        if !pinned.is_some_and(|pins| pins.iter().any(|&pin| pin)) {
            scale_sampled_layout(positions, options.edge_length);
        }
        return;
    }

    let mut rng = LayoutRng::new(
        0x5647_534e_4150_5446_u64
            ^ node_count as u64
            ^ (directed_edges.len() as u64).rotate_left(29),
    );
    for index in (1..directed_edges.len()).rev() {
        let other = rng.index(index + 1);
        directed_edges.swap(index, other);
    }

    const NEGATIVE_SAMPLES: usize = 3;
    const MIN_STEP: f32 = 0.01;
    let epochs = options.iterations.clamp(1, 50);
    let cooling = 1.0 - 0.02_f32.powf(1.0 / epochs as f32);
    let mut step = 1.0_f32;
    for _ in 0..epochs {
        for &(source, target) in &directed_edges {
            let delta = positions[target] - positions[source];
            let distance_squared = delta.length_squared();
            let attraction = 0.1 + 0.8 / (1.0 + distance_squared);
            move_sampled_pair(
                positions,
                pinned,
                source,
                target,
                delta * (step * attraction),
            );

            for _ in 0..NEGATIVE_SAMPLES {
                // Sampling N - 1 and skipping over `source` removes a retry
                // branch and guarantees exactly K useful negatives.
                let sampled = rng.index(node_count - 1);
                let other = sampled + usize::from(sampled >= source);
                let delta = positions[other] - positions[source];
                let denominator = 1.0 + delta.length_squared();
                let repulsion = -1.0 / (denominator * denominator);
                move_sampled_pair(positions, pinned, source, other, delta * (step * repulsion));
            }
        }
        step += (MIN_STEP - step) * cooling;
    }

    if pinned.is_some_and(|pins| pins.iter().any(|&pin| pin)) {
        // Return to the caller's coordinate system so fixed nodes remain true
        // anchors. GraphWorkspace preserves them again during animated fit.
        if original_extent > 0.000_1 && original_extent.is_finite() {
            let inverse_scale = original_extent / 20.0;
            for position in positions.iter_mut() {
                *position = *position * inverse_scale + original_center;
            }
        }
    } else {
        scale_sampled_layout(positions, options.edge_length);
    }
}

fn move_sampled_pair(
    positions: &mut [Vec2],
    pinned: Option<&[bool]>,
    left: usize,
    right: usize,
    movement: Vec2,
) {
    let left_pinned = pinned.is_some_and(|pins| pins[left]);
    let right_pinned = pinned.is_some_and(|pins| pins[right]);
    match (left_pinned, right_pinned) {
        (false, false) => {
            positions[left] += movement;
            positions[right] += movement * -1.0;
        }
        (false, true) => positions[left] += movement * 2.0,
        (true, false) => positions[right] += movement * -2.0,
        (true, true) => {}
    }
}

fn scale_sampled_layout(positions: &mut [Vec2], edge_length: f32) {
    recenter(positions);
    let bounds = Rect::from_points(positions);
    let extent = bounds.width().max(bounds.height());
    if extent <= 0.000_1 || !extent.is_finite() {
        return;
    }
    // Area grows with node count, keeping density and picking tolerances
    // stable across graph sizes. Final Studio fitting still controls how much
    // of that world-space canvas is visible at once.
    let target_extent = edge_length.max(8.0) * (positions.len() as f32).sqrt();
    let scale = target_extent / extent;
    for position in positions {
        *position = *position * scale;
    }
}

struct LayoutRng {
    state: u64,
}

impl LayoutRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64 is small, reproducible on every target, and has ample
        // quality for edge shuffling and unbiased negative sampling.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper_bound: usize) -> usize {
        debug_assert!(upper_bound > 0);
        ((u128::from(self.next_u64()) * upper_bound as u128) >> 64) as usize
    }
}

/// Lays out each connected component with Omega's low-rank resistance-distance
/// stress model, then packs the independent components without inventing edges
/// between them. This mode deliberately favors global topology and community
/// faithfulness over the latency of the default sampled-force arrangement.
pub(super) fn run_resistance_structure_layout(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    options: LayoutOptions,
) {
    let node_count = positions.len();
    if node_count < 3 || edges.sources.is_empty() {
        run_spatial_force_layout(positions, edges, options, None);
        recenter(positions);
        return;
    }

    let mut union = LayoutUnionFind::new(node_count);
    let mut unique_pairs = Vec::with_capacity(edges.sources.len());
    let mut pair_seen = HashSet::with_capacity(edges.sources.len());
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let source = source as usize;
        let target = target as usize;
        if source >= node_count || target >= node_count || source == target {
            continue;
        }
        let pair = if source < target {
            (source, target)
        } else {
            (target, source)
        };
        if pair_seen.insert(pair) {
            union.join(pair.0, pair.1);
            unique_pairs.push(pair);
        }
    }
    if unique_pairs.is_empty() {
        layout_orbits(
            positions,
            &(0..node_count as NodeId).collect::<Vec<_>>(),
            &vec![Arc::from("Node"); node_count],
            &vec![0; node_count],
            options.edge_length,
        );
        return;
    }

    let mut component_nodes = BTreeMap::<usize, Vec<usize>>::new();
    for node in 0..node_count {
        component_nodes
            .entry(union.root(node))
            .or_default()
            .push(node);
    }
    let mut components: Vec<_> = component_nodes.into_values().collect();
    components.sort_unstable_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });

    let mut component_of = vec![usize::MAX; node_count];
    let mut local_index = vec![usize::MAX; node_count];
    for (component_index, nodes) in components.iter().enumerate() {
        for (local, &global) in nodes.iter().enumerate() {
            component_of[global] = component_index;
            local_index[global] = local;
        }
    }
    let mut component_edges = vec![Vec::new(); components.len()];
    for &(source, target) in &unique_pairs {
        let component = component_of[source];
        debug_assert_eq!(component, component_of[target]);
        component_edges[component].push((local_index[source], local_index[target]));
    }

    let mut drawings = Vec::with_capacity(components.len());
    for (component_index, nodes) in components.into_iter().enumerate() {
        let coordinates = resistance_component(
            nodes.len(),
            &component_edges[component_index],
            nodes[0] as u64,
            options,
        );
        let radius = coordinates
            .iter()
            .map(|position| position.length())
            .fold(0.0_f32, f32::max)
            .max(0.5);
        drawings.push(PackedComponent {
            nodes,
            coordinates,
            center: Vec2::ZERO,
            radius,
        });
    }
    pack_structure_components(&mut drawings, options.edge_length.max(8.0) * 0.12);

    for component in drawings {
        for (&node, coordinate) in component.nodes.iter().zip(component.coordinates) {
            positions[node] = coordinate + component.center;
        }
    }
    scale_sampled_layout(positions, options.edge_length);
}

fn resistance_component(
    node_count: usize,
    edges: &[(usize, usize)],
    seed: u64,
    options: LayoutOptions,
) -> Vec<Vec2> {
    match node_count {
        0 => return Vec::new(),
        1 => return vec![Vec2::ZERO],
        2 => return vec![Vec2::new(-0.5, 0.0), Vec2::new(0.5, 0.0)],
        _ => {}
    }

    let mut graph = Graph::<(), (), Undirected>::with_capacity(node_count, edges.len());
    for _ in 0..node_count {
        graph.add_node(());
    }
    for &(source, target) in edges {
        graph.add_edge(NodeIndex::new(source), NodeIndex::new(target), ());
    }

    // The paper's quality/performance operating point is rank 10, fifty
    // sampled pairs per vertex, epsilon_d=0.01, and fifteen SGD passes. Rank is
    // naturally capped for tiny connected components.
    let rank = 10.min(node_count - 1);
    let mut rng = StdRng::seed_from_u64(
        0x4f4d_4547_4152_4453_u64 ^ seed ^ (node_count as u64).rotate_left(17),
    );
    let embedding = RdMds::new()
        .d(rank)
        .shift(1e-6)
        .eigenvalue_max_iterations(96)
        .cg_max_iterations(64)
        .eigenvalue_tolerance(1e-5)
        .cg_tolerance(1e-5)
        .embedding(&graph, |_| 1.0_f32, &mut rng);

    let mut drawing = DrawingEuclidean2d::<NodeIndex, f32>::new(&graph);
    for index in 0..node_count {
        let node = NodeIndex::new(index);
        _ = drawing.set_x(node, embedding[[index, 0]]);
        _ = drawing.set_y(node, if rank > 1 { embedding[[index, 1]] } else { 0.0 });
    }

    let mut omega = Omega::new();
    omega.k(50).min_dist(0.01);
    let mut sgd = omega.build(&graph, &embedding, &mut rng);
    let iterations = (options.iterations / 4).clamp(8, 15);
    let mut scheduler = sgd.scheduler::<SchedulerExponential<f32>>(iterations, 0.1);
    scheduler.run(&mut |eta| {
        sgd.shuffle(&mut rng);
        sgd.apply(&mut drawing, eta);
    });

    let mut coordinates: Vec<_> = (0..node_count)
        .map(|index| {
            let node = NodeIndex::new(index);
            Vec2::new(
                drawing.x(node).unwrap_or(0.0),
                drawing.y(node).unwrap_or(0.0),
            )
        })
        .collect();
    if coordinates
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        // Numerical failure should never poison the Studio scene. A stable
        // phyllotaxis component remains inspectable and makes the fallback
        // visually explicit instead of returning a collapsed origin.
        const GOLDEN_ANGLE: f32 = 2.399_963_1;
        for (index, position) in coordinates.iter_mut().enumerate() {
            let radius = (index as f32 + 1.0).sqrt();
            let angle = index as f32 * GOLDEN_ANGLE + stable_unit(seed ^ index as u64);
            *position = Vec2::new(angle.cos(), angle.sin()) * radius;
        }
    }
    recenter(&mut coordinates);
    let extent = Rect::from_points(&coordinates)
        .width()
        .max(Rect::from_points(&coordinates).height())
        .max(0.000_1);
    let target_extent = (node_count as f32).sqrt().max(1.0);
    for position in &mut coordinates {
        *position = *position * (target_extent / extent);
    }

    coordinates
}

struct PackedComponent {
    nodes: Vec<usize>,
    coordinates: Vec<Vec2>,
    center: Vec2,
    radius: f32,
}

fn pack_structure_components(components: &mut [PackedComponent], gap: f32) {
    let Some((largest, rest)) = components.split_first_mut() else {
        return;
    };
    largest.center = Vec2::ZERO;
    if rest.is_empty() {
        return;
    }

    let mut ring_radius = largest.radius + rest[0].radius + gap;
    let mut ring_height = rest[0].radius;
    let mut cursor = 0.0_f32;
    let mut previous_radius = rest[0].radius;
    for component in rest {
        let chord = previous_radius + component.radius + gap;
        let angular_gap = 2.0 * (chord / (2.0 * ring_radius).max(chord)).asin();
        if cursor > 0.0 && cursor + angular_gap > std::f32::consts::TAU {
            ring_radius += ring_height + component.radius + gap;
            ring_height = component.radius;
            cursor = 0.0;
        }
        let angle = cursor + angular_gap * 0.5;
        component.center = Vec2::new(angle.cos(), angle.sin()) * ring_radius;
        cursor += angular_gap;
        ring_height = ring_height.max(component.radius);
        previous_radius = component.radius;
    }
}

struct LayoutUnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl LayoutUnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn root(&mut self, node: usize) -> usize {
        let parent = self.parent[node];
        if parent != node {
            self.parent[node] = self.root(parent);
        }
        self.parent[node]
    }

    fn join(&mut self, left: usize, right: usize) {
        let mut left_root = self.root(left);
        let mut right_root = self.root(right);
        if left_root == right_root {
            return;
        }
        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] = self.rank[left_root].saturating_add(1);
        }
    }
}

#[derive(Debug)]
pub struct GraphWorkspace {
    positions: Vec<Vec2>,
    base_targets: Vec<Vec2>,
    targets: Vec<Vec2>,
    pub(crate) velocities: Vec<Vec2>,
    pinned: Vec<bool>,
    adjacency: Vec<Vec<u32>>,
    moving: bool,
}

impl GraphWorkspace {
    pub fn new(snapshot: &SceneSnapshot) -> Self {
        let mut adjacency = vec![Vec::new(); snapshot.nodes.ids.len()];
        for (&source, &target) in snapshot.edges.sources.iter().zip(&snapshot.edges.targets) {
            let source = source as usize;
            let target = target as usize;
            if source < adjacency.len() && target < adjacency.len() && source != target {
                adjacency[source].push(target as u32);
                adjacency[target].push(source as u32);
            }
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let positions = snapshot.nodes.positions.clone();
        Self {
            base_targets: positions.clone(),
            targets: positions.clone(),
            velocities: vec![Vec2::ZERO; positions.len()],
            pinned: vec![false; positions.len()],
            adjacency,
            positions,
            moving: false,
        }
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.positions
    }

    /// Stable overview targets, excluding temporary search/context layouts.
    pub fn overview_positions(&self) -> &[Vec2] {
        &self.base_targets
    }

    pub fn position(&self, index: usize) -> Option<Vec2> {
        self.positions.get(index).copied()
    }

    pub fn presentation_bounds(&self) -> Rect {
        Rect::from_points(&self.positions)
    }

    pub fn is_pinned(&self, index: usize) -> bool {
        self.pinned.get(index).copied().unwrap_or(false)
    }

    pub fn pinned_count(&self) -> usize {
        self.pinned.iter().filter(|&&pinned| pinned).count()
    }

    pub fn pins(&self) -> &[bool] {
        &self.pinned
    }

    pub fn is_moving(&self) -> bool {
        self.moving
    }

    pub fn begin_drag(&mut self, index: usize) -> bool {
        let was_pinned = self.is_pinned(index);
        if index < self.positions.len() {
            self.pinned[index] = true;
            self.targets[index] = self.positions[index];
            self.velocities[index] = Vec2::ZERO;
        }
        was_pinned
    }

    pub fn drag_to(&mut self, index: usize, position: Vec2, edge_length: f32) {
        if index >= self.positions.len() || !position.x.is_finite() || !position.y.is_finite() {
            return;
        }
        self.positions[index] = position;
        self.targets[index] = position;
        self.velocities[index] = Vec2::ZERO;

        // Direct manipulation remains O(degree), not O(graph size). A bounded
        // one-hop physical response keeps connected structure attached to the
        // pointer without making a hub node capable of blowing the frame.
        const NEIGHBOR_BUDGET: usize = 512;
        let desired_length = edge_length.max(8.0);
        for &neighbor in self.adjacency[index].iter().take(NEIGHBOR_BUDGET) {
            let neighbor = neighbor as usize;
            if self.pinned[neighbor] {
                continue;
            }
            let mut delta = self.positions[neighbor] - position;
            let distance = delta.length();
            if distance < 0.001 {
                let angle = stable_unit(neighbor as u64 ^ index as u64) * std::f32::consts::TAU;
                delta = Vec2::new(angle.cos(), angle.sin());
            }
            let direction = delta / delta.length().max(0.001);
            let desired = position + direction * desired_length;
            let influence = if distance > desired_length * 2.5 {
                0.34
            } else {
                0.18
            };
            let target = self.targets[neighbor] * (1.0 - influence) + desired * influence;
            self.targets[neighbor] = target;
            self.base_targets[neighbor] = target;
        }
        self.moving = true;
    }

    pub fn restore_pin(&mut self, index: usize, was_pinned: bool) {
        if index >= self.positions.len() {
            return;
        }
        self.pinned[index] = was_pinned;
        self.targets[index] = if was_pinned {
            self.positions[index]
        } else {
            self.base_targets[index]
        };
        self.moving |= !was_pinned && self.targets[index] != self.positions[index];
    }

    pub fn set_pinned(&mut self, index: usize, pinned: bool) {
        if index >= self.positions.len() {
            return;
        }
        self.pinned[index] = pinned;
        self.velocities[index] = Vec2::ZERO;
        self.targets[index] = if pinned {
            self.positions[index]
        } else {
            self.base_targets[index]
        };
        self.moving |= !pinned && self.targets[index] != self.positions[index];
    }

    pub fn retarget_layout(&mut self, mut targets: Vec<Vec2>, frame: Rect) -> bool {
        if targets.len() != self.positions.len()
            || targets
                .iter()
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            return false;
        }
        fit_positions_to_frame(&mut targets, frame);
        self.base_targets = targets;
        for index in 0..self.positions.len() {
            if self.pinned[index] {
                self.targets[index] = self.positions[index];
            } else {
                self.targets[index] = self.base_targets[index];
            }
        }
        self.moving = self.positions.iter().zip(&self.targets).enumerate().any(
            |(index, (&position, &target))| {
                !self.pinned[index] && (target - position).length_squared() > 0.000_4
            },
        );
        true
    }

    /// Retargets a bounded neighborhood without changing the active base
    /// arrangement. Switching focus preserves spring velocity; restoring the
    /// overview returns every unpinned node to its base target.
    pub fn retarget_focus(&mut self, focus: &SceneFocus, spacing: f32) -> Option<Rect> {
        if focus.roots.is_empty()
            || focus.nodes.is_empty()
            || focus
                .nodes
                .iter()
                .chain(&focus.roots)
                .any(|&index| index >= self.positions.len())
        {
            return None;
        }
        self.restore_targets();

        let anchor = focus
            .roots
            .iter()
            .map(|&index| self.positions[index])
            .fold(Vec2::ZERO, |sum, position| sum + position)
            / focus.roots.len() as f32;
        let spacing = spacing.max(18.0);
        for (slot, &root) in focus.roots.iter().enumerate() {
            if self.pinned[root] {
                continue;
            }
            self.targets[root] = if focus.roots.len() == 1 {
                anchor
            } else {
                let offset = (slot as f32 - (focus.roots.len() - 1) as f32 * 0.5) * spacing * 1.8;
                anchor + Vec2::new(offset, 0.0)
            };
        }

        let mut parent_of = vec![usize::MAX; self.positions.len()];
        for &(child, parent) in &focus.parents {
            if child < parent_of.len() && parent < parent_of.len() {
                parent_of[child] = parent;
            }
        }
        let mut node_angles = vec![f32::NAN; self.positions.len()];
        for (slot, &root) in focus.roots.iter().enumerate() {
            node_angles[root] = std::f32::consts::TAU * slot as f32 / focus.roots.len() as f32;
        }
        let mut previous_outer_radius = if focus.roots.len() > 1 { spacing } else { 0.0 };
        for (depth, layer) in focus.layers.iter().enumerate().skip(1) {
            let mut ordered = layer.clone();
            ordered.sort_unstable_by(|&left, &right| {
                let left_parent = parent_of[left];
                let right_parent = parent_of[right];
                let left_angle = node_angles.get(left_parent).copied().unwrap_or(f32::NAN);
                let right_angle = node_angles.get(right_parent).copied().unwrap_or(f32::NAN);
                left_angle
                    .total_cmp(&right_angle)
                    .then_with(|| left_parent.cmp(&right_parent))
                    .then_with(|| left.cmp(&right))
            });

            // Each semantic hop owns a separate radial band. Dense layers may
            // use several physical rings inside their band; this keeps nodes
            // pickable without making a high-degree first hop push the second
            // hop off-screen. Angle is global across the band, avoiding the
            // repeated spokes produced by restarting at zero for every ring.
            let mut radius = previous_outer_radius + spacing * 1.45;
            let mut rings = Vec::new();
            let mut total_capacity = 0_usize;
            while total_capacity < ordered.len() {
                let capacity =
                    ((std::f32::consts::TAU * radius / (spacing * 0.92)).floor() as usize).max(8);
                rings.push((radius, capacity));
                total_capacity += capacity;
                radius += spacing * 1.05;
            }
            let phase = if depth == 1 {
                stable_unit(
                    focus.roots[0] as u64
                        ^ (depth as u64).rotate_left(17)
                        ^ (ordered.len() as u64).rotate_left(31),
                ) * 0.28
            } else {
                // Rotate the child sequence to best align its contiguous
                // parent groups with their parents. This circular mean removes
                // avoidable parent-child crossings without an iterative solve.
                let mut direction = Vec2::ZERO;
                for (slot, &node) in ordered.iter().enumerate() {
                    let base = std::f32::consts::TAU * slot as f32 / ordered.len() as f32;
                    let parent_angle = node_angles[parent_of[node]];
                    let delta = parent_angle - base;
                    direction += Vec2::new(delta.cos(), delta.sin());
                }
                direction.y.atan2(direction.x)
            };
            let mut occupancy = vec![0_usize; rings.len()];
            for (slot, &node) in ordered.iter().enumerate() {
                let ring = (0..rings.len())
                    .filter(|&ring| occupancy[ring] < rings[ring].1)
                    .min_by(|&left, &right| {
                        let left_fill = occupancy[left] as f32 / rings[left].1 as f32;
                        let right_fill = occupancy[right] as f32 / rings[right].1 as f32;
                        left_fill.total_cmp(&right_fill).then(left.cmp(&right))
                    })
                    .unwrap_or(0);
                occupancy[ring] += 1;
                let angle = std::f32::consts::TAU * slot as f32 / ordered.len() as f32 + phase;
                node_angles[node] = angle.rem_euclid(std::f32::consts::TAU);
                if !self.pinned[node] {
                    self.targets[node] =
                        anchor + Vec2::new(angle.cos(), angle.sin()) * rings[ring].0;
                }
            }
            previous_outer_radius = rings.last().unwrap().0 + spacing * 0.45;
        }

        self.moving = self.positions.iter().zip(&self.targets).enumerate().any(
            |(index, (&position, &target))| {
                !self.pinned[index] && (target - position).length_squared() > 0.000_4
            },
        );
        let focus_targets: Vec<_> = focus
            .nodes
            .iter()
            .map(|&index| {
                if self.pinned[index] {
                    self.positions[index]
                } else {
                    self.targets[index]
                }
            })
            .collect();
        Some(Rect::from_points(&focus_targets))
    }

    pub fn restore_layout(&mut self) {
        self.restore_targets();
        self.moving = self.positions.iter().zip(&self.targets).enumerate().any(
            |(index, (&position, &target))| {
                !self.pinned[index] && (target - position).length_squared() > 0.000_4
            },
        );
    }

    fn restore_targets(&mut self) {
        for index in 0..self.positions.len() {
            self.targets[index] = if self.pinned[index] {
                self.positions[index]
            } else {
                self.base_targets[index]
            };
        }
    }

    pub fn step(&mut self, elapsed_seconds: f32, reduce_motion: bool) -> bool {
        if !self.moving {
            return false;
        }
        if reduce_motion {
            for index in 0..self.positions.len() {
                if !self.pinned[index] {
                    self.positions[index] = self.targets[index];
                }
                self.velocities[index] = Vec2::ZERO;
            }
            self.moving = false;
            return false;
        }

        let elapsed = elapsed_seconds.clamp(0.0, 0.05);
        if elapsed <= f32::EPSILON {
            return true;
        }
        const MAX_STEP: f32 = 1.0 / 120.0;
        const STIFFNESS: f32 = 175.0;
        const DAMPING: f32 = 27.0;
        let steps = (elapsed / MAX_STEP).ceil().max(1.0) as usize;
        let dt = elapsed / steps as f32;
        for _ in 0..steps {
            for index in 0..self.positions.len() {
                if self.pinned[index] {
                    continue;
                }
                let displacement = self.targets[index] - self.positions[index];
                let acceleration = displacement * STIFFNESS - self.velocities[index] * DAMPING;
                self.velocities[index] += acceleration * dt;
                self.positions[index] += self.velocities[index] * dt;
            }
        }

        let mut moving = false;
        for index in 0..self.positions.len() {
            if self.pinned[index] {
                continue;
            }
            let distance_squared = (self.targets[index] - self.positions[index]).length_squared();
            let speed_squared = self.velocities[index].length_squared();
            if distance_squared <= 0.000_4 && speed_squared <= 0.002_5 {
                self.positions[index] = self.targets[index];
                self.velocities[index] = Vec2::ZERO;
            } else if self.positions[index].x.is_finite()
                && self.positions[index].y.is_finite()
                && self.velocities[index].x.is_finite()
                && self.velocities[index].y.is_finite()
            {
                moving = true;
            } else {
                self.positions[index] = self.targets[index];
                self.velocities[index] = Vec2::ZERO;
            }
        }
        self.moving = moving;
        moving
    }
}

struct ForestTopology {
    children: Vec<Vec<u32>>,
    roots: Vec<u32>,
    weights: Vec<u64>,
    subtree_sizes: Vec<usize>,
}

fn directed_forest_topology(node_count: usize, edges: &SceneEdges) -> Option<ForestTopology> {
    if node_count == 0 || edges.ids.len() >= node_count {
        return None;
    }
    let mut children = vec![Vec::<u32>::new(); node_count];
    let mut indegree = vec![0_u8; node_count];
    for (&source, &target) in edges.sources.iter().zip(&edges.targets) {
        let source = source as usize;
        let target = target as usize;
        if source >= node_count || target >= node_count || source == target {
            return None;
        }
        indegree[target] = indegree[target].saturating_add(1);
        if indegree[target] > 1 {
            return None;
        }
        children[source].push(target as u32);
    }

    let roots: Vec<u32> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index as u32))
        .collect();
    if roots.is_empty() {
        return None;
    }

    // Produce a parent-before-child order while proving that every node is
    // reachable exactly once. That rejects cycles and disconnected cycles.
    let mut order = Vec::with_capacity(node_count);
    let mut visited = vec![false; node_count];
    let mut stack = roots.clone();
    while let Some(node) = stack.pop() {
        let index = node as usize;
        if std::mem::replace(&mut visited[index], true) {
            return None;
        }
        order.push(node);
        stack.extend(children[index].iter().copied());
    }
    if order.len() != node_count {
        return None;
    }

    // Leaves own one unit of angular space; internal nodes own the sum of
    // their descendants. The resulting dendrogram keeps branch identity at
    // overview scale without a quadratic force simulation.
    let mut weights = vec![1_u64; node_count];
    let mut subtree_sizes = vec![1_usize; node_count];
    for &node in order.iter().rev() {
        let index = node as usize;
        if !children[index].is_empty() {
            weights[index] = children[index]
                .iter()
                .fold(0_u64, |sum, &child| {
                    sum.saturating_add(weights[child as usize])
                })
                .max(1);
            subtree_sizes[index] += children[index]
                .iter()
                .map(|&child| subtree_sizes[child as usize])
                .sum::<usize>();
        }
    }
    Some(ForestTopology {
        children,
        roots,
        weights,
        subtree_sizes,
    })
}

pub(super) fn try_layout_radial_forest(
    positions: &mut [Vec2],
    edges: &SceneEdges,
    edge_length: f32,
) -> bool {
    let Some(topology) = directed_forest_topology(positions.len(), edges) else {
        return false;
    };
    let total_weight = topology
        .roots
        .iter()
        .fold(0_u64, |sum, &root| {
            sum.saturating_add(topology.weights[root as usize])
        })
        .max(1) as f32;

    let radial_step = edge_length.max(12.0);
    let mut root_cursor = 0.0_f32;
    let mut assignments = Vec::with_capacity(positions.len());
    for &root in &topology.roots {
        let span = std::f32::consts::TAU * topology.weights[root as usize] as f32 / total_weight;
        assignments.push((root, root_cursor, root_cursor + span, 0_u32));
        root_cursor += span;
    }

    while let Some((node, start, end, depth)) = assignments.pop() {
        let index = node as usize;
        let angle = (start + end) * 0.5;
        let radius = if topology.roots.len() == 1 && depth == 0 {
            0.0
        } else {
            (depth + 1) as f32 * radial_step
        };
        positions[index] = Vec2::new(angle.cos(), angle.sin()) * radius;

        let child_total = topology.weights[index].max(1) as f32;
        let mut cursor = start;
        for &child in &topology.children[index] {
            let child_span = (end - start) * topology.weights[child as usize] as f32 / child_total;
            assignments.push((child, cursor, cursor + child_span, depth + 1));
            cursor += child_span;
        }
    }

    true
}

pub(super) fn try_layout_clustered_forest(
    positions: &mut [Vec2],
    ids: &[NodeId],
    edges: &SceneEdges,
    edge_length: f32,
) -> bool {
    let Some(topology) = directed_forest_topology(positions.len(), edges) else {
        return false;
    };
    let (branches, central_root) = if topology.roots.len() == 1 {
        let root = topology.roots[0];
        let branches = topology.children[root as usize].clone();
        (branches, Some(root))
    } else {
        (topology.roots.clone(), None)
    };
    if !(2..=64).contains(&branches.len()) {
        return false;
    }

    // A large forest is more legible as a constellation of top-level
    // subtrees than as global depth rings. Each subtree gets a deterministic
    // topology-preserving phyllotaxis seed; the ordinary bounded force pass
    // then relaxes edges without erasing the top-level partition.
    const GOLDEN_ANGLE: f32 = 2.399_963_1;
    let local_spacing = (edge_length * 0.24).max(7.0);
    // Leave enough negative space for the force relaxation to expand each
    // disk without visually welding neighboring top-level branches together.
    let gap = edge_length.max(24.0) * 10.0;
    let mut ranked: Vec<_> = branches
        .iter()
        .copied()
        .map(|branch| (branch, topology.subtree_sizes[branch as usize]))
        .collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| ids[left.0 as usize].cmp(&ids[right.0 as usize]))
    });

    let count = ranked.len();
    let mut slots = vec![0_usize; count];
    for (rank, slot) in slots.iter_mut().enumerate() {
        *slot = if rank.is_multiple_of(2) {
            rank / 2
        } else {
            count.div_ceil(2) + rank / 2
        };
    }
    let mut radii_by_slot = vec![0.0_f32; count];
    for ((_, size), &slot) in ranked.iter().zip(&slots) {
        radii_by_slot[slot] = local_spacing * (*size as f32).sqrt();
    }
    let chord = 2.0 * (std::f32::consts::PI / count as f32).sin();
    let mut ring_radius = 0.0_f32;
    for slot in 0..count {
        let next = (slot + 1) % count;
        ring_radius =
            ring_radius.max((radii_by_slot[slot] + radii_by_slot[next] + gap) / chord.max(0.1));
    }
    ring_radius = ring_radius.max(radii_by_slot.iter().copied().fold(0.0_f32, f32::max) + gap);

    if let Some(root) = central_root {
        positions[root as usize] = Vec2::ZERO;
    }
    for ((branch, _), &slot) in ranked.iter().zip(&slots) {
        let angle = std::f32::consts::TAU * slot as f32 / count as f32;
        let center = Vec2::new(angle.cos(), angle.sin()) * ring_radius;
        let phase = stable_unit(ids[*branch as usize]) * std::f32::consts::TAU;
        let mut local_index = 0_usize;
        let mut stack = vec![*branch];
        while let Some(node) = stack.pop() {
            let radius = local_spacing * (local_index as f32).sqrt();
            let local_angle = local_index as f32 * GOLDEN_ANGLE + phase;
            positions[node as usize] =
                center + Vec2::new(local_angle.cos(), local_angle.sin()) * radius;
            local_index += 1;
            stack.extend(topology.children[node as usize].iter().rev().copied());
        }
    }
    true
}

pub(super) fn symbol_or_unknown(read: &vecgra::ReadGuard<'_>, symbol: u32) -> Arc<str> {
    read.symbol(symbol).unwrap_or("Unknown").into()
}

pub(super) fn owned_properties(
    read: &vecgra::ReadGuard<'_>,
    properties: &[vecgra::Property],
) -> Arc<[SceneProperty]> {
    properties
        .iter()
        .map(|property| SceneProperty {
            key: symbol_or_unknown(read, property.key),
            value: property.value.clone(),
        })
        .collect()
}

pub(super) fn initialize_by_label(
    positions: &mut [Vec2],
    ids: &[NodeId],
    node_labels: &[Arc<str>],
) {
    let mut labels = BTreeMap::<Arc<str>, usize>::new();
    for label in node_labels {
        let next = labels.len();
        labels.entry(label.clone()).or_insert(next);
    }
    let label_count = labels.len().max(1);
    let mut label_offsets = vec![0_usize; label_count];
    let cluster_radius = 180.0 + (ids.len() as f32).sqrt() * 4.0;
    const GOLDEN_ANGLE: f32 = 2.399_963_1;

    for index in 0..ids.len() {
        let group = labels[&node_labels[index]];
        let group_angle = std::f32::consts::TAU * group as f32 / label_count as f32;
        let center = Vec2::new(group_angle.cos(), group_angle.sin()) * cluster_radius;
        let local_index = label_offsets[group];
        label_offsets[group] += 1;
        let local_radius = 8.0 * (local_index as f32 + 1.0).sqrt();
        let local_angle = local_index as f32 * GOLDEN_ANGLE + stable_unit(ids[index]);
        positions[index] = center + Vec2::new(local_angle.cos(), local_angle.sin()) * local_radius;
    }
}

pub(super) fn layout_orbits(
    positions: &mut [Vec2],
    ids: &[NodeId],
    labels: &[Arc<str>],
    degrees: &[u32],
    edge_length: f32,
) {
    let mut order: Vec<_> = (0..positions.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        degrees[right]
            .cmp(&degrees[left])
            .then_with(|| labels[left].cmp(&labels[right]))
            .then_with(|| ids[left].cmp(&ids[right]))
    });

    if let Some(&center) = order.first() {
        positions[center] = Vec2::ZERO;
    }
    let spacing = edge_length.max(16.0);
    let mut cursor = 1_usize;
    let mut ring = 1_usize;
    while cursor < order.len() {
        let radius = ring as f32 * spacing;
        let capacity = ((std::f32::consts::TAU * radius / spacing).floor() as usize).max(6);
        let count = capacity.min(order.len() - cursor);
        let phase = if ring.is_multiple_of(2) {
            std::f32::consts::PI / count as f32
        } else {
            0.0
        };
        for offset in 0..count {
            let angle = std::f32::consts::TAU * offset as f32 / count as f32 + phase;
            positions[order[cursor + offset]] = Vec2::new(angle.cos(), angle.sin()) * radius;
        }
        cursor += count;
        ring += 1;
    }
}

fn fit_positions_to_frame(positions: &mut [Vec2], frame: Rect) {
    if positions.is_empty() {
        return;
    }
    let source = Rect::from_points(positions);
    let scale = (frame.width() / source.width())
        .min(frame.height() / source.height())
        .max(0.000_1)
        * 0.94;
    let source_center = source.center();
    let frame_center = frame.center();
    for position in positions {
        *position = (*position - source_center) * scale + frame_center;
    }
}

pub(super) fn stable_unit(mut value: u64) -> f32 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value as u32) as f32 / u32::MAX as f32
}

fn cell_key(position: Vec2, cell_size: f32) -> (i32, i32) {
    (
        (position.x / cell_size).floor() as i32,
        (position.y / cell_size).floor() as i32,
    )
}

pub(super) fn recenter(positions: &mut [Vec2]) {
    if positions.is_empty() {
        return;
    }
    let center = positions
        .iter()
        .copied()
        .fold(Vec2::ZERO, |sum, position| sum + position)
        / positions.len() as f32;
    for position in positions {
        *position = *position - center;
    }
}
