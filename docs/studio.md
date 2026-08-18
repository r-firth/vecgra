# VectorGraph Studio

VectorGraph Studio is the native explorer for VectorGraph. It lives in this
workspace, opens the same portable `.vg` container directly, and treats graph
structure and vector relevance as two views of one result rather than two
separate products.

This document is the initial product, interaction, rendering, and visual
contract. It deliberately distinguishes the first vertical slice from release
claims.

## Product contract

| Surface | Decision |
|---|---|
| Product name | VectorGraph Studio |
| Binary | `vg-studio` |
| Application ID | `dev.vectorgraph.studio` (provisional; release blocker) |
| Workspace | Sibling crates beside the headless database and CLI |
| Database access | Direct, in-process, read-only open for the first slice |
| Initial platform | macOS 26 on Apple Silicon, verified locally |
| Intended platforms | macOS, Windows, and Linux after native launch evidence |
| Distribution | Unresolved; raw development binary is not a package |
| Durable data | The selected `.vg` file; Studio preferences are separate |
| Cache | Rebuildable layout and level-of-detail data, never graph truth |
| Network | None for text/hash search; Qwen query embedding uses OpenRouter when explicitly selected |

`gpui-component` is pinned to commit
`222cf9644fbfd9bdff970a0308f9ce216b2c0818`. Its manifest currently names
Zed's GPUI Git source without a revision, so the application uses the identical
source declaration and pins the resolved Zed commit in `Cargo.lock`. Using a
separate `rev` declaration creates two incompatible GPUI type universes. An
upstream revision-pinning mechanism remains a release-hardening item.

The database crate remains headless. Building the database and CLI must not
require the desktop stack, so they remain the workspace default members.

## Interaction contract

The unit of exploration is a query result or bounded expansion, not an
attempt to draw every record as a giant hairball.

1. Open a `.vg` file and show useful structure immediately.
2. Search semantically, structurally, or with both in one command surface.
3. Pan and zoom without waiting for database work or graph layout.
4. Select a node or relationship and inspect its label, properties, vectors,
   neighbors, and provenance without losing the current view.
5. Expand, contract, pin, hide, and compare bounded neighborhoods.
6. Move smoothly between overview, communities, individual elements, and
   evidence paths using semantic level of detail.
7. Make query cost, result count, vector score, and structural distance
   inspectable without turning the main canvas into a monitoring dashboard.

Direct manipulation has a separate presentation model from graph truth. A
press on a node wins over canvas pan, preserves the exact world-space grab
offset, and follows the pointer without easing. Crossing the drag threshold
pins that node; a click does not. Up to 512 one-hop neighbors receive a bounded
physical response, so interaction work is O(degree) rather than O(graph size).
`Release` returns the selected pin to the active layout. Force, Structure, and
Orbit arrangements compute off the application thread, preserve existing pins,
and retarget the current spring without zeroing velocity. Structure follows the
[Omega](https://arxiv.org/abs/2512.21901) pipeline: low-rank resistance-distance
spectral coordinates followed by random-pair sparse stress SGD. Disconnected
components are solved independently and packed, rather than inventing edges.
A new drag can interrupt settling at its current rendered position. Reduced
Motion keeps direct dragging functional and snaps arrangement transitions to
their semantic endpoint.

Zoom is pointer-anchored from 8% through 102,400% of fit scale. Large scenes
use aggregate community bins at ordinary distances, then switch back to true
individual elements at deep zoom (12x for 12k nodes, 24x for 25k+). In that
element view, ordinary edges whose endpoints are both offscreen are culled;
selected and emphasized paths retain their dedicated rendering passes. This
keeps structurally equivalent Omega coordinates close without making them
permanently uninspectable or distorting the resistance-distance objective.

Loading, layout, and query work are explicit states: idle, loading, ready,
empty, or failed. Each request has a generation; superseded work cannot replace
a newer scene. CPU layout and database reads never run on the GPUI application
thread.

The title-bar input is a maintained Unicode/IME-aware component. `Cmd-K`
focuses it; Enter runs the selected Text, Semantic, or Hybrid mode; Up/Down
moves the ranked selection; a second Enter or click selects and frames the
result. At widths below 1,120 logical pixels, the single command rail becomes
two rows so search, layout, pin, and camera controls remain present at the
760-pixel minimum instead of clipping or silently disappearing. Text retrieval
scans properties with a bounded top-K heap rather than materializing every
match. Semantic retrieval uses the database's adaptive native vector index
across nodes and relationships. Hybrid retrieval fuses normalized lexical and
vector signals and deduplicates multiple vector facets at the whole-element
level. The selected result exposes labelled Text and Vector contribution rails,
so relevance is not encoded by a single opaque score or color alone. Search
failures are visible, and Hybrid can retain text results with an explicit
warning if its remote embedding provider is unavailable.

A successful query retargets an interruptible camera spring to frame its
visible matches and fades in a score-weighted graph lens. Nodes receive ranked
halos; matching relationships remain directed, colored, and captioned; all
other structure recedes continuously rather than disappearing in a scene
swap. Activating a result retargets from the current presentation and velocity:
the selected node, or both endpoints of a selected relationship, becomes the
root of a bounded one-hop constellation while its neighbors fan into stable
rings. Double-clicking a node, or pressing Enter after selecting it, opens a
112-node/168-edge two-hop context lens. Expansion rotates across frontier
branches and relationship types, reserves capacity for the second hop, and
places semantic hops in separate radial bands. Child groups retain their
parent’s circular ordering; discovery edges remain prominent while cross-links
stay visible but quieter. The underlying Auto/Force/Structure/Orbit targets are
preserved. Escape or the
Overview row springs positions, camera, and emphasis back to that saved
presentation. Reduced Motion snaps to the same semantic endpoints.

The relationship and node-label taxonomies are controls rather than a passive
legend. Activating a relationship row raises every visible relationship of
that type and both endpoints into the relevance field. Activating a node-label
row raises matching nodes while retaining quieter incident evidence. The active
row carries a focusable toggle semantic, a left-edge structural marker, and a
literal `LENS` label, so state is not encoded by color alone. Selecting it
again, pressing Escape, or choosing Overview clears the lens. Starting a search
retires any taxonomy/context lens first, preventing stale emphasis behind empty
or failed search states. A command-selected label outside the usual top-ten
summary replaces the last row while active, so the current control never falls
out of view. `facet node <label>` and
`facet relationship <label>` provide the same path through the command box and
deterministic visual harness.

Exact paths are a separate evidence mode rather than another generic filter.
For direct manipulation, selecting a node exposes `Set as path origin` in the
Inspector. The origin receives a persistent, text-labelled canvas marker and
the evidence rail enters an explicit destination-picking state. The same rail
sets traversal to either direction, outgoing from the origin, or incoming to
the origin, plus a one-, two-, four-, or six-hop bound. Selected controls,
literal direction text, and the `≤ N` bound duplicate color state. Selecting a
different node places its copper `TO` endpoint and the full-width `Trace exact
path` action directly in the evidence rail. That rail owns execution at every
supported width; Enter runs the same action from a ready destination, while the
wide Inspector mirrors it as a convenience. Direction or hop-limit changes
preserve that destination-ready state, and both endpoint markers remain on the
canvas while background search is pending. Escape or Overview cancels the
draft. These are ordinary focusable buttons, so the workflow does not depend
on pointer precision, color recognition, or a panel that disappears at the
760-pixel minimum.

`path-start <node> [both|out|in] [max-hops]` drives the same state for the
command surface and visual harness.

The `path <start> <end> [both|out|in] [relationship-label|-] [max-hops]`
command searches the complete database off the application thread, then
hydrates an owned report—including properties and vector counts—after the exact
engine result is known. Multi-hop searches adapt between the two frontiers and
expose the selected physical plan. The left panel becomes an ordered evidence
rail: endpoints, each relationship step, traversal orientation, plan, hop
bound, elapsed time, and visited/read work map directly to the bright directed
chain on the canvas. A compact planner ledger splits expanded nodes into
Celadon `FROM` and Copper `TO` counts connected by a rule, so adaptive work is
inspectable without relying on color or an ambiguous percentage. Step rows are
stable, focusable controls; pointer, Enter, or Space selects and frames the
matching relationship for inspection.

The path state owns both a generation and its held task. A new path, search,
facet, context, Escape, or Overview retires stale work and prevents a late
background result from overwriting the current presentation. `Found`,
`NotFoundWithinHops`, and `ExpansionLimit` have different language: the latter
is explicitly incomplete and never presented as graph absence. When an exact
path includes elements outside the bounded snapshot, a background scene step
injects their fully hydrated records and exact relationships before
presentation. Existing overview positions and pins are preserved; missing path
nodes are placed deterministically between the nearest visible anchors. The
bounded scene may therefore grow by the small path size, but the canvas and
rail never disagree about found evidence. Evidence framing may use the full
24× focus-navigation range so a short chain remains readable against a very
large overview. Overview restores the saved camera and layout, then clears path
selection and emphasis.

## Rendering contract

The app chrome uses ordinary GPUI elements and GPUI Component behavior. The
graph is one custom canvas/custom element, not one retained UI element per
node or edge.

The first renderer groups visible edges into a small number of `Path` batches
by a stable relationship-type palette and emits visible nodes as quads inside
one paint layer. GPUI uploads and draws those primitives on the GPU. The
application thread only performs bounded projection, culling, hit-region
preparation, and scene submission. Relationship captions and arrowheads are
sparse overlays rather than retained elements per edge.

Rendering levels are semantic:

- overview: density and communities;
- middle distance: supernodes, dominant relationships, and result contours;
- detail: individual nodes, relationships, labels, and vector scores;
- inspection: selected evidence paths and properties above the base scene.

Relationship meaning follows the same levels: the overview exposes the
taxonomy and counts; middle distance gives each readable type a representative
caption and arrow; selecting a node labels its readable incident edges; and a
selected edge is always drawn, captioned, directed, and inspectable even when
the base edge layer is aggregated. Text and direction duplicate the palette's
meaning, so type is not encoded by color alone.

Initial budgets are 50,000 individually drawn nodes and 150,000 visible edge
segments. Above either budget, the renderer must aggregate instead of silently
dropping random elements. Direct manipulation targets 120 Hz on the development
machine, with application-thread frame preparation below 4 ms p95 and no
continuous frame requests while idle.

GPUI's pinned public API does not expose custom application shaders or external
GPU instance buffers. If measurements show its CPU scene construction is the
limiter, the next renderer is a narrow upstreamable `GraphPrimitive`: immutable
structure-of-arrays node/edge buffers, backend-specific instanced shaders, and
GPUI-owned clipping/composition. It is not a second overlapping swapchain and
does not introduce a GPU-to-CPU texture copy.

## Architecture

```text
vectorgraph                       vectorgraph-embedding
durable graph/vector truth        selected query embedding adapter
and query execution               (hash or Qwen/OpenRouter)
              \                    /
vectorgraph-studio-core
    bounded search, owned snapshots, layout, LOD, projection, hit testing
        |
vectorgraph-studio-ui
    GPUI entities, canvas, command surface, inspector, themes
        |
vectorgraph-studio
    process startup, platform policy, menus, windows, diagnostics
```

Scene snapshots are compact, immutable, and independently revisioned. They use
structure-of-arrays for hot geometry and IDs; strings and properties are cold
inspection data. The UI swaps an `Arc<SceneSnapshot>` after background work
completes. A UI-owned `GraphWorkspace` holds presentation positions, layout
targets, velocities, adjacency, and pins; moving a node never mutates the
database snapshot. Camera movement never rebuilds database state or graph
layout.

## Visual direction

The interface should feel like a precise scientific instrument rather than a
generic admin dashboard. The canvas receives nearly all visual drama; chrome
is quiet, dense, and legible.

Palette:

- Graphite `#0B1116`: canvas and deepest surface;
- Strata `#162129`: panels and raised controls;
- Mist `#D9E3E8`: primary text and selected labels;
- Cobalt `#4D8DFF`: vector relevance and semantic search;
- Copper `#E49562`: structural emphasis and one member of the stable
  relationship-type palette;
- Celadon `#65D1A5`: current selection, success, and pinned truth.

Typography uses the platform system sans for navigation and inspection, and a
system monospace face for queries, IDs, scores, and plans. Type size and weight
carry hierarchy; cards, outlines, and uppercase labels do not.

```text
+-----------------------------------------------------------------------+
| VG / database       semantic + graph command                     4.8ms |
+--------------+--------------------------------------+-----------------+
| views        |                                      | inspector       |
| labels       |          GPU GRAPH SCENE             | identity        |
| saved query  |                                      | properties      |
| history      |       semantic lens / minimap        | evidence path   |
|              |                                      | query plan      |
+--------------+--------------------------------------+-----------------+
| 2,391 nodes / 5,018 edges   overview > cluster > detail     ready     |
+-----------------------------------------------------------------------+
```

The signature element is the **semantic lens**. Relationship types use a
restrained stable palette with caption and direction redundancy; vector
relevance appears as a Cobalt field around the current result or selection. At
distance it becomes a density contour; close up it becomes ranked halos and
score marks. This makes VectorGraph's native fusion visible without decorating
every node.

## Design self-critique

A dark three-column developer tool, blue accent, command bar, and node-link
canvas are all familiar choices. They become generic if every region is a
rounded card, if the canvas is a random force-directed hairball, or if color is
only decoration. Studio therefore spends boldness only on the semantic lens,
uses continuous panel planes instead of card grids, keeps visible primitives
query-bounded, and assigns color stable meaning. Docking is optional
customization, not the default information architecture.

## Verification gates

- Pure tests: snapshot ownership, layout determinism, projection, LOD, hit
  testing, selection, and stale-generation rules.
- GPUI tests: actions, focus, pointer/keyboard selection, zoom anchoring,
  loading/error states, and idle frame behavior.
- Runtime: debug and release launch, real `.vg` file, resize, pan, zoom,
  selection, close/reopen, and idle.
- Visual: deterministic database at fixed bounds, default/selected/loading/
  empty/error states, dark/light/high-contrast, and small/target/wide windows.
- Performance: application-thread preparation, draw/present time, primitive
  count, memory, first useful scene, and interaction latency at each LOD.

## Current vertical-slice evidence

The first end-to-end slice is implemented in this repository. It is deliberately
not presented as a finished explorer:

- `vectorgraph-studio-core` opens the portable file read-only, takes a bounded
  owned snapshot, and chooses a topology-aware Auto layout. Small directed
  forests use a radial hierarchy; large forests use an O(V+E) top-level
  subtree analysis to seed a packed constellation before sampled refinement.
  Small general graphs retain a bounded cell-force pass; graphs with 256 or
  more nodes use a deterministic portable-Rust implementation of the
  [SNAP-tFDP](https://arxiv.org/abs/2608.01907) objective: bounded
  t-distribution forces and edge-centric negative sampling, with O(E) work per
  epoch and no spatial grid to leak a lattice into the drawing. It also owns
  deterministic Auto/Force/Structure/Orbit targets and the tested presentation
  spring/pin/neighbor-physics model;
- `vectorgraph-studio-ui` swaps the completed scene on the application thread,
  paints the graph as one canvas, aggregates fitted large scenes into density
  bins, batches edges by relationship type, and adds bounded captions and
  arrowheads. It supports pointer/trackpad navigation, direct node
  manipulation, persistent pin/release, animated arrangement, node/edge
  selection, relationship inspection, ranked whole-element search, animated
  result framing, reversible one-hop focus constellations, and the branch-aware
  two-hop node context lens. Relationship and node-label rows are focusable,
  stable-ID taxonomy toggles that drive the same animated graph lens. Ranked
  rows use stable graph-element identities,
  list semantics, selected state, and descriptive accessibility labels. The
  exact-path draft keeps direction, hop limit, destination evidence, and its
  primary execution action together in the responsive evidence rail. The
  native title-bar component reserves platform window-control space and owns
  drag/double-click behavior;
- `vectorgraph-studio` owns application identity, window policy, key bindings,
  assets, and an opt-in Metal offscreen capture harness.

The primary product-shaped fixture now comes from the GitHub engineering-history
importer rather than an AST. A 37 MB Zed snapshot contains 12,601 nodes, 25,569
typed relationships, and 40,209 node/relationship vectors: issues, PRs,
discussions, comments/replies, people, reviews, commits, files, releases, and
taxonomy. After the sampled-force layout landed, the default capture loaded and
laid out the complete graph in 30.3 ms; its three startup draws measured 3.326
ms p50 and 8.552 ms p95/max. A 900% detail capture measured 3.326 ms p50 and
6.795 ms p95/max and visually removed the former Cartesian bands. A selected
`CLOSES` relationship capture measured 4.000 ms p50 and 6.611 ms p95/max and
showed the type, direction, and endpoint identities in the inspector. Explicit
Force target generation took 15.773 ms (the prior cell-force measurement was
23.043 ms); the 120 Hz spring simulation settled in 173 frames using 4.482 ms
total CPU. The optional resistance-distance Structure arrangement took about
one second end-to-end off the application thread on the same 12,601/25,569
scene; its
settled overview capture drew in 1.236 ms p50 and 7.922 ms p95/max across four
startup draws. The layout benchmark found no duplicate coordinates; projected
p10 nearest-node clearance rises from 1.91 px at the former 80x zoom ceiling to
24.42 px at 1,024x. A centered deep-element capture drew in 3.635 ms p50 and
11.493 ms p95/max across four startup draws. The two-hop context lens around a
157-degree node retained 37
direct and 74 second-hop nodes plus 168 typed relationships; its settled
capture drew in 5.018 ms p50 and 7.840 ms p95/max across three startup draws.
These are local release-mode smoke
measurements with deterministic hash embeddings, not sustained-frame,
cross-machine, or semantic-quality claims.

The same complete Zed fixture returned 24 bounded Hybrid results for `memory
leak` in 25.7 ms in a release capture, including whole-element score fusion
across node and relationship vectors. The result-state capture drew in 2.096 ms
p50 and 7.905 ms p95/max across four startup draws. This measures the current
scan-plus-native-vector path on one machine; it is not yet a full-text-index or
latency-distribution claim.

The responsive search state was also captured from the same fixture at the
1,340×820 target and exact 760×520 minimum logical bounds on a 2× display. The
minimum layout retained every search, arrangement, release, and camera control,
kept the ranked result list scrollable, and left a 466-point-wide graph canvas;
the selected Hybrid result exposed labelled Text and Vector rails without
clipping. These are structural visual checks, not cross-platform evidence.

The active `AUTHORED` relationship facet was captured from a deterministic
193-node/296-edge fixture at the same 1,340×820 target and 760×520 minimum
logical bounds on a 2× display. Both captures retain the active row’s blue
structural stroke and `LENS` marker, make all 103 matching directed edges and
their endpoints legible, keep unrelated topology as dim context, and preserve
every toolbar control. The release harness recorded 4.538 ms p50 across three
startup draws at the target size and 3.619 ms p50 at the minimum. The p95/max
values were 11.158 ms and 9.839 ms respectively; these are visual smoke samples,
not sustained-frame latency distributions.

With the animated result lens settled on an `AUTHORED` relationship, the
bounded 11-element constellation capture drew in 2.018 ms p50 and 8.131 ms
p95/max across four startup draws. The visual harness advances the same spring
math deterministically before capture; interactive motion remains driven by
display frames in the normal application.

The larger structural stress fixture was generated from VectorGraph's own Rust
source using the Tree-sitter importer and deterministic hash embedder. It contains
110,303 nodes, 110,302 relationships, and 441,157 node/relationship vectors.
Studio's bounded view loaded and laid out 50,000 nodes plus 49,999 relationships
in 86.5 ms using the topology-aware constellation plus sampled refinement in a
warm-file optimized run on the development machine. An explicit 50k-node Force
rearrangement computed in 35.3 ms off-thread; the higher-quality Structure
arrangement computed in 1,585.5 ms off-thread on the same scene. Its 120 Hz
spring solver consumed
0.099 ms CPU per simulated frame and settled in 171 frames. The three final
startup draws measured 2.945 ms p50 and 10.174 ms p95/max.
With the relationship layer enabled on the same 50k-node view, a 200% middle-
distance capture measured 4.424 ms p50 and 9.273 ms p95/max across three draws;
the fitted selected-edge capture measured 4.182 ms p50 and 8.675 ms p95/max.
These tiny capture samples are smoke evidence, not latency distributions.
The unoptimized load/layout
path was 584.5 ms after replacing force layout for this forest-shaped dataset,
down from 17,970.6 ms. These are local smoke measurements, not sustained-frame
or cross-machine claims.

The exact path from repository node 0 to sampled-out syntax node 100000 adds
seven nodes and seven relationships to that 50k scene before presenting the
complete seven-hop chain. The adaptive either-direction plan reports 129 unique
visited nodes, 62 expansions, and 188 relationship reads; the former one-sided
plan reported 22,928, 15,293, and 38,219 respectively for the identical chain.
The final 24×-framed release captures drew in 3.211 ms p50 at 1,340×840 and
2.574 ms p50 at the exact 760×520 minimum; their p95/max values were 10.437 ms
and 7.676 ms.
Separate origin and destination-candidate captures preserve the explicit
`ORIGIN` marker and path intent at target width. The configurable
intent captures drew in 7.598 ms p50 at 1,340×840, 4.551 ms p50 at the exact
minimum, and 5.902 ms p50 with a destination candidate selected; p95/max was
9.470 ms, 7.492 ms, and 8.077 ms respectively. After moving destination
execution into the rail, final candidate captures drew in 5.874 ms p50 and
9.642 ms p95/max at 1,340×820, and 4.123 ms p50 and 7.676 ms p95/max at the
exact 760×520 minimum. These remain startup-draw smokes, not
interaction-latency distributions.

The final planner-ledger captures of the same 27-from/35-to query drew in 3.152
ms p50 and 10.117 ms p95/max at 1,340×820, and 2.787 ms p50 and 7.623 ms
p95/max at 760×520. The minimum-width destination draft with persistent
`ENTER RUNS` guidance drew in 4.198 ms p50 and 7.934 ms p95/max. These are
also startup-draw smokes rather than interaction-latency distributions.

The repeatable layout benchmark uses visited, bounded graph characterization,
so it is safe for cyclic graphs and DAGs with shared descendants as well as
forests:

```sh
cargo run --release -p vectorgraph-studio-core \
  --example layout_bench -- graph.vg

cargo run --release -p vectorgraph-studio-core \
  --example layout_bench -- graph.vg structure
```

The repeatable visual path is:

```sh
cargo build --release -p vectorgraph-studio --features visual-test

VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio.png \
  target/release/vg-studio graph.vg

# Capture a deterministic inspection state.
VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio-edge.png \
VG_STUDIO_CAPTURE_COMMAND='edge 0' \
  target/release/vg-studio graph.vg

# Wait for search, activate ranked result 0, settle every spring, then capture.
VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio-focus.png \
VG_STUDIO_CAPTURE_COMMAND='memory leak' \
VG_STUDIO_CAPTURE_RESULT=0 \
  target/release/vg-studio graph.vg

# Exercise the exact minimum supported window geometry.
VG_STUDIO_WINDOW_WIDTH=760 VG_STUDIO_WINDOW_HEIGHT=520 \
VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio-minimum.png \
  target/release/vg-studio graph.vg

# Activate a taxonomy row through the same command path as pointer input.
VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio-facet.png \
VG_STUDIO_CAPTURE_COMMAND='facet relationship AUTHORED' \
  target/release/vg-studio graph.vg

# Set a path origin, then select a destination candidate without executing it.
VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio-path-origin.png \
VG_STUDIO_CAPTURE_COMMAND='path-start 0 out 4' \
VG_STUDIO_CAPTURE_CENTER=1 \
  target/release/vg-studio graph.vg

# Wait for the complete database path, settle its lens, then capture the
# evidence rail and its one-for-one directed canvas chain. Sampled-out path
# records are injected before the frame is drawn.
VG_STUDIO_CAPTURE=/tmp/vectorgraph-studio-path.png \
VG_STUDIO_CAPTURE_COMMAND='path 0 42 out SUPPORTS 6' \
  target/release/vg-studio graph.vg
```

The next evidence gates are pointer-to-photon interaction-latency distributions,
pinned-state captures, multi-selection and marquee movement, and Windows/Linux
native launch evidence. A
custom GPUI graph primitive remains contingent on those measurements; the
current public quad/path pipeline is already fast enough for the fitted
50k-node overview smoke case.
