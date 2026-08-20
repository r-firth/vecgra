# Vecgra Studio

Vecgra Studio is the native explorer for Vecgra. It lives in this
workspace, opens the same portable `.vg` container directly, and treats graph
structure and vector relevance as two views of one result rather than two
separate products.

This document describes the product, its interaction model, and its renderer.

## Product contract

| Area | Decision |
|---|---|
| Product name | Vecgra Studio |
| Binary | `vecgra-studio` |
| Application ID | `dev.vecgra.studio` |
| Workspace | Sibling crates beside the headless database and CLI |
| Database access | Direct, in-process, read-only open |
| Initial platform | macOS 26 on Apple Silicon, verified locally |
| Intended platforms | macOS, Windows, and Linux after native launch evidence |
| Distribution | Unresolved; raw development binary is not a package |
| Durable data | The selected `.vg` file; Studio preferences are separate |
| Cache | Rebuildable layout and level-of-detail data, never graph truth |
| Network | None for text/hash search; Qwen query embedding uses OpenRouter when explicitly selected |

Studio uses [Bezel](https://github.com/crabtalk/bezel) for its material-aware
application chrome and pins commit
`49ee7b1d422d761551b3f6951d74cdfb87314241`. Bezel's GPUI fork is pinned at
`e0b415b4bcffe4a2f05221544a788d482f5f6f50`; the workspace patches
`gpui-component` onto that same source so the desktop graph has one GPUI
runtime type universe. The maintained `gpui-component` input remains the
Unicode/IME-aware command field; Bezel owns the segmented modes, graph control
bar, view navigation, focus rings, and keyboard traversal.

GPUI also resolves disabled GPL tracing facades and the obsolete `block` 0.1.6
FFI declaration. The workspace replaces the disabled tracing API with
Apache-2.0 no-op compatibility crates and applies a narrow MIT-licensed source
patch to `block`; neither changes Studio behavior. The exact provenance,
licence boundary, and removal conditions are in
[the dependency policy](dependencies.md).

The database crate remains headless. Building the database and CLI must not
require the desktop stack, so they remain the workspace default members.

## Interaction contract

The unit of exploration is a query result or bounded expansion, not an
attempt to draw every record as a giant hairball.

1. Open a `.vg` file and show useful structure immediately.
2. Search semantically, structurally, or with both in one command box.
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
result. At widths below 1,280 logical pixels, the single command rail becomes
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
parent's circular ordering; discovery edges remain prominent while cross-links
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
deterministic screenshot runner.

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
command box and screenshot runner.

The `path <start> <end> [both|out|in] [relationship-label|-] [max-hops]`
command searches the complete database off the application thread. It then
hydrates a report with properties and vector counts after the engine returns
an exact result. Multi-hop searches adapt between the two frontiers and
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

The app chrome combines Bezel's material, control, and navigation vocabulary
with GPUI Component's maintained text-input behavior. The graph is one custom
canvas/custom element, not one retained UI element per node or edge.

The renderer groups visible edges into a small number of `Path` batches
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
vecgra                       vecgra-embedding
durable graph/vector truth        selected query embedding adapter
and query execution               (hash or Qwen/OpenRouter)
              \                    /
vecgra-studio-core
    bounded search, owned snapshots, layout, LOD, projection, hit testing
        |
vecgra-studio-ui
    GPUI entities, canvas, command box, inspector, themes
        |
vecgra-studio
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

- Graphite `#0B1116`: canvas and darkest background;
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
| Vecgra / database   semantic + graph command                     4.8ms |
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
score marks. This makes Vecgra's native fusion visible without decorating
every node.
