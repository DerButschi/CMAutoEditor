# OSM road/network recovery plan — TDD milestones after refactor regression

**Repository:** `DerButschi/CMAutoEditor`  
**Old reference branch:** `feature/cm_terrain_extractor`  
**Current refactored branch:** `cm_terrain_editor_refactoring`  
**Basis:** `osm_refactor_state_comparison(2).md`  
**Plan focus:** restore and improve road/network extraction semantics through test-driven milestones  
**Primary goal:** make the new refactored architecture produce road networks that are at least as topologically correct as the old graph-based implementation, while keeping the new cleaner infrastructure.

---

## 0. Executive intent

The refactor improved structure, typing, occupancy, output assembly, and testability, but the road/network core is not yet a faithful replacement for the old code. The old implementation was slow and messy, but it carried several important invariants:

```text
continuous OSM topology
  -> routed square topology
  -> persistent route/connection state
  -> tile-feasible intersection constraints
  -> priority-sensitive occupancy
```

The current refactored implementation has the right module names and phases, but several contracts are too weak:

1. **Topology anchors are selected too naively.**
2. **Route node vs road-tile-cell semantics are ambiguous or wrong.**
3. **There is no explicit raster-spine / line-support contract.**
4. **Tile assignment happens too late and does not feed back strongly enough into routing.**
5. **Priority/stage semantics for linear features are weaker than in the old implementation.**
6. **Accepted routes do not update a persistent road-connection state equivalent to the old graph-node `connections`.**
7. **Tests are too component-local and do not yet lock down whole-pipeline road semantics.**

This plan turns those findings into milestones. Each milestone is intentionally decision-complete enough to use as a project/Codex planning file.

---

## 1. Guiding principles

### 1.1 Keep the new architecture, repair the semantics

Do **not** revert to the old `processing.py` / `path_search.py` design wholesale. Keep the new module split and typed infrastructure:

```text
osm_extraction/
  config_schema.py
  grid_index.py
  occupancy.py
  network_topology.py
  network_routing.py
  tile_assignment.py
  building_fitter.py
  output_rows.py
```

But strengthen the network contracts until they encode the same essential semantics the old graph stack encoded.

### 1.2 Tests must run against extraction output, not just helper functions

Component tests are useful, but the observed failures are integration failures. The main acceptance tests must operate on representative GeoJSON fixtures and inspect:

- final CM output rows,
- debug route paths,
- topology nodes,
- selected anchors,
- route step cells,
- required tile direction sets,
- selected tile variants,
- failure diagnostics.

A road test is not passed merely because the router returns a path. It is passed only if the final output rows encode a legal, connected Combat Mission road network.

### 1.3 Make contracts explicit before optimizing

Runtime matters, but the immediate problem is correctness. Milestones should first create observable contracts:

```text
source line -> raster support/spine -> topology anchors -> routed nodes
            -> step cells -> required dirs -> legal tile rows
```

Once those are enforced and tested, performance can be improved without changing behavior.

### 1.4 Prefer hard failures over broken output

For roads, a route that cannot be assigned legal tiles should not silently generate a broken network. It should become a structured failure and trigger repair:

```text
reroute -> shift anchor -> split intersection -> drop lower-priority branch with diagnostic
```

---

## 2. Test taxonomy

The recovery should be driven by three levels of tests.

### 2.1 Unit contract tests

These verify small pure contracts, for example:

- `GridIndex.projected_to_cell` and `cell_center` are inverse-consistent.
- `line_to_raster_spine` returns ordered cells close to a segment.
- `route_nodes_to_step_cells` returns exactly one step cell per node transition.
- `CompiledTileCatalog` can answer whether direction set `{N, E, S}` has a legal road tile.
- `LinearNetworkState` stores connection bits consistently.

### 2.2 Scenario integration tests

These run the extraction pipeline on small GeoJSON fixtures and inspect final output rows plus debug artifacts. These are the most important tests.

Example scenario test:

```text
Input: one horizontal road and one vertical road crossing at the map center.
Expected:
  - exactly one logical intersection cluster,
  - all four arms connect to that cluster,
  - final output contains a legal 4-way or explicit legal split-intersection pattern,
  - no road arm terminates one cell before the intersection,
  - no diagonal continuation is used if the selected road tile catalog cannot represent it.
```

### 2.3 Regression comparison tests

Where practical, run both the old and new extraction on the same fixtures and compare semantic metrics rather than exact row equality.

Useful metrics:

- connectivity component count,
- number of failed network edges,
- route-to-source distance distribution,
- number of legal/illegal intersections,
- road cells with impossible direction sets,
- building-road collision count,
- output row count deltas within expected tolerance.

The old output should not be treated as perfect. It is a reference for important invariants, not a golden truth.

---

## 3. Core debug artifacts required before algorithm changes

Before repairing algorithms, the pipeline must expose the following debug layers for every network extraction:

| Debug layer | Purpose |
|---|---|
| `source_lines` | projected OSM linework after clipping and normalization |
| `topology_nodes` | continuous-space intersection/end/snap nodes |
| `topology_edges` | noded continuous edges |
| `raster_spines` | ordered support cells/cost support per topology edge |
| `anchor_candidates` | candidate grid anchors per topology node with scores/reasons |
| `selected_anchors` | chosen anchor or split-intersection anchor group |
| `routed_nodes` | lattice-node path per topology edge |
| `route_step_cells` | actual CM road tile cells per route transition |
| `connection_bits` | per-cell N/E/S/W connection bitset after route reservation |
| `tile_required_dirs` | direction set required for each output road cell |
| `selected_tiles` | chosen CM tile variant per cell |
| `tile_failures` | missing tile / impossible transition / conflict diagnostics |

These can initially be emitted as JSON/GeoJSON-like debug dictionaries in tests. Full UI visualization can come later.

---

## 4. Representative scenario fixtures

Create fixtures under something like:

```text
tests/terrain_extraction/osm_extraction/fixtures/network_recovery/
```

Each fixture should be tiny and deterministic. Use synthetic projected or GeoJSON coordinates with a known bbox so output can be reasoned about.

### 4.1 `straight_road_2pt.geojson`

**Shape:** one road `LineString` with two endpoints only.  
**Purpose:** prove that the extractor does not depend on intermediate OSM vertices.

Expected invariants:

- one connected road component,
- road cells form a continuous path,
- route follows the source line within tolerance,
- no diagonal road steps if road catalog does not allow diagonal continuation.

### 4.2 `diagonalish_2pt_equal_length.geojson`

**Shape:** two-point road from approximately `(0, 0)` to `(2, 2)` in grid terms, but using metric coordinates.  
**Purpose:** catch equal-length shortest-path ambiguity.

Expected invariants:

- among equal-length Manhattan alternatives, the path closer to the source line wins,
- mean/max distance to source line is below a strict threshold,
- route intersects or tracks the raster spine cells,
- path does not take a long L-shaped detour with the same step count.

This fixture directly targets the issue: endpoint-only shortest path is insufficient.

### 4.3 `ninety_degree_bend.geojson`

**Shape:** one road with a clear 90° turn.  
**Purpose:** verify route-step-cell direction extraction and legal bend tile assignment.

Expected invariants:

- exactly one bend location,
- bend cell has required direction set `{A, B}` for adjacent cardinal directions,
- selected tile exists in the road catalog,
- no duplicate/overlapping bend cells.

### 4.4 `four_way_crossing.geojson`

**Shape:** two roads crossing exactly.  
**Purpose:** enforce legal shared intersection.

Expected invariants:

- topology graph has one degree-4 intersection node,
- selected anchor or split-intersection pattern is legal,
- all four arms connect,
- final road graph has one connected component,
- there is no pair of roads crossing geometrically without a shared road connection unless explicitly configured as overpass/underpass.

### 4.5 `t_junction.geojson`

**Shape:** one road ending on another.  
**Purpose:** distinguish T-junction from crossing.

Expected invariants:

- one degree-3 topology node,
- one legal 3-way tile or legal split pattern,
- the terminating branch actually connects to the through road,
- no one-cell gap at the junction.

### 4.6 `staggered_near_miss.geojson`

**Shape:** two roads whose endpoints nearly meet but are separated by less/greater than snap tolerance in two variants.  
**Purpose:** test snap tolerance semantics.

Expected invariants:

- below tolerance: connected junction,
- above tolerance: separate endpoints or explicit no-connect diagnostic,
- behavior changes only when tolerance changes.

### 4.7 `minor_meets_major.geojson`

**Shape:** lower-priority/minor road meets major road near an awkward grid location.  
**Purpose:** priority-aware routing and anchor selection.

Expected invariants:

- major road stays closer to source line,
- minor road may shift more,
- minor road connects legally to the major road,
- major road cells are not overwritten by minor road cells.

### 4.8 `parallel_close_roads.geojson`

**Shape:** two near-parallel roads separated by roughly one or two cells.  
**Purpose:** ensure occupancy/connection state does not collapse distinct roads.

Expected invariants:

- two distinct route cell sequences,
- no unintended merging,
- if separation is impossible, lower-priority road fails or shifts with diagnostic instead of silently sharing invalid cells.

### 4.9 `road_near_building.geojson`

**Shape:** one road and one building footprint that would overlap under naive placement.  
**Purpose:** road occupancy must be valid before building fitting.

Expected invariants:

- road is placed first and legal,
- building avoids road cells,
- no building-road collision,
- building shift/IoU diagnostics are emitted.

### 4.10 `road_stream_crossing.geojson`

**Shape:** road and stream cross or meet depending config.  
**Purpose:** cross-network priority/stage semantics.

Expected invariants depend on intended CM semantics:

- if streams block roads, road shifts/bridge/failure is explicit;
- if crossings are allowed, both outputs remain valid;
- priority/stage order is observable.

---

## 5. Milestone M0 — Establish a semantic regression harness

### Purpose

Create the testing and debug foundation. Do not change road algorithms yet except to expose debug information.

### Decision

The current branch should not receive more road-algorithm patches before the team can observe exactly where a scenario fails:

```text
topology? anchor? route? step cell? tile assignment? output assembly?
```

### Deliverables

1. Add fixture directory and at least these initial fixtures:
   - `straight_road_2pt`
   - `diagonalish_2pt_equal_length`
   - `ninety_degree_bend`
   - `four_way_crossing`
   - `t_junction`
   - `road_near_building`

2. Add a test runner helper:

```python
run_osm_extraction_fixture(
    fixture_name: str,
    profile: str,
    config_name: str,
    bbox: BoundingBox | FixtureBBox,
    *,
    seed: int = 0,
    debug: bool = True,
) -> ExtractionTestResult
```

3. `ExtractionTestResult` should expose:
   - final output rows,
   - debug network artifacts,
   - diagnostics/stats,
   - helper methods for graph reconstruction from output rows.

4. Add output-row graph reconstruction for roads:
   - parse output road tile rows,
   - infer required/actual connection directions,
   - build a cell adjacency graph,
   - detect connected components,
   - detect illegal direction sets.

### Explanation

Most failures are currently visually obvious but not test-locatable. The test harness must make failures machine-checkable. For example, an intersection can be wrong because:

- topology missed the intersection,
- anchor snapped to a bad cell,
- route missed the anchor,
- step-cell extraction was wrong,
- tile assignment could not find a 4-way tile,
- output conflict cleanup deleted a road cell.

Those are different bugs and need different fixes.

### Acceptance tests

- Each fixture can run through the current refactored pipeline.
- Tests can fail with structured diagnostics.
- At least one current bad road case is reproduced as a failing test.
- The test result can print a compact ASCII grid for small fixtures.

Example ASCII debug output:

```text
..|..
..|..
--+--
..|..
..|..
```

or for failure:

```text
..|..
..x..
--.--
..|..
..|..
```

where `x` marks an illegal/missing connection.

### Done when

There is a red/green test loop for road fixtures. The first result may be red; that is expected.

---

## 6. Milestone M1 — Define and fix route node vs route tile-cell semantics

### Purpose

Eliminate the ambiguous `RouteRecord.cells` behavior. This is the most concrete likely bug and should be fixed before anchor/routing sophistication.

### Decision

A route must distinguish:

```python
RouteRecord.nodes       # lattice points / routing graph nodes
RouteRecord.step_cells  # CM road tile cells, one per node-to-node transition
```

Do **not** use the same object for both.

### Required contract

For every routed edge:

```python
len(route.nodes) >= 2
len(route.step_cells) == len(route.nodes) - 1
```

Each step cell corresponds to the CM square/tile occupied by the movement from `nodes[k]` to `nodes[k + 1]`.

### Key design question: how to map node transition to cell

For a cardinal transition between adjacent lattice nodes, the occupied road cell should be defined consistently with the existing CM grid coordinate convention.

You must decide and document one of these models:

#### Model A — node-as-cell-center model

If route nodes are actually cell centers, then a transition from cell center to adjacent cell center occupies the destination or current cell depending on convention. This is simple but often produces ambiguous endpoints/intersections.

#### Model B — lattice-corner / edge-to-cell model

If route nodes are grid lattice points and road tiles are cells between/adjacent to nodes, then each node-to-node transition maps to the cell on one side of the edge. This better matches board-tile semantics but needs explicit orientation.

#### Model C — cell-path model

Do not route on nodes at all. Route directly over road tile cells. Then each path item is a tile cell, and directions are derived from adjacent cells. This may be the simplest for CM roads if tiles occupy cells and connect to neighboring cells.

### Recommendation

Use **Model C** unless existing coordinate conventions make it impossible. Road tiles are ultimately placed in cells. A cell-path representation avoids the confusing “node path plus step cell” conversion.

However, if the current router is already node-based and quick repair is desired, implement Model B temporarily but name it explicitly.

### Deliverables

1. Rename existing ambiguous fields:
   - `RouteRecord.nodes`
   - `RouteRecord.step_cells` or `RouteRecord.tile_cells`
   - remove or deprecate `RouteRecord.cells`.

2. Add conversion function:

```python
def route_nodes_to_step_cells(nodes: Sequence[GridNode], grid: GridIndex) -> tuple[GridCell, ...]:
    ...
```

or switch router to emit cell paths directly:

```python
def route_edge_as_cell_path(...) -> tuple[GridCell, ...]:
    ...
```

3. Update tile assignment to consume only `tile_cells` / cell paths.

4. Add assertions:
   - no route with ambiguous cell semantics can reach output assembly,
   - no fallback bypass based on `len(route.cells) == len(route.nodes)` remains.

### Tests

Unit tests:

- horizontal two-node transition maps to expected cell,
- vertical transition maps to expected cell,
- path of 5 nodes maps to 4 step cells,
- no duplicate step cell unless route intentionally doubles back,
- diagonal transitions are rejected for normal road catalog unless explicitly legal.

Scenario tests:

- `straight_road_2pt` produces continuous output cells.
- `ninety_degree_bend` produces exactly one bend cell.
- Existing failing intersection fixture changes from “wrong/ambiguous” to either “legal” or “explicit failure”.

### Done when

Tile assignment receives a clear ordered cell path. Any test failure after this milestone is no longer due to node/cell ambiguity.

---

## 7. Milestone M2 — Add raster-spine / line-support extraction

### Purpose

Make the source geometry influence the path more strongly than endpoint anchors alone.

### Problem

A two-point road line from roughly `(0,0)` to `(2,2)` can have multiple equal-length grid routes. A generic shortest-path search may choose an L-shaped path that is much farther from the source line than another equal-length path. The old grid-intersection logic, although not always active in the final road path, captured an important idea: the source line induces an ordered set of grid cells intersecting or supporting the line.

### Decision

Every topology edge must produce a **raster spine** before route search:

```python
@dataclass(frozen=True)
class RasterSpine:
    topology_edge_id: int
    cells: tuple[GridCell, ...]
    progress: tuple[float, ...]       # normalized 0..1 along source line
    distance_m: tuple[float, ...]     # cell center or cell polygon distance to line
    source_length_m: float
```

The router may deviate from the spine, but deviations must be penalized and bounded.

### Algorithm options

#### Option A — polygon/cell intersection

For each topology edge:

1. compute integer candidate bbox around line bounds,
2. test candidate cell polygons against the line or a thin line buffer,
3. select cells intersecting the line/buffer,
4. order by projection distance along the line.

Pros:
- close to the old explicit grid-intersection behavior,
- robust for sparse OSM vertices.

Cons:
- uses Shapely cell geometry tests, but only inside a small candidate bbox.

#### Option B — supercover line rasterization in grid-local coordinates

Transform line coordinates to grid-local coordinates and run a supercover/Bresenham-like rasterization that includes all cells touched by the line.

Pros:
- fast,
- deterministic,
- avoids Shapely in hot path.

Cons:
- more implementation care for rotated grid and arbitrary polylines.

#### Option C — sampled polyline support

Sample the line at sub-cell intervals, snap samples to cells, deduplicate, order.

Pros:
- easiest.

Cons:
- can miss thin intersections or produce sampling artifacts unless oversampled.

### Recommendation

Implement **Option B** eventually, but start with **Option A** for correctness and tests. Once tests lock behavior, optimize to supercover if needed.

### Routing cost integration

The route cost should include distance to the raster spine, not just distance to the continuous line:

```text
cost += w_spine * min_distance_to_spine_cells
cost += w_progress * monotonic_progress_penalty
```

The route should be encouraged to pass through or near the ordered support cells.

### Tests

Unit tests:

- two-point diagonalish line produces middle support cell(s),
- support cells are ordered by progress,
- no gap larger than one adjacency step in raster spine unless source line jumps across map,
- rotated bbox does not miss support cells.

Scenario tests:

- `diagonalish_2pt_equal_length` chooses the route closer to source line,
- detour route with same number of cells fails distance threshold,
- route max distance to spine is below configured limit.

### Acceptance criteria

- Every routed topology edge has a raster spine.
- Router diagnostics report:
  - number of spine cells,
  - mean/max distance of route cells to spine/source line,
  - route cells skipped from spine,
  - extra detour cells.
- Equal-length alternatives are resolved by geometry support, not arbitrary neighbor order.

### Done when

Endpoint-only roads are no longer underconstrained. Road shape is controlled by the source line over its full length.

---

## 8. Milestone M3 — Compile tile catalog as a feasibility oracle

### Purpose

Make tile feasibility queryable before and during routing, not only after routing.

### Decision

`CompiledTileCatalog` must provide fast answers to these questions:

```python
catalog.has_tile(required_dirs: frozenset[Direction]) -> bool
catalog.best_tile(required_dirs: frozenset[Direction], road_class: ...) -> TileVariant | None
catalog.allowed_step_dirs() -> frozenset[Direction]
catalog.can_extend(existing_dirs: frozenset[Direction], new_dir: Direction) -> bool
```

Tile assignment remains responsible for choosing exact variants, but routing and anchor selection must be able to reject impossible direction sets early.

### Required direction model

For every road output cell, define required directions as connections to neighboring road cells:

```text
N, E, S, W
```

If diagonal road tiles are not valid in the catalog, diagonal movement must not appear as a direct continuation. A diagonal-looking road must be represented as a sequence of legal cardinal steps and bend tiles.

### Deliverables

1. Extend `CompiledTileCatalog`.
2. Add direction-set normalization.
3. Add tests that enumerate all direction sets:
   - straight: `{N, S}`, `{E, W}`
   - bend: `{N, E}`, etc.
   - T: `{N, E, S}`, etc.
   - 4-way: `{N, E, S, W}`
   - dead-end if supported/unsupported.
4. Produce a startup diagnostic listing missing direction sets for each process/catalog.

### Explanation

A CM tile catalog is not a cosmetic lookup table. It defines the legal grammar of the road network. The algorithm must know whether a proposed local topology can be expressed in that grammar before accepting it.

### Tests

- `CompiledTileCatalog` matches the profile data.
- If a required fixture needs a 4-way tile and the profile lacks one, the test expects split-intersection behavior rather than a fake 4-way.
- Diagonal moves are rejected unless catalog explicitly supports diagonal connection columns and output tile semantics.

### Done when

The router and anchor selector can ask tile feasibility questions without constructing final output rows.

---

## 9. Milestone M4 — Introduce `LinearNetworkState`

### Purpose

Replace the old hidden persistent `grid_graph.nodes[node]["connections"]` semantics with an explicit, fast, debuggable dense state.

### Decision

During linear-network routing, accepted routes must update a shared state immediately. Later routes must see previous accepted routes.

Suggested state:

```python
class LinearNetworkState:
    occupied: np.ndarray[bool]
    connection_bits: np.ndarray[np.uint8]  # N/E/S/W bits
    route_id_at_cell: np.ndarray[np.int32]
    priority_at_cell: np.ndarray[np.int16]
    process_at_cell: np.ndarray[np.int16]
    intersection_kind_at_cell: np.ndarray[np.int8]
```

### Semantics

For each accepted road cell:

- `occupied[cell] = True`
- `connection_bits[cell]` stores the directions this cell connects to.
- `priority_at_cell` stores the winning network rank.
- A new route may:
  - connect to the cell if catalog feasibility allows the union of directions,
  - be blocked if it would overwrite a stronger route illegally,
  - be allowed to share only as a deliberate intersection/merge.

### Conflict policy

The state must distinguish:

```text
same road continuing through same cell       -> allowed if direction union legal
minor road joining major road                -> allowed as legal intersection
parallel road trying to reuse major cells    -> usually blocked
stream/fence crossing road                   -> process-specific policy
building trying to occupy road cell          -> blocked later by occupancy
```

### Deliverables

1. `linear_network_state.py` or add to `network_routing.py`.
2. Methods:
   - `can_enter_cell(cell, incoming_dir, outgoing_dir, process, priority)`
   - `reserve_path(route, process, priority)`
   - `release_path(route_id)`
   - `required_dirs(cell)`
   - `as_debug_layer()`
3. Integrate with routing:
   - accepted route updates state,
   - failed/retried route releases tentative reservations,
   - route search sees state as cost/blocking input.

### Explanation

The old persistent grid graph was ugly but important. It meant roads were not routed independently. `LinearNetworkState` should preserve that idea without NetworkX.

### Tests

Unit tests:

- reserving a straight path sets correct N/S or E/W bits,
- adding a T branch creates legal `{N,E,S}` set,
- adding a fourth arm creates `{N,E,S,W}` only if catalog supports it,
- illegal direction union is rejected,
- lower-priority overwrite is rejected.

Scenario tests:

- `four_way_crossing` produces one legal connected intersection or legal split,
- `parallel_close_roads` does not collapse,
- `minor_meets_major` attaches without destroying major route.

### Done when

Road routing is stateful across edges and this state is inspectable in debug output.

---

## 10. Milestone M5 — Make topology anchor selection tile-aware

### Purpose

Stop snapping OSM topology nodes directly to the closest grid cell if that cell produces an ugly or impossible CM intersection.

### Decision

Each topology node must go through candidate-anchor search:

```python
@dataclass(frozen=True)
class AnchorCandidate:
    topology_node_id: int
    cell: GridCell
    score: float
    required_dirs_estimate: frozenset[Direction]
    tile_feasible: bool
    occupancy_feasible: bool
    reasons: tuple[str, ...]
```

### Candidate generation

For each topology node:

1. Convert projected coordinate to nearest cell.
2. Generate candidates within radius:
   - start radius: 1 cell,
   - retry radius: 2–3 cells,
   - configurable maximum.
3. For each candidate, estimate directions of incident arms:
   - use vector from topology node along each incident edge,
   - or use short raster spine prefix/suffix,
   - map to cardinal direction bins.
4. Query tile catalog feasibility for direction set.
5. Score:
   - distance from original topology point,
   - angle/direction distortion,
   - road class priority,
   - occupancy conflicts,
   - tile feasibility,
   - alignment with raster spines.

### Single-tile vs split intersections

If no candidate can represent all incident directions as one tile, the algorithm must create a **split-intersection plan** instead of forcing an impossible tile.

Example split strategy:

```text
one topology node with degree 4
  -> small cluster of 2–4 nearby anchor cells
  -> connector cells between them
  -> each cell has legal direction set
```

This can represent awkward intersections or missing catalog tiles.

### Deliverables

1. `anchor_selection.py` or equivalent.
2. `AnchorPlan` model:
   - `SingleAnchorPlan`
   - `SplitAnchorPlan`
   - `FailedAnchorPlan`
3. Debug export of all candidates and selected anchor plan.
4. Integrate anchor plans into routing.

### Explanation

A real OSM intersection is a continuous point, but a CM intersection is a tile pattern. The closest cell may be wrong. Anchor selection must optimize for legal tile representation, not only geometric proximity.

### Tests

- `four_way_crossing` candidate set contains at least one tile-feasible anchor if catalog supports 4-way.
- `t_junction` chooses a legal 3-way anchor.
- artificially remove 4-way tile from test catalog → split-intersection plan is selected.
- anchor does not move beyond configured limit unless diagnostics mark relaxation.

### Done when

Every topology node has a documented anchor decision before edge routing.

---

## 11. Milestone M6 — Integrate tile feasibility into routing

### Purpose

Prevent the router from producing paths that tile assignment cannot legally represent.

### Decision

Routing should be constrained by local tile feasibility and `LinearNetworkState`, not only by corridor distance and occupancy.

### Routing state

For cell-path routing, A* state should include enough information to evaluate turns:

```text
(current_cell, previous_direction)
```

For each candidate next cell:

1. determine new direction,
2. derive local required direction set for current cell,
3. query `CompiledTileCatalog.can_extend(...)`,
4. query `LinearNetworkState.can_enter_cell(...)`,
5. compute geometric/spine/corridor cost.

### Cost function

Suggested cost:

```text
cost =
    step_cost
  + w_source_distance * distance_to_source_line^2
  + w_spine_distance  * distance_to_raster_spine^2
  + w_turn            * turn_penalty
  + w_priority        * priority_displacement_penalty
  + w_state           * occupied_or_near_occupied_penalty
```

Hard constraints:

- invalid tile direction set,
- illegal overwrite of stronger route,
- outside corridor unless in relaxation mode,
- impossible endpoint/intersection anchor.

### Deliverables

1. Modify router to route from anchor plan to anchor plan.
2. Use raster-spine distance in cost.
3. Use tile feasibility in neighbor expansion.
4. Use `LinearNetworkState` as live reservation/check state.
5. Add retry modes:
   - widen corridor,
   - increase anchor radius,
   - split edge at midpoint,
   - mark edge failed.

### Explanation

This is the point where the new router becomes a real replacement for old `custom_weight`. The old code checked existing node connections and tile feasibility during graph search. The new implementation should do the same more cleanly and faster.

### Tests

Scenario tests:

- `diagonalish_2pt_equal_length`: closest-support path wins.
- `ninety_degree_bend`: route produces legal bend.
- `four_way_crossing`: all arms route into selected anchor.
- `parallel_close_roads`: state blocks unintended merge.
- `minor_meets_major`: branch connects legally.

Failure tests:

- impossible catalog → structured route failure, no broken output rows.
- blocked corridor → retry/widen diagnostic emitted.

### Done when

Tile assignment failures become rare and meaningful. The router no longer emits obviously untileable paths.

---

## 12. Milestone M7 — Restore priority/stage semantics for linear features

### Purpose

Recover the old behavior where major/higher-priority networks claim space first and later/lower-priority networks route around or connect to them intentionally.

### Decision

Do not route all linear features as one undifferentiated batch. Use deterministic staged processing.

Suggested order:

```text
1. major roads / highest-priority road classes
2. secondary/minor roads
3. rail if configured as stronger or independent
4. streams/watercourses
5. fences/hedges/walls
6. dependent linear-area effects, e.g. road surface/shoulder
```

The exact order should come from config `priority`/`rank` plus process-specific rules.

### Important distinction

There are two types of interaction:

1. **Connective interaction:** minor road connects to major road.
2. **Avoidance interaction:** building/fence/stream avoids road, or lower-priority line shifts around higher-priority line.

The state must encode which is allowed per process pair.

### Deliverables

1. `LinearProcessingPlan`:
   - groups topology edges by process/config/rank,
   - preserves source feature metadata,
   - decides whether to node across process classes.
2. Stage-by-stage routing:
   - route group,
   - validate/tile group,
   - commit group to `LinearNetworkState` and `OccupancyModel`,
   - proceed to next group.
3. Configurable process-pair interaction policy.

### Explanation

The old pipeline’s priority/stage model was not just output conflict sorting. It influenced noding, routing, and occupancy. The refactor must make that explicit.

### Tests

- major road remains near source while minor road shifts/attaches,
- lower-priority road cannot overwrite major road,
- fences do not force roads to reroute unless configured,
- road/stream behavior matches chosen process-pair policy.

### Done when

Routing order and priority effects are visible in debug diagnostics and stable under repeated runs.

---

## 13. Milestone M8 — Make tile assignment a validator and finalizer, not a repair dump

### Purpose

Tile assignment should finalize already-valid routed cell paths, not try to rescue arbitrary paths.

### Decision

After M3–M7, tile assignment receives:

- legal cell paths,
- selected anchor plans,
- connection bits,
- process/rank/source metadata.

It should:

1. choose the exact CM tile variant per cell,
2. validate every required direction set,
3. produce placement records,
4. fail hard if a supposedly valid path is impossible.

### Deliverables

1. Tile assignment consumes `LinearNetworkState.connection_bits`.
2. Intersection cells are selected from the same state as path cells.
3. Synthetic config names like `{process}_intersection` should be avoided unless explicitly mapped.
4. Every output road cell gets metadata:
   - source process/config,
   - source feature IDs contributing to it,
   - connection dirs,
   - selected tile ID,
   - whether it is intersection/bend/straight/dead-end.

### Explanation

The old code had separate edge compatibility graphs and intersection candidate intersections. The new architecture can be cleaner: once `LinearNetworkState` stores required directions, tile assignment is mostly a catalog lookup.

### Tests

- Every road output row corresponds to a legal catalog tile.
- Every required direction set in state has a selected tile.
- No missing/gap tile is emitted silently.
- Removing a needed tile from a test catalog causes route/intersection failure before output.

### Done when

Tile assignment is deterministic, linear-ish in number of road cells, and acts as a strict validator.

---

## 14. Milestone M9 — Whole-pipeline output invariants and visual debug

### Purpose

Lock in road quality at the final output level.

### Decision

Tests should not only inspect internal records. They must reconstruct road connectivity from final CM rows and verify semantic invariants.

### Output-level validators

Implement:

```python
validate_road_output_rows(rows, profile) -> RoadValidationReport
```

Report should include:

- illegal tile labels,
- cells with no valid connection interpretation,
- disconnected components,
- one-cell gaps at topology junctions,
- dangling arms not corresponding to source endpoints,
- duplicate mutually exclusive cells,
- diagonal continuations where unsupported,
- intersections without legal tile pattern,
- road/building overlap.

### Visual debug

For failing small fixtures, generate:

1. ASCII grid.
2. Optional SVG/PNG or GeoJSON layers:
   - source line,
   - raster spine,
   - selected cell path,
   - final tile directions.

### Tests

All core fixtures must pass final-output validators:

- `straight_road_2pt`
- `diagonalish_2pt_equal_length`
- `ninety_degree_bend`
- `four_way_crossing`
- `t_junction`
- `minor_meets_major`
- `parallel_close_roads`
- `road_near_building`

### Done when

A road regression can be caught from final output rows alone, while internal debug artifacts explain the cause.

---

## 15. Milestone M10 — Building and area revalidation after road repair

### Purpose

Ensure repairing roads does not break the improvements in area rasterization/building fitting.

### Decision

Do this after road state and occupancy semantics stabilize. Building fitting depends on road occupancy; testing it before road repair gives misleading results.

### Deliverables

1. Re-run area rasterization tests:
   - area coverage threshold,
   - rotated bbox boundary,
   - defaults,
   - conflict layers.
2. Re-run building tests:
   - simple rectangle,
   - diagonal building,
   - building near road,
   - small village cluster.
3. Add cross-feature tests:
   - buildings avoid final road cells,
   - roads are not deleted by output conflict resolution,
   - area layers do not overwrite roads/buildings.

### Explanation

The refactor likely improved area/building structure, but road placement errors corrupt the occupancy basis. Once roads are valid, building and area behavior can be judged fairly.

### Done when

The refactored pipeline passes road + building + area integration tests with deterministic output.

---

## 16. Milestone M11 — Consolidate orchestration into `ExtractionPipeline`

### Purpose

The current architecture is not fully consolidated if real orchestration still lives in `OSMProcessor._run_typed_processors` while `ExtractionPipeline.run()` is a stub or thin placeholder.

### Decision

Move real extraction orchestration into `osm_extraction.pipeline.ExtractionPipeline`.

`OSMProcessor` should become a compatibility adapter:

```text
OSMProcessor.preprocess_osm_data -> pipeline.load/match/init
OSMProcessor.run_processors      -> pipeline.run
OSMProcessor.get_output          -> pipeline.result.to_output_rows
OSMProcessor.get_geometries      -> pipeline.result.to_debug_geometries
```

### Deliverables

1. Real `ExtractionPipeline.run()`.
2. `OSMProcessor` delegates to pipeline.
3. Tests call both:
   - public compatibility API,
   - direct pipeline API.
4. Remove duplicated orchestration code.

### Explanation

A partially migrated architecture makes future Codex sessions risky because the “real” path is not where the module layout says it is. Consolidation improves maintainability and makes tests less brittle.

### Done when

There is one authoritative orchestration path.

---

## 17. Milestone M12 — Performance pass after correctness

### Purpose

Recover/improve runtime after correctness contracts are locked.

### Decision

Only optimize after M0–M9 tests are green. Performance changes must preserve scenario test output or semantic metrics.

### Targets

1. Raster spine:
   - replace Shapely cell intersection with supercover grid rasterization if needed.
2. Routing:
   - restrict A* to smaller corridor windows,
   - cache distance fields,
   - use NumPy arrays for closed/open masks,
   - reduce heap allocations.
3. Tile assignment:
   - precompute direction-set → tile choices,
   - avoid pandas in hot path.
4. Debug:
   - emit debug layers only in debug mode.

### Benchmarks

Run representative fixtures plus medium real OSM extracts and report:

```text
feature_matching
topology_noding
raster_spine
anchor_selection
routing
tile_assignment
building_fitting
output_assembly
```

### Done when

Network-heavy scenarios are faster than the old branch and produce equal or better semantic metrics.

---

## 18. Recommended milestone order

The most likely successful order is:

```text
M0  Semantic regression harness
M1  Route node vs tile-cell contract
M2  Raster-spine / line-support extraction
M3  Tile catalog feasibility oracle
M4  LinearNetworkState
M5  Tile-aware anchor selection
M6  Tile-feasible routing
M7  Priority/stage linear processing
M8  Tile assignment validator/finalizer
M9  Output-level road invariants/debug
M10 Building/area revalidation
M11 Pipeline orchestration consolidation
M12 Performance pass
```

If time is limited, the shortest high-value path is:

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M8 -> M9
```

That would not fully solve anchor selection, but it should expose and fix the most concrete route-cell and illegal-tile issues.

---

## 19. Suggested Codex task prompts

These are task boundaries, not final prompts.

### Prompt 1 — Harness and road fixture diagnostics

Implement M0 only. Add the road recovery fixtures, a fixture runner, output-row road graph reconstruction, and debug artifact capture. Do not change road algorithms except to expose debug info. Ensure at least one current road/intersection bug is reproduced as a failing test.

### Prompt 2 — Route cell contract

Implement M1 only. Replace ambiguous `RouteRecord.cells` with explicit `nodes` and `tile_cells`/`step_cells`, or convert the router to direct cell-path routing. Update tile assignment to consume only tile cells. Add unit and scenario tests.

### Prompt 3 — Raster spine

Implement M2 only. Add `RasterSpine` generation for each topology edge and integrate its distance/progress into routing cost. Add tests for endpoint-only diagonalish lines and equal-length route ambiguity.

### Prompt 4 — Tile feasibility oracle

Implement M3 only. Extend `CompiledTileCatalog` with direction-set feasibility queries. Add exhaustive direction-set tests against road tile catalog. No broad routing rewrite yet.

### Prompt 5 — LinearNetworkState

Implement M4 only. Add connection-bit state and route reservation/release. Integrate state minimally with routing and tile assignment. Add tests for straight, bend, T, 4-way, and illegal direction union.

### Prompt 6 — Tile-aware anchors

Implement M5 only. Candidate-search topology anchors within a configurable radius and choose anchors using tile feasibility, geometry score, occupancy, and road class. Add split-intersection placeholder/failure diagnostics even if full split routing is not complete.

### Prompt 7 — Tile-feasible routing

Implement M6 only. Make routing neighbor expansion consult raster spine cost, tile feasibility, and `LinearNetworkState`. Add retry diagnostics. Make impossible tile routes fail instead of emitting broken rows.

### Prompt 8 — Priority/stage restoration

Implement M7 only. Route linear features in priority/config groups and commit each accepted group before routing later groups. Add major/minor road and road/fence/stream fixtures.

### Prompt 9 — Final tile/output validation

Implement M8 and M9. Make tile assignment strict. Add final output validators and ASCII/debug export for failed fixtures.

### Prompt 10 — Revalidate buildings/areas and consolidate pipeline

Implement M10 and M11. Re-run integration tests and move orchestration into `ExtractionPipeline`.

---

## 20. Definition of done for the road recovery

Road recovery is complete when:

1. The road fixtures pass final-output validation.
2. Every road output cell has a legal tile variant for its required connection directions.
3. Every OSM topology junction is represented by either:
   - one legal CM intersection tile, or
   - an explicit legal split-intersection pattern, or
   - a structured failure diagnostic.
4. Endpoint-only roads follow their source geometry via raster spine support, not arbitrary equal-length shortest paths.
5. No diagonal continuation is emitted for road catalogs that only support cardinal continuation.
6. Accepted road routes update shared connection/occupancy state before later routes are planned.
7. Priority/rank affects routing and occupancy, not only final row conflict resolution.
8. Tile assignment failures are hard failures for roads and trigger repair or diagnostics.
9. Debug artifacts can explain every failed fixture.
10. The public `OSMProcessor` path and direct `ExtractionPipeline` path run the same implementation.

---

## 21. Notes on expected tradeoffs

### 21.1 Legal topology may require visible displacement

Some real OSM intersections cannot be represented exactly by CM tile constraints. The correct behavior is not to force them into broken output. The correct behavior is:

```text
preserve major-road continuity,
shift/split lower-priority arms,
emit a legal nearby pattern,
record diagnostics.
```

### 21.2 The old output is not the golden truth

The old implementation sometimes made poor choices, was slow, and had bugs. The goal is not exact reproduction. The goal is to preserve the useful old invariants while using the new typed/grid/occupancy architecture.

### 21.3 Tests should allow intentional legal alternatives

For example, a 4-way source intersection may be represented as:

- one 4-way tile,
- two adjacent T-junction tiles,
- a small legal split intersection,

depending on catalog and geometry. Tests should assert legality/connectivity/quality metrics, not one exact cell pattern unless the fixture is designed to require it.

---

## 22. Compact architecture target

The corrected road pipeline should look like this:

```text
matched OSM road features
  -> clipped/noded topology graph
  -> raster spine per topology edge
  -> candidate anchors per topology node
  -> tile-feasible anchor or split-intersection plan
  -> priority-ordered cell-path routing
       uses:
         - raster-spine distance
         - tile feasibility oracle
         - LinearNetworkState
         - occupancy/process-pair policy
  -> connection-bit state
  -> strict tile assignment
  -> final road placements
  -> output-row validation
```

Key invariant:

> No road placement may reach final output unless its cell path, connection directions, and selected CM tile variants form a legal connected road network under the active profile catalog.
