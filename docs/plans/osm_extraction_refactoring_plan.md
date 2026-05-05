# OSM extraction refactoring plan — performance, quality, and bug fixing

**Repository:** `DerButschi/CMAutoEditor`  
**Branch:** `feature/cm_terrain_extractor`  
**Primary entry point to preserve externally:** `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`  
**Basis document:** `osm_extraction_function_map.md`  
**Plan date:** 2026-05-05

This document is a proposed refactoring and redesign plan for the OSM extraction functionality. It deliberately does **not** try to preserve legacy internals. The desired preservation boundary is the *external behavior needed by the app*: given an OSM/GeoJSON input, a bounding box, a profile, and a config, produce CM AutoEditor-compatible terrain rows and debug geometries. Everything inside that boundary may be replaced if the replacement is faster, more robust, or produces better Combat Mission maps.

The plan is written as a project file for later implementation work. It is intentionally implementation-oriented and opinionated.

---

## 1. Goals, non-goals, and design thesis

### 1.1 Goals

The refactor has three equally important goals:

1. **Improve runtime performance**
   - Reduce or eliminate NetworkX searches over full map grids.
   - Reduce repeated GeoDataFrame/GeoSeries spatial operations in hot paths.
   - Replace repeated `pandas.concat` with chunk accumulation.
   - Move Shapely geometry operations out of inner loops wherever possible.
   - Use dense integer grids, masks, and vectorized NumPy/Shapely operations for repeated work.

2. **Improve generated map quality**
   - Preserve road/rail/stream/fence network topology more reliably.
   - Keep major road intersections coherent.
   - Shift lower-priority features around higher-priority features intentionally rather than by accidental post-process conflict resolution.
   - Improve building placement/fitting and preserve complex footprints where possible.
   - Make output quality measurable through explicit metrics.

3. **Fix known and likely bugs**
   - Fix high-confidence code defects from the function map.
   - Replace unfinished or misleading functions.
   - Make behavior deterministic under a seed.
   - Make config/process/profile mappings validated instead of stringly typed.

### 1.2 Non-goals

The following are explicitly **not** goals:

- Preserving legacy internal classes/functions.
- Keeping stage numbers or string-dispatched functions.
- Keeping NetworkX in hot paths.
- Keeping the current mutable `OSMProcessor` state model internally.
- Keeping current random output behavior if it is nondeterministic.
- Keeping unfinished functions merely because they exist.

### 1.3 Design thesis

The best refactor is not a cleanup of the existing `processing.py` / `path_search.py` design. The current pipeline has the right high-level idea — classify OSM features, build grids, route networks, fit buildings, export rows — but the internals mix UI, mutable global state, GeoDataFrames, NetworkX graphs, output rows, and occupancy rules in ways that make both speed and correctness hard.

The recommended replacement is a staged, typed pipeline:

```text
GeoJSON / OSM features
        │
        ▼
Config + feature matching
        │
        ▼
Projected FeatureStore
        │
        ▼
GridIndex + OccupancyModel
        │
        ├──────────────► Area rasterization
        │
        ├──────────────► Network topology noding
        │                    │
        │                    ▼
        │              Integer-grid routing
        │                    │
        │                    ▼
        │              Tile assignment DP/CSP
        │
        └──────────────► Building candidate fitting
                             │
                             ▼
                    Layered output model
                             │
                             ▼
                    CM AutoEditor rows + debug geometries
```

The main architectural change is to separate three concepts that are currently entangled:

1. **Geometry evidence**: projected OSM points/lines/polygons.
2. **Discrete placement state**: integer grid cells, sub-cells, diagonal cells, occupancy masks.
3. **CM output rows**: the final serialized rows, created only after placement has been decided.

---

## 2. Target module layout

The exact filenames can change during implementation, but the refactor should move toward this shape:

```text
cm_terrain_extractor_app/terrain_extraction/osm_extraction/
  __init__.py
  pipeline.py              # Main extraction pipeline and compatibility adapter
  models.py                # Dataclasses / typed records used between stages
  config_schema.py         # Config validation, compiled selectors, process enums
  grid_index.py            # Rotated grid math, integer <-> projected mapping, lazy geometries
  occupancy.py             # Layered occupancy masks and conflict policy
  feature_matcher.py       # OSM tag/id matching, projection, FeatureStore construction
  area_rasterizer.py       # Land cover / area / point placement
  network_topology.py      # Line clipping, noding, intersection graph, chain collapse
  network_routing.py       # Fast integer-grid routing, no NetworkX hot path
  tile_assignment.py       # Catalog compilation, per-path DP, intersection CSP
  building_fitter.py       # Building candidate generation, scoring, placement
  output_rows.py           # Final row assembly, map extents, CSV-compatible output
  debug_export.py          # Debug geometries and diagnostic layers
  stats.py                 # Timings, counters, quality metrics
```

Existing public entry points can remain as thin wrappers:

```text
terrain_extraction/osm_processor.py
  OSMProcessor  # compatibility wrapper around osm_extraction.pipeline.ExtractionPipeline
```

This allows the app to keep calling `OSMProcessor` while the internal implementation changes substantially.

---

## 3. Core data contracts

### 3.1 `ExtractionConfig`

Replace raw nested JSON dict use in hot code with a validated compiled config.

Suggested fields:

```python
@dataclass(frozen=True)
class ConfigEntry:
    name: str
    active: bool
    process: ProcessKind
    priority: int
    selectors: list[TagSelector]
    required_tags: dict[str, str | list[str]]
    exclude_tags: dict[str, str | list[str]]
    allowed_ids: set[str]
    exclude_ids: set[str]
    cm_types: list[CMType]
    modifiers: Mapping[str, Any]
```

Important behavioral changes:

- Make `any` vs `all` tag matching explicit.
- Validate that all `process` names are known.
- Validate building process names against profile building type mappings.
- Validate tile/building catalog references at startup, not during late processing.
- Compile weighted choices once into arrays.
- Carry a deterministic RNG object through the pipeline.

### 3.2 `FeatureRecord` and `FeatureStore`

Use typed records instead of raw dicts in `matched_elements`:

```python
@dataclass(frozen=True)
class FeatureRecord:
    feature_id: str | int | None
    source_index: int
    config_name: str
    process: ProcessKind
    priority: int
    geometry: BaseGeometry       # projected CRS
    source_tags: Mapping[str, str]
    source_properties: Mapping[str, Any]
```

The `FeatureStore` groups records by process and config name. This replaces the current stage collection that stores indices/config names in nested dicts.

### 3.3 `GridIndex`

`GridIndex` should own all coordinate math:

- Projected CRS.
- Rotated bbox origin and axes.
- Map width/height in 8 m cells.
- Conversion from projected coordinates to grid-local coordinates.
- Conversion from grid indices to projected centers/polygons.
- Normal grid, sub-square grid, and diagonal grid index conventions.

The key performance improvement is that nearest-grid queries should not use a spatial index. The grid is regular and rotated, so snapping a point should be simple affine math:

```text
local = R^-1 · (projected_point - origin)
cell_x = floor(local_x / 8)
cell_y = floor(local_y / 8)
nearest_node_x = round(local_x / 8)
nearest_node_y = round(local_y / 8)
```

GeoDataFrames should become **lazy debug/export views**, not the primary internal representation.

### 3.4 `OccupancyModel`

Replace `occupancy_gdf` plus late `post_process` with a single layered occupancy model.

Suggested layers:

```text
GROUND          # default grass, farmland, mud, gravel, etc.
FOLIAGE         # trees, bushes, orchards, etc.
LINEAR_SURFACE  # roads/rail/streams where they consume terrain cells
LINEAR_OBJECT   # fences, hedges, walls if represented as mutually exclusive objects
BUILDING        # building placements on sub-square/diagonal grid
POINT_OBJECT    # individual objects
RESERVED        # temporary routing reservations / anchor cells
```

Each layer should have:

- a dense integer array for placed object id or `-1`,
- optional priority/rank arrays,
- optional metadata mapping object id → source feature/config/type.

Conflict handling becomes an explicit function:

```python
def can_place(candidate: Placement, occupancy: OccupancyModel) -> PlacementDecision:
    ...
```

This replaces the current situation where conflicts are sometimes prevented by `occupancy_gdf`, sometimes resolved by deleting rows in `post_process`, and sometimes allowed to leak into output.

### 3.5 `PlacementRecord`

Processors should emit placement records, not final output rows:

```python
@dataclass(frozen=True)
class PlacementRecord:
    layer: LayerKind
    grid_kind: GridKind
    cells: tuple[GridCell, ...]
    config_name: str
    feature_id: str | int | None
    priority: int
    cm_type: CMType | None
    score: float | None
    diagnostics: Mapping[str, Any]
```

Final rows are assembled only in `output_rows.py` once conflicts have been resolved.

---

## 4. Refactoring work packages

The work packages below are ordered to reduce risk. The most likely best path is **not** to rewrite everything at once. Instead, build a new internal pipeline beside the old one, redirect one feature class at a time, and compare outputs.

---

## WP0 — Baseline tests, metrics, and profiling harness

### Purpose

Before rewriting algorithms, create a small but useful test and measurement harness. This prevents refactoring from becoming blind.

### Scope

Add synthetic fixtures and timing instrumentation around the current implementation.

### Deliverables

1. `tests/terrain_extraction/osm_extraction/fixtures/`
   - `simple_area.geojson`: one land-cover polygon.
   - `single_road.geojson`: one road with a bend.
   - `crossroads.geojson`: two roads crossing.
   - `road_near_building.geojson`: one road and one building that would collide under naive rasterization.
   - `village_cluster.geojson`: several buildings, one primary road, smaller roads.
   - `multi_network.geojson`: road + stream + fence/hedge interaction.

2. A baseline runner:

```text
python -m cm_terrain_extractor_app.terrain_extraction.osm_extraction_benchmark \
  --fixture crossroads \
  --profile CMBS \
  --config default_osm_config.json
```

3. Metrics emitted as JSON:

```json
{
  "timings": {
    "preprocess": 0.0,
    "area_rasterization": 0.0,
    "network_noding": 0.0,
    "network_routing": 0.0,
    "tile_assignment": 0.0,
    "building_fitting": 0.0,
    "post_process": 0.0
  },
  "counts": {
    "features_in": 0,
    "matched_features": 0,
    "network_edges": 0,
    "routed_edges": 0,
    "failed_edges": 0,
    "buildings_in": 0,
    "buildings_placed": 0,
    "building_failures": 0,
    "output_rows": 0
  },
  "quality": {
    "network_disconnected_components": 0,
    "mean_network_distance_m": 0.0,
    "p95_network_distance_m": 0.0,
    "collision_count": 0,
    "mean_building_iou": 0.0
  }
}
```

4. Golden-output tests for the tiny fixtures.
   - Do not overfit exact random choices.
   - Assert invariants: no building-road collision, road intersection connected, all rows inside map, no duplicate mutually exclusive rows.

### Acceptance criteria

- The current implementation can be run repeatedly on the fixtures.
- A timing breakdown exists even if stage timings are approximate at first.
- At least five high-level invariants are tested.

### Notes

This package should be done first even if the old output is flawed. It creates a baseline and exposes which stages actually dominate runtime.

---

## WP1 — Pipeline skeleton and compatibility wrapper

### Purpose

Introduce the new architecture without yet changing all algorithms.

### Scope

Create `osm_extraction/` modules, typed records, and a compatibility wrapper that lets `OSMProcessor` delegate to the new pipeline.

### Deliverables

1. `osm_extraction/models.py`
   - `ProcessKind`
   - `GridKind`
   - `LayerKind`
   - `CMType`
   - `FeatureRecord`
   - `PlacementRecord`
   - `ExtractionResult`
   - `ExtractionStats`

2. `osm_extraction/pipeline.py`
   - `ExtractionPipeline`
   - `ExtractionContext`
   - `ProgressCallback` protocol
   - deterministic RNG initialization

3. Compatibility class in `osm_processor.py`
   - Preserve existing constructor and public methods where practical:
     - `preprocess_osm_data`
     - `run_processors`
     - `post_process`
     - `write_to_file`
     - `get_output`
     - `get_geometries`
   - Internally delegate to new pipeline step by step.

4. Remove Streamlit from core code paths.
   - UI code may pass a callback.
   - Core extraction must run headless in tests.

### Acceptance criteria

- Existing app entry point still works.
- Tests can import and run the new pipeline without Streamlit.
- No algorithmic behavior needs to improve yet.

---

## WP2 — Config schema, deterministic matching, and early bug fixes

### Purpose

Make OSM-to-process classification explicit, validated, and deterministic.

### Scope

Replace raw config dict access in new code. Fix obvious bugs that do not require major algorithm changes.

### Deliverables

1. `config_schema.py`
   - Load JSON.
   - Validate known process names.
   - Validate `cm_types` structure.
   - Validate selector semantics.
   - Validate profile-dependent building and tile catalogs.

2. `feature_matcher.py`
   - Extract tags consistently from either `properties['tags']` or direct properties.
   - Apply required/excluded/allowed-id logic in one place.
   - Return `FeatureRecord` objects.

3. Deterministic random source:

```python
rng = np.random.default_rng(seed)
```

4. Immediate bug fixes:
   - Rename `self.path_to_congih` → `self.path_to_config`.
   - Fix `type_from_barn_outlines` vs `type_from_barn_outline` mapping.
   - Include `rail_tiles` in linear-feature checks/debug geometry handling.
   - Verify and fix diagonal/sub-square grid selection in `get_geometries`.
   - Fix `BoundingBox.__init__` non-WGS84 branch if `crs.to_epsg` is missing `()`.
   - Remove duplicated `get_matched_cm_type` implementation.
   - Validate or fix hard-coded `'Road Tile ...'` labels for rail/stream/fence tile outputs.
   - Replace or test the output extent row logic using `self.gdf.total_bounds[2:]`.

### Acceptance criteria

- Config errors fail early with useful messages.
- Matching is covered by unit tests for required/excluded/allowed-id cases.
- Random outputs are reproducible under a seed.
- The listed bug fixes are covered by regression tests or explicit assertions.

---

## WP3 — Fast `GridIndex` and dense `OccupancyModel`

### Purpose

Remove GeoDataFrames from the core hot path and replace implicit output-row conflicts with explicit occupancy.

### Scope

Implement regular-grid math and dense occupancy masks.

### Deliverables

1. `grid_index.py`
   - `GridIndex.from_bbox(bbox, cell_size=8)`
   - `projected_to_cell(x, y)`
   - `projected_to_nearest_node(x, y)`
   - `cell_center(i, j)`
   - `cell_polygon(i, j)`
   - `subcell_center(...)`
   - `diagonal_cell_center(...)`
   - bounds clipping utilities
   - lazy GeoDataFrame creation only for debug/export

2. `occupancy.py`
   - Dense arrays per layer.
   - `reserve`, `place`, `release`, `is_blocked`, `can_place`.
   - Priority/rank policy.
   - Human-readable conflict reasons for diagnostics.

3. Replace `occupancy_gdf` for new code paths.
   - Keep a debug export method that can produce GeoDataFrames if needed.

### Algorithmic details

For normal-grid cells, convert projected coordinates to local grid coordinates by applying the inverse bbox rotation and translation:

```python
local = inverse_rotation @ (point_xy - origin_xy)
i = floor(local[0] / 8)
j = floor(local[1] / 8)
```

For nearest routing nodes:

```python
i = round(local[0] / 8)
j = round(local[1] / 8)
```

For polygon candidate ranges, transform the polygon bounds or polygon vertices into grid-local coordinates, then compute an integer candidate bounding box. This avoids querying a huge spatial index just to find nearby cells.

### Acceptance criteria

- Point-to-cell and cell-to-polygon conversions match the old grid on representative rotated bboxes.
- Building sub-square and diagonal coordinate conventions are tested explicitly.
- Occupancy can answer blocking queries without Shapely geometry operations in the common case.

---

## WP4 — Area rasterization rewrite

### Purpose

Make polygon/area assignment faster, deterministic, and less dependent on output-row post-processing.

### Scope

Replace `get_grid_cells_to_fill`, `assign_type_from_tag`, random area assignment, point assignment, and default filling in the new pipeline.

### Deliverables

1. `area_rasterizer.py`
   - `rasterize_polygon_to_cells(geometry, grid_index, threshold_area)`
   - `rasterize_line_buffer_to_cells(...)` if needed for dependent linear-area effects.
   - `place_point_object(...)`
   - deterministic weighted choice helpers.

2. Default layer handling:
   - Defaults should be filled after real placements or represented as base layer values.
   - Do not emit default rows into the same conflict pool as real features.

3. Real clustered random assignment:
   - Delete or replace `assign_type_in_random_clusters`.
   - Recommended replacement: seeded low-resolution random field over the map, bilinear/nearest upsampling, optional smoothing, then per-cell category sampling from spatially correlated values.
   - No hard-coded `500 × 500` arrays.

### Algorithmic details

Candidate cells should be computed by integer bbox first:

```text
polygon bounds → grid-local bounds → min/max cell index → candidate slice
```

Only candidate cells are converted to Shapely polygons or vectorized Shapely arrays for intersection-area tests.

For many small polygons, batch processing should group by config entry where possible.

### Acceptance criteria

- Area rasterization on fixtures matches old behavior within reasonable boundary tolerance.
- Runtime is lower on large polygons and large bboxes.
- No mutually exclusive conflict is left for `post_process` to fix.
- Cluster random assignment visibly creates clusters and is deterministic under seed.

---

## WP5 — Network topology noding rewrite

### Purpose

Replace pairwise line intersection logic with faster and more robust topology construction.

### Scope

Rewrite `collect_network_data` and `create_line_graph` into typed topology construction.

### Deliverables

1. `network_topology.py`
   - `collect_network_features(records, bbox_polygon)`
   - `node_network_lines(...)`
   - `build_topology_graph(...)`
   - `collapse_degree_two_chains(...)`

2. Topology data model:

```python
@dataclass(frozen=True)
class TopologyNode:
    id: int
    xy: tuple[float, float]
    source: NodeSource  # endpoint, intersection, snapped_cluster, synthetic_split

@dataclass(frozen=True)
class TopologyEdge:
    id: int
    u: int
    v: int
    geometry: LineString
    config_name: str
    priority: int
    feature_id: str | int | None
    road_class_rank: int | None
```

3. Use spatial index or Shapely unary operations instead of all-pairs intersections.

Recommended strategy:

- Clip all lines to `effective_bbox_polygon`.
- Normalize `LineString`, `MultiLineString`, and polygon boundaries into line records.
- Use an `STRtree` to find candidate intersections.
- Split only candidates whose bounding boxes intersect.
- Coalesce near-identical endpoints/intersections within a configurable tolerance.
- Preserve source metadata when splitting.
- Collapse degree-2 chains with identical config/type metadata for routing to reduce the number of routing calls.

### Important design decision: noding tolerance

OSM features often have tiny gaps or near-misses. The current code only handles exact intersections. The new code should support a small configurable snap tolerance:

```json
"network_snap_tolerance_m": 0.5
```

Potential values to test:

- `0.25 m`: conservative, fixes numerical slivers only.
- `1.0 m`: likely practical for OSM line endpoint near-misses.
- `4.0 m`: half a CM tile; useful but may create false intersections.

Default recommendation: start with `1.0 m`, expose config.

### Acceptance criteria

- Crossroads fixture produces one topology node at the crossing.
- T-junction fixture produces a degree-3 node.
- Near-miss fixture behavior is controlled by snap tolerance.
- Topology construction is measurably faster than pairwise Shapely intersections on dense linework.

---

## WP6 — Fast integer-grid network routing

### Purpose

Replace the current `path_search.search_path` NetworkX full-grid routing with a custom, corridor-limited integer-grid router.

### Scope

Implement a routing algorithm that works on integer grid nodes/cells and dense occupancy masks.

### Deliverables

1. `network_routing.py`
   - `compile_move_set(tile_catalog)`
   - `choose_node_anchors(topology_graph, grid_index, occupancy)`
   - `route_topology_edge(edge, anchors, grid_index, occupancy, routing_params)`
   - `route_network(topology_graph, ...)`

2. A route result model:

```python
@dataclass(frozen=True)
class RoutedEdge:
    topology_edge_id: int
    path: tuple[GridNode, ...]
    status: RouteStatus
    cost: float
    diagnostics: Mapping[str, Any]
```

3. Routing metrics:
   - success/failure count,
   - mean/p95 distance to source line,
   - mean/max detour ratio,
   - number of split intersections,
   - number of forced relaxations.

### Algorithmic design

#### 6.1 Route only in a corridor

For each topology edge, construct a candidate corridor around its source geometry:

```text
corridor = source_line.buffer(max_route_deviation_m)
```

But avoid Shapely inside the A* loop:

1. Use the corridor bounds to compute a candidate integer grid window.
2. Precompute candidate cell/node centers.
3. Compute distance to source line once per candidate node, vectorized if possible.
4. Store distances in a dense local array.

Default starting parameters:

```json
{
  "max_route_deviation_m": 32,
  "max_route_deviation_minor_m": 48,
  "max_route_deviation_major_m": 24,
  "turn_penalty": 0.15,
  "distance_penalty": 1.0,
  "occupied_hard_block": true,
  "minor_road_can_shift": true
}
```

#### 6.2 Use a custom A* / Dijkstra over small local arrays

State should include at least:

```text
(node_i, node_j, incoming_direction)
```

Incoming direction is needed for turn penalties and tile feasibility. It also makes later tile assignment easier.

The cost function should be explicit:

```text
cost = step_cost
     + distance_weight * normalized_distance_to_source^2
     + turn_weight * turn_penalty
     + diagonal_weight * diagonal_penalty
     + occupancy_penalty
     + anchor_alignment_penalty
```

For roads that cannot use diagonal tiles, the compiled move set simply excludes diagonal moves.

#### 6.3 Avoid Shapely and NetworkX in the inner loop

The inner loop should operate on:

- integer node coordinates,
- small local boolean masks,
- precomputed float distance arrays,
- a compiled move table,
- a heap queue from Python stdlib.

This is the single most important runtime refactor.

#### 6.4 Anchor topology nodes before routing edges

The current implementation snaps every edge endpoint independently, then tries to repair intersection issues later. The new implementation should select one grid anchor per topology node before routing incident edges.

For each topology node:

1. Generate candidate anchors in a radius around the projected node coordinate.
2. Score anchors by:
   - distance to source coordinate,
   - whether required incident directions are tile-compatible,
   - occupancy conflicts,
   - spacing from nearby anchors,
   - preserving major-road alignment.
3. Choose one anchor for normal nodes.
4. For high-degree or tile-impossible intersections, explicitly split into multiple anchors connected by short connector paths.

This makes the “major roads meet at the same intersection” requirement a first-class constraint instead of an emergent property.

#### 6.5 Route order

Process networks in a deterministic order:

1. Higher-priority/lower-rank network classes first.
2. Within roads, major roads before minor roads.
3. Longer edges before shorter edges, unless an edge is incident to a high-degree intersection anchor.
4. Fences/hedges and other low-priority linear objects after roads/streams.

This makes lower-priority roads shift around larger roads intentionally.

#### 6.6 Retry and relaxation policy

If routing fails:

1. Increase corridor width for the failed minor feature.
2. Allow soft crossing near low-priority occupancy if allowed by config.
3. Split a long edge at its midpoint and route sub-edges.
4. As last resort, mark feature as dropped with diagnostics.

Never silently fail.

### Acceptance criteria

- No NetworkX dependency in the new hot routing path.
- Crossroads and T-junction fixtures route with shared anchors.
- Road-near-building fixture routes roads first and makes buildings avoid roads later.
- Minor road can shift around major road occupancy when necessary.
- Routing metrics are emitted and visible in tests/benchmark output.
- Runtime is significantly lower than current `search_path` on dense line fixtures.

---

## WP7 — Tile assignment rewrite: catalog automaton + intersection CSP

### Purpose

Replace per-edge NetworkX compatibility graphs with a compiled tile-catalog solver.

### Scope

Rewrite `assign_tiles_to_network` and related direction/tile helpers.

### Deliverables

1. `tile_assignment.py`
   - `compile_tile_catalog(tile_df)`
   - `required_dirs_for_path(path)`
   - `candidate_tiles_for_required_dirs(required_dirs, catalog)`
   - `solve_intersection_tiles(...)`
   - `assign_tiles_to_routed_network(...)`

2. Compiled catalog model:

```python
@dataclass(frozen=True)
class CompiledTileCatalog:
    by_required_dirs: dict[frozenset[Direction], list[TileVariant]]
    legal_moves: frozenset[Direction]
    transition_costs: np.ndarray | None
```

3. Debug report for tile assignment failures.

### Algorithmic design

Once routing has produced a path, the required tile directions for each occupied cell/node are known:

- Interior path cell: directions to previous and next path cells.
- Endpoint/intersection cell: directions to all incident path segments.
- Loop: special handling for start=end if needed.

For many tile catalogs, assignment can be reduced to:

```python
required_dirs = frozenset({"u", "d"})
candidates = catalog.by_required_dirs[required_dirs]
choose min-cost or weighted candidate
```

If the catalog has extra transition constraints beyond direction compatibility, use dynamic programming over tile states along the path:

```text
dp[position, tile_variant] = min previous cost + tile cost + transition cost
```

Intersections should be solved once, not separately per incident edge:

```text
for each intersection anchor:
    required_dirs = union of incident edge directions
    choose one tile variant satisfying all directions
    if impossible:
        request intersection split from routing/repair stage
```

### Benefits over current implementation

- No per-edge NetworkX graph construction.
- Intersections are globally consistent by construction.
- Tile assignment failure produces clear diagnostics.
- The same compiled catalog can be reused for every edge.

### Acceptance criteria

- Current road/rail/stream/fence tile catalogs compile once per run.
- Intersections use a single tile choice shared by all incident paths.
- Per-path assignment runs in linear time in path length for simple catalogs.
- Failures are explicit and tested.

---

## WP8 — Building fitter v2

### Purpose

Improve building placement quality and speed while using the new occupancy model.

### Scope

Replace the current minimum-rotated-rectangle-only building approximation and `occupancy_gdf`-based collision filtering.

### Deliverables

1. `building_fitter.py`
   - `collect_building_candidates(...)`
   - `classify_building_orientation(...)`
   - `generate_candidate_placements(...)`
   - `score_building_candidate(...)`
   - `fit_building_cluster(...)`
   - `place_buildings(...)`

2. Building metrics:
   - placed/failed count,
   - IoU distribution,
   - centroid shift distribution,
   - orientation mismatch count,
   - road/building collision count,
   - number of modular buildings used.

3. Explicit fallback behavior for failed buildings.

### Algorithmic design

#### 8.1 Keep the real footprint longer

The current fitter reduces every polygon to an equal-area minimum rotated rectangle. This is often acceptable for simple rectangular houses, but it discards L-shaped and complex farm/industrial buildings too early.

New approach:

1. Clean and clip original polygon.
2. Compute shape descriptors:
   - area,
   - oriented bounding rectangle,
   - rectangularity = area / MRR area,
   - elongation,
   - main orientation,
   - distance to nearest road.
3. Choose fitting mode:
   - high rectangularity → direct rectangle/catalog fit,
   - low rectangularity but small building → best single catalog rectangle,
   - complex/large footprint → rectangular cover using actual grid footprint mask.

#### 8.2 Generate placements, then score

Candidate placements should vary:

- orientation: axis-aligned vs diagonal,
- local shift: e.g. within ±1 to ±3 sub-cells,
- footprint size from available catalog entries,
- modular vs non-modular composition,
- story/type choice after geometric placement is selected.

Candidate score:

```text
score = + w_iou              * IoU(candidate, source_polygon)
        - w_centroid_shift   * distance(candidate_centroid, source_centroid)
        - w_angle_error      * angle_difference
        - w_area_error       * abs(candidate_area - source_area) / source_area
        - w_collision        * hard/soft occupancy conflicts
        - w_road_overlap     * road/building conflict
        - w_modular_penalty  * number_of_pieces
```

Hard constraints:

- No overlap with roads/rail/streams unless explicitly allowed.
- Stay inside effective bbox.
- Use only profile-supported building catalog footprints.

#### 8.3 Cluster-aware placement

Local per-building placement can fail in dense villages because each building greedily consumes space. Use cluster-aware ordering:

1. Build building clusters with an STRtree or grid-bucket adjacency.
2. Within each cluster, place constrained buildings first:
   - buildings close to roads,
   - larger buildings,
   - buildings with fewer valid candidates.
3. If a later building fails, allow limited local repair:
   - try alternate candidate for nearby lower-priority building,
   - otherwise drop the lowest-quality candidate with diagnostic.

This is not a full global optimizer, but it should improve villages substantially without exploding runtime.

#### 8.4 Branch-and-bound replacement or bounded reuse

The existing `branch_and_bound` can be kept only if bounded and isolated:

- Add maximum states / maximum time budget per building.
- Use greedy rectangle-cover fallback.
- Use actual footprint masks, not only minimum-rotated rectangles.
- Remove legacy `_match_building` path.

### Acceptance criteria

- Buildings near roads shift away from roads rather than being dropped unnecessarily.
- Simple rectangular buildings retain or improve current IoU.
- Complex buildings are not always collapsed to a single MRR if modular catalog entries exist.
- Large/difficult buildings have bounded runtime.
- All building placement decisions produce diagnostics.

---

## WP9 — Layered conflict resolution and final output rows

### Purpose

Make final output assembly deterministic and remove `post_process` as the primary conflict mechanism.

### Scope

Replace row-group conflict deletion with placement-layer assembly.

### Deliverables

1. `output_rows.py`
   - `placements_to_output_rows(...)`
   - `append_extent_marker(...)`
   - `normalize_output_coordinates(...)`
   - `validate_output_rows(...)`

2. Conflict validation:
   - No duplicate mutually exclusive rows for the same cell/layer.
   - No building-road collisions.
   - No invalid coordinates.
   - All `cat2` labels valid for profile/process.

3. Minimal compatibility `post_process`:
   - Can remain as a no-op or final assertion/cleanup step.
   - Should not be responsible for core semantics.

### Algorithmic details

Defaults should be emitted last or represented as implicit base values. Real feature placements should not compete with default rows in a DataFrame groupby.

Output order should be stable:

```text
extent marker
base/default ground
area features
linear networks
buildings
objects
```

or whatever order the CM AutoEditor import expects. The chosen order should be documented and tested.

### Acceptance criteria

- Running output assembly twice with the same seed yields identical rows.
- `post_process` no longer deletes large numbers of conflicting rows under normal operation.
- Output validation catches deliberately injected duplicate/conflicting placements.

---

## WP10 — Debug geometry and quality visualization

### Purpose

Keep the extractor inspectable while decoupling debug geometry from the main algorithm.

### Scope

Rebuild `get_geometries` around final placement records and grid index utilities.

### Deliverables

1. `debug_export.py`
   - output OSM source features by matched config,
   - routed network paths,
   - topology nodes and anchors,
   - building candidate vs selected footprint,
   - occupancy layers,
   - final reconstructed geometries.

2. Fix current debug issues:
   - rail handling,
   - diagonal/sub-square grid selection,
   - barn process mapping,
   - swallowed exceptions.

3. Optional debug GeoJSON output:

```text
*_source.geojson
*_topology.geojson
*_routed_paths.geojson
*_occupancy.geojson
*_buildings.geojson
*_final.geojson
```

### Acceptance criteria

- Debug geometry export is not used by the hot path.
- A failed route/building has enough diagnostics to understand why it failed.
- Debug export works for all process classes, including rail and barns.

---

## WP11 — Legacy removal and documentation consolidation

### Purpose

After replacement code paths are active, delete or quarantine code that obscures the pipeline.

### Scope

Remove active references to old mutable processors and dead helpers.

### Candidates for deletion or quarantine

- `create_square_graph`
- `create_octagon_graph`
- `line_graph_to_square_graph`
- `handle_square_graph_duplicate_edges`
- `remove_degree_two_nodes_from_graph` if still stubbed
- `search_path2`
- `_match_building`
- obsolete geometry rectangulation helpers unless reintroduced deliberately
- old duplicate `get_matched_cm_type`
- unfinished `assign_type_in_random_clusters`

### Deliverables

1. `docs/osm_extraction_architecture.md`
   - New pipeline overview.
   - Data contracts.
   - Priority/occupancy semantics.
   - Quality metrics.

2. `docs/osm_extraction_config_schema.md`
   - Config fields.
   - Selector semantics.
   - Process kinds.
   - Modifiers.

3. Removal PR/work package with tests proving the new paths are active.

### Acceptance criteria

- There is one documented active path for each feature class.
- Legacy helpers are either removed or explicitly marked as historical reference.
- New contributors can understand the OSM extraction pipeline from docs plus module names.

---

## 5. Bug-fix checklist

This checklist can be implemented early or as part of the relevant work packages.

| Priority | Issue | Proposed fix | Test |
|---:|---|---|---|
| P0 | `type_from_barn_outlines` vs `type_from_barn_outline` mismatch | Use one enum/process name everywhere; validate mapping at config load | Barn fixture reconstructs debug geometry |
| P0 | `rail_tiles` omitted from linear debug handling | Include rail in linear process set | Rail fixture exports as line/debug geometry |
| P0 | Possible `sgdf`/`dgdf` inversion in building reconstruction | Verify convention; fix and add coordinate tests | Axis/diagonal building debug geometry test |
| P0 | `BoundingBox.__init__` likely `crs.to_epsg` bug | Use `crs.to_epsg()` and test non-4326 input | Non-WGS84 bbox fixture |
| P0 | Output extent row uses coordinate bounds as positional index | Replace with explicit extent marker generation | Output dimensions test |
| P1 | `path_to_congih` typo | Rename and keep alias only if needed | Constructor smoke test |
| P1 | Duplicated `get_matched_cm_type` | Move to config/matching module | Unit test single implementation |
| P1 | `assign_type_in_random_clusters` misleading | Replace or remove | Cluster fixture deterministic output |
| P1 | Hard-coded `'Road Tile ...'` in all tile assignment | Validate profile convention; generate process-specific label if needed | Rail/stream/fence output test |
| P2 | Broad exceptions in `get_geometries` | Replace with diagnostics | Debug export failure test |
| P2 | Streamlit coupling in core | Progress callback | Headless test import without Streamlit |

---

## 6. Performance strategy in detail

### 6.1 Expected current hotspots

The function map points to these likely bottlenecks:

1. Pairwise Shapely intersections during line graph creation.
2. NetworkX path search per line segment over a large grid.
3. Dynamic Shapely distance calculations inside routing loops.
4. Per-edge NetworkX compatibility graph construction for tile assignment.
5. Repeated GeoDataFrame spatial queries and DataFrame concatenation.
6. Building branch-and-bound spikes for large footprints.

### 6.2 Replacement strategy

| Current mechanism | Replacement | Expected benefit |
|---|---|---|
| Spatial-index nearest query for grid snapping | Direct affine inverse grid math | Large speedup, simpler code |
| Full-map NetworkX grid graph | Corridor-limited custom A* on integer arrays | Large speedup, less memory |
| Shapely distance inside A* weight | Precomputed distance field per corridor | Large speedup |
| Per-edge tile NetworkX graph | Compiled catalog lookup + DP/CSP | Large speedup, better diagnostics |
| `occupancy_gdf` geometry intersection | Dense layer masks | Large speedup, explicit conflicts |
| repeated `pandas.concat` | collect row chunks, concat once | Moderate speedup |
| MRR-only building fit | candidate scoring against original footprint | Better quality |

### 6.3 Profiling targets

Track at least these stage times:

```text
feature_matching
projection
normal_grid_init
area_rasterization
network_collection
network_noding
network_anchor_selection
network_routing
network_tile_assignment
building_candidate_generation
building_fitting
output_assembly
```

Track these memory-sensitive counts:

```text
normal_grid_cells
sub_grid_cells
network_source_lines
network_topology_nodes
network_topology_edges
routed_path_cells
building_candidates_evaluated
placement_records
output_rows
```

### 6.4 Performance acceptance targets

Exact numbers require current benchmark data, but initial targets should be concrete after WP0. A reasonable first target:

- Small fixtures: no regression, near-instant.
- Medium village extract: at least **2× faster** after WP3–WP5.
- Network-heavy extract: at least **5× faster** after WP6–WP7.
- Large map: no pathological memory growth from NetworkX full-grid graph.

---

## 7. Quality strategy in detail

### 7.1 Network quality metrics

For each routed network:

```text
mean_distance_to_source_m
p95_distance_to_source_m
max_distance_to_source_m
detour_ratio = routed_path_length / source_line_length
topology_node_preservation_rate
failed_edge_count
split_intersection_count
unintended_crossing_count
```

A good route is not merely close to the source line. It must also preserve topology. For example, two major roads crossing in OSM should either:

- meet at one CM intersection tile, or
- be explicitly represented as a split intersection because the tile catalog cannot support the original geometry.

### 7.2 Building quality metrics

For each building:

```text
IoU(selected_footprint, source_polygon)
centroid_shift_m
angle_error_deg
area_error_fraction
collision_status
candidate_count
fit_mode
```

Aggregate:

```text
placed_fraction
mean_iou
p10_iou
p90_centroid_shift_m
collision_count
failed_count_by_reason
```

### 7.3 Area quality metrics

For each area class:

```text
area_preservation_fraction
boundary_error_cells
conflict_count
```

### 7.4 Human-inspection outputs

Add a debug mode that writes layers suitable for visual comparison:

- source OSM features colored by matched config,
- topology graph,
- selected anchors,
- routed network paths,
- building candidates and selected footprints,
- final CM cell output.

This is essential because “better result” is partly visual and domain-specific.

---

## 8. Implementation sequencing recommendations

### Recommended path

The most likely best path is:

1. **WP0 baseline tests/profiling**
2. **WP1 pipeline skeleton**
3. **WP2 config/matching and early bug fixes**
4. **WP3 grid/occupancy**
5. **WP4 area rasterization**
6. **WP5 network topology**
7. **WP6 routing**
8. **WP7 tile assignment**
9. **WP8 building fitter**
10. **WP9 output assembly**
11. **WP10 debug export**
12. **WP11 legacy removal/docs**

### Why this order

- WP0 gives a safety net.
- WP1–WP3 create the architecture needed for performance.
- WP4 is relatively low risk and validates the new grid/occupancy model.
- WP5–WP7 tackle the main performance and result-quality bottleneck: networks.
- WP8 then uses the final road/stream/fence occupancy model for better building placement.
- WP9 removes the old `post_process` semantics only once placements are trustworthy.

### Alternative order

If the current runtime is so dominated by network routing that nothing else matters, an aggressive variant is:

1. WP0
2. WP3 minimal `GridIndex`
3. WP6 prototype integer-grid routing behind a feature flag
4. WP7 tile assignment
5. WP1/WP2 architecture cleanup afterwards

This is riskier because it may create another semi-isolated algorithm beside the old design. I recommend the main order unless immediate runtime relief is urgent.

---

## 9. Feature flags and migration plan

To avoid a single huge switch-over, add feature flags:

```json
{
  "experimental": {
    "use_new_feature_matcher": true,
    "use_new_grid_index": true,
    "use_new_area_rasterizer": true,
    "use_new_network_topology": false,
    "use_new_network_router": false,
    "use_new_tile_assignment": false,
    "use_new_building_fitter": false,
    "use_layered_output": false
  }
}
```

During migration:

- Run old and new implementations on the same fixtures.
- Compare metrics, not only exact output rows.
- Enable new code class by class.
- Remove flags only after the new path is stable.

---

## 10. Suggested Codex implementation prompts

These are not full prompts, but task boundaries that should map well to focused Codex sessions.

### Prompt A — Baseline harness

Implement WP0 only. Add synthetic GeoJSON fixtures, a benchmark runner, and stage timing around the current OSM extraction path. Do not refactor algorithms yet. Add tests for basic invariants.

### Prompt B — New model/config skeleton

Implement WP1 and WP2 model/config skeletons. Add `osm_extraction/models.py`, `config_schema.py`, and `feature_matcher.py`. Keep old algorithm behavior available. Fix the early bug checklist items that do not require algorithm changes.

### Prompt C — GridIndex and occupancy

Implement WP3. Add affine rotated-grid conversion tests against the existing GeoDataFrame grid. Add dense occupancy masks and conflict-policy tests. Do not yet rewrite networks.

### Prompt D — Area rasterizer

Implement WP4 using `GridIndex` and `OccupancyModel`. Replace area/default/point placement behind a feature flag. Add deterministic random tests.

### Prompt E — Network topology

Implement WP5. Replace line collection/noding behind a feature flag. Add tests for crossings, T-junctions, near-misses, clipping, and degree-2 chain collapse.

### Prompt F — Integer-grid router

Implement WP6. Add custom corridor-limited A* over integer arrays. No NetworkX in the new routing hot path. Add route metrics and fixture tests.

### Prompt G — Tile assignment

Implement WP7. Compile tile catalogs once and replace per-edge NetworkX compatibility graphs. Add intersection CSP tests.

### Prompt H — Building fitter

Implement WP8 using occupancy masks and candidate scoring. Add building-road collision tests and simple/complex footprint tests.

### Prompt I — Layered output and debug export

Implement WP9 and WP10. Assemble final rows from placements. Rebuild debug geometry export. Turn `post_process` into validation rather than core conflict resolution.

### Prompt J — Legacy deletion

Implement WP11. Remove or quarantine old functions once tests prove new paths are active. Update architecture/config docs.

---

## 11. Definition of done

The refactor should be considered complete when:

1. The app can still run OSM extraction through `OSMProcessor` or an explicitly migrated equivalent.
2. Core extraction can run headless without Streamlit.
3. Config is validated before processing.
4. Random output is deterministic under a seed.
5. Routing no longer uses NetworkX over the full map grid in the main path.
6. Tile assignment no longer builds NetworkX graphs per edge in the main path.
7. Buildings use explicit candidate scoring and dense occupancy, not only `occupancy_gdf` intersection.
8. Final conflict handling is explicit in `OccupancyModel` / placement assembly, not primarily `post_process` row deletion.
9. The known bug checklist is closed.
10. Stage timings and quality metrics are emitted for every extraction run.
11. Synthetic fixture tests cover areas, points, roads, intersections, streams/fences, buildings, and collisions.
12. Legacy code is removed or clearly quarantined.

---

## 12. Open design decisions

These should be decided explicitly during implementation, ideally after WP0 metrics and a few visual inspections.

1. **Priority semantics**
   - Current behavior suggests lower positive priority wins.
   - Recommendation: rename internally to `rank`, where smaller rank means stronger placement claim.

2. **Network snap tolerance**
   - Start with `1.0 m`, tune with fixtures.

3. **Maximum route deviation**
   - Major roads should stay closer to source geometry than minor roads.
   - Suggested initial values: 24 m major, 32–48 m minor.

4. **Intersection splitting**
   - Need clear rules for when a real OSM intersection cannot be represented by one CM tile.

5. **Building shift limit**
   - Suggested initial cap: ±1 normal tile or ±2 sub-cells for normal buildings; allow larger shifts only when avoiding roads.

6. **What can coexist in one CM square**
   - Encode this in `LayerKind` and `ConflictPolicy` rather than ad hoc row deletion.

7. **External output extent marker**
   - Confirm the exact CM AutoEditor convention and test it explicitly.

---

## 13. Highest-value first fixes if time is limited

If only a limited refactor can be done, prioritize this subset:

1. Add baseline fixtures and stage timing.
2. Fix high-confidence bugs.
3. Introduce `GridIndex` affine snapping.
4. Replace full-map NetworkX routing with corridor-limited integer A* for roads only.
5. Replace per-edge tile NetworkX graph with compiled tile catalog lookup for roads only.
6. Move building collision checks to dense occupancy masks.
7. Defer stream/fence/rail/building quality improvements until road extraction is stable.

This would attack the most likely runtime bottleneck while also improving the most visually important feature class: roads.

---

## 14. Summary recommendation

The strongest recommendation is to treat the refactor as a **replacement of the extraction engine**, not as a cleanup of `processing.py` and `path_search.py`.

The new engine should be based on:

- typed feature/config/placement records,
- affine grid math instead of spatial-index grid snapping,
- dense occupancy masks instead of `occupancy_gdf` plus late DataFrame conflict cleanup,
- topology-first network handling,
- corridor-limited integer-grid routing,
- compiled tile-catalog assignment,
- candidate-scored building fitting,
- deterministic randomness,
- fixture-driven metrics and debug exports.

This approach should be faster because the hot path becomes integer-array based rather than NetworkX/Shapely/GeoDataFrame-heavy. It should also produce better maps because intersections, occupancy, lower-priority feature displacement, and building fitting become explicit optimization/decision problems rather than side effects of rasterization and post-processing.
