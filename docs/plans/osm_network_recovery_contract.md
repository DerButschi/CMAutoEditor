# OSM Network Recovery Contract

This document is authoritative for OSM network recovery. Its content must not be changed unless the user explicitly approves the contract change. If executing any prompt, plan, milestone, or task would violate this contract, stop immediately and ask the user for permission before continuing.

Source plan: `docs/plans/osm_network_recovery_tdd_plan.md`.

Execution plan: `docs/plans/osm_network_recovery_exec_plan.md`.

Status ledger: `docs/plans/osm_network_recovery_status.md`.

Related refactor contract: `docs/plans/osm_extraction_refactoring_contract.md`.

## Purpose

Recover and improve OSM road and linear-network extraction semantics after the typed OSM refactor. The corrected pipeline must produce Combat Mission road networks that are topologically legal, tile-feasible, priority-aware, and explainable through debug artifacts.

The recovery keeps the typed architecture introduced by the refactor. It must not revert wholesale to legacy `osm_utils/processing.py` or `osm_utils/path_search.py`, but it must restore the useful old invariants:

```text
continuous OSM topology
  -> routed square topology
  -> persistent route/connection state
  -> tile-feasible intersection constraints
  -> priority-sensitive occupancy
```

The public compatibility boundary remains the application-facing OSM extraction behavior: given OSM or GeoJSON input, a bounding box, a profile, and an OSM config file, the extractor produces CM AutoEditor-compatible rows and debug geometries through the existing app integration.

## Scope Boundaries

In scope:

- Code under `cm_terrain_extractor_app/terrain_extraction/`, especially `osm_processor.py` and `osm_extraction/`.
- Existing typed modules in `cm_terrain_extractor_app/terrain_extraction/osm_extraction/`.
- New focused modules under `cm_terrain_extractor_app/terrain_extraction/osm_extraction/` when a milestone needs a clearer owner.
- Tests under `tests/cm_terrain_extractor/osm_extraction/`.
- Synthetic road recovery fixtures under `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/`.
- Durable documentation under `docs/` and `docs/plans/`.

Out of scope unless the user approves a contract change:

- Removing `cm_terrain_extractor_app/terrain_extraction/osm_processor.py` as the app-facing compatibility entry point.
- Changing CM AutoEditor CSV column names or coordinate normalization semantics without explicit migration tests and user approval.
- Requiring live OSM network access in unit tests or scenario tests.
- Treating CPU-only restricted-sandbox performance results as authoritative for GPU or unrestricted Windows behavior.
- Reverting to the old graph implementation as the long-term architecture.
- Silently emitting broken road rows when routing, anchor selection, or tile assignment cannot satisfy this contract.

## Ownership Boundaries

### App Compatibility Boundary

Owned by `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`.

Responsibilities:

- Preserve `OSMProcessor` as the public compatibility adapter.
- Preserve public app-facing methods unless a user-approved contract change says otherwise.
- Delegate network recovery behavior to the typed pipeline once the pipeline owns the relevant stage.
- Return CM AutoEditor-compatible output rows and debug geometries through the existing app integration.

Invariants:

- Application callers must not import recovery internals directly.
- Compatibility helpers may remain for existing tests, but active extraction must have one clear owner.
- Streamlit remains a UI-edge concern, not a required dependency for headless unit tests.

### Typed OSM Extraction Package

Owned by `cm_terrain_extractor_app/terrain_extraction/osm_extraction/`.

Responsibilities:

- Keep typed records, grid math, network topology, routing, tile assignment, occupancy, output rows, debug export, and pipeline orchestration in focused modules.
- Preserve deterministic behavior under an explicit seed.
- Expose enough structured diagnostics that a failing road fixture can identify the failing stage.

Primary module ownership:

- `network_topology.py`: line clipping, normalization, noding, topology nodes, topology edges, snapping, and source-line topology diagnostics.
- `network_routing.py`: route planning, route records, route diagnostics, route retry policy, and integration with raster spines, anchor plans, tile feasibility, and linear state.
- `tile_assignment.py`: compiled tile catalogs, direction-set feasibility, strict tile variant selection, and tile failure diagnostics.
- `occupancy.py`: dense layer occupancy used by roads, buildings, areas, and conflict policy.
- `output_rows.py`: final CM row assembly and output conflict validation.
- `debug_export.py`: debug layers, app geometry export, and optional GeoJSON-like diagnostic views.
- `pipeline.py`: authoritative orchestration path once consolidation is complete.
- New focused modules, when introduced:
  - `raster_spine.py`: source-line support cell extraction and route/spine metrics.
  - `linear_network_state.py`: persistent per-cell linear connection state.
  - `anchor_selection.py`: tile-aware topology anchor candidate scoring and selected anchor plans.
  - `road_output_validation.py`: final-output graph reconstruction and semantic road validation.

Invariants:

- Add focused modules when they clarify ownership. Do not create a second parallel architecture.
- Network recovery must strengthen the typed pipeline, not expand quarantined legacy helpers.
- Public row semantics remain owned by `output_rows.py`; debug layers remain owned by `debug_export.py`.

### Tests and Fixtures

Owned by `tests/cm_terrain_extractor/osm_extraction/`.

Responsibilities:

- New road recovery fixtures live under `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/`.
- Unit tests cover pure contracts such as route cell semantics, raster-spine generation, tile feasibility, and connection-state updates.
- Scenario tests run representative GeoJSON fixtures through extraction and inspect final rows plus debug artifacts.
- Regression comparison tests, when added, compare semantic metrics rather than exact legacy row equality.

Invariants:

- Every production-code milestone starts with failing tests for the behavior being introduced or repaired.
- Unit and scenario tests must not require live OSM downloads or internet access.
- New recovery tests must inspect final output rows where possible, because a road route is not correct until the final rows encode a legal connected CM road network.
- Existing tests under `tests/cm_terrain_extractor/` remain valid unless a user-approved contract change updates the behavior.

## Core Network Contracts

### Route Node and Tile-Cell Semantics

`RouteRecord` must not expose ambiguous route-cell semantics.

Required outcome:

- Either route directly over CM road tile cells as the primary path representation, or keep route nodes and tile cells as separate explicit fields.
- If route nodes are kept, the contract is:

```python
len(route.nodes) >= 2
len(route.step_cells) == len(route.nodes) - 1
```

- Tile assignment, output rows, and validation must consume only explicit CM road tile cells, not ambiguous lattice nodes.

Forbidden behavior:

- Using `RouteRecord.cells` to sometimes mean nodes and sometimes mean occupied road cells.
- Letting a route reach tile assignment without an explicit cell-path or route-step-cell contract.
- Relying on `len(route.cells) == len(route.nodes)` as a hidden compatibility fallback.

### Raster-Spine and Line-Support Contract

Every routed topology edge must have a source-line support contract before route acceptance.

Required `RasterSpine` semantics:

```python
@dataclass(frozen=True)
class RasterSpine:
    topology_edge_id: int
    cells: tuple[GridCell, ...]
    progress: tuple[float, ...]
    distance_m: tuple[float, ...]
    source_length_m: float
```

Invariants:

- `cells`, `progress`, and `distance_m` have equal length.
- `progress` is ordered from source start to source end and normalized from `0.0` to `1.0`.
- Endpoint-only source lines must still produce enough support cells to constrain equal-length route alternatives.
- Router diagnostics report spine cell count, mean and max route distance to source/spine, skipped support cells, and detour cells.

### Tile Catalog Feasibility Contract

`CompiledTileCatalog` must be a feasibility oracle before final tile assignment.

Required interface:

```python
catalog.has_tile(required_dirs: frozenset[Direction]) -> bool
catalog.best_tile(required_dirs: frozenset[Direction], road_class: object | None = None) -> TileVariant | None
catalog.allowed_step_dirs() -> frozenset[Direction]
catalog.can_extend(existing_dirs: frozenset[Direction], new_dir: Direction) -> bool
```

Invariants:

- Required directions are normalized as connections to neighboring cells: `N`, `E`, `S`, `W`.
- Diagonal continuation is forbidden unless the active profile catalog explicitly supports diagonal connection semantics and output rows can represent them.
- Missing 3-way or 4-way tiles must produce split-intersection planning or structured failure, not fake tiles.
- Startup or test diagnostics must identify catalog gaps for required direction sets.

### Persistent Linear Connection State

Accepted routes must update persistent shared state before later routes are routed.

Required owner:

- `linear_network_state.py`, or an equivalent focused owner if explicitly recorded in the status ledger.

Required state:

```python
class LinearNetworkState:
    occupied: object
    connection_bits: object
    route_id_at_cell: object
    priority_at_cell: object
    process_at_cell: object
    intersection_kind_at_cell: object
```

Required methods:

```python
can_enter_cell(cell, incoming_dir, outgoing_dir, process, priority)
reserve_path(route, process, priority)
release_path(route_id)
required_dirs(cell)
as_debug_layer()
```

Invariants:

- Same-road continuation is allowed only when the resulting direction union is catalog-legal.
- Minor roads may connect to major roads only through deliberate legal intersections or merges.
- Lower-priority routes must not overwrite stronger routes silently.
- Parallel roads must not collapse into shared cells unless that sharing is an intentional legal merge.
- Buildings and areas must see final road occupancy before fitting or output conflict resolution can delete roads.

### Tile-Aware Anchor Selection

Topology nodes must be converted to route anchors through candidate planning, not closest-cell snapping alone.

Required models:

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

Required anchor-plan outcomes:

- `SingleAnchorPlan`
- `SplitAnchorPlan`
- `FailedAnchorPlan`

Invariants:

- Every topology node has a documented anchor decision before edge routing.
- Candidate scoring accounts for distance from source node, direction distortion, road class priority, occupancy conflicts, tile feasibility, and raster-spine alignment.
- If one cell cannot legally represent all incident directions, the algorithm chooses a legal split-intersection plan or structured failure.
- Anchor movement beyond configured limits requires explicit diagnostic relaxation.

### Tile-Feasible Routing Contract

Routing must consider geometry, raster spines, tile feasibility, occupancy, and persistent linear state during neighbor expansion.

Required behavior:

- Route search state includes enough direction history to evaluate turns and local direction sets.
- Neighbor expansion queries `CompiledTileCatalog` and `LinearNetworkState`.
- Invalid direction sets, illegal overwrites, impossible anchors, and blocked corridors are hard constraints unless a recorded retry mode permits relaxation.
- Retry diagnostics identify corridor widening, anchor-radius increase, midpoint split, soft crossing, or final route failure.

### Priority and Stage Semantics

Linear features must not be routed as one undifferentiated batch.

Invariants:

- Config priority or rank affects topology grouping, route order, occupancy, and conflict policy, not only final row sorting.
- Stronger or major roads claim space before weaker or minor roads.
- Process-pair policy distinguishes connective interaction from avoidance interaction.
- Road, rail, stream, fence, and wall interactions must be deterministic and visible in diagnostics.

### Strict Tile Assignment Contract

Tile assignment is a validator and finalizer.

Invariants:

- Tile assignment consumes legal cell paths and `LinearNetworkState.connection_bits`.
- Every road output cell has a catalog-backed tile variant for its required direction set.
- Missing tile variants are hard failures for roads and must not produce broken final rows.
- Synthetic config names such as `{process}_intersection` are forbidden unless explicitly mapped.
- Output placement diagnostics include source process/config, source feature IDs, connection directions, selected tile ID, and whether the cell is straight, bend, T, intersection, or dead-end.

### Final Output Road Validation Contract

Final output rows must reconstruct into a legal connected road network or emit structured failures.

Required validator:

```python
validate_road_output_rows(rows, profile) -> RoadValidationReport
```

Required report fields:

- illegal tile labels,
- cells with no valid connection interpretation,
- disconnected components,
- one-cell gaps at topology junctions,
- dangling arms that do not correspond to source endpoints,
- duplicate mutually exclusive cells,
- unsupported diagonal continuations,
- intersections without legal tile pattern,
- road/building overlap.

Invariants:

- Road recovery is not complete until core fixtures pass final-output validation.
- Tests may allow one legal 4-way tile, two adjacent T-junctions, or a small split intersection when all required connectivity and tile legality are preserved.
- A geometric crossing without a shared road connection is invalid unless explicitly configured as an overpass or underpass.

## Required Debug Artifacts

Every network extraction in debug mode must expose these layers or structured equivalents:

| Debug artifact | Required purpose |
| --- | --- |
| `source_lines` | Projected OSM linework after clipping and normalization. |
| `topology_nodes` | Continuous-space intersections, endpoints, and snapped topology nodes. |
| `topology_edges` | Noded continuous edges before raster routing. |
| `raster_spines` | Ordered source-line support cells and line-distance metrics. |
| `anchor_candidates` | Candidate grid anchors with scores, feasibility, and reasons. |
| `selected_anchors` | Chosen single anchor, split-intersection group, or failed anchor plan. |
| `routed_nodes` | Lattice-node route path if a node path is still part of the route model. |
| `route_step_cells` | Actual CM road tile cells for route transitions or the direct cell path. |
| `connection_bits` | Per-cell `N/E/S/W` connection bitset after route reservation. |
| `tile_required_dirs` | Direction set required for every output road cell. |
| `selected_tiles` | Chosen CM tile variant for every placed road cell. |
| `tile_failures` | Missing tile, impossible transition, conflict, or hard-failure diagnostics. |

## Representative Fixture Contract

Core network recovery fixtures must include:

- `straight_road_2pt.geojson`
- `diagonalish_2pt_equal_length.geojson`
- `ninety_degree_bend.geojson`
- `four_way_crossing.geojson`
- `t_junction.geojson`
- `staggered_near_miss.geojson`
- `minor_meets_major.geojson`
- `parallel_close_roads.geojson`
- `road_near_building.geojson`
- `road_stream_crossing.geojson`

The first implementation milestone may start with a subset, but the status ledger must record any missing fixture and its owning milestone.

## Validation and Environment Contract

Use the approved Conda tools:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe
```

Validation rules:

- Run the smallest relevant tests that exercise the milestone.
- Run scenario integration tests when final output semantics are touched.
- Run ruff on every touched Python file.
- Do not create ad-hoc sanity scripts when pytest is the appropriate verification path.
- Do not run performance tests in CPU mode and report them as authoritative.
- Record exact validation commands and results in `docs/plans/osm_network_recovery_status.md`.

## Contract Change Procedure

If implementation reveals that this contract is wrong or too restrictive:

1. Stop before making the violating change.
2. Record the issue as a contract-change request in `docs/plans/osm_network_recovery_status.md` if that edit itself does not change the contract.
3. Ask the user for explicit approval.
4. Change this contract only after user approval.
5. Update the exec plan and status ledger to match the approved contract.
