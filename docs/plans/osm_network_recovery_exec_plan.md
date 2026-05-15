# OSM Network Recovery Exec Plan

> **For agentic workers:** This plan is executable milestone by milestone. Use an execution workflow, update `docs/plans/osm_network_recovery_status.md` after every milestone, and obey `docs/plans/osm_network_recovery_contract.md` as the authoritative contract.

**Goal:** Restore and improve OSM road and linear-network extraction so final CM output rows form legal, connected, tile-feasible networks under the active profile catalog.

**Architecture:** Keep the typed OSM extraction architecture and strengthen its network contracts. `OSMProcessor` remains the public compatibility adapter, while network recovery behavior lives in `cm_terrain_extractor_app/terrain_extraction/osm_extraction/` with focused owners for topology, raster spines, anchor selection, routing, persistent linear state, tile assignment, output validation, debug export, and pipeline orchestration.

**Tech Stack:** Python, pytest, ruff, NumPy, Pandas, GeoPandas, Shapely, PyProj, typed OSM extraction modules, CM AutoEditor CSV output, synthetic GeoJSON fixtures.

**Authoritative contract:** `docs/plans/osm_network_recovery_contract.md`. If a milestone conflicts with the contract, stop and ask the user before editing code.

**Source plan:** `docs/plans/osm_network_recovery_tdd_plan.md`.

Reference style: This plan follows an ExecPlan style suitable for controlled Codex execution and plan-execute workflows. The [OpenAI ExecPlans cookbook](https://developers.openai.com/cookbook/articles/codex_exec_plans) describes plans as self-contained living documents with progress, discoveries, decision logs, concrete steps, validation, and recovery notes. The [OpenAI shell guidance](https://developers.openai.com/api/docs/guides/tools-shell) describes the local inspect-edit-validate loop. The [Codex docs](https://developers.openai.com/codex/cloud) describe delegating coding tasks to an agent that can read, modify, and run code.

## Global Rules

- Do not edit Python production code before writing a failing test for the behavior being introduced or repaired.
- Unit and scenario tests must not require live OSM downloads, internet access, or external services.
- Keep new recovery tests under `tests/cm_terrain_extractor/osm_extraction/`.
- Keep new recovery fixtures under `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/`.
- Use the approved Conda tools:
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe`
- Run ruff on every touched Python file. Use `--no-cache` if sandbox cache writes fail.
- Update `docs/plans/osm_network_recovery_status.md` after every milestone with progress, decisions, debt, blockers, validation commands, results, and residual risk.
- Preserve `OSMProcessor` as the public app boundary.
- Preserve CM AutoEditor row shape and coordinate normalization unless the user approves a contract change.
- Treat broken road output as a hard failure or structured diagnostic, not as acceptable degraded output.

## Current Implementation Facts

- `RouteRecord` currently has `nodes` and ambiguous `cells` in `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`.
- `NetworkRouter` currently records route cells directly from route nodes in `network_routing.py`.
- `TileAssigner` currently derives route direction sets during assignment in `tile_assignment.py`.
- `CompiledTileCatalog` currently has `candidates_for`, but lacks the full feasibility oracle surface from the source plan.
- `NetworkRouter` already uses integer-grid A* with state `(xidx, yidx, incoming_direction)`, but tile feasibility and persistent connection state are not yet first-class search constraints.
- `DebugExport` already exposes typed debug layers, but it does not yet expose all road recovery artifacts required by the contract.
- The typed OSM pipeline exists, but the source plan identifies possible orchestration drift between `OSMProcessor` and `ExtractionPipeline`.

## Dependency Graph

```text
Milestone M0 Semantic regression harness
  -> Milestone M1 Route node vs tile-cell contract
    -> Milestone M2 Raster-spine / line-support extraction
      -> Milestone M3 Tile catalog feasibility oracle
        -> Milestone M4 LinearNetworkState
          -> Milestone M5 Tile-aware anchor selection
            -> Milestone M6 Tile-feasible routing
              -> Milestone M7 Priority/stage linear processing
                -> Milestone M8 Tile assignment validator/finalizer
                  -> Milestone M9 Output-level road invariants/debug
                    -> Milestone M10 Building/area revalidation
                      -> Milestone M11 Pipeline orchestration consolidation
                        -> Milestone M12 Performance pass
```

Shortest high-value path if scope must be reduced:

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M8 -> M9
```

This shorter path does not fully solve tile-aware anchor selection, staged routing, or pipeline consolidation, but it exposes and fixes the most concrete route-cell and illegal-tile failures.

## Do Not Proceed Conditions

- The milestone would violate `docs/plans/osm_network_recovery_contract.md`.
- A production-code change cannot be covered by a failing test first.
- A fixture or test would require live OSM network access.
- Output row shape, coordinate normalization, or the public `OSMProcessor` boundary would change without user approval.
- A route failure would be hidden by emitting partial or broken road rows.
- Current behavior is ambiguous and cannot be inferred from code, fixtures, source plan, or contract.
- Performance validation would be run in CPU-only restricted mode and treated as authoritative.

## Likely Blockers

- Ambiguous `RouteRecord.cells` usage affects routing, tile assignment, debug export, tests, and output assembly.
- Existing tests may assert the current ambiguous cell behavior and need migration to the new explicit contract.
- Test catalogs may not contain all direction sets needed by fixtures, so missing catalog data must become expected split-intersection or structured-failure behavior.
- Cross-process road, stream, rail, and fence interaction policy may not be explicit enough in current config and may need a narrowly documented default.
- Some legal CM intersection patterns may require visible displacement from exact OSM geometry.
- Debug artifacts may need schema changes across `NetworkRoutingResult`, `TileAssignmentResult`, `DebugExport`, and scenario helpers.
- Windows sandbox cache or temp permissions may cause pytest or ruff invocation failures; retry once with `--no-cache` or corrected invocation, then record the environment failure.

## Rollback Strategy

- Keep each milestone independently revertible.
- Prefer adding tests and focused modules before rewiring existing modules.
- If a milestone fails validation, revert that milestone's production changes and keep tests only if they accurately describe the approved target behavior.
- Record rollback decisions and remaining failing tests in `docs/plans/osm_network_recovery_status.md`.

## Milestone M0: Semantic Regression Harness

**Goal:** Add road recovery fixtures, fixture runner helpers, debug capture, road graph reconstruction, and at least one reproducible failing road scenario without changing road algorithms.

**Files:**

- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/straight_road_2pt.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/diagonalish_2pt_equal_length.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/ninety_degree_bend.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/four_way_crossing.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/t_junction.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/road_near_building.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/network_recovery_helpers.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py` only to expose existing debug information if tests cannot observe it.
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add fixture tests that fail because `run_osm_extraction_fixture` and `ExtractionTestResult` do not exist.
- Implement `run_osm_extraction_fixture(fixture_name, profile, config_name, bbox, *, seed=0, debug=True)`.
- Implement `ExtractionTestResult` with final output rows, debug network artifacts, diagnostics/stats, road graph reconstruction, connected component checks, illegal direction-set checks, and compact ASCII grid output.
- Add at least one test that reproduces a current bad road or intersection case as red or xfail with a precise failure reason.

**Acceptance criteria:**

- Each initial fixture can run through the current refactored pipeline.
- Harness failures include structured diagnostics identifying topology, anchor, route, step-cell, tile, or output-stage symptoms.
- Small fixture failures can print an ASCII grid.
- No road algorithm is changed except non-invasive debug exposure.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check tests\cm_terrain_extractor\osm_extraction\network_recovery_helpers.py tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py --no-cache
```

**Status update:** Record fixture list, known red cases, debug artifact gaps, validation results, and any missing scenarios deferred to later milestones.

## Milestone M1: Route Node vs Tile-Cell Contract

**Goal:** Remove ambiguous route-cell semantics so routing, tile assignment, debug export, and output validation agree on actual CM road tile cells.

**Files:**

- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Modify: existing network routing, tile assignment, debug export, and output tests that mention `RouteRecord.cells`.
- Add or modify: `tests/cm_terrain_extractor/osm_extraction/test_route_cell_contract.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add unit tests that assert the selected model.
- Prefer direct cell-path routing if it can be introduced narrowly. If not, introduce explicit `RouteRecord.step_cells` or `RouteRecord.tile_cells` and keep `nodes` only for lattice diagnostics.
- Add conversion tests if node routing remains:
  - horizontal transition maps to one expected step cell,
  - vertical transition maps to one expected step cell,
  - path of five nodes maps to four step cells,
  - diagonal transition is rejected unless catalog support is explicit.
- Update tile assignment to consume only explicit tile cells.
- Remove ambiguous fallback behavior that treats route nodes as output cells.

**Acceptance criteria:**

- No production road output path consumes ambiguous `RouteRecord.cells`.
- `straight_road_2pt` produces continuous output cells.
- `ninety_degree_bend` produces exactly one bend cell or a structured failure unrelated to node/cell ambiguity.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_route_cell_contract.py tests\cm_terrain_extractor\osm_extraction\test_network_routing.py tests\cm_terrain_extractor\osm_extraction\test_tile_assignment.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\models.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\tile_assignment.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record the selected route representation model and any compatibility debt left around old `cells` references.

## Milestone M2: Raster-Spine / Line-Support Extraction

**Goal:** Make full source geometry constrain routes, especially endpoint-only and equal-length Manhattan alternatives.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/raster_spine.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Add: `tests/cm_terrain_extractor/osm_extraction/test_raster_spine.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add unit tests for `RasterSpine` generation from two-point diagonalish, horizontal, vertical, and rotated-bbox lines.
- Implement correctness-first Shapely cell-intersection support extraction inside a small candidate bbox.
- Order support cells by projection progress along the source line.
- Add route diagnostics for source/spine distances, skipped support cells, and extra detour cells.
- Add routing cost terms for distance to source line and raster spine without changing tile feasibility yet.

**Acceptance criteria:**

- Every routed topology edge has a raster spine or a structured failure explaining why not.
- `diagonalish_2pt_equal_length` chooses the path closer to the source line or fails with precise missing capability.
- Route diagnostics report spine statistics.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_raster_spine.py tests\cm_terrain_extractor\osm_extraction\test_network_routing.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\raster_spine.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\models.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record the initial raster-spine algorithm, known performance debt, and any fixture still failing for non-spine reasons.

## Milestone M3: Tile Catalog Feasibility Oracle

**Goal:** Make tile direction-set feasibility queryable before and during routing.

**Files:**

- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/config_schema.py` only if catalog normalization belongs there.
- Add or modify: `tests/cm_terrain_extractor/osm_extraction/test_tile_catalog_feasibility.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_tile_assignment.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add exhaustive direction-set tests for straight, bend, T, 4-way, and dead-end direction sets.
- Add tests that missing 4-way support triggers split-intersection or structured failure expectations instead of fake 4-way assignment.
- Add `has_tile`, `best_tile`, `allowed_step_dirs`, and `can_extend` to `CompiledTileCatalog`.
- Normalize required direction sets as `N`, `E`, `S`, `W`.
- Add catalog-gap diagnostics.

**Acceptance criteria:**

- Routing and anchor-selection code can ask feasibility questions without constructing final output rows.
- Diagonal moves are rejected unless the active catalog explicitly supports them.
- Catalog tests clearly identify supported and unsupported direction sets.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_tile_catalog_feasibility.py tests\cm_terrain_extractor\osm_extraction\test_tile_assignment.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\tile_assignment.py tests\cm_terrain_extractor\osm_extraction\test_tile_catalog_feasibility.py tests\cm_terrain_extractor\osm_extraction\test_tile_assignment.py --no-cache
```

**Status update:** Record catalog gaps discovered in active profiles and the chosen direction normalization model.

## Milestone M4: LinearNetworkState

**Goal:** Replace hidden persistent graph-node connection semantics with explicit shared linear connection state.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/linear_network_state.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Add: `tests/cm_terrain_extractor/osm_extraction/test_linear_network_state.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add unit tests for reserving straight, bend, T, and 4-way paths.
- Add tests for illegal direction union, lower-priority overwrite rejection, and route release after tentative failure.
- Implement dense arrays or equivalent deterministic storage for occupied cells, connection bits, route id, priority, process, and intersection kind.
- Integrate accepted-route reservation into routing at the minimal point that preserves existing behavior.
- Expose `connection_bits` debug output.

**Acceptance criteria:**

- Later routes see previously accepted routes.
- `four_way_crossing`, `parallel_close_roads`, and `minor_meets_major` failures are explainable through state diagnostics.
- Tile assignment can consume state-derived required directions.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_linear_network_state.py tests\cm_terrain_extractor\osm_extraction\test_network_routing.py tests\cm_terrain_extractor\osm_extraction\test_tile_assignment.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\linear_network_state.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\tile_assignment.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record state shape, bit encoding, integration points, and any remaining legacy route-state debt.

## Milestone M5: Tile-Aware Anchor Selection

**Goal:** Select topology anchors through tile-feasible candidate planning instead of closest-cell snapping alone.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/anchor_selection.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Add: `tests/cm_terrain_extractor/osm_extraction/test_anchor_selection.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add tests for candidate generation within radius 1 and retry radius 2 to 3.
- Add tests that `four_way_crossing` finds a feasible single anchor when the catalog supports 4-way.
- Add tests that a catalog with no 4-way tile produces a split-intersection plan or failed anchor plan.
- Implement `AnchorCandidate`, `SingleAnchorPlan`, `SplitAnchorPlan`, and `FailedAnchorPlan`.
- Add selected-anchor and anchor-candidate debug layers.
- Route edges from selected anchor plans without widening milestone scope into full routing rewrite.

**Acceptance criteria:**

- Every topology node has a documented anchor decision before edge routing.
- `t_junction` chooses a legal 3-way anchor when catalog support exists.
- Anchor movement beyond configured limits is diagnostic-visible.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_anchor_selection.py tests\cm_terrain_extractor\osm_extraction\test_network_routing.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\anchor_selection.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record anchor scoring factors, split-intersection support level, and fixtures still blocked by incomplete split routing.

## Milestone M6: Tile-Feasible Routing

**Goal:** Make route search reject paths that tile assignment cannot legally represent.

**Files:**

- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/linear_network_state.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/raster_spine.py`
- Add or modify: `tests/cm_terrain_extractor/osm_extraction/test_tile_feasible_routing.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add tests for diagonalish support path, legal bend routing, four-way anchor routing, blocked corridor retry, and impossible-catalog failure.
- Extend route search neighbor expansion to query raster-spine cost, catalog feasibility, and `LinearNetworkState.can_enter_cell`.
- Add retry diagnostics for corridor widening, anchor radius, midpoint split, and final failure.
- Ensure impossible routes fail before output rows are emitted.

**Acceptance criteria:**

- Tile assignment failures become rare and meaningful.
- Router does not emit obviously untileable paths.
- Failure tests produce structured route diagnostics and no broken output rows.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_tile_feasible_routing.py tests\cm_terrain_extractor\osm_extraction\test_network_routing.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\linear_network_state.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\tile_assignment.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\raster_spine.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record route cost terms, hard constraints, retry modes, and remaining fixtures blocked by anchor or catalog limits.

## Milestone M7: Priority/Stage Linear Processing

**Goal:** Restore priority and stage semantics so major or stronger networks claim space before weaker networks route, attach, avoid, or fail.

**Files:**

- Create or modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/linear_processing_plan.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_topology.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/linear_network_state.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Add: `tests/cm_terrain_extractor/osm_extraction/test_linear_processing_plan.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add tests for major road preserving source alignment while minor road shifts or attaches.
- Add tests for lower-priority overwrite rejection.
- Add tests for road/fence and road/stream process-pair policy using deterministic defaults.
- Implement `LinearProcessingPlan` to group topology edges by process, config, rank, and feature metadata.
- Route, validate, tile, and commit each group before later groups.
- Make connective interaction distinct from avoidance interaction.

**Acceptance criteria:**

- Priority affects routing and occupancy, not only final row conflict sorting.
- Road, stream, rail, fence, and wall interactions are deterministic and diagnostic-visible.
- Repeated runs with the same seed produce stable route order and output.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_linear_processing_plan.py tests\cm_terrain_extractor\osm_extraction\test_network_routing.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\linear_processing_plan.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_topology.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\linear_network_state.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\pipeline.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record default process-pair policy, stage order, config assumptions, and any user-confirmation needs.

## Milestone M8: Tile Assignment Validator/Finalizer

**Goal:** Make tile assignment strictly finalize already-valid routed cell paths from connection state.

**Files:**

- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/linear_network_state.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Add or modify: `tests/cm_terrain_extractor/osm_extraction/test_tile_assignment_validator.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add tests that every state direction set receives a selected catalog tile.
- Add tests that removing a needed tile causes route or intersection failure before output.
- Make tile assignment consume `LinearNetworkState.connection_bits`.
- Add placement metadata for source process/config, contributing feature IDs, connection dirs, selected tile ID, and role.
- Remove repair behavior that hides invalid route paths.

**Acceptance criteria:**

- Every road output placement corresponds to a legal catalog tile.
- Tile failures are hard failures with structured diagnostics.
- Tile assignment is deterministic for the same state and profile.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_tile_assignment_validator.py tests\cm_terrain_extractor\osm_extraction\test_tile_assignment.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\tile_assignment.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\linear_network_state.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\models.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record tile metadata shape, removed repair behavior, and any catalog gaps.

## Milestone M9: Output-Level Road Invariants and Visual Debug

**Goal:** Lock road quality at the final output level with validators and failure visualization.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/road_output_validation.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/output_rows.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Create or modify: `tests/cm_terrain_extractor/osm_extraction/test_road_output_validation.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_network_recovery_harness.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add tests for illegal labels, disconnected components, one-cell junction gaps, unsupported diagonal continuations, duplicate cells, invalid intersections, and road/building overlap.
- Implement `validate_road_output_rows(rows, profile) -> RoadValidationReport`.
- Add ASCII grid rendering for small failed fixtures.
- Add optional debug geometry or SVG/PNG hook only if it is needed for tests or app diagnostics.
- Run all core road fixtures through final-output validators.

**Acceptance criteria:**

- Core fixtures pass final-output validation:
  - `straight_road_2pt`
  - `diagonalish_2pt_equal_length`
  - `ninety_degree_bend`
  - `four_way_crossing`
  - `t_junction`
  - `minor_meets_major`
  - `parallel_close_roads`
  - `road_near_building`
- Internal debug artifacts explain any remaining structured failures.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_road_output_validation.py tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\road_output_validation.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\output_rows.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record final-output validator coverage, fixture pass/fail matrix, and any accepted legal alternatives.

## Milestone M10: Building and Area Revalidation After Road Repair

**Goal:** Ensure corrected roads do not regress area rasterization, building fitting, road occupancy, or output conflict behavior.

**Files:**

- Modify: `tests/cm_terrain_extractor/osm_extraction/test_area_rasterizer.py`
- Modify: `tests/cm_terrain_extractor/osm_extraction/test_building_fitter.py`
- Modify or add: `tests/cm_terrain_extractor/osm_extraction/test_cross_feature_integration.py`
- Modify production modules only if tests expose a real integration defect.
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add cross-feature tests that buildings avoid final road cells and roads are not deleted by output conflict cleanup.
- Re-run area threshold, rotated bbox, default layer, and conflict-layer tests.
- Re-run simple building, diagonal building, building near road, and village-cluster tests.
- Fix only defects exposed by these tests; avoid broad building or area redesign.

**Acceptance criteria:**

- Road, building, and area integration tests pass deterministically.
- Road occupancy is committed before building fitting.
- Area layers do not overwrite road or building layers.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_area_rasterizer.py tests\cm_terrain_extractor\osm_extraction\test_building_fitter.py tests\cm_terrain_extractor\osm_extraction\test_cross_feature_integration.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Status update:** Record cross-feature results, any area/building fixes, and remaining integration risk.

## Milestone M11: Pipeline Orchestration Consolidation

**Goal:** Ensure there is one authoritative typed orchestration path in `ExtractionPipeline`, with `OSMProcessor` acting as compatibility adapter.

**Files:**

- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Modify or add: `tests/cm_terrain_extractor/osm_extraction/test_pipeline.py`
- Modify: `tests/cm_terrain_extractor/test_osm_processor.py`
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add tests that direct `ExtractionPipeline` API and public `OSMProcessor` API run the same implementation and produce equivalent final rows/debug layers.
- Move real extraction orchestration into `ExtractionPipeline.run()` if it still lives in `OSMProcessor`.
- Keep `OSMProcessor.preprocess_osm_data`, `run_processors`, `get_output`, and `get_geometries` compatible while delegating.
- Remove duplicated orchestration only after tests prove equivalence.

**Acceptance criteria:**

- There is one authoritative orchestration path.
- Public app behavior remains compatible.
- Direct pipeline tests and `OSMProcessor` tests agree on road recovery behavior.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_pipeline.py tests\cm_terrain_extractor\test_osm_processor.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\pipeline.py cm_terrain_extractor_app\terrain_extraction\osm_processor.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py tests\cm_terrain_extractor --no-cache
```

**Status update:** Record the final orchestration owner, removed duplication, and any compatibility shims retained.

## Milestone M12: Performance Pass After Correctness

**Goal:** Improve runtime only after road correctness fixtures and output validators are green.

**Files:**

- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/raster_spine.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Modify or add: benchmark or regression tests under `tests/regression/` only if a durable performance check is appropriate.
- Modify: `docs/plans/osm_network_recovery_status.md`

**TDD task boundaries:**

- Add semantic regression tests that lock current green fixture behavior before optimizing.
- Replace Shapely raster-spine cell intersection with supercover grid rasterization only if profiling shows it matters.
- Tighten routing corridor windows, cache distance fields, and reduce heap allocation only while preserving semantic metrics.
- Precompute direction-set to tile-choice maps.
- Emit debug layers only in debug mode.

**Acceptance criteria:**

- Road recovery semantic tests remain green.
- Network-heavy scenarios are faster than the pre-optimization typed path.
- Performance results are not reported as authoritative if run in CPU-only restricted mode.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction tests\cm_terrain_extractor\osm_extraction --no-cache
```

Run benchmark commands only in an environment where their results are meaningful, then record exact command lines, scenario names, timings, and hardware/context in the status ledger.

**Status update:** Record baseline, optimized timings, semantic comparison results, and any performance debt left intentionally.

## Status Update Procedure

After each milestone, update `docs/plans/osm_network_recovery_status.md`:

- Mark milestone status and date.
- Record decisions made and who approved them.
- Record contract-change requests separately from normal implementation decisions.
- Record technical debt, reason, and paydown target.
- Record exact validation commands and results.
- Record blockers, residual risk, and fixtures still failing.

## Definition of Done

OSM network recovery is complete when:

- Core road fixtures pass final-output validation.
- Every road output cell has a legal tile variant for its required connection directions.
- Every OSM topology junction becomes one legal CM intersection tile, a legal split-intersection pattern, or a structured failure diagnostic.
- Endpoint-only roads follow their source geometry through raster spine support.
- Road catalogs without diagonal continuation never emit unsupported diagonal road output.
- Accepted road routes update shared connection and occupancy state before later routes are planned.
- Priority/rank affects routing and occupancy.
- Tile assignment failures are hard failures for roads.
- Debug artifacts explain every failed fixture.
- `OSMProcessor` and direct `ExtractionPipeline` paths run the same implementation.
