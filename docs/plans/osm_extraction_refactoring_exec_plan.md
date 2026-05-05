# OSM Extraction Refactoring Exec Plan

> **For agentic workers:** This plan is executable milestone by milestone. Use an execution workflow, update `docs/plans/osm_extraction_refactoring_status.md` after every milestone, and obey `docs/plans/osm_extraction_refactoring_contract.md` as the authoritative contract.

**Goal:** Replace the current OSM extraction internals with a typed, deterministic, test-covered pipeline while preserving the `OSMProcessor` app-facing compatibility boundary.

**Architecture:** `OSMProcessor` remains the public adapter. New engine code lives under `cm_terrain_extractor_app/terrain_extraction/osm_extraction/` and separates config, features, grid math, occupancy, placement, routing, output rows, debug export, and stats. Migration happens behind feature flags so each feature class can be validated against fixtures before legacy paths are removed.

**Tech Stack:** Python, pytest, ruff, Pandas, GeoPandas, Shapely, NumPy, PyProj, NetworkX for legacy comparison only, Streamlit at the UI edge only, CM AutoEditor CSV output.

**Authoritative contract:** `docs/plans/osm_extraction_refactoring_contract.md`. If a milestone conflicts with the contract, stop and ask the user before editing code.

Reference style: This plan follows an ExecPlan style suitable for controlled Codex execution and plan-execute workflows. The [OpenAI ExecPlans cookbook](https://developers.openai.com/cookbook/articles/codex_exec_plans) describes plans as self-contained living documents with progress and decision logs. The [OpenAI shell guidance](https://developers.openai.com/api/docs/guides/tools-shell) describes the local plan-execute loop of inspecting, editing, and validating. The [Codex docs](https://developers.openai.com/codex/cloud) describe delegating coding tasks to an agent that can read, modify, and run code.

## Global Rules

- Do not edit Python production code before writing the failing test for the behavior being moved, introduced, or fixed.
- All touched Python production code must be covered by tests after the milestone.
- Place new OSM extraction tests under `tests/cm_terrain_extractor/osm_extraction/`.
- Preserve existing tests under `tests/cm_terrain_extractor/`; the old outline's statement that no test suite exists is stale.
- Use the approved Conda tools:
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe`
- Run ruff on every touched Python file. Use `--no-cache` if sandbox cache writes fail.
- Unit tests must not require live OSM downloads or internet access.
- Update `docs/plans/osm_extraction_refactoring_status.md` after each milestone with progress, decisions, debt, blockers, validation commands, results, and residual risk.
- Stop and ask the user before changing contract semantics, public CSV semantics, the app-facing `OSMProcessor` compatibility boundary, or milestone scope.

## Current Implementation Facts

- Current app-facing OSM code is `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`.
- Legacy algorithms live mainly in `cm_terrain_extractor_app/terrain_extraction/osm_utils/processing.py` and `cm_terrain_extractor_app/terrain_extraction/osm_utils/path_search.py`.
- Current code still uses `matched_elements` dicts, `processing_stages`, GeoDataFrame grids, `occupancy_gdf`, NetworkX graph routing, and Streamlit progress in core preprocessing.
- Current known defects or drift to cover are `path_to_congih`, `type_from_barn_outline` versus `type_from_barn_outlines`, the non-WGS84 `BoundingBox.__init__` `crs.to_epsg` call bug, duplicate `get_matched_cm_type`, hard-coded `Road Tile ...` labels, and full-grid NetworkX routing.
- Rail debug/export handling must be tested before changing it because current `get_geometries` handles non-building rows generically.

## Dependency Graph

```text
Milestone 0 baseline fixtures and metrics
  -> Milestone 1 pipeline skeleton and compatibility wrapper
    -> Milestone 2 config schema, feature matcher, deterministic seed, early bug fixes
      -> Milestone 3 GridIndex and OccupancyModel
        -> Milestone 4 area rasterizer
        -> Milestone 5 network topology
          -> Milestone 6 integer-grid routing
            -> Milestone 7 tile assignment and intersections
              -> Milestone 8 building fitter
                -> Milestone 9 layered output rows
                  -> Milestone 10 debug export and quality visualization
                    -> Milestone 11 legacy quarantine/removal and docs
```

Milestones 4 and 5 both depend on Milestone 3. Milestone 8 depends on road/linear occupancy from Milestones 5 through 7. Milestone 9 must wait until the main placement producers exist. Milestone 11 must wait until fixture tests prove the new paths are active.

## Do Not Proceed Conditions

- The milestone would violate `docs/plans/osm_extraction_refactoring_contract.md`.
- A production-code change cannot be covered by a failing test first.
- A fixture or benchmark would require network access.
- A migration flag would leave both old and new behavior active without a clear owner or status entry.
- Output row semantics or CSV columns would change without explicit tests and user approval.
- Performance validation is attempted in CPU-only restricted mode and treated as authoritative.
- Current behavior is ambiguous and cannot be inferred from code, fixtures, or the contract.

## Likely Blockers

- `osm_processor.py` imports Streamlit directly and creates progress bars during preprocessing.
- Existing terrain modules use absolute `terrain_extraction.*` imports, so tests often add `cm_terrain_extractor_app` to `sys.path`.
- `processing.py` is large and mixes area rasterization, network topology, tile assignment, building fitting, and output-row side effects.
- `path_search.py` builds full NetworkX grid graphs and contains custom A* logic coupled to GeoDataFrame grid rows.
- Building catalog mappings currently disagree with config process spelling for barns.
- Current output extent marker behavior is not clearly documented and must be captured by tests before replacement.
- Windows sandbox temp/cache permissions may make pytest or ruff fail before running code; retry once with corrected `--basetemp` or `--no-cache`, then record the environment failure.

## Rollback Strategy

- Keep each milestone small enough that its touched files can be reverted independently.
- Prefer adding new modules and adapting `OSMProcessor` over rewriting legacy helpers in place.
- Keep feature flags defaulting to legacy behavior until new fixtures and invariants pass.
- If validation fails after implementation, revert only that milestone's production changes and keep tests or docs if they accurately describe the desired behavior.
- Record rollback decisions in the status ledger.

## Milestone 0: Baseline Fixtures, Invariant Tests, Benchmark Runner, and Metrics JSON

**Goal:** Create a repeatable baseline for current OSM behavior before algorithm replacement.

**Files:**

- Create: `tests/cm_terrain_extractor/osm_extraction/__init__.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/simple_area.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/single_road.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/crossroads.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/road_near_building.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/village_cluster.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/fixtures/multi_network.geojson`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_baseline_invariants.py`
- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction_benchmark.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py` only for non-invasive timing hooks if needed.
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First add invariant tests that fail because fixtures, runner, or metrics output do not exist. Do not refactor algorithms in this milestone.

**Tasks:**

- Add small synthetic GeoJSON fixtures for area, bend road, crossing roads, road-building collision, village buildings, and road/stream/fence interaction.
- Add tests asserting at least five invariants: rows inside map, no duplicate mutually exclusive cell rows after `post_process`, crossroads output has connected linear cells, building-road collision fixture reports or prevents collision, and same seed produces stable output.
- Add `osm_extraction_benchmark.py` with CLI options `--fixture`, `--profile`, `--config`, `--seed`, and `--json-output`.
- Emit metrics JSON with `timings`, `counts`, and `quality` keys. If legacy stage timings are approximate, record that in `stats["diagnostics"]`.
- Ensure benchmark runner can execute fixture data without network access.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_baseline_invariants.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m cm_terrain_extractor_app.terrain_extraction.osm_extraction_benchmark --fixture crossroads --profile cold_war --config default_osm_config.json --seed 123
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction_benchmark.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Fixtures exist, invariant tests pass against current behavior or record expected legacy defects as xfail with reasons, benchmark emits valid JSON, and status records the baseline.

## Milestone 1: Pipeline Skeleton and Compatibility Wrapper

**Goal:** Introduce the new package and pipeline orchestration without changing extraction algorithms yet.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/__init__.py`
- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/stats.py`
- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_models.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_pipeline.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** Write tests for importability, immutable records, no Streamlit import requirement, progress callback recording, and deterministic RNG initialization before implementing modules.

**Tasks:**

- Define `ProcessKind`, `GridKind`, `LayerKind`, `CMType`, `GridCell`, `GridNode`, `FeatureRecord`, `PlacementRecord`, `ExtractionResult`, and `ExtractionStats`.
- Define `ProgressCallback` and `ExtractionContext`.
- Implement `ExtractionPipeline` with stub stages that can wrap or call the legacy processor while returning `ExtractionResult`.
- Add a compatibility path in `OSMProcessor` that can hold a pipeline instance without changing public method names.
- Add a no-op progress callback for headless tests.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_models.py tests\cm_terrain_extractor\osm_extraction\test_pipeline.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction cm_terrain_extractor_app\terrain_extraction\osm_processor.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** The new package imports headlessly, `OSMProcessor` public methods still exist, and no algorithmic behavior changes are required.

## Milestone 2: Config Schema, Feature Matcher, Deterministic Seed, and Early Bug Fixes

**Goal:** Make feature classification validated, deterministic, and covered by focused regression tests.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/config_schema.py`
- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/feature_matcher.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_config_schema.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_feature_matcher.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/bbox_utils.py`
- Modify: `profiles/__init__.py`
- Modify only if tests prove it necessary: `cm_terrain_extractor_app/terrain_extraction/osm_utils/processing.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write failing tests for required tags, excluded tags, allowed IDs, unknown process validation, deterministic seed behavior, barn process spelling, non-WGS84 `BoundingBox`, and `path_to_config`.

**Tasks:**

- Implement config loading and validation for known processes, selectors, CM types, building mappings, tile catalogs, and modifiers.
- Implement `FeatureMatcher` that extracts tags from either nested `properties["tags"]` or direct properties and returns `FeatureRecord` objects.
- Add deterministic `np.random.default_rng(seed)` ownership to `ExtractionContext`.
- Rename `path_to_congih` to `path_to_config`, keeping a temporary read-only compatibility alias only if existing tests or callers require it.
- Reconcile barn process spelling by supporting the config's `type_from_barn_outline` and profile mapping consistently.
- Fix `BoundingBox.__init__` so non-WGS84 projection passes `crs.to_epsg()`.
- Move duplicated matching/type logic toward the new schema module where possible without breaking legacy callers.
- Add regression coverage for hard-coded tile label behavior if it can be fixed safely in this milestone; otherwise record as M7 debt.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_config_schema.py tests\cm_terrain_extractor\osm_extraction\test_feature_matcher.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction cm_terrain_extractor_app\terrain_extraction\osm_processor.py cm_terrain_extractor_app\terrain_extraction\bbox_utils.py profiles\__init__.py tests\cm_terrain_extractor\osm_extraction tests\cm_terrain_extractor\test_osm_processor.py --no-cache
```

**Acceptance criteria:** Config errors fail early with useful messages, feature matching is deterministic and headless, listed early bugs are either fixed with tests or recorded as deferred debt with a milestone owner.

## Milestone 3: GridIndex and Dense OccupancyModel

**Goal:** Add regular-grid affine math and dense occupancy arrays for new code paths.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/grid_index.py`
- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/occupancy.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_grid_index.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_occupancy.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests comparing normal, sub-square, and diagonal coordinates against current representative grids and tests for layer conflict decisions.

**Tasks:**

- Implement `GridIndex.from_bbox`, `projected_to_cell`, `projected_to_nearest_node`, `cell_center`, `cell_polygon`, subcell helpers, diagonal helpers, and clipped integer windows.
- Add lazy GeoDataFrame debug/export view creation; keep GeoDataFrames out of core snapping.
- Implement `OccupancyModel` with dense arrays per layer, priority/rank arrays, metadata, `reserve`, `place`, `release`, `is_blocked`, and `can_place`.
- Define conflict decisions with readable reasons.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_grid_index.py tests\cm_terrain_extractor\osm_extraction\test_occupancy.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\grid_index.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\occupancy.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\models.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Grid conversions match the old grid within tested tolerances, occupancy conflict checks do not require Shapely in common cases, and conventions are documented in tests.

## Milestone 4: Area Rasterizer

**Goal:** Move area/default/point placement behind the new grid and occupancy model.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/area_rasterizer.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_area_rasterizer.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify only for feature-flag adapter: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for polygon rasterization thresholds, deterministic weighted choices, default handling, point placement, and no duplicate mutually exclusive rows.

**Tasks:**

- Implement integer bounding-window candidate selection for polygons.
- Intersect only candidate cells against polygons, using vectorized Shapely where practical.
- Implement deterministic weighted choice helpers using the pipeline RNG.
- Replace `assign_type_in_random_clusters` behavior for new paths with seeded spatially correlated clusters.
- Represent defaults as base-layer placements or emit them after real placements so they do not compete with real features.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_area_rasterizer.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_baseline_invariants.py -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\area_rasterizer.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\pipeline.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Area fixtures pass invariants, random area output is deterministic under seed, and feature flag status is recorded.

## Milestone 5: Network Topology and Noding

**Goal:** Build robust typed topology from OSM linework before routing.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_topology.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_network_topology.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for crossroads, T-junctions, near-miss snap tolerance, clipping, MultiLineString and GeometryCollection normalization, and degree-2 chain collapse.

**Tasks:**

- Implement `TopologyNode`, `TopologyEdge`, and topology graph records.
- Clip all lines to the effective bbox.
- Normalize LineString, MultiLineString, and polygon boundaries into source line records.
- Use `STRtree` or Shapely unary operations to find candidate intersections; avoid all-pairs intersections.
- Coalesce near-identical endpoints/intersections with default snap tolerance `1.0 m`.
- Collapse degree-2 chains with identical metadata for fewer routing calls.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_topology.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_topology.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\models.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Crossroads produce one topology node, near-miss behavior is controlled by tolerance, and topology construction emits diagnostics.

## Milestone 6: Corridor-Limited Integer-Grid Routing

**Goal:** Replace full-map NetworkX grid routing in the new path with bounded A* or Dijkstra over integer arrays.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/network_routing.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_network_routing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/stats.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for route success, blocked occupancy, shared intersection anchors, minor-road relaxation, failed-route diagnostics, and absence of NetworkX in the new routing module.

**Tasks:**

- Implement move-set compilation from tile catalog constraints.
- Select one anchor per topology node before routing incident edges.
- Build corridor windows from source-line bounds and precompute local distance arrays.
- Implement A* or Dijkstra with state `(node_i, node_j, incoming_direction)` using `heapq`.
- Add route ordering by priority/rank, network class, edge length, and intersection importance.
- Add retry policy: widen minor corridor, allow configured soft crossing, split long edge, then drop with diagnostics.
- Emit route metrics for success/failure count, source-line distance, detour ratio, split intersections, and forced relaxations.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_routing.py -v
Select-String -Path cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py -Pattern "networkx","nx."
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\network_routing.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\stats.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** New routing hot path does not import NetworkX, fixtures route through shared anchors, failures are explicit, and metrics are recorded.

## Milestone 7: Tile Catalog Assignment and Intersection Solving

**Goal:** Replace per-edge NetworkX tile graphs with compiled catalog lookup, path assignment, and intersection solving.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/tile_assignment.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_tile_assignment.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify if legacy compatibility bug is fixed here: `cm_terrain_extractor_app/terrain_extraction/osm_utils/processing.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for catalog compilation, required directions, intersection union directions, road/rail/stream/fence label generation, impossible intersection diagnostics, and deterministic candidate choice.

**Tasks:**

- Implement `CompiledTileCatalog`, `TileVariant`, direction extraction, and required-direction lookup.
- Assign tiles to simple paths in linear time when no transition constraints exist.
- Add dynamic programming only if the profile catalog requires transition constraints.
- Solve intersection anchors once using unioned incident directions.
- Fix hard-coded `Road Tile ...` labels for rail, stream, and fence if tests prove current behavior wrong; otherwise record the convention as validated.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_tile_assignment.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\tile_assignment.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Catalogs compile once per run, intersections use one shared tile decision, and failure diagnostics name missing directions or catalog gaps.

## Milestone 8: Building Fitter v2

**Goal:** Improve building placement quality using candidates, scoring, and occupancy rather than `occupancy_gdf` intersection.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/building_fitter.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_building_fitter.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/models.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/stats.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for simple rectangular IoU, complex footprint candidate scoring, road avoidance, bounded runtime fallback, and cluster ordering.

**Tasks:**

- Collect and clean building polygons.
- Compute descriptors: area, oriented rectangle, rectangularity, elongation, orientation, nearest-road distance.
- Generate candidate placements across orientation, local shifts, catalog footprint sizes, and modular options.
- Score candidates by IoU, centroid shift, angle error, area error, collision, road overlap, and modular penalty.
- Place clusters in constrained-first order and allow bounded local repair.
- Emit diagnostics for every placed or dropped building.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_building_fitter.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\building_fitter.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\stats.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Buildings avoid linear occupancy when possible, simple buildings retain or improve IoU, complex buildings are not blindly collapsed, and difficult cases have bounded diagnostics.

## Milestone 9: Layered Output Rows and Conflict Validation

**Goal:** Assemble final CM rows from placements and make `post_process` a compatibility validation step.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/output_rows.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_output_rows.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for stable output order, coordinate normalization, explicit extent marker, duplicate conflict validation, deterministic rows under seed, and `OSMProcessor.get_output` compatibility.

**Tasks:**

- Implement `placements_to_output_rows`, `append_extent_marker`, `normalize_output_coordinates`, and `validate_output_rows`.
- Make defaults implicit or emit them after real features in a stable order.
- Validate no duplicate mutually exclusive rows, no building-road collisions, valid coordinates, and valid profile labels.
- Update `OSMProcessor` adapter to use new row assembly when `use_layered_output` is enabled.
- Keep legacy `post_process` available for old path; in new path it should assert/validate rather than delete core conflicts.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_output_rows.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\output_rows.py cm_terrain_extractor_app\terrain_extraction\osm_processor.py tests\cm_terrain_extractor\osm_extraction tests\cm_terrain_extractor\test_osm_processor.py --no-cache
```

**Acceptance criteria:** New output assembly is deterministic, explicit conflict validation catches injected defects, and app-facing output remains compatible.

## Milestone 10: Debug Export and Quality Visualization

**Goal:** Rebuild debug geometries and diagnostics from the new records and placements.

**Files:**

- Create: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/debug_export.py`
- Create: `tests/cm_terrain_extractor/osm_extraction/test_debug_export.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/pipeline.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_extraction/stats.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First write tests for source features, routed paths, topology nodes, anchors, occupancy layers, building selected footprints, final reconstructed geometries, rail handling, barn handling, and diagnostic propagation.

**Tasks:**

- Implement debug layers from `FeatureStore`, topology, routes, occupancy, placements, and output rows.
- Add optional GeoJSON output for source, topology, routed paths, occupancy, buildings, and final rows.
- Replace swallowed exceptions with diagnostics in new debug export.
- Revalidate rail debug/export behavior rather than assuming the legacy bug.
- Update `OSMProcessor.get_geometries` adapter under the new debug flag.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_debug_export.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\debug_export.py cm_terrain_extractor_app\terrain_extraction\osm_processor.py tests\cm_terrain_extractor\osm_extraction --no-cache
```

**Acceptance criteria:** Debug export is not used by hot-path algorithms, failed routes/buildings have useful diagnostics, and `get_geometries` remains app-compatible.

## Milestone 11: Legacy Quarantine/Removal and Durable Docs

**Goal:** Remove or quarantine inactive legacy paths and document the final architecture.

**Files:**

- Create: `docs/osm_extraction_architecture.md`
- Create: `docs/osm_extraction_config_schema.md`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_utils/processing.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_utils/path_search.py`
- Modify: `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`
- Modify: `docs/plans/osm_extraction_refactoring_status.md`

**TDD requirement:** First add or update tests proving all feature classes use new active paths before deleting or quarantining old helpers.

**Tasks:**

- Remove or quarantine unused legacy helpers such as old graph builders, `search_path2`, old building matching, duplicate `get_matched_cm_type`, and obsolete random cluster logic.
- Keep any retained legacy helper explicitly marked as historical reference or compatibility-only.
- Remove feature flags only after tests prove the new path is stable.
- Document the new pipeline, data contracts, priority/rank semantics, occupancy policy, config schema, modifiers, extension points, and quality metrics.
- Run broad OSM extraction tests and nearby app tests.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py tests\cm_terrain_extractor\test_actions_smoke.py -q
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction tests\cm_terrain_extractor\osm_extraction tests\cm_terrain_extractor\test_osm_processor.py --no-cache
Select-String -Path docs\osm_extraction_architecture.md,docs\osm_extraction_config_schema.md -Pattern "GridIndex","OccupancyModel","FeatureRecord","PlacementRecord","ExtractionStats","ProcessKind"
```

**Acceptance criteria:** There is one documented active path for every feature class, legacy code is removed or explicitly quarantined, and final docs are sufficient for a new contributor.

## Status Update Procedure

After each milestone:

- Change milestone status to `In progress`, `Blocked`, or `Done`.
- Add owner and exact date.
- Record decisions and contract-change requests.
- Record technical debt introduced intentionally.
- Record exact validation commands and concise results.
- Record manual smoke or visual checks separately from automated tests.
- Record unresolved risks before handing off.
