# OSM Extraction Refactoring Contract

This document is authoritative for the CMTerrainExtractor OSM extraction refactor. Its content must not be changed unless the user explicitly approves the contract change. If executing any prompt, plan, milestone, or task would violate this contract, stop immediately and ask the user for permission before continuing.

Source outline: `docs/plans/osm_extraction_refactoring_plan.md`.

Execution plan: `docs/plans/osm_extraction_refactoring_exec_plan.md`.

Status ledger: `docs/plans/osm_extraction_refactoring_status.md`.

## Purpose

Refactor the OSM extraction engine into a typed, deterministic, test-covered pipeline that is faster, easier to inspect, and produces better Combat Mission terrain output. The refactor preserves the application's external OSM processing surface while replacing the current internals that mix Streamlit progress, mutable processor state, GeoDataFrames, NetworkX routing, occupancy geometry, and final CSV rows.

The compatibility boundary is the public behavior needed by the app: given OSM or GeoJSON input, a bounding box, a profile, and an OSM config file, the extractor must produce CM AutoEditor-compatible terrain rows and debug geometries through the existing app integration.

## Scope Boundaries

In scope:

- Code under `cm_terrain_extractor_app/terrain_extraction/`, especially `osm_processor.py` and `osm_utils/`.
- A new internal package at `cm_terrain_extractor_app/terrain_extraction/osm_extraction/`.
- New tests under `tests/cm_terrain_extractor/osm_extraction/`.
- Targeted regression tests in `tests/cm_terrain_extractor/test_osm_processor.py` when preserving legacy compatibility behavior.
- Durable documentation under `docs/` or `docs/plans/`.

Out of scope unless the user approves a contract change:

- Removing `cm_terrain_extractor_app/terrain_extraction/osm_processor.py` as the app-facing compatibility entry point.
- Changing CM AutoEditor CSV column names or coordinate normalization semantics without explicit migration coverage.
- Changing the Streamlit UI workflow as part of the extraction-engine refactor.
- Requiring live OSM network access in unit tests.
- Running performance tests in CPU mode and treating the results as authoritative for GPU or unrestricted Windows behavior.
- Deleting legacy OSM helpers before replacement paths are active and covered by tests.

## Current-State Corrections

The older outline remains the architectural source, but these current repository facts override stale statements in it:

- The repository now has an active pytest suite under `tests/cm_terrain_extractor/`; the refactor still needs a new OSM-specific TDD suite under `tests/cm_terrain_extractor/osm_extraction/`.
- Current OSM extraction still uses `OSMProcessor`, `terrain_extraction/osm_utils/processing.py`, `terrain_extraction/osm_utils/path_search.py`, NetworkX, GeoDataFrame grids, `occupancy_gdf`, and Streamlit progress in core paths.
- Confirmed defects or drift to fix through the plan include `path_to_congih`, `type_from_barn_outline` versus `type_from_barn_outlines`, `BoundingBox.__init__` passing `crs.to_epsg` instead of `crs.to_epsg()` in the non-WGS84 branch, duplicate `get_matched_cm_type`, hard-coded `Road Tile ...` labels in shared tile assignment, and full-grid NetworkX routing.
- Rail debug/export handling must be revalidated rather than assumed broken, because current `get_geometries` has generic non-building geometry handling.

## Ownership Boundaries

### Compatibility Entry Point

Owned by `cm_terrain_extractor_app/terrain_extraction/osm_processor.py`.

Responsibilities:

- Preserve the constructor signature `OSMProcessor(profile: str, bbox: BoundingBox, path_to_config: str = "default_osm_config.json")`.
- Preserve app-facing methods: `preprocess_osm_data`, `run_processors`, `post_process`, `write_to_file`, `get_output`, and `get_geometries`.
- Act as a thin compatibility adapter once the new pipeline is available.
- Accept an optional progress callback or internal no-op progress adapter so extraction can run headless in tests.

Invariants:

- Application callers must not need to import the new internal package directly.
- The compatibility wrapper may keep temporary legacy attributes only while milestones still need them. Any retained legacy state must be recorded in the status ledger as technical debt.
- Core extraction code must not require Streamlit to be importable after the pipeline skeleton milestone.

### New OSM Extraction Package

Owned by `cm_terrain_extractor_app/terrain_extraction/osm_extraction/`.

Responsibilities:

- Typed records, config validation, feature matching, grid math, occupancy, processors, row assembly, debug export, and stats.
- Deterministic pipeline execution under an explicit seed.
- Headless execution suitable for pytest and benchmark runs.

Initial module ownership:

- `models.py`: enums and immutable records exchanged between stages.
- `config_schema.py`: config loading, validation, selector compilation, process validation, and catalog validation.
- `feature_matcher.py`: OSM tag and ID matching, projection, and `FeatureStore` construction.
- `grid_index.py`: rotated regular-grid math and lazy debug GeoDataFrame views.
- `occupancy.py`: dense layered occupancy arrays and conflict policy.
- `area_rasterizer.py`: area, default, random, and point placement logic.
- `network_topology.py`: line clipping, noding, snapping, topology graph construction, and chain collapse.
- `network_routing.py`: corridor-limited integer-grid routing with no NetworkX hot path.
- `tile_assignment.py`: compiled tile catalogs, per-path assignment, and intersection solving.
- `building_fitter.py`: candidate generation, scoring, cluster ordering, and placement.
- `output_rows.py`: final row assembly, extent markers, normalization, and validation.
- `debug_export.py`: debug geometry and optional GeoJSON export.
- `stats.py`: timings, counters, quality metrics, and JSON serialization.
- `pipeline.py`: `ExtractionPipeline`, `ExtractionContext`, progress callback protocol, feature flags, and compatibility orchestration.

### Legacy OSM Utils

Owned by `cm_terrain_extractor_app/terrain_extraction/osm_utils/` until replacement paths are proven.

Responsibilities:

- Remain available for compatibility and comparison during migration.
- Serve as historical reference for current behavior and regression tests.

Invariants:

- Legacy helpers may be called behind feature flags while a milestone migrates one feature class at a time.
- Legacy helpers must not be expanded into a second long-term architecture.
- Removal or quarantine must happen only after tests prove the new path covers the behavior.

### Tests

Owned by `tests/cm_terrain_extractor/`.

Responsibilities:

- Existing app tests remain valid.
- New extraction tests go under `tests/cm_terrain_extractor/osm_extraction/`.
- Synthetic GeoJSON fixtures go under `tests/cm_terrain_extractor/osm_extraction/fixtures/`.

Invariants:

- Every milestone that touches Python production code must start with failing tests.
- All touched Python production code must be covered by tests after the milestone.
- Live OSM downloads are not unit tests. Use fixtures, fakes, or marked integration checks.

## Public and Internal Interfaces

### Config

`ExtractionConfig` must be a validated, compiled representation of the JSON config. Raw nested JSON dict access must not be used in new hot-path code.

Required records:

- `ProcessKind`: enum of known process names, including road, rail, stream, fence, area, point, random, default, and building-outline processes.
- `TagSelector`: compiled selector with explicit `any` or `all` semantics.
- `CMType`: normalized CM menu/category/direction/catalog record.
- `ConfigEntry`: name, active flag, process kind, priority or rank, selectors, required tags, excluded tags, allowed IDs, excluded IDs, CM types, and modifiers.
- `ExtractionConfig`: ordered entries, feature flags, seed, routing parameters, snap tolerance, catalog mappings, and validation diagnostics.

Invariants:

- Unknown process names fail before processing.
- `type_from_barn_outline` and `type_from_barn_outlines` must be reconciled through one canonical process name, with compatibility handling only if tests require it.
- Building process names must be validated against profile building catalog mappings.
- Tile catalog references must be validated once at startup.
- Weighted random choices must be compiled once and driven by the pipeline RNG.

### Feature Records

`FeatureRecord` must replace ad hoc `matched_elements` dicts in new code.

Required fields:

- `feature_id: str | int | None`
- `source_index: int`
- `config_name: str`
- `process: ProcessKind`
- `priority: int`
- `geometry: BaseGeometry`
- `source_tags: Mapping[str, str]`
- `source_properties: Mapping[str, Any]`

`FeatureStore` must group records by process and config name without losing source order.

Invariants:

- Tags are extracted consistently from either `properties["tags"]` or direct properties.
- Projection happens only after a feature matches at least one active config entry.
- Source feature metadata is retained for diagnostics and debug export.

### GridIndex

`GridIndex` owns all rotated grid math.

Required behavior:

- Create from `BoundingBox` and `cell_size_m=8`.
- Convert projected coordinates to normal cells with affine inverse rotation and floor division.
- Convert projected coordinates to nearest routing nodes with affine inverse rotation and rounding.
- Return normal cell centers and polygons.
- Return sub-square and diagonal centers and polygons using documented conventions.
- Clip integer windows to map bounds.
- Produce GeoDataFrame views lazily for debug/export only.

Invariants:

- New core code must not use spatial-index nearest queries for regular-grid snapping.
- Normal, sub-square, and diagonal grid conventions must be tested against current representative grids before replacing output/debug behavior.

### OccupancyModel

`OccupancyModel` owns conflict policy before final output rows are assembled.

Required layers:

- `GROUND`
- `FOLIAGE`
- `LINEAR_SURFACE`
- `LINEAR_OBJECT`
- `BUILDING`
- `POINT_OBJECT`
- `RESERVED`

Required behavior:

- Dense integer arrays per layer with `-1` for empty.
- Optional priority/rank arrays.
- Metadata mapping placed object ID to source feature, config, process, and diagnostics.
- `reserve`, `place`, `release`, `is_blocked`, and `can_place`.
- Human-readable conflict reasons.

Invariants:

- `occupancy_gdf` must not be the normal conflict mechanism for new code.
- Road, rail, stream, building, and object conflicts must be resolved before final rows.
- `post_process` must become validation or compatibility cleanup, not the primary conflict resolver.

### Placement and Output

Processors emit `PlacementRecord` objects instead of final rows.

Required fields:

- `layer`
- `grid_kind`
- `cells`
- `config_name`
- `feature_id`
- `priority`
- `cm_type`
- `score`
- `diagnostics`

`output_rows.py` converts placements to CM AutoEditor-compatible rows.

Invariants:

- Output order must be stable and tested.
- Extent marker generation must be explicit and tested.
- Duplicate mutually exclusive rows for one cell/layer must be validation failures.
- Running the same input with the same seed must produce identical rows.

### Stats and Debug

`ExtractionStats` must include timings, counts, and quality metrics. It must serialize to JSON for benchmark output.

Minimum timings:

- `feature_matching`
- `projection`
- `grid_init`
- `area_rasterization`
- `network_noding`
- `network_routing`
- `tile_assignment`
- `building_fitting`
- `output_assembly`
- `debug_export`

Minimum quality metrics:

- network disconnected components,
- route distance to source line,
- route detour ratio,
- collision count,
- building placed/failed counts,
- building IoU and centroid shift where available.

Debug export must be derived from records, placements, stats, and `GridIndex`; it must not be required by hot-path algorithms.

### Progress and Headless Execution

`ProgressCallback` is an optional callable or protocol owned by `pipeline.py`.

Invariants:

- Core extraction must run in pytest without importing or monkeypatching Streamlit.
- UI code may adapt Streamlit progress to the callback.
- Tests may use a no-op or recording progress callback.

## Feature Flags and Migration

The pipeline may use feature flags to move one feature class at a time:

- `use_new_feature_matcher`
- `use_new_grid_index`
- `use_new_area_rasterizer`
- `use_new_network_topology`
- `use_new_network_router`
- `use_new_tile_assignment`
- `use_new_building_fitter`
- `use_layered_output`
- `use_new_debug_export`

Invariants:

- Flags are temporary migration tools, not a permanent dual implementation.
- Every enabled flag must have tests and status-ledger notes.
- Old and new outputs may be compared by metrics and invariants rather than exact random row equality.

## Validation Requirements

Use only the approved Conda tools:

- `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe`
- `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe`
- `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe`

Before a milestone is marked done:

- The smallest relevant pytest target must pass.
- Ruff must pass on touched Python files.
- Status must record exact validation commands and results.
- New behavior, config semantics, or contracts must be reflected in durable docs.
- Any unverified manual or performance risk must be recorded explicitly.

## Contract Change Procedure

If implementation reveals that this contract is wrong or too restrictive:

1. Stop work before making the violating change.
2. Record the proposed contract change in `docs/plans/osm_extraction_refactoring_status.md` only if that edit itself does not change the contract.
3. Ask the user for approval.
4. Change this contract only after explicit user approval.
5. Update the exec plan and status ledger to match the approved contract.
