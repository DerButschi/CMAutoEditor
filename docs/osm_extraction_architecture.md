# OSM Extraction Architecture

The OSM extractor keeps `OSMProcessor` as the app-facing compatibility boundary and runs the active internals through `terrain_extraction.osm_extraction`. The compatibility class still owns public methods such as `preprocess_osm_data`, `run_processors`, `post_process`, `get_output`, and `get_geometries`, but `run_processors` now dispatches to typed processors instead of the historical `osm_utils.processing` stage table.

## Active Pipeline

`OSMProcessor.preprocess_osm_data` projects matched fixture or OSM geometries and keeps a compatibility `matched_elements` list. `run_processors` converts those entries to `FeatureRecord` objects, creates a `GridIndex`, creates an `OccupancyModel`, and runs the typed stages:

1. `AreaRasterizer` handles area, random, point, and default feature classes.
2. `NetworkTopologyBuilder` nodes road, rail, stream, and fence features while preserving source LineString vertices between topology nodes.
3. `NetworkRouter` routes those topology edges on integer output-cell centers without NetworkX.
4. `TileAssigner` converts successful routes to process-specific tile `PlacementRecord` objects.
5. The linear-dependent placement pass handles `type_from_linear` entries from existing linear placements.
6. `BuildingFitter` scores and places building outlines against current occupancy.
7. `output_rows.py` validates layered conflicts, emits explicit extent markers, validates road structure according to `road_validation_mode`, and normalizes rows for CM AutoEditor.
8. `debug_export.py` builds debug layers for source features, topology, routes, occupancy, buildings, and final rows.

The active data path is therefore:

```text
OSMProcessor
  -> FeatureRecord
  -> GridIndex + OccupancyModel
  -> PlacementRecord
  -> output rows + ExtractionStats
  -> debug layers
```

## Data Contracts

`FeatureRecord` is the source-facing record exchanged between matching and processors. It stores source identity, config name, `ProcessKind`, priority, projected Shapely geometry, source tags, and source properties. The app adapter clips non-point geometries to the effective map bounding polygon before typed processing and drops points outside that polygon.

`PlacementRecord` is the processor-facing output record. It stores layer, grid kind, cells, config name, feature ID, priority, CM type, score, and diagnostics. Final CSV rows are derived from placements rather than written directly by feature processors.

`ExtractionStats` stores timings, counts, quality metrics, and diagnostics. Benchmark JSON uses the same top-level shape, with `timings`, `counts`, `quality`, and `diagnostics`.

## Grid And Occupancy

`GridIndex` owns rotated 8 m grid math. Core processors use affine coordinate conversion for normal cells. Network routes use output-cell center indices, so `RouteRecord.cells` names the CSV cells occupied by the route instead of cells inferred later from grid-line edges. GeoDataFrame views are lazy debug/export views, not the hot-path snapping mechanism.

`OccupancyModel` owns dense layered conflict state. Layers include ground, foliage, linear surface, linear object, building, point object, and reserved. Empty cells use `-1`; metadata maps object IDs back to source feature/config/process diagnostics. Smaller positive priority values are stronger ranks, zero/negative area priorities are weaker than positive priorities, and default rows are the weakest compatibility fills. Defaults are suppressed when a real placement owns the same layer/cell.

Linear tile assignment uses the profile catalog's direction, cost, and side-connection signatures. Tuple signatures such as `(2, 3)` are exact connector identities, not sets of allowed side tokens; a shared edge is compatible only when both tiles open toward each other and expose the same normalized signature on that edge. Candidate routes are solved as least-cost compatible tile paths, and finalized `LinearNetworkState` components are solved as compatible tile assignments before placements are emitted, so adjacent road, stream, rail, and fence cells must satisfy metadata compatibility rather than merely sharing a broad north/south/east/west direction set. Adjacent road cells from disjoint known source features are ambiguous in row-only validation and are left to topology/source-aware diagnostics instead of being treated as hard output defects. Intersections are anchored only at shared routed topology endpoints with three or more incident directions whose arms continue into the next grid square from the selected intersection cell; one-cell stubs are ignored. If no single legal tile can represent an intersection, anchor selection degrades locally: it preserves the deterministic highest-priority road pair, attaches lower-priority arms as nearby legal T-junctions when available, and drops only unattachable arms with `intersection_fallback_failures` diagnostics. Ordinary bends and intermediate path nodes stay as corner or straight route cells. Cardinal-only catalogs reject diagonal route steps instead of coercing them into north/south or east/west tiles. Tile assignment failures are recorded as `tile_assignment_failures` diagnostics with process, route ID when known, cell, required directions, and reason; impossible component-level side-signature assignments use `no_compatible_tile_component`. In warn mode, failed local route pieces are suppressed before occupancy reservation and output assembly while unrelated finalized placements remain eligible for output; in strict mode, tile assignment failures raise `TileAssignmentError`.

Layered output assembly appends the extent marker, clips rows to `idx_bbox`, and only then runs row, layer-conflict, and road validation. This preserves the legacy output contract for fractional diagonal or sub-square placements whose display coordinate can fall just outside the map edge even when their source grid cell was a valid in-bounds candidate. Building placements are stricter before row collapse: they must carry explicit CM output coordinates, and their blocked normal cells are checked against linear cells so a multi-cell footprint cannot hide behind its emitted anchor row. Row and layer-conflict validation remain strict. Road-structure validation supports `"strict"` and `"warn"` modes: strict raises `OutputRowValidationError`, while warn returns the rows and records `road_validation` plus `road_validation_status` diagnostics with mode, validity, summary, and hard issue count.

Building fitting is mode based. Ordinary rectangular residential-like outlines use `single_rect`: the source polygon is summarized by its minimum rotated rectangle, catalog footprints are compared by physical half-cell or diagonal dimensions, and only the best few footprint types are placed around the source centroid with small local shifts. Large, special, or clearly low-rectangularity outlines use `modular_cover`, which is bounded by piece, state, and time limits and keeps a single-rectangle fallback. Road and other linear overlap checks remain local to candidate windows and use the reconstructed CM footprint polygon, not just the output anchor. Selected building geometry keeps the CM output coordinate and grid kind separate from the blocked 8 m normal cells derived from the footprint polygon. Per-building diagnostics report fit mode, MRR dimensions, rectangularity, preferred orientation, best single-rectangle fit, placement counts, selected footprint units, catalog row/column/direction, centroid shift, area error, road overlap, modular use, fallback reason, blocked normal cells, and selected Shapely footprint geometry for debug/export layers.

## Legacy Quarantine

`terrain_extraction.osm_utils.processing` and `terrain_extraction.osm_utils.path_search` are retained as compatibility-only historical references for direct regression tests and old behavior comparison. They are marked with `LEGACY_QUARANTINE_REASON` and `LEGACY_COMPATIBILITY_HELPERS`.

Do not add new architecture to those modules. New behavior belongs under `terrain_extraction.osm_extraction`, with tests under `tests/cm_terrain_extractor/osm_extraction/`.

## Extension Points

Add a new feature class by adding a `ProcessKind` mapping, compiling config entries in `config_schema.py`, emitting `FeatureRecord` objects, implementing a typed processor that returns `PlacementRecord` objects, and wiring it through `ExtractionPipeline` or the `OSMProcessor` typed orchestration.

Add debug output by extending `debug_export.py` with a layer builder that consumes typed records. Debug export must remain outside hot-path processing.

Add quality metrics by extending `ExtractionStats` helpers or benchmark metrics. Metrics should be deterministic for fixture inputs and must not require live OSM downloads.
