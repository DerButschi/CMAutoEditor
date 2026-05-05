# OSM Extraction Architecture

The OSM extractor keeps `OSMProcessor` as the app-facing compatibility boundary and runs the active internals through `terrain_extraction.osm_extraction`. The compatibility class still owns public methods such as `preprocess_osm_data`, `run_processors`, `post_process`, `get_output`, and `get_geometries`, but `run_processors` now dispatches to typed processors instead of the historical `osm_utils.processing` stage table.

## Active Pipeline

`OSMProcessor.preprocess_osm_data` projects matched fixture or OSM geometries and keeps a compatibility `matched_elements` list. `run_processors` converts those entries to `FeatureRecord` objects, creates a `GridIndex`, creates an `OccupancyModel`, and runs the typed stages:

1. `AreaRasterizer` handles area, random, point, and default feature classes.
2. `NetworkTopologyBuilder` nodes road, rail, stream, and fence features.
3. `NetworkRouter` routes those topology edges on integer grid nodes without NetworkX.
4. `TileAssigner` converts successful routes to process-specific tile `PlacementRecord` objects.
5. The linear-dependent placement pass handles `type_from_linear` entries from existing linear placements.
6. `BuildingFitter` scores and places building outlines against current occupancy.
7. `output_rows.py` validates layered conflicts, emits explicit extent markers, and normalizes rows for CM AutoEditor.
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

`FeatureRecord` is the source-facing record exchanged between matching and processors. It stores source identity, config name, `ProcessKind`, priority, projected Shapely geometry, source tags, and source properties.

`PlacementRecord` is the processor-facing output record. It stores layer, grid kind, cells, config name, feature ID, priority, CM type, score, and diagnostics. Final CSV rows are derived from placements rather than written directly by feature processors.

`ExtractionStats` stores timings, counts, quality metrics, and diagnostics. Benchmark JSON uses the same top-level shape, with `timings`, `counts`, `quality`, and `diagnostics`.

## Grid And Occupancy

`GridIndex` owns rotated 8 m grid math. Core processors use affine coordinate conversion for normal cells and routing nodes. GeoDataFrame views are lazy debug/export views, not the hot-path snapping mechanism.

`OccupancyModel` owns dense layered conflict state. Layers include ground, foliage, linear surface, linear object, building, point object, and reserved. Empty cells use `-1`; metadata maps object IDs back to source feature/config/process diagnostics. Smaller positive priority values are stronger ranks. Default rows are weaker compatibility fills and are suppressed when a real placement owns the same layer/cell.

## Legacy Quarantine

`terrain_extraction.osm_utils.processing` and `terrain_extraction.osm_utils.path_search` are retained as compatibility-only historical references for direct regression tests and old behavior comparison. They are marked with `LEGACY_QUARANTINE_REASON` and `LEGACY_COMPATIBILITY_HELPERS`.

Do not add new architecture to those modules. New behavior belongs under `terrain_extraction.osm_extraction`, with tests under `tests/cm_terrain_extractor/osm_extraction/`.

## Extension Points

Add a new feature class by adding a `ProcessKind` mapping, compiling config entries in `config_schema.py`, emitting `FeatureRecord` objects, implementing a typed processor that returns `PlacementRecord` objects, and wiring it through `ExtractionPipeline` or the `OSMProcessor` typed orchestration.

Add debug output by extending `debug_export.py` with a layer builder that consumes typed records. Debug export must remain outside hot-path processing.

Add quality metrics by extending `ExtractionStats` helpers or benchmark metrics. Metrics should be deterministic for fixture inputs and must not require live OSM downloads.
