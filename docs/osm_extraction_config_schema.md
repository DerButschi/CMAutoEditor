# OSM Extraction Config Schema

OSM extraction config is loaded from JSON and compiled by `ExtractionConfig` in `terrain_extraction.osm_extraction.config_schema`. New hot-path code should use compiled entries rather than raw nested dict access.

## Entry Shape

Each active OSM config entry is keyed by name and must include:

| Field | Type | Purpose |
| --- | --- | --- |
| `tags` | list of `[key, value]` pairs | Candidate selector. Any listed pair may match. |
| `process` | list of legacy process strings | Compiled to `ProcessKind`. Unknown process values fail validation. |
| `priority` | integer | Placement rank; smaller positive values are stronger. |
| `cm_types` | list of objects | CM AutoEditor type candidates. Each real candidate needs `menu` and `cat1`. |

Optional fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `active` | boolean | Defaults to true. Inactive entries are ignored. |
| `required_tags` | list of `[key, value]` pairs | All listed pairs must match. |
| `exclude_tags` | list of `[key, value]` pairs | Any listed pair excludes the feature. |
| `allowed_ids` | list | If present, only those feature IDs match. |
| `exclude_ids` | list | Feature IDs to skip. |
| `modifiers` | object | Processor-specific settings such as linear source names or stride values. |

## ProcessKind Mapping

`ProcessKind` is the canonical internal process enum. Compatibility strings remain accepted in config:

| Legacy process | ProcessKind | Active owner |
| --- | --- | --- |
| `type_from_tag` | `AREA` | `AreaRasterizer` |
| `type_random_area` | `AREA` | `AreaRasterizer` weighted choices |
| `type_random_individual` | `RANDOM` | `AreaRasterizer` per-cell choices |
| `type_random_clusters` | `RANDOM` | `AreaRasterizer` seeded spatial clusters |
| `single_object_random` | `POINT` | `AreaRasterizer` point placement |
| `default_ground`, `default_foliage` | `DEFAULT` | `AreaRasterizer` default emission |
| `road_tiles` | `ROAD` | topology, routing, tile assignment |
| `rail_tiles` | `RAIL` | topology, routing, tile assignment |
| `stream_tiles` | `STREAM` | topology, routing, tile assignment |
| `fence_tiles` | `FENCE` | topology, routing, tile assignment |
| `type_from_linear` | `LINEAR` | typed linear-dependent placement pass |
| `type_from_residential_building_outline` | `BUILDING_OUTLINE` | `BuildingFitter` |
| `type_from_church_outline` | `BUILDING_OUTLINE` | `BuildingFitter` |
| `type_from_barn_outline`, `type_from_barn_outlines` | `BUILDING_OUTLINE` | `BuildingFitter` |

## CM Types

`cm_types` compile to `CMType`. Required real fields are:

| Field | Meaning |
| --- | --- |
| `menu` | CM AutoEditor menu or layer label. |
| `cat1` | Primary category. |
| `cat2` | Optional secondary category. |
| `direction` | Optional direction label or catalog direction. |
| `id` | Optional tile or object ID. |
| `tags` | Optional selector used to choose a CM type from matched source tags. |
| `weight` | Optional weighted random choice weight. Defaults to `1.0`. |
| `dummy` | Optional true value meaning "choose nothing" in weighted choices. |

Weighted choices are made with the pipeline RNG. Fixture and benchmark runs should set an explicit seed when deterministic replay is required.

## Modifiers

Known modifier patterns:

| Modifier | Used by | Meaning |
| --- | --- | --- |
| `linear_name` | `type_from_linear` | Source placement config name whose cells receive the derived placement. |
| `stride_x`, `stride_y` | random area compatibility | Optional spacing filter retained from old configs. |

New modifiers should be validated in `config_schema.py` when they become part of a stable processor contract. Processor-only experimental keys should be documented here once they affect output semantics.

## Output And Debug Contracts

Processors emit `PlacementRecord` objects. `output_rows.py` validates conflicts, appends the explicit extent marker, and normalizes coordinates. Public CSV columns remain `x`, `y`, `z`, `menu`, `cat1`, `cat2`, `direction`, `id`, `name`, and `priority`.

`debug_export.py` can expose source features, topology, routes, occupancy, building footprints, and final rows. Debug layers are derived views and must not become required hot-path inputs.

`ExtractionStats` should include timing, count, quality, and diagnostic fields for new stages. Add metrics when a new config process changes routing, placement, conflict handling, or output assembly.
