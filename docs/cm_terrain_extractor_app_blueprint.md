# CM Terrain Extractor App Blueprint

This is a concise reconstruction blueprint for the active code path launched by `cm_terrain_extractor_app/cm_terrain_extractor_app.py`.

## Product Shape

Build a Streamlit desktop/web app that lets a user:

1. Draw or manually enter a rotated rectangular Combat Mission map footprint.
2. Validate the footprint against practical map limits.
3. Extract terrain elevation data from one of several public DEM/DSM sources.
4. Download/import OSM GeoJSON for the same footprint.
5. Convert OSM features into Combat Mission tile rows.
6. Preview the selected bbox, extracted elevation overlay, and processed OSM geometries on a Folium map.
7. Download elevation and OSM outputs as CSV.

## Main Concepts

### Bounding Box

Represent the selected map as a `BoundingBox` object built from a user polygon in WGS84.

The object must:

- Choose an appropriate local projected CRS, currently a UTM EPSG based on bbox centroid/longitude.
- Project the polygon to that CRS.
- Compute the minimum rotated rectangle in projected meters.
- Preserve the user's first selected point as the Combat Mission origin as closely as possible.
- Expose the rectangle in both WGS84 and projected CRS.
- Provide x/y lengths in meters, area, bounds, coordinate lists, a dataframe of corner points, the origin point, two axis reference points, a 100 m buffer, and rotation angle.
- Allow cycling the origin around the rectangle corners.

All downstream elevation and OSM work assumes this bbox is the authoritative map contract.

### Elevation Data Sources

Use one common data-source interface:

```text
get_data(bounding_box, cache_dir) -> pandas.DataFrame[x, y, z]
intersects_bounding_box(bounding_box) -> bool
get_png(bounding_box, cache_dir) -> path_to_overlay_png
```

Each concrete source owns:

- Display metadata: name, country/region, terrain model type, resolution, source format.
- A local spatial index of downloadable tiles.
- The native CRS of the source.
- A cache subdirectory and download/extraction rules.

The standard extraction pipeline is:

1. Check the source index for tiles intersecting the selected bbox.
2. Download missing source files into `data_cache/<source>/`.
3. Extract/decompress archives when needed.
4. Collect cached source rasters/XYZ files that overlap the bbox.
5. Merge overlapping files.
6. Reproject to the bbox projected CRS at 1 m resolution.
7. Clip to the bbox bounds plus buffer.
8. Rotate/crop to the Combat Mission map axes.
9. Downscale from 1 m samples to 8 m Combat Mission cells.
10. Return a dataframe with local `x`, `y`, `z` values.
11. Produce a transparent PNG overlay in WGS84 for Folium preview.

GeoTIFF, XYZ, and ASC sources share most behavior but differ in how files are read and merged.

### OSM Processing

Use an `OSMProcessor` object initialized with:

```text
profile: Combat Mission title key
bbox: BoundingBox
path_to_config: JSON config path
```

The config is the domain model. Each config entry defines:

- OSM tags to match.
- Optional excluded/required tags and allowed/excluded ids.
- Whether the entry is active.
- Priority.
- Processing methods, such as `type_from_tag`, `road_tiles`, `stream_tiles`, or building-outline processors.
- Combat Mission output type choices (`menu`, `cat1`, optional `cat2`, direction, etc.).
- Optional visualization style.

The processor should:

1. Project OSM geometries from WGS84 into the bbox CRS.
2. Match each OSM feature against active config entries.
3. Add synthetic full-map entries such as default ground/foliage if configured.
4. Create a rotated 8 m Combat Mission square grid plus sub-square/diagonal grids.
5. Convert matched features into grid-cell output rows using config-selected processor functions.
6. Resolve duplicate/conflicting rows by priority and CM type rank.
7. Return a CSV-shaped dataframe with `x`, `y`, `z`, `menu`, `cat1`, `cat2`, `direction`, `id`, `name`, and `priority`.
8. Reconstruct preview geometries for non-default processed entries.

Network processors build graph representations of OSM linear features, snap paths to valid CM tile grids, and assign road/rail/stream/fence tile types. Linear tile assignment preserves mandatory side-connection compatibility between neighboring tiles. State components are solved by graph shape: path DP for paths, cycle DP for simple cycles, tree DP for tree components, and a bounded configurable cutset solver for small loopy components. Components above the configured `tile_assignment_solver` limits report structured diagnostics and are contained by the pipeline's `warn`/`strict` validation mode instead of running unbounded exponential search. Building processors collect outlines, decompose or match them to known CM building footprints from `profiles`, and write tile rows.

## UI Blueprint

The app has two tabs:

- `Map View`: primary map and workflow.
- `Options`: data-source inclusion toggles and cache status.

The sidebar switches among three modes:

- `Bounding Box Selection`: edit/draw the map footprint, cycle the CM origin, and display dimensions.
- `Elevations`: search included sources intersecting the bbox, choose one, extract data, and download CSV.
- `OpenStreetMap`: choose CM title/profile and config, download OSM data or upload GeoJSON, process it, and download CSV.

Streamlit session state is the app state store. Important keys:

- `map_mode`: active sidebar mode.
- `selectable_data_sources`: sources enabled in Options.
- `bbox`, `bbox_object`, `projected_bbox_object`: current map footprint.
- `len_x`, `len_y`, `bbox_origin`, `selected_area_valid`: bbox validation/display state.
- `data_sources`, `selected_data_source`, `elevation_in_bbox`: elevation workflow state.
- `osm_config_file`, `osm_config`, `osm_profile_str`, `osm_data`, `osm_bbox_object`, `osm_output`, `osm_geometries`: OSM workflow state.
- `map_center`, `map_zoom`, `map_key`, `center_cache`, `zoom_cache`: Folium/Streamlit map persistence.

The map should use OpenTopoMap tiles, drawing tools only in bbox mode, geocoder/measure/fullscreen controls, red bbox edge labels for CM axes, optional dashed OSM imported-data bbox, optional elevation image overlay, and optional processed OSM preview geometries sorted by priority.

## Quality Notes

- Treat CRS conversion and origin orientation as core correctness concerns.
- Keep elevation extraction cache-aware because source files are large.
- Keep OSM processing config-driven; adding a new terrain feature should usually mean adding JSON config and a processor function, not editing app UI.
- Keep processor output dataframe-compatible with the CSV expected by the Combat Mission auto-editor workflow.
- Avoid CPU-heavy work on each Streamlit rerun; expensive actions should live behind buttons and session-state caching.
