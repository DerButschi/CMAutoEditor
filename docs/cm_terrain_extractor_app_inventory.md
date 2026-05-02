# CM Terrain Extractor App Inventory

This inventory covers the files under `cm_terrain_extractor_app/` as reached from `cm_terrain_extractor_app/cm_terrain_extractor_app.py`. Generated `__pycache__` entries are listed as generated artifacts, not source of behavior.

## Entrypoint and UI

| File | Responsibility |
| --- | --- |
| `cm_terrain_extractor_app/cm_terrain_extractor_app.py` | Packaged-app bootstrapper. Changes the working directory to the app folder, prepends `dll/` to `PATH`, loads Streamlit config, and runs the Streamlit script at `cmterrainextractor.py` on port 8501. |
| `cm_terrain_extractor_app/cmterrainextractor.py` | Main Streamlit app. Owns UI state, map drawing, bounding-box selection, data-source selection, elevation extraction, OSM download/import, OSM processing, Folium visualization, CSV downloads, and data-source option toggles. |
| `cm_terrain_extractor_app/__init__.py` | Empty package marker. |

## Core Terrain Extraction Package

| File | Responsibility |
| --- | --- |
| `terrain_extraction/bbox_utils.py` | Defines `BoundingBox`, the central geometry contract. Converts drawn/user polygons to WGS84 and a local UTM CRS, computes the minimum rotated rectangle, preserves/cycles Combat Mission origin orientation, returns dimensions, bounds, coordinates, dataframes, buffers, reference points, and rotation angle. |
| `terrain_extraction/data_source_utils.py` | Shared elevation-data framework. Defines the abstract `DataSource` contract plus `GeoTiffDataSource`, `XYZDataSource`, and `ASCDataSource` base behavior. Provides dataframe/raster conversions, clipping, reprojection, rotation/cropping to the selected map, height-map downscaling to 8 m Combat Mission cells, PNG overlay generation, GeoTIFF merging, nodata filling, and archive-integrity checks. |
| `terrain_extraction/elevation_map.py` | Older/unused elevation crop helper. Contains a standalone `cut_out_bounding_box` implementation that reads raw `x/y/z` data, rotates/crops/rescales it, and writes a cache CSV. The active app path uses `DataSource.cut_out_bounding_box` instead. |
| `terrain_extraction/grid_utils.py` | Small affine helper for rotated raster grids. Builds a `rasterio.transform.Affine` from a `BoundingBox` and pixel size. |
| `terrain_extraction/osm_processor.py` | Main OSM-to-Combat-Mission conversion orchestrator. Loads an OSM config, initializes 8 m and sub-square grids, matches OSM features to config entries, schedules processor stages by priority, calls functions in `osm_utils.processing`, post-processes duplicate/conflicting tiles, exports output rows, and reconstructs geometries for map preview. |
| `terrain_extraction/projection_utils.py` | CRS and reprojection helpers. Chooses a UTM EPSG from a WGS84 bbox, transforms Shapely polygons/points/linestrings, and reprojects raster arrays with rasterio. |
| `terrain_extraction/utils.py` | Legacy/incomplete duplicate `BoundingBox` shell. Not used by the active app path. |
| `terrain_extraction/visualization_utils.py` | Converts Shapely geometries into Folium polygons/lines using optional style dictionaries and tooltips. Used to render processed OSM geometries on the map. |
| `terrain_extraction/__init__.py` | Empty package marker. |
| `terrain_extraction/hessen.qmd` | Quarto/source note related to Hessen index work. Not imported by the app. |

## Data Source Implementations

Each active data source exposes metadata (`name`, `country`, `model_type`, `resolution`, `data_type`), an outline/index GeoJSON, and methods to identify missing cached tiles, download them, collect files overlapping the selected `BoundingBox`, merge/reproject them, crop to Combat Mission size, and cache the result.

| File | Responsibility |
| --- | --- |
| `terrain_extraction/data_sources/__init__.py` | Package marker for data-source modules. |
| `terrain_extraction/data_sources/aw3d30/data_source.py` | `AW3D30DataSource`, global ALOS World 3D DSM GeoTIFF source at about 30 m resolution. Uses FTP-style progress handling, intersects against `aw3d30.geojson`, downloads missing archives/files, and extracts merged data for the bbox. |
| `terrain_extraction/data_sources/aw3d30/aw3d30.geojson` | Spatial index/coverage polygons for AW3D30 tiles. |
| `terrain_extraction/data_sources/aw3d30/aw3d30_file.json` | AW3D30 file metadata/index used by the source. |
| `terrain_extraction/data_sources/aw3d30/process_aw3d30_index.py` | Offline index-generation script for AW3D30 metadata. Not imported by the app. |
| `terrain_extraction/data_sources/bavaria_dgm1/data_source.py` | `BavariaDataSource`, Bavaria DGM1 DTM GeoTIFF source at 1 m resolution. Uses `bavaria_dgm1.geojson` to find overlapping download URLs. |
| `terrain_extraction/data_sources/bavaria_dgm1/bavaria_dgm1.geojson` | Bavaria tile coverage/download index. |
| `terrain_extraction/data_sources/bavaria_dgm1/process_bavaria_dgm1_index.py` | Offline Bavaria index-generation script. |
| `terrain_extraction/data_sources/hessen_dgm1/data_source.py` | `HessenDataSource`, Hessen DGM1 DTM GeoTIFF source at 1 m resolution. Uses a Hessen tile index and overrides intersection behavior. |
| `terrain_extraction/data_sources/hessen_dgm1/__init__.py` | Package marker. |
| `terrain_extraction/data_sources/hessen_dgm1/Hessen DGM1 - DTM 1m.geojson` | Original/raw Hessen DGM1 index export. |
| `terrain_extraction/data_sources/hessen_dgm1/hessen_dgm1.geojson` | Processed Hessen DGM1 tile coverage/download index used by the data source. |
| `terrain_extraction/data_sources/hessen_dgm1/process_hessen_index.py` | Offline Hessen index-processing script. |
| `terrain_extraction/data_sources/lower_saxony_dgm1/data_source.py` | `LowerSaxonyDataSource`, Lower Saxony DGM1 DTM GeoTIFF source at 1 m resolution. |
| `terrain_extraction/data_sources/lower_saxony_dgm1/__init__.py` | Package marker. |
| `terrain_extraction/data_sources/lower_saxony_dgm1/lgln-opengeodata-dgm1.geojson` | Raw/source Lower Saxony index. |
| `terrain_extraction/data_sources/lower_saxony_dgm1/lower_saxony_dgm1.geojson` | Processed Lower Saxony tile coverage/download index. |
| `terrain_extraction/data_sources/lower_saxony_dgm1/process_lower_saxony_dgm1_index.py` | Offline Lower Saxony index-processing script. |
| `terrain_extraction/data_sources/netherlands_dtm05/data_source.py` | `NetherlandsDataSource`, Netherlands AHN DTM GeoTIFF source at 0.5 m resolution. Uses a JSON tile index and maps cached download names back to index rows. |
| `terrain_extraction/data_sources/netherlands_dtm05/netherlands_dtm05.json` | Netherlands tile/download index. |
| `terrain_extraction/data_sources/nrw_dgm1/data_source.py` | `NRWDataSource`, North Rhine-Westphalia DGM1 DTM GeoTIFF source at 1 m resolution. Downloads and extracts overlapping files from the NRW index. |
| `terrain_extraction/data_sources/nrw_dgm1/nrw_dgm1.geojson` | NRW tile coverage/download index. |
| `terrain_extraction/data_sources/nrw_dgm1/process_nrw_gdm1_index.py` | Offline NRW index-processing script. |
| `terrain_extraction/data_sources/poland_dtm1/process_poland_dtm1_index.py` | Offline Poland DTM1 index-processing script. No active `data_source.py` exists, so it is not selectable in the app. |
| `terrain_extraction/data_sources/rge_alti/data_source.py` | `FranceDataSource`, France RGE Alti DTM source at 5 m resolution. Uses ASC data and the `rge_alti.geojson` index. |
| `terrain_extraction/data_sources/rge_alti/__init__.py` | Package marker. |
| `terrain_extraction/data_sources/rge_alti/rge_alti.geojson` | RGE Alti coverage/download index. |
| `terrain_extraction/data_sources/rge_alti/process_rge_alti_index.py` | Offline France RGE Alti index-processing script. |
| `terrain_extraction/data_sources/thuringia_dgm1/data_source.py` | `ThuringiaDataSource`, Thuringia DGM1 DTM XYZ source at 1 m resolution. Downloads/extracts XYZ files and reprojects them through `XYZDataSource`. |
| `terrain_extraction/data_sources/thuringia_dgm1/raw_data_index.geojson` | Raw Thuringia index. |
| `terrain_extraction/data_sources/thuringia_dgm1/raw_data_index.qmd` | Quarto/source note for raw Thuringia index processing. |
| `terrain_extraction/data_sources/thuringia_dgm1/thuringia_dgm1.geojson` | Processed Thuringia tile coverage/download index. |
| `terrain_extraction/data_sources/thuringia_dgm1/process_thuringia_dgm1.py` | Offline Thuringia index-processing script. |

## OSM Utility Modules

| File | Responsibility |
| --- | --- |
| `terrain_extraction/osm_utils/__init__.py` | Package marker. |
| `terrain_extraction/osm_utils/io.py` | GeoJSON file/object loading plus `BoundingBox` derivation from uploaded OSM data. |
| `terrain_extraction/osm_utils/grid.py` | Builds the Combat Mission grid family for the selected map: 8 m square grid, diagonal sub-square grid, axis-aligned sub-square grid, rotation, and reference rectangle alignment. |
| `terrain_extraction/osm_utils/processing.py` | Processor library called dynamically by `OSMProcessor`. Handles tag-to-type assignment, random area/square/cluster placement, linear network graph creation, road/rail/stream/fence tile assignment, building outline collection and decomposition, rectangulation/matching, occupancy conflict handling, and output-row appending. |
| `terrain_extraction/osm_utils/path_search.py` | Specialized path-search implementation for snapping linear OSM features to valid Combat Mission network tiles. Provides A*/custom graph search, node selection, tile validity checks, and CM type matching. |
| `terrain_extraction/osm_utils/geometry.py` | Geometry decomposition helpers for OSM building/area processing: concave vertex detection, chord discovery, polygon splitting, appendage removal, and rectangulation. |

## Raw Data Indices and Reference Files

These files are not imported directly by the Streamlit app, but they are source/reference material for processed data-source indices.

| File | Responsibility |
| --- | --- |
| `terrain_extraction/raw_data_indices/bavaria_meta.xml` | Bavaria source metadata. |
| `terrain_extraction/raw_data_indices/departements.geojson` | France department boundaries/reference data. |
| `terrain_extraction/raw_data_indices/hessen.geojson` | Raw Hessen coverage/index data. |
| `terrain_extraction/raw_data_indices/poland_2018.geojson` | Poland DTM raw index for 2018. |
| `terrain_extraction/raw_data_indices/poland_2018.qmd` | Notes/source processing for Poland 2018 index. |
| `terrain_extraction/raw_data_indices/poland_2019.geojson` | Poland DTM raw index for 2019. |
| `terrain_extraction/raw_data_indices/poland_2019.qmd` | Notes/source processing for Poland 2019 index. |
| `terrain_extraction/raw_data_indices/poland_2020.geojson` | Poland DTM raw index for 2020. |
| `terrain_extraction/raw_data_indices/poland_2020.qmd` | Notes/source processing for Poland 2020 index. |
| `terrain_extraction/raw_data_indices/poland_2021.geojson` | Poland DTM raw index for 2021. |
| `terrain_extraction/raw_data_indices/poland_2021.qmd` | Notes/source processing for Poland 2021 index. |
| `terrain_extraction/raw_data_indices/poland_2022.geojson` | Poland DTM raw index for 2022. |
| `terrain_extraction/raw_data_indices/poland_2022.qmd` | Notes/source processing for Poland 2022 index. |
| `terrain_extraction/raw_data_indices/poland_2023.geojson` | Poland DTM raw index for 2023. |
| `terrain_extraction/raw_data_indices/poland_2023.qmd` | Notes/source processing for Poland 2023 index. |
| `terrain_extraction/raw_data_indices/rgealti.html` | RGE Alti raw/source index page snapshot or reference. |

## Generated Artifacts

`__pycache__/` files appear under the app root, `terrain_extraction/`, `data_sources/`, individual data-source folders, and `osm_utils/`. They are Python bytecode caches from previous runs for Python 3.10 and 3.12. They should not be edited and do not define source behavior.
