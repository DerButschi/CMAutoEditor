# Streamlit Refactoring Inventory

Milestone 1 behavior snapshot for the CMTerrainExtractor Streamlit refactor.

Authoritative contract: `docs/plans/streamlit_refactoring_contract.md`.

Primary source files inspected:

- `cm_terrain_extractor_app/cmterrainextractor.py`
- `cm_terrain_extractor_app/cm_terrain_extractor_app.py`
- `cm_terrain_extractor_app.spec`
- Supporting behavior references in `terrain_extraction/bbox_utils.py`, `terrain_extraction/data_source_utils.py`, `terrain_extraction/osm_utils/io.py`, `terrain_extraction/osm_processor.py`, and `terrain_extraction/visualization_utils.py`.

## Session State Inventory

Current state is stored directly in `st.session_state` from `cmterrainextractor.py`.

| Current key | Current behavior and source | Planned `AppState` field | Planned owner | Reset or invalidation behavior |
| --- | --- | --- | --- | --- |
| `selectable_data_sources` | Set unconditionally at import from all `data_sources` at lines 30-41; edited in Options tab at lines 531-548. | `selectable_data_sources` | `app_core/state.py`, initialized through `streamlit_ui/session_adapter.py`; edited by `streamlit_ui/options` later. | Defaults to all sources. Options tab replaces it with selected sources. No current automatic reset. |
| `map_mode` | Read by sidebar and map view; line 574 initializes to `"bounding_box"`, then radio writes one of `"Bounding Box Selection"`, `"Elevations"`, `"OpenStreetMap"` at lines 172-186. | `map_mode` | `AppState`, UI adapter | Default should follow contract value `"Bounding Box Selection"`. Mode switch currently does not clear mode-specific outputs. |
| `bbox` | Lat/lon coordinate list from `BoundingBox.get_coordinates(xy=False)` at lines 66, 83, 135; used by Folium lines 420-444. | `bbox_coordinates` | `app_core/state.py`; derived by `app_core/validation.py` | Replaced when drawing, data-editor update, debug load, or origin cycling changes bbox. |
| `bbox_object` | `BoundingBox` object stored at lines 67 and 84; read by data source search, extraction, OSM download/process, table editor, map layers. | `bbox_object` | `AppState`; set by `app_core/validation.update_state_from_bbox` | Replaced when drawing or data-editor update succeeds. |
| `projected_bbox_object` | Projected Shapely polygon stored at lines 68, 85, 136. | `projected_bbox_object` | `AppState`; validation/metrics helper | Recomputed with bbox. |
| `len_x` | W-E length in meters from `BoundingBox.get_length_xaxis()` at lines 69, 86, 137; used for validation and metrics at lines 156-168, 216-226. | `len_x` | `AppState`; validation/metrics helper | Recomputed with bbox or origin cycle. |
| `len_y` | S-N length in meters from `BoundingBox.get_length_yaxis()` at lines 70, 87, 138; used for validation and metrics. | `len_y` | `AppState`; validation/metrics helper | Recomputed with bbox or origin cycle. |
| `bbox_origin` | Set to `0` at lines 71 and 88. Current origin cycling mutates the `BoundingBox` but does not update this key at lines 132-138. | `bbox_origin` | `AppState`; validation/state helper | Reset to `0` on new bbox. Unknown whether it should increment on cycle; inspect lines 132-138 before M4. |
| `selected_area_valid` | Set in sidebar from max length/area checks at lines 146-170; read to enable bbox, elevation, and OSM actions at lines 213, 239-245, 286-306. | `selected_area_valid` | `app_core/validation.py` | Recomputed from `len_x`, `len_y`, and area. False when no length exists. |
| `data_sources` | Available elevation sources after search, written at lines 95-103 and read by selectbox at line 242. | `available_data_sources` | `app_core/actions.py` result stored in `AppState` | Replaced by search. Should be cleared by bbox changes because availability depends on bbox. |
| `selected_data_source` | Selected from Streamlit selectbox and written at lines 242-245; read by elevation extraction lines 113-126. | `selected_data_source` | `AppState`; UI writes, actions read | Should clear `elevation_in_bbox` when source changes. Current code does not explicitly clear on source change. |
| `elevation_in_bbox` | Elevation CSV dataframe written at line 125; read by CSV download lines 247-253 and map overlay lines 409-427. Debug path can set OSM outputs but not this key. | `elevation_in_bbox` | `AppState`; `app_core/actions.extract_elevation_data` | Deleted only on new bbox at lines 89-90. Should also clear when data source changes. |
| `height_map_layer` | Marker flag set after first overlay draw at lines 409-415; controls map key/center refresh. | No contract field yet; likely temporary UI/map view flag or replaced by `map_key` dirty handling. | `map_view` or `streamlit_ui` | Unknown reset. New bbox deletes elevation data but not this flag. Inspect lines 409-415 and 73-90 in M7/M8. |
| `map_center` | Initialized to `[0,0]` at lines 372-376; passed to `st_folium` at line 493; restored from `center_cache` after elevation layer at lines 412-413. | `map_center` | `AppState`; map view/UI | Default currently list `[0,0]`; contract says tuple `(0.0, 0.0)`. Updated from `center_cache` when adding height map. |
| `map_zoom` | Initialized to `2` at lines 372-380; passed to `st_folium` at line 494; restored from `zoom_cache` after elevation layer at line 412. | `map_zoom` | `AppState`; map view/UI | Default `2`. Updated from `zoom_cache` when adding height map. |
| `map_key` | Initialized to `0` at lines 379-380; incremented when height map layer first appears at line 414; passed to `st_folium` at line 497. | `map_key` | `AppState`; `mark_map_dirty` | Increment when map needs remount. Current trigger is first height-map overlay only. |
| `zoom_cache` | Written from `st_folium` payload at lines 508-509; read when elevation overlay first appears at line 412. | Not in contract; temporary UI cache or folded into `map_zoom`. | `streamlit_ui` / `map_view` | Updated every map render when `zoom` is in payload. |
| `center_cache` | Written from `st_folium` payload at lines 510-511; read when elevation overlay first appears at line 413. | Not in contract; temporary UI cache or folded into `map_center`. | `streamlit_ui` / `map_view` | Updated every map render when `center` is in payload. |
| `drawn_coordinates` | Last drawing coordinate array written at lines 500-506 to avoid repeated bbox update/rerun. | Not in contract; likely map parser/UI local cache. | `map_view/drawing.py` plus UI adapter if needed | Replaced when drawing payload shape or values change. |
| `edited_df` | Data editor output written at lines 198-212 and deleted by `update_bbox_from_df` at lines 140-144; read after map render at lines 516-523. | Not in contract; transient UI form state. | `streamlit_ui` | Deleted after applying if valid. |
| `osm_config_file` | Debug default at line 60; selected config filename written at line 279; read for processing/download at lines 318-319 and 336-338. | `osm_config_file` | `AppState`; resources/config UI | Replaced on config select. Current code does not clear already loaded/processed OSM outputs when config changes. |
| `osm_config` | Loaded JSON dict at lines 61-62 and 281-282; used for OSM geometry visualization at lines 466-470. | `osm_config` | `app_core/actions.load_osm_config` and `AppState` | Replaced on config select. Should clear OSM processing outputs when config/profile changes. |
| `osm_profile_str` | Debug/profile select value at lines 64 and 280; read by `OSMProcessor` at line 319. | `osm_profile` | `AppState` | Replaced on profile select. Contract field name differs from current key. |
| `osm_data` | Uploaded GeoJSON object from `read_file_object` line 296 or downloaded GeoJSON from OSMnx line 354; required to enable processing at lines 302-306. | `osm_data` | `AppState`; `app_core/actions` | Replaced on upload or download. Should clear OSM output/geometries when changed. |
| `osm_bbox_object` | Bounding box of uploaded OSM file written at lines 291-296; rendered as dashed bbox lines at lines 446-457. | `osm_bbox_object` | `AppState`; OSM upload action/map view | Replaced on upload. Current download flow does not set this key. |
| `osm_output` | OSM CSV dataframe from debug path lines 58-59 or processing lines 329-330; read by download button lines 307-312. | `osm_output` | `AppState`; `app_core/actions.process_osm_data` | Should clear when bbox, OSM data, config, or profile changes. Current code only overwrites on process. |
| `osm_geometries` | Geometry dict from debug path line 59 or processing line 330; rendered in map lines 459-481. | `osm_geometries` | `AppState`; `map_view` consumes | Same invalidation as `osm_output`. |
| `currently_processing_data` | Tuple set before elevation extraction at lines 117-120, deleted at line 129; final warning uses and deletes it at lines 589-593 if processing was interrupted. | `currently_processing_data` | `AppState` | Set during long actions; deleted on successful completion or next run warning. |

## Top-Level Function Inventory

All current functions live in `cmterrainextractor.py`.

| Function | Lines | Current responsibility | Target ownership |
| --- | --- | --- | --- |
| `update_bounding_box(points)` | 73-92 | Convert drawn/editor points to Shapely polygon, build `BoundingBox`, update bbox-derived session keys, delete elevation result, rerun. | Split between `map_view/drawing.py` for payload parsing, `app_core/validation.update_state_from_bbox`, and Streamlit UI rerun handling. |
| `find_data_sources_in_bbox(status_update_area)` | 95-104 | UI spinner plus data-source intersection search. | Pure search in `app_core/actions.find_data_sources_in_bbox`; status/spinner stays in `streamlit_ui`. |
| `get_data_source_label(data_source)` | 106-107 | Format selectbox label from data-source metadata. | `streamlit_ui` display helper. |
| `dataframe2csv(df)` | 109-111 | Cached CSV export bytes using `df.to_csv().encode("utf-8")`. | `app_core/exports.dataframe_to_csv_bytes`; optional Streamlit caching at UI edge only. |
| `extract_data_in_bbox(status_update_area)` | 113-130 | UI status, data cache directory creation, data-source `get_data`, PNG generation, state writes. | `app_core/actions.extract_elevation_data` for backend work; UI handles status and stores result. |
| `permute_bbox()` | 132-138 | Cycle `BoundingBox` origin and recompute bbox coordinates/metrics. | `app_core/validation` or `app_core/state` helper called by UI. |
| `update_bbox_from_df()` | 140-144 | Apply non-null data-editor rows to bbox update. | `streamlit_ui` table handler plus `app_core/validation.update_state_from_bbox`. |
| `draw_sidebar(status_update_area)` | 146-313 | Validation calculation, mode selector, bbox editor, metrics, elevation controls/download, OSM profile/config/upload/process/download. | Split across `streamlit_ui/sidebar.py`, `streamlit_ui/elevation.py`, `streamlit_ui/osm.py`; validation/export/actions delegated to `app_core`. |
| `process_osm_data(status_update_area)` | 316-334 | Create `OSMProcessor`, run preprocess/process/postprocess with Streamlit status, write output/geometries. | `app_core/actions.process_osm_data`; status stays UI. |
| `get_osm_data(status_update_area)` | 336-354 | Build OSMnx tag dict from config, download OSM features, drop heavy columns, convert to GeoJSON. | `app_core/actions.download_osm_data`; UI calls and stores result. |
| `map_view_tab()` | 356-525 | Render map header, map defaults, Folium map/plugins/layers, `st_folium`, drawing payload handling, map cache, edited bbox handling. | Folium construction and drawing parser in `map_view/`; `st_folium` call and rerun/session handling in `streamlit_ui`. |
| `options_tab()` | 527-565 | Select included data sources and display cache size/clear button. | `streamlit_ui/options.py`; data source list remains app resources/state. Cache operations likely `app_core/resources.py` or action helper. |

## Widget and UI Inventory

Top-level page wiring:

- `st.set_page_config` at lines 568-571 sets title, wide layout, and bug-report menu.
- `st.empty` at line 577 creates a shared status area passed into sidebar actions.
- `st.tabs(["Map View", "Options"])` at lines 581-587 owns the two main screens.
- Interruption warning at lines 589-593 appears if `currently_processing_data` survived a rerun.

Sidebar widgets and state:

- Map mode `st.radio` at lines 172-186 writes `map_mode`.
- Bbox `st.data_editor` at lines 190-212 writes transient `edited_df`.
- `Cycle bounding box origin` button at line 213 calls `permute_bbox`; disabled unless selected area is valid.
- Metric columns at lines 214-235 display W-E length, S-N length, and selected area; deltas turn inverse when limits are exceeded.
- `Find available data sources` button at line 241 calls `find_data_sources_in_bbox`.
- `Data sources` selectbox at line 242 reads `data_sources` and writes `selected_data_source` at line 244.
- `Extract elevation data` button at line 245 calls `extract_data_in_bbox`.
- Elevation download button at lines 247-253 writes `elevation_data.csv` using `dataframe2csv`.
- OSM profile selectbox at lines 262-266 writes `osm_profile_str`.
- OSM config selectbox at lines 267-282 writes `osm_config_file` and `osm_config`.
- `Download OpenStreeMap data` button at line 288 calls `get_osm_data`.
- OSM GeoJSON file uploader at lines 290-296 writes `osm_bbox_object` and `osm_data`.
- `Process OpenStreeMap data` button at lines 297-306 calls `process_osm_data`.
- OSM download button at lines 307-312 writes `osm_data.csv` using `dataframe2csv`.

Map View widgets and state:

- Header/subheader text changes by `map_mode` at lines 356-369.
- `st_folium` at lines 491-498 passes `center`, `zoom`, `feature_group_to_add`, width `1200`, and `key`.
- Draw payload handling at lines 500-506 reads `last_active_drawing.geometry.coordinates`, stores `drawn_coordinates`, and calls `update_bounding_box(coordinates[0])`.
- Map zoom and center payloads are cached in `zoom_cache` and `center_cache` at lines 508-511.
- Edited bbox dataframe is reconciled after map render at lines 516-523.

Options tab:

- Data-source inclusion editor at lines 531-548 mutates `selectable_data_sources`.
- Cache size is computed by walking `data_cache_path` at lines 550-563.
- `Clear Cache (...)` button at line 564 has no current `on_click`; behavior is unknown/no-op.

## Folium and Map Behavior Inventory

Current map behavior lives in `cmterrainextractor.py` lines 356-525.

- Base map: `folium.Map` with OpenTopoMap tiles and attribution at lines 387-392.
- Feature group: one `folium.FeatureGroup("bbox")` at line 394 is passed to `st_folium` as `feature_group_to_add`, not added directly to the map before that call.
- Draw controls: `folium.plugins.Draw` is only added in `"Bounding Box Selection"` mode at lines 396-403. Polyline, circle, marker, and circlemarker are disabled. Polygon/rectangle remain enabled by default.
- Utility plugins: Geocoder, MeasureControl, and Fullscreen are added at lines 405-407.
- Elevation overlay: if `elevation_in_bbox` exists, add `folium.raster_layers.ImageOverlay` named `"Elevation data"` from `data_cache_path/current_height_map.png` with opacity `0.9` and bounds from `bbox` at lines 409-427. On first overlay render, cached zoom/center are copied back and `map_key` increments at lines 410-415.
- CM bbox layer: if `bbox` exists, add four red `PolyLine`s, W-E and S-N `PolyLineTextPath` labels, and a red `CircleMarker` at the origin at lines 430-444.
- OSM bbox layer: if `osm_bbox_object` exists, add four dashed red `PolyLine`s and an `"OSM data"` label at lines 446-457.
- OSM geometry layer: if `osm_geometries` exists, convert each geometry with `shapely2folium`, using optional visualization and priority from `osm_config`, then add higher numeric priorities first at lines 459-481.
- Layer control is currently commented out at line 489.
- Drawn geometry payload is interpreted as GeoJSON coordinate order from `st_folium`: `np.array(last_active_drawing["geometry"]["coordinates"])`; first ring `coordinates[0]` is passed into `update_bounding_box` at lines 500-506.

Tests to create later:

- M7 drawing parser test for `last_active_drawing.geometry.coordinates` preserving GeoJSON coordinate order and selecting the first ring.
- M7 map builder tests/smoke checks for draw control only in bbox mode, elevation overlay bounds, CM bbox lines, OSM bbox lines, OSM geometry priority ordering, and default map center/zoom/key behavior.

## Runtime Path and PyInstaller Inventory

Streamlit app script path handling:

- In `cmterrainextractor.py`, packaged mode is detected with `getattr(sys, "frozen", False)` and `hasattr(sys, "_MEIPASS")` at lines 45-50.
- Packaged mode sets `executable_path = dirname(sys.executable)` and `data_cache_path = dirname(sys.executable)/data_cache`.
- Source mode sets `executable_path = "."` and `data_cache_path = "data_cache"`, making JSON config lookup and cache path relative to current working directory.

Launcher behavior:

- `cm_terrain_extractor_app.py` imports `streamlit.web.bootstrap` and `streamlit.runtime.scriptrunner.magic_funcs` at lines 1-4.
- Launcher changes cwd to `dirname(__file__)` at line 7, then computes `current_location = os.getcwd()` at line 8.
- It prepends `current_location/dll` to `PATH` and preserves existing PATH entries at lines 12-20.
- Streamlit flag options set server port `8501` and disable development mode at lines 22-28.
- It boots `./cm_terrain_extractor_app/cmterrainextractor.py` with `bootstrap.run(...)` at lines 29-35.

PyInstaller spec behavior:

- Entry script is `cm_terrain_extractor_app/cm_terrain_extractor_app.py` at spec line 5.
- Bundles `gdal.dll` from the approved Conda env into `dll` at line 7.
- Bundles Streamlit static files and `streamlit_folium` frontend build at lines 8-16.
- Bundles the whole local `cm_terrain_extractor_app` directory into `./cm_terrain_extractor_app` at lines 19-22.
- Bundles local `profiles` into `./profiles` at lines 24-27.
- Hidden imports include Streamlit, streamlit-folium, shapely, pyproj, geopandas, rasterio modules, py7zr, skimage modules, terrain extraction modules, fiona/rasterio shims, osmnx, geojson, profiles, and matplotlib SVG backend at lines 30-36.
- Hook path is `./hooks` at line 38.
- EXE name is `cm_terrain_extractor_app`, console enabled, UPX enabled at lines 49-63.

Tests to create later:

- M3 resource tests for source-mode paths, fake packaged-mode paths, idempotent DLL PATH preparation, and default JSON config discovery.
- M9 packaging checks for hidden imports/data paths after module movement and a manual packaged run.

## Data Flow Snapshots

### Bbox Drawing and Editing

1. User draws rectangle/polygon in Folium Draw.
2. `st_folium` returns `last_active_drawing` at lines 491-506.
3. App extracts `geometry.coordinates` into a NumPy array and compares it with `drawn_coordinates` to prevent repeated reruns.
4. First ring `coordinates[0]` is passed to `update_bounding_box`.
5. `update_bounding_box` builds `shapely.Polygon(points)` and ignores degenerate line-shaped minimum rotated rectangles at lines 73-76.
6. `BoundingBox` computes WGS84/projected boxes and dimensions; state keys `bbox`, `bbox_object`, `projected_bbox_object`, `len_x`, `len_y`, and `bbox_origin` are written at lines 83-88.
7. Existing `elevation_in_bbox` is deleted at lines 89-90; app reruns at line 92.
8. Data-editor path writes `edited_df` in the sidebar and reconciles it after map render at lines 516-523. If all cells are non-null, `update_bbox_from_df` passes zipped `(x, y)` pairs into the same bbox update flow at lines 140-144.

### Elevation Extraction

1. Sidebar validation computes `selected_area_valid` from `len_x`, `len_y`, max W-E `4160`, max S-N `4160`, and max area `18000000` at lines 146-170.
2. User clicks `Find available data sources`; the app checks each `selectable_data_sources` entry with `intersects_bounding_box(bbox_object)` and stores matches in `data_sources` at lines 95-103.
3. User selects a data source; line 244 stores it in `selected_data_source`.
4. User clicks `Extract elevation data`; app creates `data_cache_path`, sets `currently_processing_data`, calls `data_source.get_data(bounding_box, data_cache_path)`, stores `elevation_in_bbox`, calls `data_source.get_png(...)`, and clears the processing marker at lines 113-130.
5. Elevation CSV download uses `df.to_csv().encode("utf-8")` through `dataframe2csv` lines 109-111 and 247-253.
6. Map overlay reads `data_cache_path/current_height_map.png` and `bbox` bounds at lines 417-427.

### OSM Download, Upload, and Processing

Download path:

1. OSM profile/config widgets write `osm_profile_str`, `osm_config_file`, and `osm_config` at lines 262-282.
2. User clicks `Download OpenStreeMap data`; app reads selected config from `executable_path` at lines 336-338.
3. Tags are collected from each config section's `tags`, `exclude_tags`, and `required_tags` entries into one `tag_dict` at lines 339-347.
4. App calls `osmnx.features_from_polygon(bbox_object.box_wgs84, tag_dict)` at line 348.
5. `ways` column is dropped if present; `nodes` drop is attempted but not assigned at lines 349-352.
6. GeoDataFrame is converted to GeoJSON and stored as `osm_data` at line 354.

Upload path:

1. User uploads a `.geojson` file at lines 290-296.
2. `get_bounding_box_from_file_object` derives `osm_bbox_object` from the uploaded object.
3. `read_file_object` loads uploaded GeoJSON into `osm_data`.
4. `terrain_extraction/osm_utils/io.py` currently imports `streamlit.cache_data` at lines 7-18, which conflicts with the final app-core boundary if used from `app_core`.

Processing path:

1. Processing is enabled only if selected area is valid and `osm_data` exists at lines 297-306.
2. `OSMProcessor` is created from `executable_path/osm_config_file`, `bbox_object`, and `osm_profile_str` at lines 316-319.
3. UI status writes "Preprocessing data", "Running processors", and "Doing postprocessing" while calling `preprocess_osm_data`, `run_processors`, and `post_process` at lines 321-328.
4. `osm_output` and `osm_geometries` are stored from processor getters at lines 329-330.
5. CSV download uses `dataframe2csv` and filename `osm_data.csv` at lines 307-312.

### CSV Export

- The current export behavior is exactly `df.to_csv().encode("utf-8")` in `dataframe2csv` at lines 109-111.
- Elevation filename is fixed as `elevation_data.csv` at line 251.
- OSM filename is fixed as `osm_data.csv` at line 311.
- M5 tests should lock byte compatibility before replacing the Streamlit-cached helper.

## Known Unknowns and Follow-Up Windows

| Unknown | Why it matters | Exact window to inspect later |
| --- | --- | --- |
| `bbox_origin` intended value after `permute_bbox` | Contract includes `bbox_origin`, but current origin cycling does not update it. | `cm_terrain_extractor_app/cmterrainextractor.py` lines 132-138. |
| `height_map_layer` reset semantics | It controls map key/center restoration but is not cleared on bbox changes. | `cm_terrain_extractor_app/cmterrainextractor.py` lines 409-415 and 73-90. |
| Cache clear button behavior | Button currently has no callback, so "Clear Cache" appears to be a no-op. | `cm_terrain_extractor_app/cmterrainextractor.py` lines 550-565. |
| OSM config/profile invalidation | Current code reloads config/profile but does not clear existing OSM output/geometries. | `cm_terrain_extractor_app/cmterrainextractor.py` lines 262-282 and 297-330. |
| Selected data source invalidation | Current code stores the selectbox value but does not clear old elevation output if the source changes without bbox changing. | `cm_terrain_extractor_app/cmterrainextractor.py` lines 237-253. |
| Source-mode `executable_path = "."` assumptions | Config lookup depends on cwd, while launcher changes cwd only in packaged/launcher path. | `cm_terrain_extractor_app/cmterrainextractor.py` lines 45-50 and `cm_terrain_extractor_app/cm_terrain_extractor_app.py` lines 6-35. |
| `terrain_extraction/osm_utils/io.py` Streamlit cache import | App-core actions must not import Streamlit, but OSM IO helpers currently do. | `cm_terrain_extractor_app/terrain_extraction/osm_utils/io.py` lines 7-18. |

## Later Test Inventory

- M2 `test_exports.py`: assert CSV bytes equal `df.to_csv().encode("utf-8")`.
- M2/M4 `test_state.py`: defaults, bbox-dependent invalidation, source/config/profile invalidation, `mark_map_dirty`, and preservation of config/profile/map fields.
- M2/M7 `test_drawing.py`: extract `last_active_drawing`, preserve coordinate order, return first coordinate ring.
- M3 `test_resources.py`: source/package resource resolution, JSON config discovery, DLL PATH idempotency.
- M5 `test_validation.py`: selected area limits `4160`, `4160`, `18000000`, derived bbox metrics, and bbox update invalidation.
- M6 action tests: fake data source search/extraction, fake OSM processor workflow, fake uploaded/downloaded OSM data; no network dependency.
- M7 map tests: layer construction, draw-control mode gating, overlay bounds, OSM priority ordering.
- M8 UI smoke/manual checks: app starts, mode switching, bbox editing, elevation workflow, OSM upload/process controls.
- M9 packaging checks: spec hidden imports/data files and manual PyInstaller executable run.
