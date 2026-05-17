# CM Terrain Extractor App Code Flow

This trace follows the active path from `cm_terrain_extractor_app/cm_terrain_extractor_app.py`.

## 1. Process Launch

1. `cm_terrain_extractor_app.py` runs as `__main__`.
2. It changes the process working directory to the directory containing the bootstrap file.
3. It prepends `<app_dir>/dll` to `PATH` and preserves the existing `PATH`.
4. It builds Streamlit flags:
   - `server.port = 8501`
   - `global.developmentMode = False`
5. It calls:

```text
streamlit.web.bootstrap.load_config_options(flag_options)
streamlit.web.bootstrap.run("./cm_terrain_extractor_app/cmterrainextractor.py", False, [], flag_options)
```

The Streamlit script is then evaluated and rerun by Streamlit whenever widgets or callbacks change state.

## 2. Streamlit Script Initialization

When `cmterrainextractor.py` is evaluated:

1. Imports UI libraries (`streamlit`, `streamlit_folium`, `folium`), data libraries (`numpy`, `pandas`, `shapely`, `geojson`, `osmnx`), and app modules.
2. Instantiates the selectable elevation sources:
   - `HessenDataSource`
   - `NRWDataSource`
   - `FranceDataSource`
   - `NetherlandsDataSource`
   - `BavariaDataSource`
   - `ThuringiaDataSource`
   - `AW3D30DataSource`
   - `LowerSaxonyDataSource`
3. Writes those into `st.session_state["selectable_data_sources"]`.
4. Determines runtime paths:
   - frozen executable: cache beside `sys.executable`
   - normal Python run: `data_cache` and executable path `.`
5. Defines callbacks and tab render functions.

If the script is run as `__main__`, it configures the page, initializes `map_mode` when absent, creates `status_update_area`, renders the sidebar, and renders `Map View` and `Options` tabs.

## 3. Sidebar Flow

`draw_sidebar(status_update_area)` is called on every rerun.

Inputs:

- `status_update_area`: a Streamlit placeholder used for progress/status blocks.
- `st.session_state["bbox_object"]`, if present.

Steps:

1. If a bbox exists, compute `len_x`, `len_y`, area, and deltas from limits:
   - max x length: 4160 m
   - max y length: 4160 m
   - max area: 18,000,000 square meters
2. Set `selected_area_valid` based on those limits.
3. Render a mode radio:
   - `Bounding Box Selection`
   - `Elevations`
   - `OpenStreetMap`
4. Render controls for the chosen mode.

### Bounding Box Sidebar

1. If `bbox_object` exists, show its corner dataframe.
2. Otherwise, show a four-row empty dataframe.
3. Store edits in `st.session_state["edited_df"]`.
4. The bounding box CSV uploader parses a four-row `x`/`y` dataframe and updates `bbox_object`.
5. `Cycle bounding box origin` button calls `permute_bbox()`.
6. Show x length, y length, and area metrics.

Callback interfaces:

```text
handle_uploaded_bbox_file(file_object) -> None
permute_bbox() -> None
update_bbox_from_df() -> None
update_bounding_box(points) -> None
```

### Elevation Sidebar

1. `Find available data sources` calls `find_data_sources_in_bbox(status_update_area)`.
2. The selectbox displays `st.session_state["data_sources"]` using `get_data_source_label(data_source)`.
3. `Extract elevation data` calls `extract_data_in_bbox(status_update_area)`.
4. Download button serializes `st.session_state["elevation_in_bbox"]` via cached `dataframe2csv(df)`.

Callback interfaces:

```text
find_data_sources_in_bbox(status_update_area) -> None
get_data_source_label(data_source) -> str
extract_data_in_bbox(status_update_area) -> None
dataframe2csv(df) -> bytes
```

### OSM Sidebar

1. User chooses Combat Mission title/profile.
2. App lists `*.json` files in `executable_path`.
3. User chooses config file; the JSON is loaded into `st.session_state["osm_config"]`.
4. `Download OpenStreeMap data` calls `get_osm_data(status_update_area)`.
5. Alternatively, uploaded GeoJSON is read with:

```text
get_bounding_box_from_file_object(file_object) -> BoundingBox
read_file_object(file_object) -> geojson object
```

6. `Process OpenStreeMap data` calls `process_osm_data(status_update_area)`.
7. Download button serializes `st.session_state["osm_output"]`.

Callback interfaces:

```text
get_osm_data(status_update_area) -> None
process_osm_data(status_update_area) -> None
```

## 4. Bounding Box Update Flow

### From Folium Drawing

1. `map_view_tab()` calls `st_folium(...)`.
2. If `st_data["last_active_drawing"]` exists in bbox mode, extract GeoJSON coordinates.
3. If the coordinates differ from the previous draw, store `drawn_coordinates`.
4. Call `update_bounding_box(coordinates[0])`.

### From Data Editor

1. `map_view_tab()` compares `edited_df` against `bbox_object.get_dataframe()`.
2. If values changed and all are non-null, call `update_bbox_from_df()`.
3. `update_bbox_from_df()` converts the dataframe rows to `(x, y)` points and calls `update_bounding_box(points)`.

### From Bounding Box CSV Upload

1. The sidebar uploader reads the uploaded CSV bytes.
2. `parse_bbox_csv_bytes()` accepts the exported dataframe format, including an optional index column.
3. The parsed `x`/`y` rows are converted to a `shapely.Polygon`.
4. A `BoundingBox` is constructed and stored through `update_state_from_bbox()`.
5. The upload signature is cached so the retained Streamlit uploader value is ignored on unchanged reruns.

### `update_bounding_box(points)`

1. Create `shapely.Polygon(points)`.
2. Ignore degenerate line-like rectangles.
3. Construct `BoundingBox(polygon)`.
4. Store:
   - `bbox`
   - `bbox_object`
   - `projected_bbox_object`
   - `len_x`
   - `len_y`
   - `bbox_origin`
5. Delete stale `elevation_in_bbox`, if present.
6. Call `st.rerun()`.

### `BoundingBox.__init__(polygon, crs)`

1. Keep original polygon node points.
2. Ensure a WGS84 polygon.
3. Select local projected CRS with `get_projection_epsg_code_from_bbox`.
4. Transform WGS84 polygon to UTM.
5. Compute minimum rotated rectangle.
6. Make rectangle counter-clockwise.
7. Choose the rectangle node closest to the original first point as origin.
8. Store UTM and WGS84 versions of the rotated rectangle.

## 5. Elevation Extraction Flow

### Data-Source Discovery

`find_data_sources_in_bbox(status_update_area)`:

1. Iterate `st.session_state["selectable_data_sources"]`.
2. Call `data_source.intersects_bounding_box(st.session_state["bbox_object"])`.
3. Store matches in `st.session_state["data_sources"]`.

Base implementation:

```text
DataSource.intersects_bounding_box(bounding_box)
  -> quick envelope intersects check in source CRS
  -> source GeoDataFrame spatial-index query
```

Some sources override this when their index needs special handling.

### Data Extraction

`extract_data_in_bbox(status_update_area)`:

1. Read `selected_data_source` and `bbox_object`.
2. Mark `currently_processing_data` for interruption warning.
3. Ensure `data_cache_path` exists.
4. Call `data_source.get_data(bounding_box, data_cache_path)`.
5. Store dataframe as `elevation_in_bbox`.
6. Call `data_source.get_png(bounding_box, data_cache_path)`.
7. Clear `currently_processing_data`.

Concrete data-source `get_data` methods follow this shape:

```text
get_missing_files(bounding_box, source_cache_dir) -> list[str]
download_overlapping_data(missing_files, source_cache_dir) -> None
get_images_in_bounding_box(bounding_box, source_cache_dir) -> list[path]
merge/read/reproject source files -> dataframe[x, y, z]
cut_out_bounding_box(df, bounding_box) -> dataframe[x, y, z]
cache result and bbox on the instance
```

GeoTIFF path:

```text
GeoTiffDataSource.merge_image_files(image_files, out_dir, bounding_box)
GeoTiffDataSource.get_merged_dataframe(bounding_box)
DataSource.cut_out_bounding_box(df, bounding_box)
```

XYZ path:

```text
XYZDataSource.get_merged_dataframe(bounding_box, data_files)
DataSource.cut_out_bounding_box(df, bounding_box)
```

PNG overlay path:

```text
DataSource.get_png(bounding_box, cache_dir)
dataframe_in_bbox_to_png(df, bounding_box, "current_height_map.png")
height_map_to_png(height_map, file_path)
```

## 6. OSM Download/Import Flow

### Download

`get_osm_data(status_update_area)`:

1. Load selected OSM config JSON.
2. Build an OSMnx tag dictionary from every config entry's `tags`, `exclude_tags`, and `required_tags`.
3. Call `osmnx.features_from_polygon(bounding_box.box_wgs84, tag_dict)`.
4. Drop large `ways`/`nodes` columns when present.
5. Convert the GeoDataFrame to GeoJSON.
6. Store it in `st.session_state["osm_data"]`.

### Upload

Uploaded files go through `osm_utils.io`:

```text
read_file_object(file_object) -> geojson.load(file_object)
get_bounding_box_from_file_object(file_object)
  -> read_file_object(file_object)
  -> get_bounding_box(osm_data)
  -> BoundingBox(union/bounds of feature geometries)
```

The upload bbox is stored separately as `osm_bbox_object` for dashed preview.

## 7. OSM Processing Flow

`process_osm_data(status_update_area)`:

1. Read `osm_data`.
2. Construct:

```text
OSMProcessor(path_to_config, bbox=bbox_object, profile=osm_profile_str)
```

3. Call:

```text
osm_processor.preprocess_osm_data(osm_data)
osm_processor.run_processors()
osm_processor.post_process()
```

4. Store:

```text
osm_output = osm_processor.get_output()
osm_geometries = osm_processor.get_geometries()
```

### `OSMProcessor.preprocess_osm_data(osm_data)`

1. Create WGS84-to-bbox-CRS transformer.
2. Iterate GeoJSON features.
3. Convert feature geometry to Shapely and project it.
4. Extract tags from either `properties["tags"]` or direct properties.
5. For each active config entry:
   - skip excluded tags/ids
   - enforce required tags and allowed ids
   - match configured `tags`
6. Append matched entries:

```text
{"element": element, "geometry": projected_geometry, "name": config_name, "idx": element_idx}
```

7. Initialize grid with `_init_grid(bbox)`.
8. Add configured default full-map entries for ground/foliage.

### `OSMProcessor._init_grid(bbox)`

1. Get bbox origin and axis reference points in projected CRS.
2. Compute number of 8 m cells along x and y.
3. Build rotated square, diagonal, and sub-square grids with `osm_utils.grid.get_all_grids`.
4. Assign CRS to all grids.
5. Union the square grid into `effective_bbox_polygon`.
6. Store index bounds as `[0, 0, max_xidx, max_yidx]`.

### `OSMProcessor.run_processors()`

1. `_collect_stages()` groups matched elements by:
   - config priority
   - stage index
   - processor function name
   - item list
2. Iterate priorities and stage indices in ascending order.
3. Dynamically call processor functions from `terrain_extraction.osm_utils.processing`.
4. Calls are either:

```text
processor(self, self.config, matched_element)
processor(self, self.config, config_name, tqdm_string=...)
```

The processor functions append rows to `self.df` through `_append_to_df(sub_df)` and may fill supporting structures such as network graphs, occupancy data, or building outlines.

### `OSMProcessor.post_process()`

1. Drop exact duplicate rows.
2. Drop invalid rows where `menu == -1` and `z == -1`.
3. For each `(xidx, yidx)` group:
   - remove default priority rows when real content exists
   - keep the strongest applicable priority
   - remove lower-priority or negative-priority conflicts
   - when duplicate entries remain for the same priority/name, keep the earliest matching CM type rank from config
4. Drop all selected conflict rows.

### Output Interfaces

`get_output()`:

1. Add a sentinel/top-right grid row.
2. Rename `xidx/yidx` to `x/y`.
3. Clip to `idx_bbox`.
4. Normalize coordinates to start at zero.
5. Return dataframe for CSV download.

`get_geometries(crs=WGS84)`:

1. For non-default area entries, union grid-cell polygons by `name`.
2. For building entries, rebuild building outline geometry from profile tile definitions and processed rows.
3. Return `{config_name: [shapely geometries...]}` for Folium preview.

## 8. Map Rendering Flow

`map_view_tab()`:

1. Choose header/subheader from `map_mode`.
2. Initialize persistent map center/zoom/key state.
3. Create a Folium map using OpenTopoMap tiles.
4. Add draw controls only in bbox mode.
5. Add geocoder, measure, and fullscreen controls.
6. Add elevation image overlay if `elevation_in_bbox` exists.
7. Draw current bbox as red lines with CM axis labels and origin marker.
8. Draw uploaded OSM bbox as dashed red lines if present.
9. Convert `osm_geometries` to Folium geometries with `shapely2folium`, sorted by config priority.
10. Call `st_folium(...)`.
11. Process newly drawn bbox coordinates.
12. Cache current zoom/center.
13. Process edited dataframe bbox changes.

## 9. Options Tab Flow

`options_tab()`:

1. Render a data editor listing every instantiated data source and whether it is included in searches.
2. Rebuild `selectable_data_sources` from checked rows.
3. Walk `data_cache_path` and sum cached file sizes.
4. Render a cache clear button with size label.

The cache clear button currently has no callback attached in the inspected code.
