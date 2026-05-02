# CMTerrainExtractor Streamlit Refactoring Outline

## Purpose

This document describes a refactoring direction for the `cm_terrain_extractor_app` Streamlit application. It is intended as an input document for Codex or another coding agent to derive an implementation plan, work packages, and execution steps. It is **not** a direct prompt. The refactor should preserve the current Streamlit + PyInstaller deployment path while making the application easier to maintain, test, and eventually port to another GUI framework if desired.

The main target is to separate the application into:

1. **Streamlit UI shell**: widget rendering, layout, status messages, and Streamlit-specific behavior.
2. **Application state model**: explicit state object instead of many independent `st.session_state` keys.
3. **Core actions/services**: terrain/elevation/OSM operations that do not import Streamlit.
4. **Map rendering layer**: Folium/streamlit-folium logic isolated from business logic.
5. **Resource and packaging layer**: PyInstaller/runtime path handling centralized in one module.

The refactor should be incremental and behavior-preserving. The current app should remain usable after each major milestone.

---

## Scope

### In scope

The refactor focuses on the code under:

```text
cm_terrain_extractor_app/
```

Especially:

```text
cm_terrain_extractor_app/cmterrainextractor.py
cm_terrain_extractor_app/cm_terrain_extractor_app.py
cm_terrain_extractor_app/terrain_extraction/
```

The current Streamlit app appears to combine several responsibilities in `cmterrainextractor.py`:

- Streamlit page layout and widgets.
- `st.session_state` initialization and mutation.
- Folium map construction.
- `streamlit_folium.st_folium(...)` map interaction handling.
- Bounding-box drawing and coordinate conversion.
- Available elevation data-source detection.
- Elevation extraction and PNG overlay generation.
- OSM data download/import.
- OSM processing and geometry rendering.
- CSV export preparation.
- Runtime path handling for packaged execution.

The launcher `cm_terrain_extractor_app.py` should remain the executable entry point for PyInstaller-style distribution, but its responsibilities should be reduced to runtime preparation and Streamlit bootstrap.

### Out of scope

Do **not** attempt to change the GUI framework in this refactor.

Do **not** replace Folium, Streamlit, or `streamlit-folium`.

Do **not** rewrite the terrain extraction algorithms unless a minimal interface change is needed to separate GUI and backend logic.

Do **not** introduce NiceGUI, Dash, Panel, Qt, or Flet in this refactor. The refactor should make such a migration easier later, but it should not start one.

Do **not** change the user-facing workflow unless required to preserve correctness. Existing users should recognize the app.

---

## High-level goal

Convert the current design from this:

```text
cmterrainextractor.py
  ├─ Streamlit UI
  ├─ Streamlit session state
  ├─ Folium map creation
  ├─ Drawing result parsing
  ├─ bbox validation
  ├─ data-source discovery
  ├─ elevation extraction
  ├─ OSM download/import/process
  ├─ CSV export
  └─ runtime/package path handling
```

Into this:

```text
cm_terrain_extractor_app/
  cm_terrain_extractor_app.py        # packaged executable launcher
  streamlit_main.py                  # Streamlit entry point

  app_core/
    __init__.py
    state.py                         # explicit AppState dataclass
    resources.py                     # runtime/PyInstaller/resource paths
    actions.py                       # Streamlit-free business actions
    validation.py                    # bbox/data validity checks
    exports.py                       # CSV/download helpers
    errors.py                        # optional typed app errors

  map_view/
    __init__.py
    folium_map.py                    # construct complete Folium map
    folium_layers.py                 # bbox/elevation/OSM layer helpers
    drawing.py                       # parse st_folium drawing events

  streamlit_ui/
    __init__.py
    session_adapter.py               # st.session_state <-> AppState
    app.py                           # high-level Streamlit app render
    sidebar.py                       # sidebar controls
    map_page.py                      # map tab/view
    osm_page.py                      # OSM controls/view helpers, if needed
    elevation_page.py                # elevation controls/view helpers, if needed
    widgets.py                       # reusable widget helpers

  terrain_extraction/
    ... existing package ...
```

The exact file names may be adjusted after inventorying the current code, but the architectural boundaries should remain.

---

## Architectural principles

### 1. Keep Streamlit at the edges

Only files in `streamlit_ui/` and the top-level Streamlit entry point should import `streamlit`.

Allowed:

```python
# streamlit_ui/sidebar.py
import streamlit as st
```

Avoid:

```python
# app_core/actions.py
import streamlit as st  # should not happen
```

The backend should be callable from tests, command-line scripts, or a future NiceGUI/Dash UI without importing Streamlit.

### 2. Make application state explicit

Replace scattered `st.session_state[...]` keys with a single `AppState` object stored in Streamlit session state.

Streamlit session state should become an implementation detail:

```python
state = get_state(resources, data_sources)
```

Instead of:

```python
st.session_state["bbox"]
st.session_state["bbox_object"]
st.session_state["osm_data"]
st.session_state["elevation_in_bbox"]
...
```

### 3. Preserve Streamlit's rerun model, but contain it

The app should not try to fight Streamlit's rerun behavior. Instead:

- Keep state mutations explicit.
- Keep `st.rerun()` calls rare and documented.
- Use `st.form` only where it simplifies batched input updates.
- Avoid premature use of `st.fragment` until the state split is stable.
- Keep `st_folium` map key/zoom/center behavior contained in `streamlit_ui/map_page.py` and `map_view/drawing.py`.

### 4. Map rendering is a view, not business logic

Folium construction should be driven by `AppState` and resources. It should not perform downloads, extractions, or OSM processing.

Good:

```python
folium_map = build_folium_map(state, resources)
```

Bad:

```python
folium_map = build_folium_map_and_download_osm_and_update_session_state(...)
```

### 5. Actions return data; UI decides how to display status

Core actions should return results or raise exceptions. They should not call `st.status`, `st.info`, `st.error`, `st.success`, or modify Streamlit widgets.

Good:

```python
elevation_df, png_path = extract_elevation_data(...)
```

Then the Streamlit UI wraps that call in `st.status(...)`.

### 6. Runtime resource handling belongs in one place

PyInstaller/resource lookup logic should be centralized. Avoid repeated checks for `sys.frozen`, `sys.executable`, current working directory, and config paths throughout the app.

---

## Target module contracts

### `app_core/state.py`

Defines the central application state. This should be a dataclass or a small set of dataclasses. Keep it serializable-ish and simple. It does not need to be perfectly JSON-serializable because some fields may hold DataFrames, geometry objects, or data-source objects.

Suggested initial sketch:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class AppState:
    # UI mode
    map_mode: str = "Bounding Box Selection"

    # Bounding box / geometry state
    bbox_object: Any | None = None
    bbox_coordinates: list | None = None
    projected_bbox_object: Any | None = None
    len_x: float | None = None
    len_y: float | None = None
    selected_area_valid: bool = False
    bbox_origin: int = 0

    # Map state
    map_center: tuple[float, float] = (0.0, 0.0)
    map_zoom: int = 2
    map_key: int = 0

    # Elevation data-source state
    selectable_data_sources: list[Any] = field(default_factory=list)
    available_data_sources: list[Any] = field(default_factory=list)
    selected_data_source: Any | None = None
    elevation_in_bbox: pd.DataFrame | None = None
    height_map_png: Path | None = None

    # OSM state
    osm_config_file: str | None = None
    osm_config: dict | None = None
    osm_profile: str = "cold_war"
    osm_data: dict | None = None
    osm_bbox_object: Any | None = None
    osm_output: pd.DataFrame | None = None
    osm_geometries: dict | None = None

    # Processing/status state
    currently_processing_data: tuple[str, str] | None = None
```

Codex should inventory the current session-state keys before finalizing this dataclass. The first task should be to create a mapping table:

```text
Current st.session_state key -> AppState field -> owner module -> reset conditions
```

Important: reset conditions matter. For example, changing the bounding box should probably invalidate elevation data, OSM output, available sources, and overlays derived from the previous bounding box.

Recommended helper methods or functions:

```python
def clear_bbox_dependent_results(state: AppState) -> None:
    ...


def mark_map_dirty(state: AppState) -> None:
    state.map_key += 1
```

Do not over-engineer state management. A simple dataclass plus a few helper functions is enough.

---

### `app_core/resources.py`

Centralizes runtime path resolution and PyInstaller-specific behavior.

Suggested dataclass:

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppResources:
    app_root: Path
    executable_root: Path
    data_cache_path: Path
    config_dir: Path
    dll_dir: Path
```

Suggested functions:

```python
def resolve_resources() -> AppResources:
    """Resolve app paths for both source execution and PyInstaller execution."""
    ...


def prepare_runtime_environment(resources: AppResources) -> None:
    """Apply runtime environment changes, e.g. prepend bundled DLL directory to PATH."""
    ...


def find_default_osm_configs(resources: AppResources) -> list[Path]:
    ...
```

Rules:

- All `sys.frozen` / `sys.executable` / `__file__` path branching should live here.
- All DLL path setup should live here or in the launcher calling this module.
- The Streamlit app should receive an `AppResources` instance rather than rediscover paths.
- Tests should be able to create a fake `AppResources` pointing to a temp directory.

---

### `app_core/validation.py`

Owns validation and derived metrics for bounding boxes and selected areas.

Suggested functions:

```python
def update_state_from_bbox(state: AppState, bbox_object: object) -> None:
    """Set bbox fields, derived dimensions, validity flags, and invalidate dependent data."""
    ...


def is_selected_area_valid(len_x: float, len_y: float) -> bool:
    ...


def compute_bbox_metrics(bbox_object: object) -> dict[str, float]:
    ...
```

This module may import project geometry classes such as `BoundingBox`, but should not import Streamlit or Folium.

---

### `app_core/actions.py`

Owns backend actions that currently happen inside Streamlit callbacks or rendering code.

Suggested functions:

```python
def find_data_sources_in_bbox(
    *,
    bbox: object,
    selectable_sources: list,
) -> list:
    ...


def extract_elevation_data(
    *,
    data_source: object,
    bbox: object,
    data_cache_path: Path,
) -> tuple[pd.DataFrame, Path | None]:
    ...


def load_osm_config(
    *,
    config_path: Path,
) -> dict:
    ...


def download_osm_data(
    *,
    bbox: object,
    config: dict,
) -> dict:
    ...


def load_osm_data_from_uploaded_bytes(
    *,
    data: bytes,
    filename: str | None = None,
) -> dict:
    ...


def process_osm_data(
    *,
    osm_data: dict,
    bbox: object,
    config_path: Path,
    profile: str,
) -> tuple[pd.DataFrame, dict]:
    ...
```

Rules:

- No Streamlit imports.
- No Folium imports unless a strong reason is found; prefer returning project data structures.
- Status/progress messages should be returned as optional structured events only if necessary. The first version can simply let the UI show coarse status before/after each action.
- Keep existing backend APIs where possible. Do not rewrite `terrain_extraction` internals unnecessarily.

Potential error strategy:

```python
class TerrainExtractorError(Exception):
    pass

class NoDataSourceError(TerrainExtractorError):
    pass

class OSMProcessingError(TerrainExtractorError):
    pass
```

Only add typed errors if they simplify UI error handling. Do not create a large exception hierarchy prematurely.

---

### `app_core/exports.py`

Owns conversion of output DataFrames and results into downloadable bytes.

Suggested functions:

```python
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    ...


def suggest_elevation_filename(state: AppState) -> str:
    ...


def suggest_osm_filename(state: AppState) -> str:
    ...
```

Keep Streamlit `st.download_button` usage in the UI layer. This module should only prepare bytes and filenames.

---

### `map_view/drawing.py`

Owns conversion from `st_folium` return payloads to project geometry objects.

Suggested functions:

```python
def extract_last_active_drawing(st_folium_data: dict) -> dict | None:
    ...


def drawing_to_lon_lat_points(drawing: dict) -> list[tuple[float, float]]:
    ...


def drawing_to_bounding_box(drawing: dict) -> object:
    ...
```

Rules:

- This module may know about the shape of `streamlit-folium` return data.
- It should not call Streamlit directly.
- It should be unit-testable with small GeoJSON-like example payloads.
- Be careful about coordinate order. GeoJSON commonly uses `[lon, lat]`; Leaflet UI interactions often expose `[lat, lon]`. Codex must inspect the current code path and preserve current behavior.

---

### `map_view/folium_map.py`

Owns construction of the complete Folium map from state.

Suggested function:

```python
def build_folium_map(
    *,
    state: AppState,
    resources: AppResources,
) -> folium.Map:
    ...
```

This should add base tiles, controls, draw tools, and delegate overlays to `folium_layers.py`.

Rules:

- It should be mostly pure with respect to application state: read state, build map, return map.
- It may read image files such as elevation PNG overlays if required by Folium.
- It should not download data, process OSM, or mutate `AppState` except possibly through explicitly named helper functions. Prefer no mutation.
- Keep the existing OpenTopoMap tile behavior unless there is a specific reason to change it.

---

### `map_view/folium_layers.py`

Owns individual Folium layer creation.

Suggested helpers:

```python
def add_draw_control(map_obj: folium.Map) -> None:
    ...


def add_bbox_layer(map_obj: folium.Map, state: AppState) -> None:
    ...


def add_elevation_overlay(map_obj: folium.Map, state: AppState, resources: AppResources) -> None:
    ...


def add_osm_bbox_layer(map_obj: folium.Map, state: AppState) -> None:
    ...


def add_osm_geometry_layers(map_obj: folium.Map, state: AppState) -> None:
    ...
```

Layer styling should initially preserve the current appearance. Avoid large visual redesigns in this refactor.

---

### `streamlit_ui/session_adapter.py`

Owns Streamlit session initialization and state retrieval.

Suggested functions:

```python
STATE_KEY = "cm_terrain_extractor_state"


def get_state(resources: AppResources) -> AppState:
    ...


def reset_state(resources: AppResources) -> AppState:
    ...
```

Rules:

- Only this module should directly store the `AppState` object in `st.session_state`.
- It may also handle migration from old keys during a transitional phase, but that should be temporary.
- Avoid duplicating state between `AppState` and many independent `st.session_state` keys.

---

### `streamlit_ui/app.py`

High-level Streamlit app render function.

Suggested shape:

```python
def render_app(resources: AppResources) -> None:
    configure_page()
    state = get_state(resources)

    render_sidebar(state, resources)
    render_main_tabs(state, resources)
```

This module should coordinate layout but not contain heavy business logic.

---

### `streamlit_ui/sidebar.py`

Owns sidebar widgets and actions.

Likely responsibilities:

- Map mode selector.
- Bounding-box metrics.
- Data-source discovery button.
- Elevation source selector.
- Elevation extraction button.
- OSM config/profile selection.
- OSM download/import/process controls.
- Download buttons.

Rules:

- Widget callbacks may mutate `AppState`.
- Expensive work should call functions from `app_core/actions.py`.
- Status/progress should be displayed here or in a dedicated `streamlit_ui/status.py` helper.
- Keep the UI code readable; do not hide one-line widgets behind excessive wrappers.

---

### `streamlit_ui/map_page.py`

Owns map rendering and handling `st_folium` output.

Suggested shape:

```python
def render_map_page(state: AppState, resources: AppResources) -> None:
    folium_map = build_folium_map(state=state, resources=resources)

    st_data = st_folium(
        folium_map,
        center=state.map_center,
        zoom=state.map_zoom,
        width=1200,
        key=state.map_key,
    )

    handle_map_result(state, st_data)
```

`handle_map_result` should:

- Update map center/zoom if the current code does this.
- Detect new drawings.
- Convert drawing to `BoundingBox` through `map_view/drawing.py`.
- Update `AppState` via `app_core/validation.py`.
- Invalidate bbox-dependent results.
- Mark the map dirty if needed.
- Trigger `st.rerun()` only if required.

Document any remaining `st.rerun()` calls with a comment explaining why they are necessary.

---

### `cm_terrain_extractor_app.py`

Keep as the executable/packaged launcher.

Target responsibilities:

1. Resolve resources.
2. Prepare runtime environment, including DLL path setup.
3. Bootstrap Streamlit with the new `streamlit_main.py`.

Suggested shape:

```python
from app_core.resources import resolve_resources, prepare_runtime_environment


def main() -> None:
    resources = resolve_resources()
    prepare_runtime_environment(resources)

    import streamlit.web.bootstrap as bootstrap

    flag_options = {
        "server.port": 8501,
        "global.developmentMode": False,
    }

    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True

    bootstrap.run(
        str(resources.app_root / "streamlit_main.py"),
        False,
        [],
        flag_options,
    )


if __name__ == "__main__":
    main()
```

Codex must adapt this to the current package layout and working PyInstaller setup. Do not break the current executable behavior without a replacement.

---

### `streamlit_main.py`

The script passed to Streamlit bootstrap.

Suggested shape:

```python
from app_core.resources import resolve_resources
from streamlit_ui.app import render_app


resources = resolve_resources()
render_app(resources)
```

This should be intentionally small.

---

## Suggested work packages

Codex should first make a concrete plan from this outline before editing. The plan should use small work packages. Each package should leave the app runnable or include a clear rollback path.

### Work package 0: Inventory and current behavior snapshot

#### Goal

Create an inventory of the current app before moving code.

#### Deliverables

Create a short refactor planning document, for example:

```text
docs/cm_terrain_extractor_refactor_inventory.md
```

Include:

1. Current `st.session_state` keys and what they mean.
2. Current top-level functions in `cmterrainextractor.py` and their responsibilities.
3. Current imports and which modules they belong to.
4. Current Streamlit widgets and where their state goes.
5. Current Folium layers and map interactions.
6. Current runtime/PyInstaller path handling.
7. Current data flow from bbox drawing to CSV export.

#### Acceptance criteria

- No behavior changes.
- Inventory is specific enough to guide the refactor.
- Any uncertainty is explicitly marked.

---

### Work package 1: Introduce resources module and preserve launcher behavior

#### Goal

Centralize runtime path handling without changing GUI behavior.

#### Deliverables

Add:

```text
app_core/resources.py
```

Update launcher and app to use it.

#### Acceptance criteria

- Running from source still works.
- Running through the current PyInstaller workflow still works, or any required spec changes are documented.
- DLL path setup still happens exactly once.
- Existing config/data/cache paths still resolve correctly.

#### Notes

This is a good first code change because it is isolated and reduces packaging risk later.

---

### Work package 2: Introduce AppState and session adapter

#### Goal

Replace scattered Streamlit session keys with one explicit `AppState` object.

#### Deliverables

Add:

```text
app_core/state.py
streamlit_ui/session_adapter.py
```

Initially, it is acceptable to keep some legacy session keys if removing them all at once is too risky. But the direction should be clear and documented.

#### Acceptance criteria

- App starts with equivalent default state.
- Drawing/selecting bbox still works.
- Previously available actions remain available.
- State reset/invalidation behavior is preserved or improved.
- A mapping from old session keys to new fields exists in the inventory or status doc.

#### Notes

Be careful not to duplicate state indefinitely. Temporary duplication is acceptable only as a migration step.

---

### Work package 3: Extract pure backend actions

#### Goal

Move elevation and OSM backend operations into Streamlit-free functions.

#### Deliverables

Add:

```text
app_core/actions.py
app_core/exports.py
app_core/validation.py
```

Move or wrap logic for:

- Available elevation data-source lookup.
- Elevation extraction.
- PNG overlay path retrieval, if applicable.
- OSM config loading.
- OSM data download.
- OSM uploaded-file parsing.
- OSM processing.
- CSV bytes creation.
- Bbox validation and derived metrics.

#### Acceptance criteria

- These modules do not import Streamlit.
- The Streamlit UI still exposes the same actions.
- The current output CSV format is unchanged.
- Exceptions are either handled at the UI layer or documented.
- At least small unit tests or smoke tests exist for the easiest pure functions.

#### Notes

Do not rewrite data-source implementations unless necessary. This step should move orchestration logic, not algorithm internals.

---

### Work package 4: Extract Folium map builder and drawing parser

#### Goal

Move Folium map construction and drawing parsing out of Streamlit UI code.

#### Deliverables

Add:

```text
map_view/drawing.py
map_view/folium_map.py
map_view/folium_layers.py
```

Move logic for:

- Base map creation.
- Draw plugin configuration.
- Bounding-box display.
- Elevation PNG overlay.
- OSM bbox display.
- OSM geometry display.
- `st_folium` drawing payload parsing.

#### Acceptance criteria

- Map visually matches current behavior as closely as possible.
- Drawing a rectangle/polygon still updates the bbox.
- Existing elevation and OSM overlays still appear.
- Coordinate order is verified and documented.
- The map builder does not download/process data.

#### Notes

This is the highest-risk work package. It should be done after AppState and actions are stable.

---

### Work package 5: Split Streamlit UI into small modules

#### Goal

Reduce `cmterrainextractor.py` to a small entry point or replace it with `streamlit_main.py` plus modular UI files.

#### Deliverables

Add:

```text
streamlit_ui/app.py
streamlit_ui/sidebar.py
streamlit_ui/map_page.py
streamlit_ui/widgets.py
```

Optionally add:

```text
streamlit_ui/elevation_page.py
streamlit_ui/osm_page.py
streamlit_ui/status.py
```

#### Acceptance criteria

- The app layout remains recognizable.
- Sidebar actions still work.
- Map tab still works.
- OSM and elevation workflows still work.
- `cmterrainextractor.py` is either removed, replaced by a tiny compatibility wrapper, or reduced substantially.
- No app-core module imports Streamlit.

#### Notes

Avoid creating too many tiny wrapper functions. The goal is clarity, not fragmentation. A widget helper is justified only if it hides repeated Streamlit boilerplate or enforces consistent behavior.

---

### Work package 6: Packaging stabilization

#### Goal

Make the PyInstaller path explicit and stable after module splitting.

#### Deliverables

- Update the PyInstaller spec or packaging script.
- Ensure data files, configs, DLLs, and package resources are included.
- Add a packaging note, for example:

```text
docs/cm_terrain_extractor_packaging.md
```

#### Acceptance criteria

- Source run works.
- Packaged run works on the target Windows environment.
- The packaged app can find OSM configs, data cache, DLLs, and any terrain data source metadata.
- The launcher starts the refactored Streamlit entry point.

#### Notes

Do this after module movement because hidden imports and data files may change.

---

### Work package 7: Tests and smoke checks

#### Goal

Add lightweight tests that make future GUI migration safer.

#### Suggested tests

```text
tests/cm_terrain_extractor/test_resources.py
tests/cm_terrain_extractor/test_state.py
tests/cm_terrain_extractor/test_drawing.py
tests/cm_terrain_extractor/test_exports.py
tests/cm_terrain_extractor/test_actions_smoke.py
```

Focus on:

- Drawing payload to bbox conversion.
- DataFrame to CSV bytes.
- Resource resolution in source mode.
- State invalidation when bbox changes.
- OSM config loading.

Full Streamlit UI testing is not required for the first refactor.

#### Acceptance criteria

- Tests pass in the normal development environment.
- Tests do not require launching Streamlit.
- Tests do not require network access unless explicitly marked as integration tests.

---

## Desired final data flow

### Bounding-box drawing

```text
User draws bbox on Folium map
  -> st_folium returns drawing payload
  -> streamlit_ui/map_page.py receives st_data
  -> map_view/drawing.py converts drawing to coordinates/bbox
  -> app_core/validation.py updates AppState and derived metrics
  -> bbox-dependent results are invalidated
  -> map is marked dirty if needed
  -> Streamlit reruns if needed
```

### Elevation extraction

```text
User clicks extract elevation
  -> streamlit_ui/sidebar.py shows status
  -> app_core/actions.py calls selected data source
  -> action returns DataFrame and optional PNG path
  -> Streamlit UI stores results in AppState
  -> map_view/folium_layers.py shows PNG overlay on next render
  -> streamlit_ui/sidebar.py exposes CSV download
```

### OSM workflow

```text
User selects OSM config/profile
  -> state stores config/profile

User downloads OSM data or uploads GeoJSON
  -> app_core/actions.py returns OSM data dict
  -> state.osm_data and state.osm_bbox_object are updated

User processes OSM
  -> app_core/actions.py calls OSMProcessor
  -> action returns output DataFrame and geometries
  -> state.osm_output and state.osm_geometries are updated
  -> map_view/folium_layers.py renders geometries
  -> streamlit_ui/sidebar.py exposes CSV download
```

---

## Handling invalidation and dependencies

Codex should explicitly implement or document invalidation rules.

Recommended initial rules:

### When bbox changes

Invalidate:

- `available_data_sources`
- `selected_data_source`
- `elevation_in_bbox`
- `height_map_png`
- `osm_data`
- `osm_bbox_object`
- `osm_output`
- `osm_geometries`

Possibly preserve:

- `osm_config_file`
- `osm_profile`
- UI mode
- map zoom/center, unless current behavior differs

### When selected elevation data source changes

Invalidate:

- `elevation_in_bbox`
- `height_map_png`

### When OSM config/profile changes

Invalidate at least:

- `osm_output`
- `osm_geometries`

Possibly preserve:

- raw `osm_data`, if it was downloaded for the same bbox and config changes only affect processing.

### When uploaded OSM data changes

Invalidate:

- `osm_output`
- `osm_geometries`

Set:

- `osm_data`
- maybe `osm_bbox_object`, depending on current behavior

---

## Streamlit-specific implementation guidance

### Session state

Store only one main application state object:

```python
st.session_state[STATE_KEY] = AppState(...)
```

Avoid new independent state keys unless they are truly widget-local and documented.

### Forms

Consider `st.form` for batched inputs such as:

- Manual bbox coordinate editing.
- OSM config/profile selection.
- Data-source selection options.

Do not force all controls into forms. Buttons triggering actions can remain normal Streamlit buttons.

### Fragments

Do not introduce `st.fragment` in the first pass. After the refactor is stable, fragments may be considered for:

- Download/status areas.
- Expensive display regions.
- Option tables.

Avoid fragmenting the Folium map until its state flow is clean and well understood.

### Reruns

Keep explicit `st.rerun()` calls rare.

If one is needed, comment it:

```python
# Required because st_folium returns the drawing payload after the map has
# already been rendered. We update AppState and rerun once so the new bbox
# layer and dependent controls are shown consistently.
st.rerun()
```

### Status handling

Core actions should not call Streamlit. UI wrappers should use `st.status`, `st.spinner`, `st.error`, and `st.success`.

Example shape:

```python
try:
    with st.status("Extracting elevation data", expanded=True) as status:
        result = extract_elevation_data(...)
        apply_elevation_result(state, result)
        status.update(label="Elevation data extracted", state="complete")
except Exception as exc:
    st.error(f"Elevation extraction failed: {exc}")
```

---

## Coding style guidance

- Prefer readable modules over clever abstractions.
- Avoid a proliferation of trivial wrappers that only call another function with the same arguments.
- Keep app-core functions small enough to test, but not artificially tiny.
- Keep type hints where they clarify interfaces.
- Use `pathlib.Path` for paths where practical.
- Keep Pandas DataFrame formats unchanged unless there is a test and a clear reason.
- Prefer explicit named functions for state invalidation and result application.
- Keep user-facing labels/text stable unless intentionally improving them.

---

## Suggested documentation generated during refactor

Codex should maintain lightweight docs while working:

```text
docs/cm_terrain_extractor_refactor_inventory.md
```

Purpose: current state/function inventory and responsibility mapping.

```text
docs/cm_terrain_extractor_refactor_plan.md
```

Purpose: concrete work packages, dependencies, and status.

```text
docs/cm_terrain_extractor_packaging.md
```

Purpose: PyInstaller entry point, data files, DLLs, config paths, source-vs-packaged behavior.

These docs do not need to be long, but they should be kept accurate enough to resume work later.

---

## Definition of done

The refactor is complete when:

1. The app still runs through the current Streamlit workflow.
2. The packaged launcher still works or has an updated working PyInstaller spec/script.
3. `cmterrainextractor.py` is no longer a large mixed-responsibility script, or it is reduced to a compatibility shim.
4. Streamlit imports are limited to Streamlit UI/launcher modules.
5. Core actions can be imported and tested without launching Streamlit.
6. Folium map construction is isolated in `map_view/`.
7. Drawing payload parsing is unit-testable.
8. App state is represented by an explicit `AppState` object.
9. Bbox-dependent invalidation rules are implemented and documented.
10. CSV output behavior remains compatible with the current app.
11. Existing visual map behavior is preserved closely enough that users do not experience a redesign.

---

## Recommended first Codex planning task

Before editing code, Codex should create an implementation plan from this outline. The plan should:

1. Inventory current functions and session keys.
2. Propose the final file layout based on the actual repository.
3. Identify imports that may cause circular dependencies.
4. Identify the minimal first PR/work package.
5. List smoke tests/manual checks after each package.
6. Explicitly call out any behavior that is unclear from code inspection.

The implementation plan should not attempt to migrate to NiceGUI. It should keep the current Streamlit app working while making the codebase easier to maintain and easier to port later.

