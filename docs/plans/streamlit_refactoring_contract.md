# Streamlit Refactoring Contract

This document is authoritative for the CMTerrainExtractor Streamlit refactor. Its content must not be changed unless the user explicitly approves the contract change. If executing any prompt, plan, milestone, or task would violate this contract, stop immediately and ask the user for permission before continuing.

Source outline: `docs/plans/streamlit_refactoring_outline_for_codex.md`.

## Purpose

Refactor the `cm_terrain_extractor_app` Streamlit application incrementally while preserving the current Streamlit plus PyInstaller deployment path. The refactor separates UI, state, backend actions, map rendering, and runtime resource handling so the application is easier to maintain, test, and eventually port to another GUI framework.

This refactor is behavior-preserving. Existing users should recognize the app and its workflows after every milestone.

## Scope Boundaries

In scope:

- Code under `cm_terrain_extractor_app/`, especially `cmterrainextractor.py`, `cm_terrain_extractor_app.py`, and `terrain_extraction/` integration points.
- New tests under `tests/cm_terrain_extractor/`.
- Packaging files that are required to keep the current PyInstaller app working.
- Durable planning and packaging documentation under `docs/` or `docs/plans/`.

Out of scope unless the user approves a contract change:

- Changing away from Streamlit as the active GUI framework.
- Replacing Folium or `streamlit-folium`.
- Rewriting terrain extraction algorithms.
- Changing CSV output formats.
- Redesigning the user-facing workflow.
- Moving canonical ownership of terrain algorithms out of `terrain_extraction/`.

## Ownership Boundaries

### Streamlit UI

Owned by `cm_terrain_extractor_app/streamlit_ui/` and the top-level Streamlit entry script.

Responsibilities:

- Streamlit page configuration, layout, tabs, sidebar, forms, widgets, status, messages, and download buttons.
- `st_folium(...)` invocation and Streamlit rerun handling.
- Calling backend actions and applying returned results to `AppState`.

Allowed imports:

- `streamlit`
- `streamlit_folium`
- `app_core`
- `map_view`
- Existing `terrain_extraction` types only when needed for display integration.

Invariant:

- Only `streamlit_ui/` and top-level Streamlit entrypoints may import `streamlit`.

### Application Core

Owned by `cm_terrain_extractor_app/app_core/`.

Responsibilities:

- Explicit application state.
- Runtime resources and PyInstaller path resolution.
- Streamlit-free backend actions.
- Bounding-box validation and derived metrics.
- CSV/export byte preparation.
- Optional typed application errors.

Forbidden imports:

- `streamlit`
- `streamlit_folium`

Folium imports are forbidden in `app_core/` unless the user approves a contract change.

### Map View

Owned by `cm_terrain_extractor_app/map_view/`.

Responsibilities:

- Building complete Folium maps from `AppState` and `AppResources`.
- Adding Folium layers for bbox, elevation overlay, OSM bbox, OSM geometries, controls, and draw tools.
- Parsing `streamlit-folium` drawing payloads into project geometry inputs.

Forbidden behavior:

- Downloading data.
- Running terrain extraction.
- Running OSM processing.
- Mutating `st.session_state`.

### Terrain Extraction

Owned by `cm_terrain_extractor_app/terrain_extraction/`.

Responsibilities:

- Existing terrain, elevation, projection, OSM processing, data-source, and visualization algorithms.
- Existing runtime contracts used by the app unless explicitly changed by a milestone and covered by tests.

Invariant:

- The refactor may wrap or call these modules, but must not rewrite algorithms unless a minimal interface change is required and approved by tests.

### Launcher and Packaging

Owned by:

- `cm_terrain_extractor_app/cm_terrain_extractor_app.py`
- `cm_terrain_extractor_app/streamlit_main.py`
- `cm_terrain_extractor_app.spec`
- packaging docs under `docs/`

Responsibilities:

- Keep `cm_terrain_extractor_app.py` as the packaged executable launcher.
- Resolve resources and prepare runtime environment before Streamlit bootstrap.
- Preserve bundled DLL, config, Streamlit static, `streamlit_folium`, and profiles behavior.

Invariant:

- PyInstaller support must remain explicit and tested manually before the refactor is considered complete.

## Planned Interfaces

### `app_core/state.py`

Owns explicit state. Initial interface:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class AppState:
    map_mode: str = "Bounding Box Selection"
    bbox_object: Any | None = None
    bbox_coordinates: list | None = None
    projected_bbox_object: Any | None = None
    len_x: float | None = None
    len_y: float | None = None
    selected_area_valid: bool = False
    bbox_origin: int = 0
    map_center: tuple[float, float] = (0.0, 0.0)
    map_zoom: int = 2
    map_key: int = 0
    selectable_data_sources: list[Any] = field(default_factory=list)
    available_data_sources: list[Any] = field(default_factory=list)
    selected_data_source: Any | None = None
    elevation_in_bbox: pd.DataFrame | None = None
    height_map_png: Path | None = None
    osm_config_file: str | None = None
    osm_config: dict | None = None
    osm_profile: str = "cold_war"
    osm_data: dict | None = None
    osm_bbox_object: Any | None = None
    osm_output: pd.DataFrame | None = None
    osm_geometries: dict | None = None
    currently_processing_data: tuple[str, str] | None = None
```

Required helper behavior:

- `clear_bbox_dependent_results(state: AppState) -> None`
- `mark_map_dirty(state: AppState) -> None`

### `app_core/resources.py`

Owns runtime and packaging paths:

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


def resolve_resources() -> AppResources: ...
def prepare_runtime_environment(resources: AppResources) -> None: ...
def find_default_osm_configs(resources: AppResources) -> list[Path]: ...
```

All `sys.frozen`, `sys.executable`, `__file__`, DLL path, and source-vs-packaged path branching must live here or in the launcher calling this module.

### `streamlit_ui/session_adapter.py`

Owns session storage:

```python
STATE_KEY = "cm_terrain_extractor_state"


def get_state(resources: AppResources) -> AppState: ...
def reset_state(resources: AppResources) -> AppState: ...
```

Only this module may store the main `AppState` object in `st.session_state`.

### `app_core/validation.py`

Owns bbox validation and derived metrics:

```python
def update_state_from_bbox(state: AppState, bbox_object: object) -> None: ...
def is_selected_area_valid(len_x: float, len_y: float) -> bool: ...
def compute_bbox_metrics(bbox_object: object) -> dict[str, float]: ...
```

The current limits are contractual unless changed by user approval:

- Maximum W-E length: 4160 m.
- Maximum S-N length: 4160 m.
- Maximum selected area: 18,000,000 m2.

### `app_core/actions.py`

Owns Streamlit-free backend actions:

```python
def find_data_sources_in_bbox(*, bbox: object, selectable_sources: list) -> list: ...
def extract_elevation_data(*, data_source: object, bbox: object, data_cache_path: Path) -> tuple[pd.DataFrame, Path | None]: ...
def load_osm_config(*, config_path: Path) -> dict: ...
def download_osm_data(*, bbox: object, config: dict) -> dict: ...
def load_osm_data_from_uploaded_bytes(*, data: bytes, filename: str | None = None) -> dict: ...
def process_osm_data(*, osm_data: dict, bbox: object, config_path: Path, profile: str) -> tuple[pd.DataFrame, dict]: ...
```

Actions return data or raise exceptions. UI decides how to display status.

### `app_core/exports.py`

Owns downloadable bytes and filenames:

```python
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes: ...
def suggest_elevation_filename(state: AppState) -> str: ...
def suggest_osm_filename(state: AppState) -> str: ...
```

CSV compatibility invariant:

- `dataframe_to_csv_bytes(df)` must match current `df.to_csv().encode("utf-8")` behavior until a user-approved contract change.

### `map_view/drawing.py`

Owns drawing payload parsing:

```python
def extract_last_active_drawing(st_folium_data: dict) -> dict | None: ...
def drawing_to_lon_lat_points(drawing: dict) -> list[tuple[float, float]]: ...
def drawing_to_bounding_box(drawing: dict) -> object: ...
```

Coordinate-order invariant:

- Preserve current behavior. The current implementation reads `last_active_drawing["geometry"]["coordinates"]`, converts it to a NumPy array, and passes the first coordinate ring to `BoundingBox(Polygon(points))`.
- Any coordinate-order correction must be backed by a regression test and documented in the status document.

### `map_view/folium_map.py` and `map_view/folium_layers.py`

Own Folium map rendering:

```python
def build_folium_map(*, state: AppState, resources: AppResources) -> folium.Map: ...
def add_draw_control(map_obj: folium.Map) -> None: ...
def add_bbox_layer(map_obj: folium.Map, state: AppState) -> None: ...
def add_elevation_overlay(map_obj: folium.Map, state: AppState, resources: AppResources) -> None: ...
def add_osm_bbox_layer(map_obj: folium.Map, state: AppState) -> None: ...
def add_osm_geometry_layers(map_obj: folium.Map, state: AppState) -> None: ...
```

Visual invariant:

- Preserve OpenTopoMap tiles, draw controls, geocoder, measure control, fullscreen control, red bbox lines, axis labels, OSM dashed bbox, elevation overlay opacity, and OSM geometry priority ordering as closely as possible.

## State Invalidation Rules

When bbox changes, invalidate:

- `available_data_sources`
- `selected_data_source`
- `elevation_in_bbox`
- `height_map_png`
- `osm_data`
- `osm_bbox_object`
- `osm_output`
- `osm_geometries`

When selected elevation data source changes, invalidate:

- `elevation_in_bbox`
- `height_map_png`

When OSM config or profile changes, invalidate:

- `osm_output`
- `osm_geometries`

Raw `osm_data` may be preserved when the bbox and downloaded/uploaded data are still valid.

When uploaded OSM data changes, invalidate:

- `osm_output`
- `osm_geometries`

and set:

- `osm_data`
- `osm_bbox_object` when a bbox can be derived from the uploaded file.

## Testing Contract

The repository currently has fixture data under `test/`, but no active pytest suite for this app. Therefore:

- Every milestone that touches Python code must start with a failing pytest test.
- Every touched Python module must have relevant tests after the milestone.
- Tests must live under `tests/cm_terrain_extractor/`.
- Pure logic must be covered with unit tests.
- Cross-module wiring must be covered with integration-style tests that do not launch Streamlit unless explicitly necessary.
- Streamlit UI and PyInstaller behavior may use manual smoke checks in addition to automated tests.
- Network access must not be required for unit tests. Network-dependent checks must be marked or left as manual.

Validation commands must use the approved Conda environment:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe <target tests> -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check <touched Python files>
```

## Stop Conditions

Stop and ask the user before continuing if:

- A task requires changing this contract.
- A task requires changing GUI frameworks.
- A task requires replacing Streamlit, Folium, or `streamlit-folium`.
- A task requires changing CSV output compatibility.
- A task requires rewriting terrain extraction algorithms.
- A task cannot keep the app runnable or cannot provide a clear rollback path.
- Tests cannot be added for touched code.
- Packaging behavior cannot be preserved or manually validated on the target Windows environment.
