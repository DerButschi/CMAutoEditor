from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

OSM_DATA_SOURCE_DOWNLOADED = "downloaded"
OSM_DATA_SOURCE_UPLOADED = "uploaded"


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
    osm_data_source: str | None = None
    osm_uploaded_file_signature: str | None = None
    osm_bbox_object: Any | None = None
    osm_output: pd.DataFrame | None = None
    osm_geometries: dict | None = None
    osm_debug_layers: dict | None = None
    currently_processing_data: tuple[str, str] | None = None


def clear_bbox_dependent_results(state: AppState) -> None:
    state.available_data_sources = []
    state.selected_data_source = None
    clear_elevation_result(state)
    if state.osm_data_source != OSM_DATA_SOURCE_UPLOADED:
        state.osm_data = None
        state.osm_data_source = None
        state.osm_uploaded_file_signature = None
        state.osm_bbox_object = None
    clear_osm_processing_result(state)


def clear_elevation_result(state: AppState) -> None:
    had_elevation_result = state.elevation_in_bbox is not None or state.height_map_png is not None
    state.elevation_in_bbox = None
    state.height_map_png = None
    if had_elevation_result:
        mark_map_dirty(state)


def clear_osm_processing_result(state: AppState) -> None:
    state.osm_output = None
    state.osm_geometries = None
    state.osm_debug_layers = None


def mark_map_dirty(state: AppState) -> None:
    state.map_key += 1
