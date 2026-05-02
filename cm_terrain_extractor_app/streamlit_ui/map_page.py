from __future__ import annotations

import sys
from typing import Any

import numpy as np
import streamlit as st
from shapely import Polygon
from streamlit_folium import st_folium

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.app_core.state import AppState
from cm_terrain_extractor_app.app_core.validation import update_state_from_bbox
from cm_terrain_extractor_app.map_view.drawing import (
    drawing_to_bounding_box,
    extract_last_active_drawing,
)
from cm_terrain_extractor_app.map_view.folium_map import build_folium_map


def render_map_page(*, state: AppState, resources: AppResources) -> None:
    header, sub_header = _map_copy(state.map_mode)
    st.header(header)
    st.markdown(sub_header)

    _refresh_map_for_new_elevation_overlay(state)
    map_obj = build_folium_map(state=state, resources=resources)
    st_data = st_folium(
        map_obj,
        center=state.map_center,
        zoom=state.map_zoom,
        width=1200,
        key=state.map_key,
    )

    _handle_drawing_payload(state, st_data)
    _cache_map_viewport(st_data)
    _handle_bbox_editor_update(state, resources)


def _map_copy(map_mode: str) -> tuple[str, str]:
    if map_mode == "Bounding Box Selection":
        return (
            "Bounding Box Selection",
            "Select the outline of the Combat Mission map by drawing a rectangle or polygon.",
        )
    if map_mode == "Elevations":
        return (
            "Extraction of Elevation Data",
            "Check which data sources are available for your selected outline and extract the data.",
        )
    return (
        "Extraction of OpenStreetMap Data",
        "Extract map content from OpenStreetMap for your selected outline.",
    )


def _refresh_map_for_new_elevation_overlay(state: AppState) -> None:
    if state.elevation_in_bbox is not None and "height_map_layer" not in st.session_state:
        if "zoom_cache" in st.session_state:
            state.map_zoom = st.session_state["zoom_cache"]
        if "center_cache" in st.session_state:
            center_cache = st.session_state["center_cache"]
            state.map_center = (center_cache["lat"], center_cache["lng"])
        state.map_key += 1
        st.session_state["height_map_layer"] = True


def _handle_drawing_payload(state: AppState, st_data: dict[str, Any]) -> None:
    drawing = extract_last_active_drawing(st_data)
    if drawing is None or state.map_mode != "Bounding Box Selection":
        return

    coordinates = np.array(drawing["geometry"]["coordinates"])
    if (
        "drawn_coordinates" not in st.session_state
        or st.session_state["drawn_coordinates"].shape != coordinates.shape
        or not (st.session_state["drawn_coordinates"] == coordinates).all()
    ):
        st.session_state["drawn_coordinates"] = coordinates
        update_state_from_bbox(state, drawing_to_bounding_box(drawing))
        # Streamlit-Folium returns the drawing after render; rerun is required to rebuild
        # the Folium map and sidebar metrics from the newly parsed bounding box.
        st.rerun()


def _cache_map_viewport(st_data: dict[str, Any]) -> None:
    if "zoom" in st_data:
        st.session_state["zoom_cache"] = st_data["zoom"]
    if "center" in st_data:
        st.session_state["center_cache"] = st_data["center"]


def _handle_bbox_editor_update(state: AppState, resources: AppResources) -> None:
    if "edited_df" not in st.session_state:
        return
    if state.bbox_object is not None:
        edited_df = st.session_state["edited_df"]
        bbox_df = state.bbox_object.get_dataframe()
        if not (edited_df == bbox_df).all().all():
            _update_bbox_from_df(state, resources)
    else:
        _update_bbox_from_df(state, resources)


def _update_bbox_from_df(state: AppState, resources: AppResources) -> None:
    df = st.session_state["edited_df"].copy()
    del st.session_state["edited_df"]
    if (~df.isnull().any()).all():
        _update_bounding_box(
            state=state,
            resources=resources,
            points=list(zip(df.x.values, df.y.values, strict=False)),
        )


def _update_bounding_box(
    *,
    state: AppState,
    resources: AppResources,
    points: list[tuple[float, float]],
) -> None:
    polygon = Polygon(points)
    if polygon.minimum_rotated_rectangle.geom_type == "LineString":
        return

    update_state_from_bbox(state, _bounding_box_class(resources)(polygon))
    # Data-editor changes happen outside the map render path; rerun keeps the map,
    # sidebar metrics, and stored AppState synchronized in the same Streamlit cycle.
    st.rerun()


def _bounding_box_class(resources: AppResources) -> type:
    _ensure_legacy_app_path(resources)
    from terrain_extraction.bbox_utils import BoundingBox

    return BoundingBox


def _ensure_legacy_app_path(resources: AppResources) -> None:
    app_root = str(resources.app_root)
    if app_root not in sys.path:
        sys.path.append(app_root)
