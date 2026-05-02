from __future__ import annotations

from typing import Any

from cm_terrain_extractor_app.app_core.state import (
    AppState,
    clear_bbox_dependent_results,
)

MAX_LEN_X_METERS = 4160
MAX_LEN_Y_METERS = 4160
MAX_SELECTED_AREA_SQUARE_METERS = 18_000_000


def is_selected_area_valid(len_x: float, len_y: float) -> bool:
    return (
        len_x <= MAX_LEN_X_METERS
        and len_y <= MAX_LEN_Y_METERS
        and len_x * len_y <= MAX_SELECTED_AREA_SQUARE_METERS
    )


def compute_bbox_metrics(bbox_object: Any) -> dict[str, float]:
    len_x = bbox_object.get_length_xaxis()
    len_y = bbox_object.get_length_yaxis()
    return {
        "len_x": len_x,
        "len_y": len_y,
        "area": len_x * len_y,
    }


def update_state_from_bbox(state: AppState, bbox_object: Any) -> None:
    metrics = compute_bbox_metrics(bbox_object)
    state.bbox_coordinates = bbox_object.get_coordinates(xy=False)
    state.bbox_object = bbox_object
    state.projected_bbox_object = bbox_object.get_box(bbox_object.crs_projected)
    state.len_x = metrics["len_x"]
    state.len_y = metrics["len_y"]
    state.selected_area_valid = is_selected_area_valid(state.len_x, state.len_y)
    state.bbox_origin = 0
    clear_bbox_dependent_results(state)
