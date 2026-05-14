from pathlib import Path

import pandas as pd


class FakeBBox:
    crs_projected = "EPSG:3857"

    def __init__(self, *, len_x: float, len_y: float) -> None:
        self.len_x = len_x
        self.len_y = len_y
        self.coordinates = [(7.0, 51.0), (7.1, 51.0), (7.1, 51.1), (7.0, 51.1)]
        self.projected_box = object()

    def get_coordinates(self, *, xy: bool) -> list[tuple[float, float]]:
        assert xy is False
        return self.coordinates

    def get_box(self, crs: str) -> object:
        assert crs == self.crs_projected
        return self.projected_box

    def get_length_xaxis(self) -> float:
        return self.len_x

    def get_length_yaxis(self) -> float:
        return self.len_y


def test_is_selected_area_valid_covers_contractual_limits() -> None:
    from cm_terrain_extractor_app.app_core.validation import is_selected_area_valid

    assert is_selected_area_valid(4160, 4160) is True
    assert is_selected_area_valid(4160.1, 1000) is False
    assert is_selected_area_valid(1000, 4160.1) is False
    assert is_selected_area_valid(4500, 4000) is False


def test_compute_bbox_metrics_uses_bounding_box_methods() -> None:
    from cm_terrain_extractor_app.app_core.validation import compute_bbox_metrics

    metrics = compute_bbox_metrics(FakeBBox(len_x=1200.5, len_y=3000.0))

    assert metrics == {
        "len_x": 1200.5,
        "len_y": 3000.0,
        "area": 3_601_500.0,
    }


def test_update_state_from_bbox_sets_metrics_and_clears_bbox_dependent_results() -> None:
    from cm_terrain_extractor_app.app_core.state import OSM_DATA_SOURCE_DOWNLOADED, AppState
    from cm_terrain_extractor_app.app_core.validation import update_state_from_bbox

    bbox = FakeBBox(len_x=1200, len_y=1000)
    state = AppState(
        bbox_origin=3,
        available_data_sources=["source-a"],
        selected_data_source="source-a",
        elevation_in_bbox=pd.DataFrame({"height": [1]}),
        height_map_png=Path("height.png"),
        osm_config_file="default_osm_config_cmcw.json",
        osm_profile="cold_war",
        osm_data={"type": "FeatureCollection", "features": []},
        osm_data_source=OSM_DATA_SOURCE_DOWNLOADED,
        osm_bbox_object=object(),
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
    )

    update_state_from_bbox(state, bbox)

    assert state.bbox_object is bbox
    assert state.bbox_coordinates == bbox.coordinates
    assert state.projected_bbox_object is bbox.projected_box
    assert state.len_x == 1200
    assert state.len_y == 1000
    assert state.selected_area_valid is True
    assert state.bbox_origin == 0
    assert state.available_data_sources == []
    assert state.selected_data_source is None
    assert state.elevation_in_bbox is None
    assert state.height_map_png is None
    assert state.osm_data is None
    assert state.osm_data_source is None
    assert state.osm_bbox_object is None
    assert state.osm_output is None
    assert state.osm_geometries is None
    assert state.osm_config_file == "default_osm_config_cmcw.json"
    assert state.osm_profile == "cold_war"


def test_update_state_from_bbox_marks_invalid_area() -> None:
    from cm_terrain_extractor_app.app_core.state import AppState
    from cm_terrain_extractor_app.app_core.validation import update_state_from_bbox

    state = AppState()

    update_state_from_bbox(state, FakeBBox(len_x=4160, len_y=4400))

    assert state.selected_area_valid is False


def test_update_state_from_bbox_preserves_uploaded_osm_data() -> None:
    from cm_terrain_extractor_app.app_core.state import OSM_DATA_SOURCE_UPLOADED, AppState
    from cm_terrain_extractor_app.app_core.validation import update_state_from_bbox

    osm_data = {"type": "FeatureCollection", "features": []}
    osm_bbox = object()
    state = AppState(
        osm_data=osm_data,
        osm_data_source=OSM_DATA_SOURCE_UPLOADED,
        osm_uploaded_file_signature="upload-a",
        osm_bbox_object=osm_bbox,
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
    )
    bbox = FakeBBox(len_x=1200, len_y=1000)

    update_state_from_bbox(state, bbox)

    assert state.bbox_object is bbox
    assert state.osm_data is osm_data
    assert state.osm_data_source == OSM_DATA_SOURCE_UPLOADED
    assert state.osm_uploaded_file_signature == "upload-a"
    assert state.osm_bbox_object is osm_bbox
    assert state.osm_output is None
    assert state.osm_geometries is None


def test_update_state_from_uploaded_osm_data_uses_geojson_bbox_when_no_bbox_selected() -> None:
    from cm_terrain_extractor_app.app_core.state import OSM_DATA_SOURCE_UPLOADED, AppState
    from cm_terrain_extractor_app.app_core.validation import update_state_from_uploaded_osm_data

    state = AppState()
    osm_data = {"type": "FeatureCollection", "features": []}
    osm_bbox = FakeBBox(len_x=1200, len_y=1000)

    update_state_from_uploaded_osm_data(
        state,
        osm_data=osm_data,
        osm_bbox_object=osm_bbox,
        upload_signature="upload-a",
    )

    assert state.bbox_object is osm_bbox
    assert state.selected_area_valid is True
    assert state.osm_data is osm_data
    assert state.osm_data_source == OSM_DATA_SOURCE_UPLOADED
    assert state.osm_uploaded_file_signature == "upload-a"
    assert state.osm_bbox_object is osm_bbox


def test_update_state_from_uploaded_osm_data_keeps_existing_bbox_selected() -> None:
    from cm_terrain_extractor_app.app_core.state import OSM_DATA_SOURCE_UPLOADED, AppState
    from cm_terrain_extractor_app.app_core.validation import update_state_from_uploaded_osm_data

    selected_bbox = FakeBBox(len_x=900, len_y=800)
    osm_bbox = FakeBBox(len_x=1200, len_y=1000)
    state = AppState(
        bbox_object=selected_bbox,
        selected_area_valid=True,
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
    )
    osm_data = {"type": "FeatureCollection", "features": []}

    update_state_from_uploaded_osm_data(
        state,
        osm_data=osm_data,
        osm_bbox_object=osm_bbox,
        upload_signature="upload-b",
    )

    assert state.bbox_object is selected_bbox
    assert state.osm_data is osm_data
    assert state.osm_data_source == OSM_DATA_SOURCE_UPLOADED
    assert state.osm_uploaded_file_signature == "upload-b"
    assert state.osm_bbox_object is osm_bbox
    assert state.osm_output is None
    assert state.osm_geometries is None
