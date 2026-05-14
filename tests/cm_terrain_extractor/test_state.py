from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_app_state_defaults_match_contract() -> None:
    from cm_terrain_extractor_app.app_core.state import AppState

    state = AppState()

    assert state.map_mode == "Bounding Box Selection"
    assert state.bbox_object is None
    assert state.bbox_coordinates is None
    assert state.projected_bbox_object is None
    assert state.len_x is None
    assert state.len_y is None
    assert state.selected_area_valid is False
    assert state.bbox_origin == 0
    assert state.map_center == (0.0, 0.0)
    assert state.map_zoom == 2
    assert state.map_key == 0
    assert state.selectable_data_sources == []
    assert state.available_data_sources == []
    assert state.selected_data_source is None
    assert state.elevation_in_bbox is None
    assert state.height_map_png is None
    assert state.osm_config_file is None
    assert state.osm_config is None
    assert state.osm_profile == "cold_war"
    assert state.osm_data is None
    assert state.osm_data_source is None
    assert state.osm_uploaded_file_signature is None
    assert state.osm_bbox_object is None
    assert state.osm_output is None
    assert state.osm_geometries is None
    assert state.currently_processing_data is None


def test_clear_bbox_dependent_results_preserves_user_map_and_osm_choices() -> None:
    from cm_terrain_extractor_app.app_core.state import (
        OSM_DATA_SOURCE_DOWNLOADED,
        AppState,
        clear_bbox_dependent_results,
    )

    state = AppState(
        available_data_sources=["source-a"],
        selected_data_source="source-a",
        elevation_in_bbox=pd.DataFrame({"height": [1]}),
        height_map_png=Path("height.png"),
        osm_config_file="default_osm_config_cmcw.json",
        osm_config={"roads": {"visualization": {}}},
        osm_profile="cold_war",
        osm_data={"type": "FeatureCollection", "features": []},
        osm_data_source=OSM_DATA_SOURCE_DOWNLOADED,
        osm_uploaded_file_signature="old-upload",
        osm_bbox_object=object(),
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
        map_mode="OpenStreetMap",
        map_center=(51.0, 7.0),
        map_zoom=11,
    )

    clear_bbox_dependent_results(state)

    assert state.available_data_sources == []
    assert state.selected_data_source is None
    assert state.elevation_in_bbox is None
    assert state.height_map_png is None
    assert state.osm_data is None
    assert state.osm_data_source is None
    assert state.osm_uploaded_file_signature is None
    assert state.osm_bbox_object is None
    assert state.osm_output is None
    assert state.osm_geometries is None
    assert state.osm_config_file == "default_osm_config_cmcw.json"
    assert state.osm_config == {"roads": {"visualization": {}}}
    assert state.osm_profile == "cold_war"
    assert state.map_mode == "OpenStreetMap"
    assert state.map_center == (51.0, 7.0)
    assert state.map_zoom == 11


def test_clear_bbox_dependent_results_preserves_uploaded_osm_data() -> None:
    from cm_terrain_extractor_app.app_core.state import (
        OSM_DATA_SOURCE_UPLOADED,
        AppState,
        clear_bbox_dependent_results,
    )

    osm_data = {"type": "FeatureCollection", "features": []}
    osm_bbox = object()
    state = AppState(
        available_data_sources=["source-a"],
        selected_data_source="source-a",
        elevation_in_bbox=pd.DataFrame({"height": [1]}),
        height_map_png=Path("height.png"),
        osm_data=osm_data,
        osm_data_source=OSM_DATA_SOURCE_UPLOADED,
        osm_uploaded_file_signature="upload-a",
        osm_bbox_object=osm_bbox,
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
    )

    clear_bbox_dependent_results(state)

    assert state.available_data_sources == []
    assert state.selected_data_source is None
    assert state.elevation_in_bbox is None
    assert state.height_map_png is None
    assert state.osm_data is osm_data
    assert state.osm_data_source == OSM_DATA_SOURCE_UPLOADED
    assert state.osm_uploaded_file_signature == "upload-a"
    assert state.osm_bbox_object is osm_bbox
    assert state.osm_output is None
    assert state.osm_geometries is None


def test_clear_elevation_result_invalidates_selected_source_output_only() -> None:
    from cm_terrain_extractor_app.app_core.state import (
        AppState,
        clear_elevation_result,
    )

    state = AppState(
        selected_data_source="source-a",
        elevation_in_bbox=pd.DataFrame({"height": [1]}),
        height_map_png=Path("height.png"),
        map_key=4,
        osm_output=pd.DataFrame({"category": ["road"]}),
    )

    clear_elevation_result(state)

    assert state.selected_data_source == "source-a"
    assert state.elevation_in_bbox is None
    assert state.height_map_png is None
    assert state.map_key == 5
    assert state.osm_output is not None


def test_clear_osm_processing_result_invalidates_config_profile_outputs_only() -> None:
    from cm_terrain_extractor_app.app_core.state import (
        AppState,
        clear_osm_processing_result,
    )

    state = AppState(
        osm_config_file="default_osm_config_cmcw.json",
        osm_config={"roads": {"visualization": {}}},
        osm_profile="cold_war",
        osm_data={"type": "FeatureCollection", "features": []},
        osm_data_source="uploaded",
        osm_uploaded_file_signature="upload-a",
        osm_bbox_object=object(),
        osm_output=pd.DataFrame({"category": ["road"]}),
        osm_geometries={"roads": []},
        elevation_in_bbox=pd.DataFrame({"height": [1]}),
    )

    clear_osm_processing_result(state)

    assert state.osm_config_file == "default_osm_config_cmcw.json"
    assert state.osm_config == {"roads": {"visualization": {}}}
    assert state.osm_profile == "cold_war"
    assert state.osm_data == {"type": "FeatureCollection", "features": []}
    assert state.osm_data_source == "uploaded"
    assert state.osm_uploaded_file_signature == "upload-a"
    assert state.osm_bbox_object is not None
    assert state.osm_output is None
    assert state.osm_geometries is None
    assert state.elevation_in_bbox is not None


def test_mark_map_dirty_increments_key() -> None:
    from cm_terrain_extractor_app.app_core.state import AppState, mark_map_dirty

    state = AppState(map_key=2)

    mark_map_dirty(state)

    assert state.map_key == 3


def test_session_adapter_get_and_reset_state(monkeypatch) -> None:
    from cm_terrain_extractor_app.app_core.resources import AppResources
    from cm_terrain_extractor_app.app_core.state import AppState
    from cm_terrain_extractor_app.streamlit_ui import session_adapter

    fake_session_state = {}
    monkeypatch.setattr(
        session_adapter,
        "st",
        SimpleNamespace(session_state=fake_session_state),
    )
    resources = AppResources(
        app_root=Path("app"),
        executable_root=Path("exe"),
        data_cache_path=Path("cache"),
        config_dir=Path("configs"),
        dll_dir=Path("dll"),
    )

    state = session_adapter.get_state(resources)

    assert state == AppState()
    assert fake_session_state == {session_adapter.STATE_KEY: state}

    state.map_mode = "Elevations"
    assert session_adapter.get_state(resources).map_mode == "Elevations"

    reset_state = session_adapter.reset_state(resources)

    assert reset_state == AppState()
    assert reset_state is fake_session_state[session_adapter.STATE_KEY]
