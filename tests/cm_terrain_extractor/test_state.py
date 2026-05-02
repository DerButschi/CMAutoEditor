from pathlib import Path

import pandas as pd


def test_clear_bbox_dependent_results_preserves_user_map_and_osm_choices() -> None:
    from cm_terrain_extractor_app.app_core.state import (
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
    assert state.osm_bbox_object is None
    assert state.osm_output is None
    assert state.osm_geometries is None
    assert state.osm_config_file == "default_osm_config_cmcw.json"
    assert state.osm_config == {"roads": {"visualization": {}}}
    assert state.osm_profile == "cold_war"
    assert state.map_mode == "OpenStreetMap"
    assert state.map_center == (51.0, 7.0)
    assert state.map_zoom == 11
