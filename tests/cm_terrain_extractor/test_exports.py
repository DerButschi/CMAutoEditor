import pandas as pd


def test_dataframe_to_csv_bytes_matches_current_streamlit_export() -> None:
    from cm_terrain_extractor_app.app_core.exports import dataframe_to_csv_bytes

    df = pd.DataFrame(
        {
            "easting": [100.0, 101.5],
            "northing": [200.25, 201.75],
            "height": [12, 14],
        }
    )

    assert dataframe_to_csv_bytes(df) == df.to_csv().encode("utf-8")


def test_download_filename_helpers_match_current_streamlit_downloads() -> None:
    from cm_terrain_extractor_app.app_core.exports import (
        suggest_elevation_filename,
        suggest_osm_filename,
    )
    from cm_terrain_extractor_app.app_core.state import AppState

    state = AppState()

    assert suggest_elevation_filename(state) == "elevation_data.csv"
    assert suggest_osm_filename(state) == "osm_data.csv"
