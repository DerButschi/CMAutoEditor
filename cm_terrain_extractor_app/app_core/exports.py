from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from cm_terrain_extractor_app.app_core.state import AppState


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def suggest_elevation_filename(_state: AppState) -> str:
    return "elevation_data.csv"


def suggest_osm_filename(_state: AppState) -> str:
    return "osm_data.csv"
