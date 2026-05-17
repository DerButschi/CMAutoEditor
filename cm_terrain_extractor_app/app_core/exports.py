from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from cm_terrain_extractor_app.app_core.state import AppState


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def parse_bbox_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(BytesIO(data))
    except Exception as exc:
        raise ValueError("Could not read the bounding box CSV file.") from exc

    if not {"x", "y"}.issubset(dataframe.columns):
        raise ValueError("The bounding box CSV file must contain x and y columns.")

    bbox_dataframe = dataframe.loc[:, ["x", "y"]].copy()
    bbox_dataframe["x"] = pd.to_numeric(bbox_dataframe["x"], errors="coerce")
    bbox_dataframe["y"] = pd.to_numeric(bbox_dataframe["y"], errors="coerce")

    if len(bbox_dataframe) != 4:
        raise ValueError("The bounding box CSV file must contain exactly four corner rows.")
    if bbox_dataframe.isnull().any().any():
        raise ValueError("The bounding box CSV file must contain numeric x and y values.")

    return bbox_dataframe


def suggest_elevation_filename(_state: AppState) -> str:
    return "elevation_data.csv"


def suggest_osm_filename(_state: AppState) -> str:
    return "osm_data.csv"
