from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from shapely import Polygon


def extract_last_active_drawing(st_folium_data: dict[str, Any]) -> dict[str, Any] | None:
    return st_folium_data.get("last_active_drawing")


def drawing_to_lon_lat_points(drawing: dict[str, Any]) -> list[tuple[float, float]]:
    coordinates = drawing["geometry"]["coordinates"]
    first_ring = coordinates[0]
    return [(float(point[0]), float(point[1])) for point in first_ring]


def drawing_to_bounding_box(drawing: dict[str, Any]) -> Any:
    return _bounding_box_class()(Polygon(drawing_to_lon_lat_points(drawing)))


def _bounding_box_class() -> type:
    _ensure_legacy_app_path()
    from terrain_extraction.bbox_utils import BoundingBox

    return BoundingBox


def _ensure_legacy_app_path() -> None:
    app_root = Path(__file__).resolve().parents[1]
    app_root_str = str(app_root)
    if app_root_str not in sys.path:
        sys.path.append(app_root_str)
