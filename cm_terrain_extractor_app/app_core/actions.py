from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geojson
import pandas as pd


def find_data_sources_in_bbox(*, bbox: object, selectable_sources: list[Any]) -> list[Any]:
    return [
        data_source
        for data_source in selectable_sources
        if data_source.intersects_bounding_box(bbox)
    ]


def extract_elevation_data(
    *,
    data_source: Any,
    bbox: object,
    data_cache_path: Path,
) -> tuple[pd.DataFrame, Path | None]:
    data_cache_path = Path(data_cache_path)
    data_cache_path.mkdir(parents=True, exist_ok=True)
    elevation_data = data_source.get_data(bbox, data_cache_path)
    png_path = data_source.get_png(bbox, data_cache_path)
    return elevation_data, Path(png_path) if png_path is not None else None


def load_osm_config(*, config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def load_osm_data_from_uploaded_bytes(*, data: bytes, filename: str | None = None) -> dict:
    return geojson.loads(data.decode("utf-8-sig"))


def download_osm_data(*, bbox: object, config: dict) -> dict:
    osm_data = _features_from_polygon(bbox.box_wgs84, _build_osmnx_tag_dict(config))
    for column in ("ways", "nodes"):
        if column in osm_data.columns:
            osm_data = osm_data.drop(columns=[column])
    return geojson.loads(osm_data.to_json())


def process_osm_data(
    *,
    osm_data: dict,
    bbox: object,
    config_path: Path,
    profile: str,
) -> tuple[pd.DataFrame, dict, dict]:
    osm_processor = _create_osm_processor(
        path_to_config=str(config_path),
        bbox=bbox,
        profile=profile,
    )
    osm_processor.preprocess_osm_data(osm_data=osm_data)
    osm_processor.run_processors()
    osm_processor.post_process()
    return (
        osm_processor.get_output(),
        osm_processor.get_geometries(),
        osm_processor.get_debug_layers(),
    )


def _build_osmnx_tag_dict(config: dict) -> dict[str, list[Any]]:
    tag_dict: dict[str, list[Any]] = {}
    for config_entry in config.values():
        if not config_entry.get("active", True):
            continue
        for key, value in config_entry.get("tags", []):
            values = tag_dict.setdefault(key, [])
            if value not in values:
                values.append(value)
    return tag_dict


def _features_from_polygon(polygon: object, tags: dict[str, list[Any]]) -> Any:
    import osmnx

    return osmnx.features_from_polygon(polygon, tags)


def _create_osm_processor(*, profile: str, bbox: object, path_to_config: str) -> Any:
    from terrain_extraction.osm_processor import OSMProcessor

    return OSMProcessor(
        path_to_config=path_to_config,
        bbox=bbox,
        profile=profile,
    )
