from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class FakeDataSource:
    def __init__(self, *, name: str, intersects: bool) -> None:
        self.name = name
        self.intersects = intersects
        self.data_calls = []
        self.png_calls = []

    def intersects_bounding_box(self, bbox: object) -> bool:
        self.checked_bbox = bbox
        return self.intersects

    def get_data(self, bbox: object, cache_path: Path) -> pd.DataFrame:
        self.data_calls.append((bbox, cache_path))
        return pd.DataFrame({"height": [12, 13]})

    def get_png(self, bbox: object, cache_path: Path) -> Path:
        self.png_calls.append((bbox, cache_path))
        return cache_path / "current_height_map.png"


def test_find_data_sources_in_bbox_uses_intersection_contract() -> None:
    from cm_terrain_extractor_app.app_core.actions import find_data_sources_in_bbox

    bbox = object()
    inside = FakeDataSource(name="inside", intersects=True)
    outside = FakeDataSource(name="outside", intersects=False)

    assert find_data_sources_in_bbox(bbox=bbox, selectable_sources=[inside, outside]) == [inside]
    assert inside.checked_bbox is bbox
    assert outside.checked_bbox is bbox


def test_extract_elevation_data_calls_source_and_png() -> None:
    from cm_terrain_extractor_app.app_core.actions import extract_elevation_data

    bbox = object()
    data_cache_path = Path(".")
    source = FakeDataSource(name="source", intersects=True)

    data, png_path = extract_elevation_data(
        data_source=source,
        bbox=bbox,
        data_cache_path=data_cache_path,
    )

    pd.testing.assert_frame_equal(data, pd.DataFrame({"height": [12, 13]}))
    assert png_path == data_cache_path / "current_height_map.png"
    assert source.data_calls == [(bbox, data_cache_path)]
    assert source.png_calls == [(bbox, data_cache_path)]


def test_load_osm_config_reads_json() -> None:
    from cm_terrain_extractor_app.app_core.actions import load_osm_config

    config_path = Path(__file__).parent / "fixtures" / "osm_config_smoke.json"

    assert load_osm_config(config_path=config_path) == {
        "roads": {"tags": [["highway", "residential"]]}
    }


def test_load_osm_data_from_uploaded_bytes_reads_geojson() -> None:
    from cm_terrain_extractor_app.app_core.actions import load_osm_data_from_uploaded_bytes

    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": None}],
    }

    assert load_osm_data_from_uploaded_bytes(
        data=json.dumps(payload).encode("utf-8"),
        filename="fixture.geojson",
    ) == payload


def test_download_osm_data_uses_active_matching_osmnx_tags_and_drops_legacy_columns(monkeypatch) -> None:
    from cm_terrain_extractor_app.app_core import actions

    class FakeBBox:
        box_wgs84 = "polygon"

    class FakeOsmFrame:
        columns = ["ways", "nodes", "geometry"]

        def __init__(self) -> None:
            self.dropped = []

        def drop(self, *, columns: list[str]) -> FakeOsmFrame:
            self.dropped.extend(columns)
            self.columns = [column for column in self.columns if column not in columns]
            return self

        def to_json(self) -> str:
            return json.dumps({"type": "FeatureCollection", "features": []})

    fake_frame = FakeOsmFrame()

    def fake_features_from_polygon(polygon: object, tags: dict[str, list[str]]) -> FakeOsmFrame:
        assert polygon == "polygon"
        assert tags == {
            "highway": ["residential"],
        }
        return fake_frame

    monkeypatch.setattr(actions, "_features_from_polygon", fake_features_from_polygon)

    assert actions.download_osm_data(
        bbox=FakeBBox(),
        config={
            "roads": {
                "tags": [["highway", "residential"]],
                "exclude_tags": [["surface", "paved"]],
                "required_tags": [["access", "private"]],
            },
            "inactive": {
                "active": False,
                "tags": [["landuse", "forest"]],
            },
        },
    ) == {"type": "FeatureCollection", "features": []}
    assert fake_frame.dropped == ["ways", "nodes"]


def test_process_osm_data_runs_processor_stages(monkeypatch) -> None:
    from cm_terrain_extractor_app.app_core import actions

    calls = []
    output = pd.DataFrame({"terrain": ["road"]})
    geometries = {"roads": ["line"]}

    class FakeProcessor:
        def __init__(self, *, profile: str, bbox: object, path_to_config: str) -> None:
            calls.append(("init", profile, bbox, path_to_config))

        def preprocess_osm_data(self, *, osm_data: dict) -> None:
            calls.append(("preprocess", osm_data))

        def run_processors(self) -> None:
            calls.append(("run",))

        def post_process(self) -> None:
            calls.append(("post",))

        def get_output(self) -> pd.DataFrame:
            return output

        def get_geometries(self) -> dict:
            return geometries

    monkeypatch.setattr(actions, "_create_osm_processor", FakeProcessor)

    bbox = object()
    osm_data = {"type": "FeatureCollection", "features": []}
    result_output, result_geometries = actions.process_osm_data(
        osm_data=osm_data,
        bbox=bbox,
        config_path=Path("config.json"),
        profile="cold_war",
    )

    assert result_output is output
    assert result_geometries is geometries
    assert calls == [
        ("init", "cold_war", bbox, "config.json"),
        ("preprocess", osm_data),
        ("run",),
        ("post",),
    ]
