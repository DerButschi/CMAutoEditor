from __future__ import annotations

from pathlib import Path

import pandas as pd
from shapely import Polygon

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.app_core.state import AppState


class FakeBbox:
    def __init__(self, coordinates: list[tuple[float, float]]) -> None:
        self.coordinates = coordinates

    def get_coordinates(self, xy: bool = True) -> list[tuple[float, float]]:
        return self.coordinates


def test_build_folium_map_adds_opentopomap_controls_and_draw_tool() -> None:
    from cm_terrain_extractor_app.map_view.folium_map import build_folium_map

    map_obj = build_folium_map(
        state=AppState(map_mode="Bounding Box Selection"),
        resources=_resources(),
    )
    rendered = map_obj.get_root().render()

    assert "tile.opentopomap.org" in rendered
    assert "OpenTopoMap" in rendered
    assert "L.Control.Draw" in rendered
    assert "L.Control.geocoder" in rendered
    assert "L.Control.Measure" in rendered
    assert "L.control.fullscreen" in rendered


def test_build_folium_map_skips_draw_tool_outside_bbox_selection() -> None:
    from cm_terrain_extractor_app.map_view.folium_map import build_folium_map

    map_obj = build_folium_map(state=AppState(map_mode="Elevations"), resources=_resources())

    assert "L.Control.Draw" not in map_obj.get_root().render()


def test_bbox_and_elevation_layers_preserve_visual_markers() -> None:
    from cm_terrain_extractor_app.map_view.folium_map import build_folium_map

    state = AppState(
        bbox_coordinates=[
            (51.20, 7.10),
            (51.20, 7.30),
            (51.35, 7.30),
            (51.35, 7.10),
            (51.20, 7.10),
        ],
        elevation_in_bbox=pd.DataFrame({"height": [1]}),
        height_map_png=Path("data_cache") / "current_height_map.png",
    )

    rendered = build_folium_map(state=state, resources=_resources()).get_root().render()

    assert "CM W\\u2194E axis" in rendered
    assert "CM S\\u2194N axis" in rendered
    assert "imageOverlay" in rendered
    assert '"opacity": 0.9' in rendered


def test_osm_layers_use_dashed_bbox_and_priority_ordering() -> None:
    from cm_terrain_extractor_app.map_view.folium_map import build_folium_map

    state = AppState(
        osm_bbox_object=FakeBbox(
            [
                (51.20, 7.10),
                (51.20, 7.30),
                (51.35, 7.30),
                (51.35, 7.10),
                (51.20, 7.10),
            ]
        ),
        osm_config={
            "low": {"priority": 1, "visualization": {"color": "#0000ff", "opacity": 0.25}},
            "high": {"priority": 5, "visualization": {"color": "#ff0000", "opacity": 0.75}},
        },
        osm_geometries={
            "low": [Polygon([(7.10, 51.20), (7.11, 51.20), (7.11, 51.21), (7.10, 51.20)])],
            "high": [Polygon([(7.20, 51.30), (7.21, 51.30), (7.21, 51.31), (7.20, 51.30)])],
        },
    )

    rendered = build_folium_map(state=state, resources=_resources()).get_root().render()

    assert "OSM data" in rendered
    assert "dashArray" in rendered
    assert rendered.index("#ff0000") < rendered.index("#0000ff")


def _resources() -> AppResources:
    app_root = Path("cm_terrain_extractor_app")
    return AppResources(
        app_root=app_root,
        executable_root=Path("."),
        data_cache_path=Path("data_cache"),
        config_dir=Path("."),
        dll_dir=app_root / "dll",
    )
