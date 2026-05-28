from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Polygon

APP_DIR = Path(__file__).parents[2] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_industrial_fences_debug_session_exposes_linear_layers() -> None:
    from terrain_extraction.osm_debug_gui import build_debug_session

    session = build_debug_session(
        fixture=Path("test/industrial_fences.geojson"),
        profile="cold_war",
        config=Path("default_osm_config.json"),
        seed=123,
    )

    layer_names = {layer["name"] for layer in session.layer_metadata()}

    assert {"original_osm", "source_features", "routed_paths", "final_rows"} <= layer_names
    assert {"diagnostics", "debug_export_diagnostics"} <= set(session.diagnostics)
    assert session.output_row_count > 0
    assert session.feature_collection("final_rows", limit=5)["features"]
    assert any(
        layer["group"] == "diagnostics" and layer["feature_count"] > 0
        for layer in session.layer_metadata()
    )


def test_buildings_ringmauer_debug_session_exposes_building_outlines() -> None:
    from terrain_extraction.osm_debug_gui import build_debug_session

    session = build_debug_session(
        fixture=Path("test/buildings_ringmauer.geojson"),
        profile="cold_war",
        config=Path("default_osm_config.json"),
        seed=123,
    )

    building_layer = session.feature_collection("building_footprints", limit=20)

    assert building_layer["features"]
    assert session.feature_collection("final_rows", limit=20)["features"]
    selected = session.selection_details(building_layer["features"][0]["id"])
    assert selected["layer"] == "building_footprints"
    assert "diagnostics" in selected["properties"]
    assert selected["related"]["final_rows"]


def test_selection_returns_related_records() -> None:
    from terrain_extraction.osm_debug_gui import build_debug_session

    session = build_debug_session(
        fixture=Path("test/industrial_fences.geojson"),
        profile="cold_war",
        config=Path("default_osm_config.json"),
        seed=123,
    )
    feature = session.feature_collection("final_rows", limit=100)["features"][0]

    selected = session.selection_details(feature["id"])

    assert selected["stable_id"] == feature["id"]
    assert selected["related"]["same_cell"] or selected["related"]["same_feature"]
    assert selected["related"]["final_rows"]


def test_layer_endpoint_filters_empty_geometries_and_bbox() -> None:
    from terrain_extraction.osm_debug_gui import DebugSession

    layer = gpd.GeoDataFrame(
        {"name": ["inside", "outside", "empty"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
            LineString(),
        ],
        crs="EPSG:4326",
    )
    session = DebugSession.from_layers(
        raw_osm={"type": "FeatureCollection", "features": []},
        layers={"final_rows": layer},
        output_rows=[],
        diagnostics={},
    )

    collection = session.feature_collection("final_rows", bbox=(0, 0, 2, 2), limit=20)

    assert [feature["properties"]["name"] for feature in collection["features"]] == ["inside"]


def test_large_grid_layer_is_viewport_filtered() -> None:
    from terrain_extraction.osm_debug_gui import DebugSession

    cells = []
    names = []
    for xidx in range(500):
        names.append(f"cell-{xidx}")
        cells.append(
            Polygon(
                [
                    (xidx * 0.0001, 0.0),
                    (xidx * 0.0001 + 0.00005, 0.0),
                    (xidx * 0.0001 + 0.00005, 0.00005),
                    (xidx * 0.0001, 0.00005),
                ]
            )
        )
    layer = gpd.GeoDataFrame({"name": names}, geometry=cells, crs="EPSG:4326")
    session = DebugSession.from_layers(
        raw_osm={"type": "FeatureCollection", "features": []},
        layers={"final_rows": layer},
        output_rows=[],
        diagnostics={},
    )

    metadata = session.layer_metadata()
    collection = session.feature_collection("final_rows", bbox=(0, 0, 0.001, 0.001), limit=5)

    assert metadata[0]["feature_count"] == 500
    assert len(collection["features"]) == 5
    assert collection["limited"] is True
