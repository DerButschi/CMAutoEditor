from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from tests.cm_terrain_extractor.osm_extraction.network_recovery_helpers import (  # noqa: E402
    ExtractionTestResult,
    run_osm_extraction_fixture,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "network_recovery"
NETWORK_RECOVERY_FIXTURES = (
    "straight_road_2pt",
    "diagonalish_2pt_equal_length",
    "ninety_degree_bend",
    "four_way_crossing",
    "t_junction",
    "road_near_building",
)


@pytest.mark.parametrize("fixture_name", NETWORK_RECOVERY_FIXTURES)
def test_network_recovery_fixtures_are_valid_feature_collections(fixture_name: str) -> None:
    fixture_path = FIXTURE_DIR / f"{fixture_name}.geojson"

    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)

    assert fixture["type"] == "FeatureCollection"
    assert len(fixture["features"]) > 0


@pytest.mark.parametrize("fixture_name", NETWORK_RECOVERY_FIXTURES)
def test_network_recovery_fixtures_run_through_current_pipeline(fixture_name: str) -> None:
    result = run_osm_extraction_fixture(
        fixture_name,
        profile="cold_war",
        config_name=Path("default_osm_config.json"),
        bbox=None,
        seed=123,
    )

    assert isinstance(result, ExtractionTestResult)
    assert result.output_rows
    assert result.road_graph.cells
    assert result.diagnostics["topology"]["edge_count"] >= 1
    assert "route" in result.diagnostics
    assert "output" in result.diagnostics
    assert result.ascii_grid()


def test_straight_road_fixture_reconstructs_one_connected_road_graph() -> None:
    result = run_osm_extraction_fixture(
        "straight_road_2pt",
        profile="cold_war",
        config_name=Path("default_osm_config.json"),
        bbox=None,
        seed=123,
    )

    assert result.road_component_count == 1
    assert result.illegal_direction_sets == ()
    assert "S" not in result.ascii_grid()


def test_harness_captures_debug_layers_and_structured_symptoms() -> None:
    result = run_osm_extraction_fixture(
        "ninety_degree_bend",
        profile="cold_war",
        config_name=Path("default_osm_config.json"),
        bbox=None,
        seed=123,
        debug=True,
    )

    assert {"source_features", "topology_edges", "routed_paths", "raster_spines", "route_anchors", "final_rows"} <= set(
        result.debug_layers
    )
    assert set(result.diagnostics) >= {"topology", "anchor", "route", "step_cell", "tile", "output"}
    assert result.diagnostics["route"]["successful_routes"] >= 1
    assert result.diagnostics["route"]["raster_spine_count"] >= 1
    assert result.diagnostics["output"]["road_components"] == result.road_component_count


@pytest.mark.xfail(
    reason=(
        "Known M0 red fixture: four-way crossings need later tile-feasible intersection and "
        "persistent linear-state milestones before one legal 4-way road cell is guaranteed."
    ),
    strict=False,
)
def test_four_way_crossing_has_a_single_legal_four_way_intersection() -> None:
    result = run_osm_extraction_fixture(
        "four_way_crossing",
        profile="cold_war",
        config_name=Path("default_osm_config.json"),
        bbox=None,
        seed=123,
    )

    assert result.road_component_count == 1, result.ascii_grid()
    assert result.illegal_direction_sets == ()
    assert result.road_graph.max_degree >= 4, result.ascii_grid()
