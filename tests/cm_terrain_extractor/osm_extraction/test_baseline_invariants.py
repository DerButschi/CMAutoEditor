from __future__ import annotations

import json
from pathlib import Path

import pytest

from cm_terrain_extractor_app.terrain_extraction.osm_extraction_benchmark import (
    FIXTURE_DIR,
    run_fixture,
)

FIXTURE_NAMES = [
    "simple_area",
    "single_road",
    "crossroads",
    "road_near_building",
    "village_cluster",
    "multi_network",
]


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_baseline_fixtures_are_valid_feature_collections(fixture_name: str) -> None:
    fixture_path = FIXTURE_DIR / f"{fixture_name}.geojson"

    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)

    assert fixture["type"] == "FeatureCollection"
    assert len(fixture["features"]) > 0


@pytest.mark.parametrize("fixture_name", ["simple_area", "single_road", "crossroads"])
def test_fixture_output_rows_stay_inside_map(fixture_name: str) -> None:
    result = run_fixture(fixture_name, profile="cold_war", config=Path("default_osm_config.json"), seed=123)

    assert result.stats["quality"]["rows_outside_map"] == 0


def test_post_process_leaves_no_duplicate_mutually_exclusive_cell_rows() -> None:
    result = run_fixture("road_near_building", profile="cold_war", config=Path("default_osm_config.json"), seed=123)

    assert result.stats["quality"]["duplicate_cells_after_post_process"] == 0


def test_crossroads_linear_cells_are_connected() -> None:
    result = run_fixture("crossroads", profile="cold_war", config=Path("default_osm_config.json"), seed=123)

    assert result.stats["quality"]["linear_connected_components"] == 1


def test_road_building_collision_is_prevented_or_reported() -> None:
    result = run_fixture("road_near_building", profile="cold_war", config=Path("default_osm_config.json"), seed=123)
    quality = result.stats["quality"]

    assert quality["building_linear_collisions"] == 0 or quality["reported_collision_cells"] > 0


def test_same_seed_produces_stable_fixture_output() -> None:
    first = run_fixture("village_cluster", profile="cold_war", config=Path("default_osm_config.json"), seed=123)
    second = run_fixture("village_cluster", profile="cold_war", config=Path("default_osm_config.json"), seed=123)

    assert first.output_rows == second.output_rows
