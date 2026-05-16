from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _road_row(xidx: int | float, yidx: int | float, *, cat2: str = "Road Tile 1", direction: str = "Direction 2"):
    return {
        "xidx": xidx,
        "yidx": yidx,
        "z": -1,
        "menu": "Roads",
        "cat1": "Paved 2",
        "cat2": cat2,
        "direction": direction,
        "id": -1,
        "name": "road",
        "priority": 4,
    }


def _building_row(xidx: int, yidx: int):
    return {
        "xidx": xidx,
        "yidx": yidx,
        "z": -1,
        "menu": "Buildings",
        "cat1": "House",
        "cat2": -1,
        "direction": -1,
        "id": -1,
        "name": "residential_building",
        "priority": 1,
    }


def test_validate_road_output_rows_accepts_connected_catalog_backed_roads() -> None:
    from terrain_extraction.osm_extraction.road_output_validation import validate_road_output_rows

    report = validate_road_output_rows(
        (
            _road_row(0, 0),
            _road_row(1, 0),
            _road_row(2, 0),
        ),
        profile="cold_war",
    )

    assert report.is_valid, report.issue_summary()
    assert report.road_cell_count == 3
    assert report.ascii_grid() == "RRR"


def test_validator_reports_illegal_labels_and_cells_without_connection_interpretation() -> None:
    from terrain_extraction.osm_extraction.road_output_validation import validate_road_output_rows

    report = validate_road_output_rows(
        (
            _road_row(0, 0, cat2="Road Tile 999", direction="Direction 99"),
            _road_row(4, 4),
        ),
        profile="cold_war",
    )

    assert [issue.reason for issue in report.illegal_tile_labels] == ["unknown_road_tile_label"]
    assert [issue.reason for issue in report.cells_without_valid_connection_interpretation] == [
        "isolated_road_cell",
    ]
    assert not report.is_valid


def test_validator_reports_one_cell_gaps_and_disconnected_components() -> None:
    from terrain_extraction.osm_extraction.road_output_validation import validate_road_output_rows

    report = validate_road_output_rows(
        (
            _road_row(0, 0),
            _road_row(2, 0),
            _road_row(8, 0),
            _road_row(9, 0),
        ),
        profile="cold_war",
    )

    assert [issue.reason for issue in report.one_cell_junction_gaps] == ["one_cell_gap"]
    assert len(report.disconnected_components) == 3
    assert not report.is_valid
    assert "G" in report.ascii_grid()


def test_validator_reports_unsupported_diagonal_duplicate_intersection_and_overlap_issues() -> None:
    from terrain_extraction.osm_extraction.road_output_validation import validate_road_output_rows

    report = validate_road_output_rows(
        (
            _road_row(0.25, 0.25),
            _road_row(0, 0),
            _road_row(0, 0),
            _road_row(1, 0),
            _road_row(0, 1, direction="Direction 1"),
            _building_row(1, 0),
        ),
        profile="cold_war",
    )

    assert [issue.reason for issue in report.unsupported_diagonal_continuations] == [
        "non_integer_road_coordinate",
    ]
    assert [issue.reason for issue in report.duplicate_mutually_exclusive_cells] == ["duplicate_road_cell"]
    assert [issue.reason for issue in report.invalid_intersections] == ["unrepresented_adjacent_road"]
    assert [issue.reason for issue in report.road_building_overlaps] == ["road_building_overlap"]
    assert not report.is_valid
