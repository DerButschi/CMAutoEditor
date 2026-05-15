from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _full_cardinal_catalog_rows():
    return (
        {"direction": 0, "row": 0, "col": 0, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 1, "col": 0, "u": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 1, "col": 1, "u": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 1, "col": 0, "d": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 1, "col": 1, "d": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 2, "col": 0, "u": (2, 3), "r": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 2, "col": 1, "u": (2, 3), "l": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 2, "col": 0, "l": (2, 3), "u": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 2, "col": 1, "l": (2, 3), "d": (2, 3), "r": (2, 3), "cost": 1.0},
        {
            "direction": 4,
            "row": 2,
            "col": 2,
            "u": (2, 3),
            "r": (2, 3),
            "d": (2, 3),
            "l": (2, 3),
            "cost": 1.0,
        },
        {"direction": 5, "row": 0, "col": 1, "u": (2, 3), "cost": 1.0},
    )


def test_catalog_oracle_enumerates_supported_cardinal_direction_sets() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    catalog = CompiledTileCatalog.from_records(_full_cardinal_catalog_rows(), process=ProcessKind.ROAD)
    supported = (
        {"N", "S"},
        {"E", "W"},
        {"N", "E"},
        {"N", "W"},
        {"S", "E"},
        {"S", "W"},
        {"N", "E", "S"},
        {"N", "S", "W"},
        {"E", "N", "W"},
        {"E", "S", "W"},
        {"E", "N", "S", "W"},
        {"N"},
    )

    for direction_set in supported:
        assert catalog.has_tile(direction_set)
        assert catalog.best_tile(direction_set) is not None


def test_catalog_oracle_reports_missing_unsupported_direction_sets() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    catalog = CompiledTileCatalog.from_records(_full_cardinal_catalog_rows()[:2], process=ProcessKind.ROAD)
    required_sets = ({"N", "S"}, {"E", "W"}, {"N", "E"}, {"E", "N", "S", "W"}, {"NE", "SW"})

    assert catalog.has_tile({"N", "E"}) is False
    assert catalog.best_tile({"E", "N", "S", "W"}) is None
    assert catalog.missing_direction_sets(required_sets) == (
        ("E", "N"),
        ("E", "N", "S", "W"),
        ("NE", "SW"),
    )
    assert catalog.catalog_gap_diagnostics(required_sets) == (
        {"process": "road", "required_directions": ("E", "N"), "failure_reason": "catalog_gap"},
        {"process": "road", "required_directions": ("E", "N", "S", "W"), "failure_reason": "catalog_gap"},
        {"process": "road", "required_directions": ("NE", "SW"), "failure_reason": "catalog_gap"},
    )


def test_allowed_step_dirs_and_can_extend_use_normalized_direction_sets() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    catalog = CompiledTileCatalog.from_records(_full_cardinal_catalog_rows(), process=ProcessKind.ROAD)

    assert catalog.allowed_step_dirs() == frozenset({"E", "N", "S", "W"})
    assert catalog.can_extend({"n"}, "e")
    assert catalog.can_extend({"N", "E"}, "S")
    assert not catalog.can_extend({"N"}, "NE")
    assert not catalog.has_tile({"N", "NE"})


def test_best_tile_uses_assignment_candidate_order() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    catalog = CompiledTileCatalog.from_records(
        (
            {"direction": 1, "row": 2, "col": 2, "r": ("wide",), "l": ("wide",), "cost": 5.0},
            {"direction": 1, "row": 0, "col": 1, "r": ("matched",), "l": ("matched",), "cost": 0.5},
            {"direction": 1, "row": 0, "col": 0, "r": ("narrow",), "l": ("narrow",), "cost": 0.5},
        ),
        process=ProcessKind.ROAD,
    )

    assert catalog.best_tile({"E", "W"}).row == 0
    assert catalog.best_tile({"E", "W"}).col == 0


def test_diagonal_steps_are_only_allowed_when_catalog_has_diagonal_semantics() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    cardinal_catalog = CompiledTileCatalog.from_records(_full_cardinal_catalog_rows(), process=ProcessKind.ROAD)
    diagonal_catalog = CompiledTileCatalog.from_records(
        ({"direction": 0, "row": 0, "col": 0, "ur": (2, 3), "dl": (2, 3), "cost": 1.0},),
        process=ProcessKind.FENCE,
    )

    assert "NE" not in cardinal_catalog.allowed_step_dirs()
    assert not cardinal_catalog.has_tile({"NE", "SW"})
    assert diagonal_catalog.allowed_step_dirs() == frozenset({"NE", "SW"})
    assert diagonal_catalog.has_tile({"NE", "SW"})
