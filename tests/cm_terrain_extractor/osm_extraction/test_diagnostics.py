from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from shapely.geometry import Point

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_json_safe_sidecar_serializes_shapely_geometry() -> None:
    from terrain_extraction.osm_extraction.diagnostics import write_diagnostics_sidecar

    test_dir = Path(".tmp") / "diagnostics_tests" / uuid.uuid4().hex
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_path = test_dir / "output.csv"
        sidecar_path = write_diagnostics_sidecar(
            output_path,
            {"geometry": Point(1, 2), "cells": {(1, 2)}},
        )

        assert sidecar_path == test_dir / "output_diagnostics.json"
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["geometry"] == "POINT (1 2)"
        assert payload["cells"] == [[1, 2]]
    finally:
        for path in test_dir.glob("*"):
            path.unlink()
        test_dir.rmdir()


def test_benchmark_timing_table_formats_stage_rows() -> None:
    from terrain_extraction.osm_extraction_benchmark import _format_timing_table

    table = _format_timing_table(
        {
            "timings": {
                "total": 0.1,
                "routing": {"elapsed_s": 0.02, "elapsed_ms": 20.0},
                "tile_assignment": {"elapsed_s": 0.03, "elapsed_ms": 30.0},
                "building_fitting": {"elapsed_s": 0.01, "elapsed_ms": 10.0},
            },
            "diagnostics": {
                "extraction": {
                    "diagnostics": {
                        "routing_diagnostics": {
                            "route_count": 2,
                            "total_a_star_expansions": 10,
                            "total_tile_feasible_rejections": 1,
                        },
                        "tile_assignment_diagnostics": {"state_component_diagnostics": ({}, {})},
                        "building_fitting_diagnostics": {
                            "building_feature_count": 3,
                            "total_candidates_scored": 7,
                        },
                    }
                }
            },
        }
    )

    assert "routing" in table
    assert "routes=2 expansions=10 tile_rejects=1" in table
    assert "tile_assignment" in table
    assert "components=2" in table
    assert "buildings=3 candidates=7" in table
