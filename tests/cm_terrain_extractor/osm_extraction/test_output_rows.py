from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _placement(
    *,
    layer,
    cell,
    config_name: str,
    priority: int,
    menu: str,
    cat1: str,
    cat2=None,
    direction=None,
    feature_id=None,
    tile_id=None,
    grid_kind=None,
):
    from terrain_extraction.osm_extraction.models import CMType, GridKind, PlacementRecord

    return PlacementRecord(
        layer=layer,
        grid_kind=grid_kind or GridKind.NORMAL,
        cells=(cell,),
        config_name=config_name,
        feature_id=feature_id,
        priority=priority,
        cm_type=CMType(menu=menu, cat1=cat1, cat2=cat2, direction=direction, tile_id=tile_id),
        score=1.0,
    )


def test_placements_to_output_rows_are_layered_stable_and_skip_shadowed_defaults() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    placements = (
        _placement(
            layer=LayerKind.GROUND,
            cell=GridCell(1, 1),
            config_name="default_ground",
            priority=-999,
            menu="Ground",
            cat1="Grass",
        ),
        _placement(
            layer=LayerKind.LINEAR_SURFACE,
            cell=GridCell(0, 0),
            config_name="road",
            priority=4,
            menu="Roads",
            cat1="Dirt",
            cat2="Road Tile 1",
            direction="Direction 2",
            feature_id="road-1",
        ),
        _placement(
            layer=LayerKind.GROUND,
            cell=GridCell(1, 1),
            config_name="field",
            priority=3,
            menu="Ground",
            cat1="Wheat",
            feature_id="field-1",
        ),
    )

    rows = placements_to_output_rows(placements)

    assert rows == (
        {
            "xidx": 0,
            "yidx": 0,
            "z": -1,
            "menu": "Roads",
            "cat1": "Dirt",
            "cat2": "Road Tile 1",
            "direction": "Direction 2",
            "id": -1,
            "name": "road",
            "priority": 4,
        },
        {
            "xidx": 1,
            "yidx": 1,
            "z": -1,
            "menu": "Ground",
            "cat1": "Wheat",
            "cat2": -1,
            "direction": -1,
            "id": -1,
            "name": "field",
            "priority": 3,
        },
    )


def test_extent_marker_and_coordinate_normalization_are_explicit() -> None:
    from terrain_extraction.osm_extraction.output_rows import (
        append_extent_marker,
        clip_output_rows_to_bounds,
        normalize_output_coordinates,
    )

    rows = (
        {"xidx": 10, "yidx": 20, "z": -1, "menu": "Ground", "cat1": "Grass", "cat2": -1, "direction": -1, "id": -1, "name": "field", "priority": 1},
        {"xidx": 9.75, "yidx": 20, "z": -1, "menu": "Ground", "cat1": "Grass", "cat2": -1, "direction": -1, "id": -1, "name": "outside", "priority": 1},
    )

    clipped = clip_output_rows_to_bounds(append_extent_marker(rows, bounds=(10, 20, 12, 22)), bounds=(10, 20, 12, 22))
    normalized = normalize_output_coordinates(clipped, bounds=(10, 20, 12, 22))

    assert [row["name"] for row in clipped] == ["field", "extent_marker"]
    assert normalized == (
        {"x": 0, "y": 0, "z": -1, "menu": "Ground", "cat1": "Grass", "cat2": -1, "direction": -1, "id": -1, "name": "field", "priority": 1},
        {"x": 2, "y": 2, "z": -1, "menu": -1, "cat1": -1, "cat2": -1, "direction": -1, "id": -1, "name": "extent_marker", "priority": -999},
    )


def test_output_rows_are_deterministic_for_same_placements() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    placements = tuple(
        reversed(
            (
                _placement(
                    layer=LayerKind.FOLIAGE,
                    cell=GridCell(1, 0),
                    config_name="trees",
                    priority=8,
                    menu="Foliage",
                    cat1="Tree",
                    feature_id="b",
                ),
                _placement(
                    layer=LayerKind.GROUND,
                    cell=GridCell(0, 0),
                    config_name="field",
                    priority=3,
                    menu="Ground",
                    cat1="Wheat",
                    feature_id="a",
                ),
            )
        )
    )

    assert placements_to_output_rows(placements) == placements_to_output_rows(placements)


def test_validation_rejects_duplicate_layer_rows_and_building_road_collisions() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.output_rows import (
        OutputRowValidationError,
        placements_to_output_rows,
    )

    duplicate_ground = (
        _placement(layer=LayerKind.GROUND, cell=GridCell(0, 0), config_name="field", priority=2, menu="Ground", cat1="Wheat"),
        _placement(layer=LayerKind.GROUND, cell=GridCell(0, 0), config_name="forest", priority=3, menu="Ground", cat1="Forest"),
    )
    building_on_road = (
        _placement(layer=LayerKind.LINEAR_SURFACE, cell=GridCell(1, 1), config_name="road", priority=2, menu="Roads", cat1="Dirt"),
        _placement(layer=LayerKind.BUILDING, cell=GridCell(1, 1), config_name="house", priority=1, menu="Buildings", cat1="House"),
    )

    with pytest.raises(OutputRowValidationError, match="duplicate mutually exclusive"):
        placements_to_output_rows(duplicate_ground)
    with pytest.raises(OutputRowValidationError, match="building-road collision"):
        placements_to_output_rows(building_on_road)


def test_pipeline_assembles_output_rows_without_migration_flag() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    pipeline = ExtractionPipeline(
        ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=123,
        )
    )

    result = pipeline.run_output_rows(
        placements=(
            _placement(
                layer=LayerKind.GROUND,
                cell=GridCell(10, 20),
                config_name="field",
                priority=3,
                menu="Ground",
                cat1="Wheat",
            ),
        ),
        bounds=(10, 20, 11, 21),
    )

    assert result.output_rows[-1]["name"] == "extent_marker"
    assert result.output_rows[0]["x"] == 0
    assert result.stats.counts["output_rows"] == 2


def test_pipeline_clips_diagonal_edge_rows_before_validation() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    pipeline = ExtractionPipeline(
        ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=123,
        )
    )

    result = pipeline.run_output_rows(
        placements=(
            _placement(
                layer=LayerKind.BUILDING,
                cell=GridCell(0, 15),
                config_name="houses",
                priority=5,
                menu="Independent Buildings",
                cat1="House",
                cat2="Building 2-3",
                direction="Direction 2",
                grid_kind=GridKind.DIAGONAL,
            ),
        ),
        bounds=(0, 0, 20, 15),
    )

    assert result.output_rows == (
        {"x": 20, "y": 15, "z": -1, "menu": -1, "cat1": -1, "cat2": -1, "direction": -1, "id": -1, "name": "extent_marker", "priority": -999},
    )


def test_osm_processor_get_output_uses_layered_rows_by_default() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor.idx_bbox = [10, 20, 11, 21]
    processor.placements = (
        _placement(
            layer=LayerKind.GROUND,
            cell=GridCell(10, 20),
            config_name="field",
            priority=3,
            menu="Ground",
            cat1="Wheat",
        ),
    )

    output = processor.get_output()

    assert isinstance(output, pd.DataFrame)
    assert output.to_dict("records") == [
        {"x": 0, "y": 0, "z": -1, "menu": "Ground", "cat1": "Wheat", "cat2": -1, "direction": -1, "id": -1, "name": "field", "priority": 3},
        {"x": 1, "y": 1, "z": -1, "menu": -1, "cat1": -1, "cat2": -1, "direction": -1, "id": -1, "name": "extent_marker", "priority": -999},
    ]
