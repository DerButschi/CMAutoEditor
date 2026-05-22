from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
ROOT_DIR = Path(__file__).parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
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
    diagnostics=None,
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
        diagnostics=diagnostics or {},
    )


def _grid(width: int = 4, height: int = 4):
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    return GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=width,
        height=height,
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
        _placement(
            layer=LayerKind.BUILDING,
            cell=GridCell(1, 1),
            config_name="house",
            priority=1,
            menu="Buildings",
            cat1="House",
            diagnostics={"output_xidx": 0.5, "output_yidx": 0.5},
        ),
    )

    with pytest.raises(OutputRowValidationError, match="duplicate mutually exclusive"):
        placements_to_output_rows(duplicate_ground)
    with pytest.raises(OutputRowValidationError, match="building-road collision"):
        placements_to_output_rows(building_on_road)


def test_building_output_rows_use_selected_sub_square_coordinates() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    rows = placements_to_output_rows(
        (
            _placement(
                layer=LayerKind.BUILDING,
                cell=GridCell(1, 1),
                config_name="houses",
                priority=5,
                menu="Independent Buildings",
                cat1="House",
                cat2="Building 1",
                direction="Direction 1",
                grid_kind=GridKind.SUB_SQUARE,
                diagnostics={"output_xidx": 0.5, "output_yidx": 0.5},
            ),
        ),
        include_internal=True,
    )

    assert len(rows) == 1
    assert rows[0]["xidx"] == 0.5
    assert rows[0]["yidx"] == 0.5
    assert rows[0]["_cell_xidx"] == 1
    assert rows[0]["_cell_yidx"] == 1
    assert rows[0]["_grid_kind"] == GridKind.SUB_SQUARE.value


def test_building_output_rows_require_explicit_selected_coordinates() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import (
        OutputRowValidationError,
        placements_to_output_rows,
    )

    with pytest.raises(OutputRowValidationError, match="building placement is missing explicit output coordinates"):
        placements_to_output_rows(
            (
                _placement(
                    layer=LayerKind.BUILDING,
                    cell=GridCell(1, 1),
                    config_name="houses",
                    priority=5,
                    menu="Independent Buildings",
                    cat1="House",
                    cat2="Building 1",
                    direction="Direction 1",
                    grid_kind=GridKind.SUB_SQUARE,
                ),
            )
        )


def test_building_output_rows_use_selected_diagonal_coordinates() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    rows = placements_to_output_rows(
        (
            _placement(
                layer=LayerKind.BUILDING,
                cell=GridCell(1, 1),
                config_name="houses",
                priority=5,
                menu="Independent Buildings",
                cat1="House",
                cat2="Building 7",
                direction="Direction 1",
                grid_kind=GridKind.DIAGONAL,
                diagnostics={"output_xidx": 0.5, "output_yidx": 1.0},
            ),
        )
    )

    assert rows == (
        {
            "xidx": 0.5,
            "yidx": 1,
            "z": -1,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 7",
            "direction": "Direction 1",
            "id": -1,
            "name": "houses",
            "priority": 5,
        },
    )


def test_building_road_validation_uses_all_blocked_building_cells() -> None:
    from shapely.geometry import Polygon
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import (
        OutputRowValidationError,
        placements_to_output_rows,
    )

    road = _placement(
        layer=LayerKind.LINEAR_SURFACE,
        cell=GridCell(2, 1),
        config_name="road",
        priority=4,
        menu="Roads",
        cat1="Dirt",
        feature_id="road-1",
    )
    building = _placement(
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 1),
        config_name="houses",
        priority=5,
        menu="Independent Buildings",
        cat1="House",
        cat2="Long House",
        direction="Direction 1",
        grid_kind=GridKind.SUB_SQUARE,
        diagnostics={
            "output_xidx": 0.5,
            "output_yidx": 0.5,
            "selected_footprint_polygon": Polygon([(8, 8), (24, 8), (24, 16), (8, 16)]),
        },
    )
    object.__setattr__(building, "cells", (GridCell(1, 1), GridCell(2, 1)))

    with pytest.raises(OutputRowValidationError, match="building-road collision"):
        placements_to_output_rows((road, building), include_internal=True)


def test_final_rows_still_emit_one_row_per_building_piece() -> None:
    from shapely.geometry import Polygon
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    building = _placement(
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 1),
        config_name="houses",
        priority=5,
        menu="Independent Buildings",
        cat1="House",
        cat2="Long House",
        direction="Direction 1",
        grid_kind=GridKind.SUB_SQUARE,
        diagnostics={
            "output_xidx": 0.5,
            "output_yidx": 0.5,
            "selected_footprint_polygon": Polygon([(8, 8), (24, 8), (24, 16), (8, 16)]),
            "selected_width_units": 4,
            "selected_height_units": 2,
            "selected_is_diagonal": False,
        },
    )
    object.__setattr__(building, "cells", (GridCell(1, 1), GridCell(2, 1)))

    rows = placements_to_output_rows((building,), include_internal=True, grid_index=_grid())

    assert len(rows) == 1
    assert rows[0]["xidx"] == 0.5
    assert rows[0]["yidx"] == 0.5


def test_geometric_building_validation_accepts_effective_swapped_dimensions() -> None:
    from shapely.geometry import Polygon
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    building = _placement(
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 1),
        config_name="houses",
        priority=5,
        menu="Independent Buildings",
        cat1="House",
        cat2="Tall House",
        direction="Direction 1",
        grid_kind=GridKind.SUB_SQUARE,
        diagnostics={
            "output_xidx": 0.5,
            "output_yidx": 0.5,
            "selected_footprint_polygon": Polygon([(8, 8), (24, 8), (24, 16), (8, 16)]),
            "selected_width_units": 4,
            "selected_height_units": 2,
            "selected_is_diagonal": False,
        },
    )
    object.__setattr__(building, "cells", (GridCell(1, 1), GridCell(2, 1)))

    rows = placements_to_output_rows((building,), include_internal=True, grid_index=_grid())

    assert len(rows) == 1


def test_geometric_building_road_validation_rejects_partial_overlap() -> None:
    from shapely.geometry import Polygon
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import (
        OutputRowValidationError,
        placements_to_output_rows,
    )

    road = _placement(
        layer=LayerKind.LINEAR_SURFACE,
        cell=GridCell(0, 1),
        config_name="road",
        priority=4,
        menu="Roads",
        cat1="Dirt",
        feature_id="road-1",
    )
    building = _placement(
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 1),
        config_name="houses",
        priority=5,
        menu="Independent Buildings",
        cat1="House",
        cat2="Building 1",
        direction="Direction 1",
        grid_kind=GridKind.SUB_SQUARE,
        diagnostics={
            "output_xidx": 0.25,
            "output_yidx": 0.5,
            "selected_footprint_polygon": Polygon([(6, 8), (14, 8), (14, 16), (6, 16)]),
            "selected_width_units": 2,
            "selected_height_units": 2,
            "selected_is_diagonal": False,
            "building_type": "residential_buildings",
        },
    )

    with pytest.raises(OutputRowValidationError, match="final building-linear geometry collision"):
        placements_to_output_rows((road, building), include_internal=True, grid_index=_grid())


def test_geometric_building_validation_uses_final_row_not_diagnostics_polygon() -> None:
    from shapely.geometry import Polygon
    from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    building = _placement(
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 1),
        config_name="houses",
        priority=5,
        menu="Independent Buildings",
        cat1="House",
        cat2="Building 1",
        direction="Direction 1",
        grid_kind=GridKind.SUB_SQUARE,
        diagnostics={
            "output_xidx": 0.5,
            "output_yidx": 0.5,
            "selected_footprint_polygon": Polygon([(12, 8), (20, 8), (20, 16), (12, 16)]),
            "selected_width_units": 2,
            "selected_height_units": 2,
            "selected_is_diagonal": False,
            "building_type": "residential_buildings",
        },
    )

    rows = placements_to_output_rows((building,), include_internal=True, grid_index=_grid())

    assert rows[0]["xidx"] == 0.5


def test_final_geometry_reconstructs_normal_half_shift_and_legacy_offset_contract() -> None:
    from terrain_extraction.osm_extraction.final_geometry import reconstruct_building_row_geometry

    grid = _grid()
    normal = reconstruct_building_row_geometry(
        grid,
        {
            "xidx": 0.5,
            "yidx": 0.5,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 1",
            "direction": "Direction 1",
            "name": "houses",
            "_building_type": "residential_buildings",
        },
    ).geometry
    half_shifted = reconstruct_building_row_geometry(
        grid,
        {
            "xidx": 0.25,
            "yidx": 0.5,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 1",
            "direction": "Direction 1",
            "name": "houses",
            "_building_type": "residential_buildings",
        },
    ).geometry

    assert normal.bounds == pytest.approx((8.0, 8.0, 16.0, 16.0))
    assert normal.area == pytest.approx(64.0)
    assert half_shifted.bounds == pytest.approx((6.0, 8.0, 14.0, 16.0))


def test_final_geometry_reconstructs_diagonal_and_direction_rotation() -> None:
    from terrain_extraction.osm_extraction.final_geometry import reconstruct_building_row_geometry

    grid = _grid()
    diagonal = reconstruct_building_row_geometry(
        grid,
        {
            "xidx": 1.5,
            "yidx": 1.0,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 7",
            "direction": "Direction 1",
            "name": "houses",
            "_building_type": "residential_buildings",
        },
    )
    direction_1 = reconstruct_building_row_geometry(
        grid,
        {
            "xidx": 0.5,
            "yidx": 0.5,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 2",
            "direction": "Direction 1",
            "name": "houses",
            "_building_type": "residential_buildings",
        },
    ).geometry
    direction_2 = reconstruct_building_row_geometry(
        grid,
        {
            "xidx": 0.5,
            "yidx": 0.5,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 2",
            "direction": "Direction 2",
            "name": "houses",
            "_building_type": "residential_buildings",
        },
    ).geometry

    assert diagonal.is_diagonal is True
    assert diagonal.geometry.bounds == pytest.approx((16.0, 4.0, 32.0, 20.0))
    assert diagonal.geometry.area == pytest.approx(128.0)
    assert direction_1.bounds == pytest.approx((8.0, 8.0, 20.0, 16.0))
    assert direction_2.bounds == pytest.approx((8.0, 8.0, 16.0, 20.0))


def test_final_row_geometry_catches_collision_that_cell_only_validation_misses() -> None:
    from terrain_extraction.osm_extraction.final_geometry import validate_final_output_geometry

    rows = (
        {
            "xidx": 0,
            "yidx": 1,
            "z": -1,
            "menu": "Roads",
            "cat1": "Dirt",
            "cat2": "Road Tile 1",
            "direction": "Direction 1",
            "id": -1,
            "name": "road",
            "priority": 4,
            "_layer": "linear_surface",
            "_cell_xidx": 0,
            "_cell_yidx": 1,
        },
        {
            "xidx": 0.25,
            "yidx": 0.5,
            "z": -1,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 1",
            "direction": "Direction 1",
            "id": -1,
            "name": "houses",
            "priority": 5,
            "_layer": "building",
            "_cell_xidx": 1,
            "_cell_yidx": 1,
            "_building_type": "residential_buildings",
        },
    )

    cell_only_overlap = (rows[0]["_cell_xidx"], rows[0]["_cell_yidx"]) == (
        rows[1]["_cell_xidx"],
        rows[1]["_cell_yidx"],
    )
    validation = validate_final_output_geometry(rows, _grid())

    assert cell_only_overlap is False
    assert validation.is_valid is False
    assert validation.issues[0].overlap_area_m2 == pytest.approx(16.0)


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


def test_pipeline_output_rows_strict_mode_rejects_invalid_road_output() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.output_rows import OutputRowValidationError
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    pipeline = ExtractionPipeline(
        ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=123,
        )
    )

    with pytest.raises(OutputRowValidationError, match="road output issues"):
        pipeline.run_output_rows(
            placements=(
                _placement(
                    layer=LayerKind.LINEAR_SURFACE,
                    cell=GridCell(0, 0),
                    config_name="road",
                    priority=4,
                    menu="Roads",
                    cat1="Paved 2",
                    cat2="Road Tile 1",
                    direction="Direction 2",
                    feature_id="road-1",
                ),
            ),
            bounds=(0, 0, 2, 2),
        )


def test_pipeline_output_rows_warn_mode_returns_invalid_road_diagnostics() -> None:
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
                layer=LayerKind.LINEAR_SURFACE,
                cell=GridCell(0, 0),
                config_name="road",
                priority=4,
                menu="Roads",
                cat1="Paved 2",
                cat2="Road Tile 1",
                direction="Direction 2",
                feature_id="road-1",
            ),
        ),
        bounds=(0, 0, 2, 2),
        road_validation_mode="warn",
    )

    assert result.output_rows[0]["name"] == "road"
    assert not result.diagnostics["road_validation"].is_valid
    assert result.diagnostics["road_validation_status"] == {
        "mode": "warn",
        "is_valid": False,
        "summary": result.diagnostics["road_validation"].issue_summary(),
        "hard_issues": len(result.diagnostics["road_validation"].hard_issues),
    }
    assert result.stats.diagnostics["road_validation_status"] == result.diagnostics["road_validation_status"]


def test_pipeline_output_rows_valid_roads_remain_unchanged() -> None:
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
        placements=tuple(
            _placement(
                layer=LayerKind.LINEAR_SURFACE,
                cell=GridCell(xidx, 0),
                config_name="road",
                priority=4,
                menu="Roads",
                cat1="Paved 2",
                cat2="Road Tile 1",
                direction="Direction 2",
                feature_id="road-1",
            )
            for xidx in range(3)
        ),
        bounds=(0, 0, 3, 1),
    )

    assert [row["x"] for row in result.output_rows if row["name"] == "road"] == [0, 1, 2]
    assert result.diagnostics["road_validation"].is_valid
    assert result.diagnostics["road_validation_status"]["mode"] == "strict"
    assert result.diagnostics["road_validation_status"]["is_valid"] is True


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
                diagnostics={"output_xidx": -0.25, "output_yidx": 15.25},
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
