from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid(width: int = 3, height: int = 2):
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    return GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=width,
        height=height,
        cell_size_m=8.0,
        crs_epsg=25832,
    )


def _feature(feature_id, config_name, process, geometry, *, priority=5):
    from terrain_extraction.osm_extraction.models import FeatureRecord

    return FeatureRecord(
        feature_id=feature_id,
        source_index=int(str(feature_id).split("-")[-1]) if "-" in str(feature_id) else 0,
        config_name=config_name,
        process=process,
        priority=priority,
        geometry=geometry,
        source_tags={"landuse": config_name},
        source_properties={"id": feature_id},
    )


def _config(raw_entries):
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig

    return ExtractionConfig.from_mapping(raw_entries)


def test_polygon_rasterization_uses_integer_window_and_area_threshold() -> None:
    from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=3, height=1)
    config = _config(
        {
            "meadow": {
                "tags": [["landuse", "meadow"]],
                "cm_types": [{"menu": "Ground 1", "cat1": "Grass"}],
                "process": ["type_from_tag"],
                "priority": 5,
            }
        }
    )
    feature = _feature(
        "area-0",
        "meadow",
        ProcessKind.AREA,
        Polygon([(0, 0), (20.8, 0), (20.8, 8), (0, 8)]),
    )

    placements = AreaRasterizer(grid, OccupancyModel.from_grid_index(grid), np.random.default_rng(1)).rasterize(
        (feature,),
        config,
    )

    assert placements[0].cells == (GridCell(0, 0), GridCell(1, 0), GridCell(2, 0))
    assert placements[0].diagnostics["candidate_window"] == (0, 0, 2, 0)
    assert placements[0].diagnostics["candidate_cells"] == 3

    below_threshold = _feature(
        "area-1",
        "meadow",
        ProcessKind.AREA,
        Polygon([(0, 0), (20.0, 0), (20.0, 8), (0, 8)]),
    )
    placements = AreaRasterizer(grid, OccupancyModel.from_grid_index(grid), np.random.default_rng(1)).rasterize(
        (below_threshold,),
        config,
    )

    assert placements[0].cells == (GridCell(0, 0), GridCell(1, 0))


def test_weighted_area_choices_are_deterministic_under_seed() -> None:
    from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=4, height=1)
    config = _config(
        {
            "field": {
                "tags": [["landuse", "field"]],
                "cm_types": [
                    {"menu": "Ground 3", "cat1": "Crop 1", "weight": 1},
                    {"menu": "Ground 3", "cat1": "Crop 2", "weight": 2},
                ],
                "process": ["type_random_area"],
                "priority": 5,
            }
        }
    )
    feature = _feature("field-0", "field", ProcessKind.AREA, Polygon([(0, 0), (32, 0), (32, 8), (0, 8)]))

    first = AreaRasterizer(grid, OccupancyModel.from_grid_index(grid), np.random.default_rng(123)).rasterize(
        (feature,),
        config,
    )
    second = AreaRasterizer(grid, OccupancyModel.from_grid_index(grid), np.random.default_rng(123)).rasterize(
        (feature,),
        config,
    )

    assert [placement.cm_type.cat1 for placement in first] == [placement.cm_type.cat1 for placement in second]
    assert len(first) == 4
    assert len({placement.cm_type.cat1 for placement in first}) > 1


def test_clustered_random_area_choices_are_seeded_and_spatially_correlated() -> None:
    from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=6, height=1)
    config = _config(
        {
            "forest": {
                "tags": [["landuse", "forest"]],
                "cm_types": [
                    {"menu": "Foliage", "cat1": "Tree A", "weight": 1},
                    {"menu": "Foliage", "cat1": "Tree B", "weight": 1},
                ],
                "process": ["type_random_clusters"],
                "priority": 5,
            }
        }
    )
    feature = _feature("forest-0", "forest", ProcessKind.AREA, Polygon([(0, 0), (48, 0), (48, 8), (0, 8)]))

    placements = AreaRasterizer(grid, OccupancyModel.from_grid_index(grid), np.random.default_rng(7)).rasterize(
        (feature,),
        config,
    )
    categories = [placement.cm_type.cat1 for placement in placements]

    assert categories == [
        placement.cm_type.cat1
        for placement in AreaRasterizer(
            grid,
            OccupancyModel.from_grid_index(grid),
            np.random.default_rng(7),
        ).rasterize((feature,), config)
    ]
    assert any(left == right for left, right in zip(categories, categories[1:], strict=False))


def test_defaults_are_emitted_after_real_placements_without_competing() -> None:
    from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind, ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=2, height=1)
    occupancy = OccupancyModel.from_grid_index(grid)
    config = _config(
        {
            "pavement": {
                "tags": [["landuse", "pavement"]],
                "cm_types": [{"menu": "Ground 2", "cat1": "Pavement"}],
                "process": ["type_from_tag"],
                "priority": 3,
            },
            "default_ground": {
                "tags": [],
                "cm_types": [{"menu": "Ground 1", "cat1": "Grass"}],
                "process": ["default_ground"],
                "priority": 99,
            },
        }
    )
    feature = _feature("area-0", "pavement", ProcessKind.AREA, Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]))

    placements = AreaRasterizer(grid, occupancy, np.random.default_rng(5)).rasterize((feature,), config)

    assert [(placement.config_name, placement.cells) for placement in placements] == [
        ("pavement", (GridCell(0, 0),)),
        ("default_ground", (GridCell(1, 0),)),
    ]
    assert occupancy.object_id_at(LayerKind.GROUND, GridCell(0, 0)) == "area-0"
    assert occupancy.object_id_at(LayerKind.GROUND, GridCell(1, 0)) == "default:default_ground:1:0"


def test_point_placement_uses_centroid_and_occupancy_conflicts() -> None:
    from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind, ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=2, height=2)
    occupancy = OccupancyModel.from_grid_index(grid)
    config = _config(
        {
            "bench": {
                "tags": [["amenity", "bench"]],
                "cm_types": [{"menu": "Flavor Objects 1", "cat1": "Bench"}],
                "process": ["single_object_random"],
                "priority": 2,
            }
        }
    )
    first = _feature("point-0", "bench", ProcessKind.POINT, Point(12, 12), priority=2)
    blocked = _feature("point-1", "bench", ProcessKind.POINT, Point(12, 12), priority=4)

    placements = AreaRasterizer(grid, occupancy, np.random.default_rng(8)).rasterize((first, blocked), config)

    assert len(placements) == 1
    assert placements[0].cells == (GridCell(1, 1),)
    assert placements[0].layer is LayerKind.POINT_OBJECT
    assert occupancy.object_id_at(LayerKind.POINT_OBJECT, GridCell(1, 1)) == "point-0"


def test_overlapping_same_layer_placements_do_not_emit_duplicate_cells() -> None:
    from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind, ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=1, height=1)
    occupancy = OccupancyModel.from_grid_index(grid)
    config = _config(
        {
            "weak": {
                "tags": [["landuse", "weak"]],
                "cm_types": [{"menu": "Ground 1", "cat1": "Grass"}],
                "process": ["type_from_tag"],
                "priority": 20,
            },
            "strong": {
                "tags": [["landuse", "strong"]],
                "cm_types": [{"menu": "Ground 2", "cat1": "Mud"}],
                "process": ["type_from_tag"],
                "priority": 5,
            },
        }
    )
    full_cell = Polygon([(0, 0), (8, 0), (8, 8), (0, 8)])
    weak = _feature("area-0", "weak", ProcessKind.AREA, full_cell, priority=20)
    strong = _feature("area-1", "strong", ProcessKind.AREA, full_cell, priority=5)

    placements = AreaRasterizer(grid, occupancy, np.random.default_rng(9)).rasterize((weak, strong), config)

    assert [(placement.config_name, placement.cells) for placement in placements] == [("strong", (GridCell(0, 0),))]
    assert occupancy.object_id_at(LayerKind.GROUND, GridCell(0, 0)) == "area-1"


def test_pipeline_runs_area_rasterizer_without_migration_flag() -> None:
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    events: list[tuple[str, float, str | None]] = []
    context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="cfg.json",
        seed=123,
        progress=lambda stage, value, message=None: events.append((stage, value, message)),
    )
    grid: GridIndex = _grid(width=1, height=1)
    config = _config(
        {
            "meadow": {
                "tags": [["landuse", "meadow"]],
                "cm_types": [{"menu": "Ground 1", "cat1": "Grass"}],
                "process": ["type_from_tag"],
                "priority": 5,
            }
        }
    )
    feature = _feature("area-0", "meadow", ProcessKind.AREA, Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]))

    result = ExtractionPipeline(context).run_area_rasterizer(
        features=(feature,),
        config=config,
        grid_index=grid,
        occupancy=OccupancyModel.from_grid_index(grid),
    )

    assert len(result.placements) == 1
    assert result.stats.counts["area_rasterizer_placements"] == 1
    assert events[-1] == ("area_rasterization", 1.0, "Area rasterization complete")
