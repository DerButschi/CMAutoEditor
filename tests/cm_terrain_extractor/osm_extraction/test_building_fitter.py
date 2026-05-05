from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid(width: int = 6, height: int = 6):
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    return GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=width,
        height=height,
    )


def _feature(feature_id: str, geometry: Polygon, *, config_name: str = "houses"):
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind

    return FeatureRecord(
        feature_id=feature_id,
        source_index=0,
        config_name=config_name,
        process=ProcessKind.BUILDING_OUTLINE,
        priority=4,
        geometry=geometry,
        source_tags={"building": "yes"},
    )


def _catalog_rows():
    return (
        {
            "width": 1,
            "height": 1,
            "row": 0,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Small House",
            "is_modular": False,
            "weight": 1.0,
        },
        {
            "width": 2,
            "height": 1,
            "row": 0,
            "col": 1,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Long House",
            "is_modular": False,
            "weight": 1.0,
        },
        {
            "width": 1,
            "height": 1,
            "row": 1,
            "col": 0,
            "direction": 0,
            "menu": "Modular Buildings",
            "cat1": "House",
            "cat2": "Modular House",
            "is_modular": True,
            "weight": 1.0,
        },
    )


def test_simple_rectangle_fits_catalog_footprint_with_exact_iou() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridCell

    grid = _grid()
    outline = Polygon([(8, 8), (24, 8), (24, 16), (8, 16)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("simple", outline),),
        catalogs={"houses": _catalog_rows()},
    )

    assert result.placed_count == 1
    assert result.dropped_count == 0
    assert result.placements[0].cells == (GridCell(1, 1), GridCell(2, 1))
    assert result.placements[0].diagnostics["iou"] == 1.0
    assert result.placements[0].cm_type.cat2 == "Long House"


def test_complex_modular_footprint_is_not_collapsed_to_one_rectangle() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridCell

    grid = _grid()
    outline = Polygon([(8, 8), (24, 8), (24, 16), (16, 16), (16, 24), (8, 24)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("l-shape", outline),),
        catalogs={"houses": _catalog_rows()},
    )

    assert result.placed_count == 1
    assert result.placements[0].cells == (GridCell(1, 1), GridCell(1, 2), GridCell(2, 1))
    assert result.placements[0].diagnostics["module_count"] == 3
    assert result.placements[0].diagnostics["iou"] == 1.0
    assert result.placements[0].cm_type.cat2 == "Modular House"


def test_road_occupancy_is_avoided_when_shifted_candidate_is_available() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import (
        CMType,
        GridCell,
        GridKind,
        LayerKind,
        PlacementRecord,
    )
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid()
    occupancy = OccupancyModel.from_grid_index(grid)
    road_cell = GridCell(2, 2)
    occupancy.place(
        PlacementRecord(
            layer=LayerKind.LINEAR_SURFACE,
            grid_kind=GridKind.NORMAL,
            cells=(road_cell,),
            config_name="road",
            feature_id="road-1",
            priority=1,
            cm_type=CMType(menu="Roads", cat1="Road"),
            score=1.0,
        ),
        object_id="road-1",
    )

    outline = Polygon([(16, 16), (24, 16), (24, 24), (16, 24)])
    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7)).fit(
        (_feature("near-road", outline),),
        catalogs={"houses": (_catalog_rows()[0],)},
    )

    assert result.placed_count == 1
    assert result.placements[0].cells != (road_cell,)
    assert result.placements[0].diagnostics["road_overlap_cells"] == 0
    assert occupancy.object_id_at(LayerKind.BUILDING, result.placements[0].cells[0]) == "near-road"


def test_candidate_generation_is_bounded_and_records_fallback_diagnostics() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid(width=10, height=10)
    outline = Polygon([(0, 0), (72, 0), (72, 72), (0, 72)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7), max_candidates_per_building=2).fit(
        (_feature("large", outline),),
        catalogs={"houses": _catalog_rows()},
    )

    diagnostics = result.diagnostics_by_feature["large"]
    assert diagnostics["candidate_limit_reached"] is True
    assert diagnostics["candidates_scored"] <= 2
    assert result.placed_count + result.dropped_count == 1


def test_constrained_building_clusters_are_placed_before_open_clusters() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid()
    occupancy = OccupancyModel.from_grid_index(grid)
    for cell in (GridCell(0, 1), GridCell(1, 0), GridCell(1, 2), GridCell(2, 1)):
        occupancy.reserve((cell,), object_id=f"reserved:{cell.xidx}:{cell.yidx}", priority=0)

    constrained = _feature("constrained", Polygon([(8, 8), (16, 8), (16, 16), (8, 16)]))
    open_feature = _feature("open", Polygon([(32, 32), (40, 32), (40, 40), (32, 40)]))
    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7)).fit(
        (open_feature, constrained),
        catalogs={"houses": (_catalog_rows()[0],)},
    )

    assert result.diagnostics["cluster_order"][:2] == ("constrained", "open")


def test_pipeline_runs_building_fitter_without_migration_flag() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    context = ExtractionContext(
        profile="cold_war",
        bbox=None,
        config_path="default_osm_config.json",
        seed=123,
        rng=np.random.default_rng(123),
    )
    outline = Polygon([(8, 8), (24, 8), (24, 16), (8, 16)])

    result = ExtractionPipeline(context).run_building_fitter(
        features=(_feature("simple", outline),),
        catalogs={"houses": _catalog_rows()},
        grid_index=_grid(),
    )

    assert result.stats.counts["buildings_placed"] == 1
    assert result.placements[0].cells == (GridCell(1, 1), GridCell(2, 1))
    assert result.diagnostics["building_fitting"].placed_count == 1
