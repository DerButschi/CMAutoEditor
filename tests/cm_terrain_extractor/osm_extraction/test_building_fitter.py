from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
ROOT_DIR = Path(__file__).parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
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


def _feature(
    feature_id: str,
    geometry: Polygon,
    *,
    config_name: str = "houses",
    source_tags: dict[str, str] | None = None,
):
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind

    return FeatureRecord(
        feature_id=feature_id,
        source_index=0,
        config_name=config_name,
        process=ProcessKind.BUILDING_OUTLINE,
        priority=4,
        geometry=geometry,
        source_tags={"building": "yes"} if source_tags is None else source_tags,
    )


def _catalog_rows():
    return (
        {
            "width": 2,
            "height": 2,
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
            "width": 4,
            "height": 2,
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


def test_catalog_width_height_match_profile_half_cell_units() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingCatalog
    from terrain_extraction.osm_extraction.building_geometry import (
        reconstruct_building_footprint_polygon,
    )

    footprint = BuildingCatalog.from_records((_catalog_rows()[0],)).footprints[0]
    polygon = reconstruct_building_footprint_polygon(_grid(), 0.5, 0.5, footprint)

    assert polygon.bounds == pytest.approx((8.0, 8.0, 16.0, 16.0))
    assert polygon.area == pytest.approx(64.0)


def test_fitter_footprint_reconstruction_matches_profile_helper() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingCatalog
    from terrain_extraction.osm_extraction.building_geometry import (
        reconstruct_building_footprint_polygon,
    )

    from profiles import get_building_outline_by_df_entry

    profile_polygon, is_diagonal = get_building_outline_by_df_entry(
        "residential_buildings",
        "Independent Buildings",
        "House",
        "Building 1",
        "Direction 1",
        profile="cold_war",
    )
    record = {
        "width": 2,
        "height": 2,
        "row": 0,
        "col": 0,
        "direction": 0,
        "menu": "Independent Buildings",
        "cat1": "House",
        "cat2": "Building 1",
        "is_diagonal": is_diagonal,
    }
    fitter_polygon = reconstruct_building_footprint_polygon(
        _grid(),
        -0.5,
        -0.5,
        BuildingCatalog.from_records((record,)).footprints[0],
    )

    assert fitter_polygon.equals_exact(profile_polygon, tolerance=0.001)


def test_diagonal_fitter_footprint_reconstruction_matches_profile_helper() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingCatalog
    from terrain_extraction.osm_extraction.building_geometry import (
        reconstruct_building_footprint_polygon,
    )

    from profiles import get_building_outline_by_df_entry

    profile_polygon, is_diagonal = get_building_outline_by_df_entry(
        "residential_buildings",
        "Independent Buildings",
        "House",
        "Building 7",
        "Direction 1",
        profile="cold_war",
    )
    record = {
        "width": 2,
        "height": 2,
        "row": 1,
        "col": 2,
        "direction": 0,
        "menu": "Independent Buildings",
        "cat1": "House",
        "cat2": "Building 7",
        "is_diagonal": is_diagonal,
    }

    fitter_polygon = reconstruct_building_footprint_polygon(
        _grid(),
        -0.5,
        -0.5,
        BuildingCatalog.from_records((record,)).footprints[0],
    )

    assert bool(is_diagonal) is True
    assert fitter_polygon.equals_exact(profile_polygon, tolerance=0.001)


def test_simple_rectangle_fits_catalog_footprint_with_exact_iou() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridCell

    grid = _grid()
    outline = Polygon([(8, 8), (24, 8), (24, 16), (8, 16)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7), debug_geometry=True).fit(
        (_feature("simple", outline),),
        catalogs={"houses": _catalog_rows()},
    )

    assert result.placed_count == 1
    assert result.dropped_count == 0
    assert result.diagnostics_by_feature["simple"]["fit_mode"] == "single_rect"
    assert result.diagnostics_by_feature["simple"]["candidate_limit_reached"] is False
    assert result.diagnostics_by_feature["simple"]["best_single_rect_footprint"] == result.placements[0].diagnostics["selected_footprint_id"]
    assert result.placements[0].cells == (GridCell(1, 1), GridCell(2, 1))
    assert result.placements[0].diagnostics["iou"] == 1.0
    assert result.placements[0].diagnostics["selected_footprint_polygon"].area == pytest.approx(128.0)
    assert result.placements[0].diagnostics["output_xidx"] == 0.5
    assert result.placements[0].diagnostics["output_yidx"] == 0.5
    assert result.placements[0].diagnostics["selected_blocked_normal_cells"] == ((1, 1), (2, 1))
    assert result.placements[0].diagnostics["selected_all_overlap_cells"] == ((1, 1), (2, 1))
    assert result.placements[0].diagnostics["selected_width_units"] == 4
    assert result.placements[0].diagnostics["selected_height_units"] == 2
    assert result.placements[0].diagnostics["selected_is_diagonal"] is False
    assert result.placements[0].diagnostics["selected_row"] == 0
    assert result.placements[0].diagnostics["selected_col"] == 1
    assert result.placements[0].diagnostics["selected_direction"] == 0
    assert result.placements[0].cm_type.cat2 == "Long House"


def test_swapped_normal_candidate_diagnostics_use_effective_output_dimensions() -> None:
    from terrain_extraction.osm_extraction.building_fitter import (
        BuildingCatalog,
        _placement_from_candidate,
    )
    from terrain_extraction.osm_extraction.building_geometry import (
        reconstruct_building_footprint_polygon,
    )
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    footprint = BuildingCatalog.from_records(
        (
            {
                "width": 2,
                "height": 4,
                "row": 0,
                "col": 0,
                "direction": 0,
                "menu": "Buildings",
                "cat1": "House",
                "cat2": "Tall House",
                "is_modular": False,
                "weight": 1.0,
            },
        )
    ).footprints[0]
    selected_polygon = reconstruct_building_footprint_polygon(
        grid,
        0.5,
        0.5,
        {"width": 4, "height": 2, "is_diagonal": False},
    )
    placement = _placement_from_candidate(
        footprint,
        cells=(GridCell(1, 1), GridCell(2, 1)),
        polygon=selected_polygon,
        iou=1.0,
        centroid_shift=0.0,
        angle_error=0.0,
        area_error=0.0,
        selected_footprint_polygon=selected_polygon,
        output_xidx=0.5,
        output_yidx=0.5,
        footprint_width_units=4,
        footprint_height_units=2,
    )

    diagnostics = placement.diagnostics
    reconstructed = reconstruct_building_footprint_polygon(
        grid,
        diagnostics["output_xidx"],
        diagnostics["output_yidx"],
        {
            "width": diagnostics["selected_width_units"],
            "height": diagnostics["selected_height_units"],
            "is_diagonal": diagnostics["selected_is_diagonal"],
        },
    )
    rows = placements_to_output_rows((placement,), include_internal=True, grid_index=grid)

    assert diagnostics["selected_width_cells"] == 2
    assert diagnostics["selected_height_cells"] == 4
    assert diagnostics["selected_width_units"] == 4
    assert diagnostics["selected_height_units"] == 2
    assert reconstructed.equals_exact(diagnostics["selected_footprint_polygon"], tolerance=0.001)
    assert len(rows) == 1


def test_small_rectangle_does_not_expand_to_two_by_two_full_cells() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridCell

    grid = _grid()
    outline = Polygon([(8, 8), (16, 8), (16, 16), (8, 16)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7), debug_geometry=True).fit(
        (_feature("small", outline),),
        catalogs={"houses": _catalog_rows()},
    )

    assert result.placed_count == 1
    assert result.placements[0].cells == (GridCell(1, 1),)
    assert result.placements[0].diagnostics["selected_footprint_polygon"].equals_exact(outline, tolerance=0.001)
    assert result.placements[0].diagnostics["module_count"] == 1
    assert result.placements[0].cm_type.cat2 == "Small House"


def test_diagonal_outline_can_select_diagonal_catalog_candidate() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridKind

    grid = _grid()
    outline = Polygon([(8, 16), (16, 8), (24, 16), (16, 24)])
    catalog = (
        {
            "width": 2,
            "height": 2,
            "row": 0,
            "col": 2,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Diagonal House",
            "is_diagonal": True,
            "is_modular": False,
            "weight": 1.0,
        },
        _catalog_rows()[0],
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("diagonal", outline),),
        catalogs={"houses": catalog},
    )

    selected = result.placements[0].diagnostics["selected_footprint_polygon"]
    assert result.placed_count == 1
    assert result.placements[0].grid_kind is GridKind.DIAGONAL
    assert result.placements[0].cm_type.cat2 == "Diagonal House"
    assert result.diagnostics_by_feature["diagonal"]["preferred_orientation"] == "diagonal"
    assert result.placements[0].diagnostics["footprint_orientation_class"] == "diagonal"
    assert result.placements[0].diagnostics["iou"] == 1.0
    assert selected.equals_exact(outline, tolerance=0.001)
    assert result.placements[0].diagnostics["selected_is_diagonal"] is True


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
    assert result.diagnostics_by_feature["l-shape"]["fit_mode"] == "modular_cover"
    assert result.diagnostics_by_feature["l-shape"]["modular_attempted"] is True
    assert result.placements[0].cells == (GridCell(1, 1), GridCell(1, 2), GridCell(2, 1))
    assert result.placements[0].diagnostics["module_count"] == 3
    assert result.placements[0].diagnostics["iou"] == 1.0
    assert result.placements[0].cm_type.cat2 == "Modular House"


def test_modular_cover_building_has_explicit_output_anchor_for_output_rows() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import GridKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    outline = Polygon([(24, 0), (40, 0), (40, 8), (24, 8)])
    catalog = (
        {
            "width": 4,
            "height": 2,
            "row": 0,
            "col": 3,
            "direction": 0,
            "menu": "Modular Buildings",
            "cat1": "1 Story",
            "cat2": "Building 4",
            "is_modular": True,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("modular", outline, source_tags={"building": "industrial"}),),
        catalogs={"houses": catalog},
    )

    placement = result.placements[0]
    rows = placements_to_output_rows(result.placements, include_internal=True)

    assert result.placed_count == 1
    assert placement.grid_kind is GridKind.SUB_SQUARE
    assert placement.diagnostics["output_xidx"] == 2.5
    assert placement.diagnostics["output_yidx"] == -0.5
    assert rows[0]["xidx"] == 2.5
    assert rows[0]["yidx"] == -0.5
    assert rows[0]["_cell_xidx"] == 3
    assert rows[0]["_cell_yidx"] == 0


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
    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7), debug_geometry=True).fit(
        (_feature("near-road", outline),),
        catalogs={"houses": (_catalog_rows()[0],)},
    )

    assert result.placed_count == 1
    assert result.placements[0].cells != (road_cell,)
    assert result.placements[0].diagnostics["road_overlap_cells"] == 0
    assert occupancy.object_id_at(LayerKind.BUILDING, result.placements[0].cells[0]) == "near-road"


def test_equivalent_building_candidates_use_profile_weights_for_variation() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid()
    outline = Polygon([(8, 8), (16, 8), (16, 16), (8, 16)])
    catalog = (
        {
            "width": 2,
            "height": 2,
            "row": 0,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Suppressed House",
            "is_modular": False,
            "weight": 0.0,
        },
        {
            "width": 2,
            "height": 2,
            "row": 0,
            "col": 1,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Weighted House",
            "is_modular": False,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("weighted", outline),),
        catalogs={"houses": catalog},
    )

    assert result.placements[0].cm_type.cat2 == "Weighted House"


def test_candidate_scoring_uses_actual_footprint_iou_not_reserved_cells() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid()
    outline = Polygon([(8, 8), (16, 8), (16, 16), (8, 16)])
    catalog = (
        {
            "width": 2,
            "height": 2,
            "row": 0,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Exact House",
            "is_modular": False,
            "weight": 1.0,
        },
        {
            "width": 1,
            "height": 1,
            "row": 0,
            "col": 1,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Tiny House",
            "is_modular": False,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("actual-iou", outline),),
        catalogs={"houses": catalog},
    )

    assert result.placements[0].cm_type.cat2 == "Exact House"
    assert result.placements[0].diagnostics["iou"] == 1.0


def test_area_shortlist_selects_medium_footprint_instead_of_first_tiny_catalog_entry() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid(width=8, height=8)
    outline = Polygon([(16, 16), (40, 16), (40, 32), (16, 32)])
    tiny_rows = tuple(
        {
            "id": f"tiny-{index}",
            "width": 1,
            "height": 1,
            "row": index,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": f"Tiny {index}",
            "is_modular": False,
            "weight": 1.0,
        }
        for index in range(20)
    )
    catalog = (
        *tiny_rows,
        {
            "id": "medium-fit",
            "width": 6,
            "height": 4,
            "row": 99,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Medium Fit",
            "is_modular": False,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("medium", outline),),
        catalogs={"houses": catalog},
    )

    diagnostics = result.placements[0].diagnostics
    assert result.placed_count == 1
    assert result.placements[0].cm_type.cat2 == "Medium Fit"
    assert diagnostics["selected_area_error_ratio"] == pytest.approx(0.0)
    assert diagnostics["selected_footprint_polygon"].area == pytest.approx(outline.area)


def test_single_rect_mode_does_not_hit_candidate_cap_for_late_area_compatible_footprints() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid(width=8, height=8)
    outline = Polygon([(16, 16), (40, 16), (40, 32), (16, 32)])
    catalog = tuple(
        {
            "id": f"tiny-{index}",
            "width": 1,
            "height": 1,
            "row": index,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": f"Tiny {index}",
            "is_modular": False,
            "weight": 1.0,
        }
        for index in range(20)
    ) + (
        {
            "id": "late-fit",
            "width": 6,
            "height": 4,
            "row": 99,
            "col": 0,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Late Fit",
            "is_modular": False,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7), max_candidates_per_building=2).fit(
        (_feature("capped", outline),),
        catalogs={"houses": catalog},
    )

    diagnostics = result.diagnostics_by_feature["capped"]
    assert result.placements[0].cm_type.cat2 == "Late Fit"
    assert diagnostics["fit_mode"] == "single_rect"
    assert diagnostics["candidate_limit_reached"] is False
    assert diagnostics["expensive_candidates_scored"] <= 18
    assert diagnostics["footprint_types_considered"] <= 5
    assert diagnostics["best_single_rect_footprint"] == "late-fit"


def test_geometric_road_overlap_penalty_avoids_half_road_cover() -> None:
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
    occupancy.place(
        PlacementRecord(
            layer=LayerKind.LINEAR_SURFACE,
            grid_kind=GridKind.NORMAL,
            cells=(GridCell(0, 1),),
            config_name="road",
            feature_id="road-1",
            priority=1,
            cm_type=CMType(menu="Roads", cat1="Road"),
            score=1.0,
        ),
        object_id="road-1",
    )
    outline = Polygon([(4, 8), (12, 8), (12, 16), (4, 16)])

    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7), debug_geometry=True).fit(
        (_feature("road-edge", outline),),
        catalogs={"houses": (_catalog_rows()[0],)},
    )

    selected = result.placements[0].diagnostics["selected_footprint_polygon"]
    assert selected.intersection(grid.cell_polygon(GridCell(0, 1))).area == pytest.approx(0.0)
    assert result.placements[0].diagnostics["road_overlap_area_m2"] == 0.0


def test_geometric_road_overlap_rejects_less_than_half_road_cover() -> None:
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
    road_cell = GridCell(0, 1)
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
    outline = Polygon([(6, 8), (14, 8), (14, 16), (6, 16)])

    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7)).fit(
        (_feature("sliver-road-edge", outline),),
        catalogs={"houses": (_catalog_rows()[0],)},
    )

    selected = result.placements[0].diagnostics["selected_footprint_polygon"]
    assert result.placed_count == 1
    assert selected.intersection(grid.cell_polygon(road_cell)).area == pytest.approx(0.0)
    assert result.placements[0].diagnostics["road_overlap_area_m2"] == 0.0


def test_fitter_accepts_only_final_row_geometry_that_avoids_roads() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.final_geometry import validate_final_output_geometry
    from terrain_extraction.osm_extraction.models import (
        CMType,
        GridCell,
        GridKind,
        LayerKind,
        PlacementRecord,
    )
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    occupancy = OccupancyModel.from_grid_index(grid)
    road = PlacementRecord(
        layer=LayerKind.LINEAR_SURFACE,
        grid_kind=GridKind.NORMAL,
        cells=(GridCell(0, 1),),
        config_name="road",
        feature_id="road-1",
        priority=1,
        cm_type=CMType(menu="Roads", cat1="Road"),
        score=1.0,
    )
    occupancy.place(road, object_id="road-1")
    outline = Polygon([(6, 8), (14, 8), (14, 16), (6, 16)])
    catalog = (
        {
            "width": 2,
            "height": 2,
            "row": 0,
            "col": 0,
            "direction": 0,
            "menu": "Independent Buildings",
            "cat1": "House",
            "cat2": "Building 1",
            "building_type": "residential_buildings",
            "is_modular": False,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7)).fit(
        (_feature("final-row-road-edge", outline),),
        catalogs={"houses": catalog},
    )
    rows = placements_to_output_rows((road, *result.placements), include_internal=True, grid_index=grid)
    validation = validate_final_output_geometry(rows, grid)

    assert result.placed_count == 1
    assert validation.is_valid is True
    selected = result.placements[0].diagnostics["selected_footprint_polygon"]
    assert selected.intersection(grid.cell_polygon(GridCell(0, 1))).area == pytest.approx(0.0)


def test_selected_output_coordinates_reconstruct_collision_footprint() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.building_geometry import (
        footprint_spec_from_diagnostics,
        reconstruct_building_footprint_polygon,
    )

    grid = _grid()
    outline = Polygon([(8, 16), (16, 8), (24, 16), (16, 24)])
    catalog = (
        {
            "width": 2,
            "height": 2,
            "row": 0,
            "col": 2,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": "Diagonal House",
            "is_diagonal": True,
            "is_modular": False,
            "weight": 1.0,
        },
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("diagonal-reconstruct", outline),),
        catalogs={"houses": catalog},
    )

    diagnostics = result.placements[0].diagnostics
    reconstructed = reconstruct_building_footprint_polygon(
        grid,
        diagnostics["output_xidx"],
        diagnostics["output_yidx"],
        footprint_spec_from_diagnostics(diagnostics),
    )
    assert reconstructed.equals_exact(diagnostics["selected_footprint_polygon"], tolerance=0.001)


def test_road_overlap_metrics_only_scan_local_candidate_window() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter
    from terrain_extraction.osm_extraction.models import (
        CMType,
        GridCell,
        GridKind,
        LayerKind,
        PlacementRecord,
    )
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    grid = _grid(width=30, height=30)
    occupancy = OccupancyModel.from_grid_index(grid)
    for index in range(100):
        cell = GridCell(20 + index % 10, 20 + index // 10)
        occupancy.place(
            PlacementRecord(
                layer=LayerKind.LINEAR_SURFACE,
                grid_kind=GridKind.NORMAL,
                cells=(cell,),
                config_name="road",
                feature_id=f"far-road-{index}",
                priority=1,
                cm_type=CMType(menu="Roads", cat1="Road"),
                score=1.0,
            ),
            object_id=f"far-road-{index}",
        )
    occupancy.place(
        PlacementRecord(
            layer=LayerKind.LINEAR_SURFACE,
            grid_kind=GridKind.NORMAL,
            cells=(GridCell(2, 2),),
            config_name="road",
            feature_id="near-road",
            priority=1,
            cm_type=CMType(menu="Roads", cat1="Road"),
            score=1.0,
        ),
        object_id="near-road",
    )
    outline = Polygon([(16, 16), (24, 16), (24, 24), (16, 24)])

    result = BuildingFitter(grid, occupancy=occupancy, rng=np.random.default_rng(7)).fit(
        (_feature("local-road", outline),),
        catalogs={"houses": (_catalog_rows()[0],)},
    )

    assert result.placed_count == 1
    assert result.placements[0].cells != (GridCell(2, 2),)
    assert result.diagnostics["shapely_overlap_evaluations"] <= 25


def test_large_industrial_building_enters_bounded_modular_cover_mode() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid(width=10, height=10)
    outline = Polygon([(8, 8), (72, 8), (72, 40), (8, 40)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7), max_candidates_per_building=2).fit(
        (_feature("large", outline, source_tags={"building": "industrial"}),),
        catalogs={"houses": _catalog_rows()},
    )

    diagnostics = result.diagnostics_by_feature["large"]
    assert diagnostics["fit_mode"] == "modular_cover"
    assert diagnostics["modular_attempted"] is True
    assert diagnostics["candidate_limit_reached"] in {False, True}
    assert diagnostics["candidates_scored"] <= 2
    assert diagnostics["elapsed_ms"] >= 0.0
    assert diagnostics["candidate_generation_ms"] >= 0.0
    assert diagnostics["placement_ms"] >= 0.0
    assert result.diagnostics["building_feature_count"] == 1
    assert result.diagnostics["total_candidates_scored"] == diagnostics["candidates_scored"]
    assert result.diagnostics["candidates_scored_per_building"]["large"] == diagnostics["candidates_scored"]
    assert result.diagnostics["candidate_limit_reached_count"] <= 1
    assert result.diagnostics["shapely_score_evaluations"] == diagnostics["candidates_scored"]
    assert result.diagnostics["shapely_overlap_evaluations"] >= 0
    assert result.placed_count + result.dropped_count == 1


def test_many_ordinary_buildings_use_fast_single_rect_mode_without_candidate_caps() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid(width=30, height=30)
    features = tuple(
        _feature(
            f"building-{index}",
            Polygon(
                    [
                        (24 * (index % 5) + 16, 16 * (index // 5) + 16),
                        (24 * (index % 5) + 32, 16 * (index // 5) + 16),
                        (24 * (index % 5) + 32, 16 * (index // 5) + 24),
                        (24 * (index % 5) + 16, 16 * (index // 5) + 24),
                ]
            ),
        )
        for index in range(50)
    )
    catalog = tuple(
        {
            "id": f"footprint-{width}-{height}",
            "width": width,
            "height": height,
            "row": width,
            "col": height,
            "direction": 0,
            "menu": "Buildings",
            "cat1": "House",
            "cat2": f"{width}x{height}",
            "is_modular": False,
            "weight": 1.0,
        }
        for width in range(1, 10)
        for height in range(1, 6)
    )

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(features, catalogs={"houses": catalog})

    scored = [diagnostics["expensive_candidates_scored"] for diagnostics in result.diagnostics_by_feature.values()]
    shifted = [diagnostics["shifted_candidates_generated"] for diagnostics in result.diagnostics_by_feature.values()]
    assert result.placed_count == len(features)
    assert result.diagnostics["candidate_limit_reached_count"] == 0
    assert all(diagnostics["fit_mode"] == "single_rect" for diagnostics in result.diagnostics_by_feature.values())
    assert max(scored) < 64
    assert max(shifted) <= 5 * 9
    assert result.diagnostics["shapely_score_evaluations"] == sum(scored)
    assert all(placement.diagnostics["selected_footprint_polygon"].area > 0 for placement in result.placements)


def test_ordinary_residential_building_does_not_enter_modular_cover_by_default() -> None:
    from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

    grid = _grid(width=8, height=8)
    outline = Polygon([(16, 16), (40, 16), (40, 32), (16, 32)])

    result = BuildingFitter(grid, rng=np.random.default_rng(7)).fit(
        (_feature("residential", outline, source_tags={"building": "residential"}),),
        catalogs={"houses": _catalog_rows()},
    )

    diagnostics = result.diagnostics_by_feature["residential"]
    assert diagnostics["fit_mode"] == "single_rect"
    assert diagnostics["modular_attempted"] is False
    assert result.placements[0].diagnostics["selected_is_modular"] is False


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
