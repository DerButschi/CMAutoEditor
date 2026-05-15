from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import LineString, Point

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid(width: int = 7, height: int = 7):
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    return GridIndex(
        origin_x=0.0,
        origin_y=0.0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=width,
        height=height,
        cell_size_m=8.0,
    )


def _catalog(*, include_four_way: bool = True):
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    rows = [
        {"direction": 0, "row": 0, "col": 0, "l": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 1, "col": 0, "l": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 1, "col": 1, "r": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 4, "row": 1, "col": 2, "l": (2, 3), "r": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 5, "row": 1, "col": 3, "l": (2, 3), "r": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 6, "row": 1, "col": 4, "l": (2, 3), "r": (2, 3), "d": (2, 3), "cost": 1.0},
    ]
    if include_four_way:
        rows.append(
            {
                "direction": 7,
                "row": 2,
                "col": 2,
                "l": (2, 3),
                "r": (2, 3),
                "u": (2, 3),
                "d": (2, 3),
                "cost": 2.0,
            }
        )
    return CompiledTileCatalog.from_records(rows, process=ProcessKind.ROAD)


def _cross_graph():
    from terrain_extraction.osm_extraction.models import (
        ProcessKind,
        TopologyEdge,
        TopologyGraph,
        TopologyNode,
    )

    return TopologyGraph(
        nodes=(
            TopologyNode(0, Point(24, 24)),
            TopologyNode(1, Point(24, 40)),
            TopologyNode(2, Point(40, 24)),
            TopologyNode(3, Point(24, 8)),
            TopologyNode(4, Point(8, 24)),
        ),
        edges=(
            TopologyEdge(0, 0, 1, LineString([(24, 24), (24, 40)]), ("n",), (0,), "primary", ProcessKind.ROAD, 1),
            TopologyEdge(1, 0, 2, LineString([(24, 24), (40, 24)]), ("e",), (1,), "primary", ProcessKind.ROAD, 1),
            TopologyEdge(2, 3, 0, LineString([(24, 8), (24, 24)]), ("s",), (2,), "primary", ProcessKind.ROAD, 1),
            TopologyEdge(3, 4, 0, LineString([(8, 24), (24, 24)]), ("w",), (3,), "primary", ProcessKind.ROAD, 1),
        ),
    )


def test_anchor_candidates_include_radius_one_cells_and_select_closest_feasible() -> None:
    from terrain_extraction.osm_extraction.anchor_selection import AnchorSelector, SingleAnchorPlan
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind

    result = AnchorSelector(
        grid_index=_grid(),
        catalogs={ProcessKind.ROAD: _catalog()},
    ).select(_cross_graph())
    plan = result.plans[0]

    assert isinstance(plan, SingleAnchorPlan)
    assert plan.anchor_cell == GridCell(3, 3)
    assert plan.selected_candidate.required_dirs_estimate == frozenset({"E", "N", "S", "W"})
    assert plan.selected_candidate.tile_feasible
    assert len([candidate for candidate in plan.candidates if candidate.search_radius == 1]) == 9


def test_anchor_selection_retries_radius_two_or_three_when_nearest_cells_are_blocked() -> None:
    from terrain_extraction.osm_extraction.anchor_selection import AnchorSelector, SingleAnchorPlan
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    occupancy = OccupancyModel.from_grid_index(_grid())
    blocked = tuple(GridCell(xidx, yidx) for xidx in range(2, 5) for yidx in range(2, 5))
    occupancy.reserve(blocked, object_id="blocked-anchor-neighborhood")

    result = AnchorSelector(
        grid_index=_grid(),
        occupancy=occupancy,
        catalogs={ProcessKind.ROAD: _catalog()},
    ).select(_cross_graph())
    plan = result.plans[0]

    assert isinstance(plan, SingleAnchorPlan)
    assert plan.selected_candidate.search_radius == 2
    assert max(abs(plan.anchor_cell.xidx - 3), abs(plan.anchor_cell.yidx - 3)) == 2
    assert result.diagnostics["anchor_retry_nodes"] == 1


def test_four_way_without_catalog_support_produces_split_anchor_plan() -> None:
    from terrain_extraction.osm_extraction.anchor_selection import AnchorSelector, SplitAnchorPlan
    from terrain_extraction.osm_extraction.models import ProcessKind

    result = AnchorSelector(
        grid_index=_grid(),
        catalogs={ProcessKind.ROAD: _catalog(include_four_way=False)},
    ).select(_cross_graph())
    plan = result.plans[0]

    assert isinstance(plan, SplitAnchorPlan)
    assert plan.reason == "single_anchor_catalog_gap"
    assert set(plan.required_dirs_estimate) == {"E", "N", "S", "W"}
    assert set(plan.split_direction_sets) == {("E", "W"), ("N", "S")}


def test_router_uses_selected_anchor_plan_cells() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    occupancy = OccupancyModel.from_grid_index(_grid())
    blocked = tuple(GridCell(xidx, yidx) for xidx in range(2, 5) for yidx in range(2, 5))
    occupancy.reserve(blocked, object_id="blocked-anchor-neighborhood")

    result = NetworkRouter(
        grid_index=_grid(),
        occupancy=occupancy,
        catalogs={ProcessKind.ROAD: _catalog()},
        corridor_deviation_m=32.0,
    ).route(_cross_graph())

    assert result.anchor_plans[0].selected_candidate.search_radius == 2
    assert max(abs(result.node_anchors[0].xidx - 3), abs(result.node_anchors[0].yidx - 3)) == 2


def test_debug_export_exposes_anchor_candidates_and_selected_plans() -> None:
    from terrain_extraction.osm_extraction.anchor_selection import AnchorSelector
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
    from terrain_extraction.osm_extraction.models import NetworkRoutingResult, ProcessKind

    selection = AnchorSelector(
        grid_index=_grid(),
        catalogs={ProcessKind.ROAD: _catalog()},
    ).select(_cross_graph())
    result = build_debug_layers(
        routing=NetworkRoutingResult(anchor_plans=selection.plans),
        grid_index=_grid(),
    )

    assert "anchor_candidates" in result.layers
    assert "selected_anchor_plans" in result.layers
    assert result.layers["selected_anchor_plans"].loc[0, "plan_kind"] == "single"
    assert {"score", "tile_feasible", "occupancy_feasible", "required_dirs_estimate"} <= set(
        result.layers["anchor_candidates"].columns
    )
