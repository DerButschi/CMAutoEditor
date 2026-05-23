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
        crs_epsg=3857,
    )


def _catalog(*, include_four_way: bool = True, include_ns: bool = True):
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    rows = [
        {"direction": 0, "row": 0, "col": 0, "l": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 1, "col": 0, "l": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 1, "col": 1, "r": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 4, "row": 1, "col": 2, "l": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 5, "row": 1, "col": 3, "r": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 6, "row": 1, "col": 4, "l": (2, 3), "r": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 7, "row": 1, "col": 5, "l": (2, 3), "r": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 8, "row": 1, "col": 6, "l": (2, 3), "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 9, "row": 1, "col": 7, "r": (2, 3), "u": (2, 3), "d": (2, 3), "cost": 1.0},
    ]
    if include_ns:
        rows.append({"direction": 1, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0})
    if include_four_way:
        rows.append(
            {
                "direction": 10,
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


def _ew_with_north_t_catalog():
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    return CompiledTileCatalog.from_records(
        (
            {"direction": 0, "row": 0, "col": 0, "l": (2, 3), "r": (2, 3), "cost": 1.0},
            {"direction": 1, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
            {"direction": 2, "row": 0, "col": 2, "l": (2, 3), "r": (2, 3), "u": (2, 3), "cost": 1.0},
        ),
        process=ProcessKind.ROAD,
    )


def _edge(edge_id, start, end, *, config_name="primary", priority=1):
    from terrain_extraction.osm_extraction.models import ProcessKind, TopologyEdge

    return TopologyEdge(
        edge_id=edge_id,
        start_node_id=start[0],
        end_node_id=end[0],
        geometry=LineString([start[1], end[1]]),
        feature_ids=(f"road-{edge_id}",),
        source_indices=(edge_id,),
        config_name=config_name,
        process=ProcessKind.ROAD,
        priority=priority,
    )


def _graph(edges):
    from terrain_extraction.osm_extraction.models import TopologyGraph, TopologyNode

    nodes = {}
    for edge in edges:
        nodes.setdefault(edge.start_node_id, Point(edge.geometry.coords[0]))
        nodes.setdefault(edge.end_node_id, Point(edge.geometry.coords[-1]))
    return TopologyGraph(
        nodes=tuple(
            TopologyNode(node_id=node_id, point=point)
            for node_id, point in sorted(nodes.items(), key=lambda entry: entry[0])
        ),
        edges=tuple(edges),
    )


def _cross_graph():
    return _graph(
        (
            _edge(0, (0, (24, 24)), (1, (24, 40))),
            _edge(1, (0, (24, 24)), (2, (40, 24))),
            _edge(2, (3, (24, 8)), (0, (24, 24))),
            _edge(3, (4, (8, 24)), (0, (24, 24))),
        )
    )


def _skewed_y_graph():
    return _graph(
        (
            _edge(0, (0, (24, 24)), (1, (40, 24))),
            _edge(1, (0, (24, 24)), (2, (24, 40))),
            _edge(2, (0, (24, 24)), (3, (28, 40)), config_name="track", priority=6),
        )
    )


def _route(edge_id, cells):
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    tile_cells = tuple(GridCell(xidx, yidx) for xidx, yidx in cells)
    return RouteRecord(
        edge_id=edge_id,
        start_node_id=edge_id * 2,
        end_node_id=edge_id * 2 + 1,
        process=ProcessKind.ROAD,
        config_name="primary",
        priority=1,
        nodes=tuple(GridNode(cell.xidx, cell.yidx) for cell in tile_cells),
        tile_cells=tile_cells,
    )


def test_router_reroutes_around_catalog_illegal_existing_connection() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    catalog = _catalog(include_four_way=False)
    state = LinearNetworkState(width=5, height=5, catalogs={ProcessKind.ROAD: catalog})
    assert state.reserve_path(_route(10, ((1, 2), (2, 2), (3, 2)))).success
    vertical = _edge(0, (0, (20, 4)), (1, (20, 36)), priority=9)

    result = NetworkRouter(
        grid_index=_grid(width=5, height=5),
        catalogs={ProcessKind.ROAD: catalog},
        linear_state=state,
        corridor_deviation_m=16.0,
    ).route(_graph((vertical,)))
    route = result.routes[0]

    assert route.success
    assert GridCell(2, 2) not in route.tile_cells
    assert state.required_dirs(GridCell(2, 2)) == frozenset({"E", "W"})


def test_impossible_catalog_route_fails_with_tile_feasibility_diagnostics() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    catalog = _catalog(include_four_way=False, include_ns=False)
    vertical = _edge(0, (0, (20, 4)), (1, (20, 36)))

    result = NetworkRouter(
        grid_index=_grid(width=5, height=5),
        catalogs={ProcessKind.ROAD: catalog},
        corridor_deviation_m=0.0,
    ).route(_graph((vertical,)))
    route = result.routes[0]

    assert not route.success
    assert route.diagnostics["failure_reason"] == "tile_catalog_gap"
    assert route.diagnostics["tile_feasible_rejections"] >= 1
    assert result.linear_state is not None
    assert not result.linear_state.occupied


def test_same_family_overlap_failure_uses_specific_diagnostic() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    state = LinearNetworkState(width=5, height=1)
    assert state.reserve_path(_route(10, ((1, 0), (2, 0), (3, 0)))).success
    secondary = _edge(0, (0, (4, 4)), (1, (28, 4)), config_name="secondary", priority=6)

    result = NetworkRouter(
        grid_index=_grid(width=5, height=1),
        linear_state=state,
        corridor_deviation_m=0.0,
    ).route(_graph((secondary,)))
    route = result.routes[0]

    assert not route.success
    assert route.diagnostics["failure_reason"] == "unplanned_same_family_overlap"
    assert route.diagnostics["conflict_family"] == "same_family"


def test_reservation_failure_is_distinct_from_route_search_failure() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind, RouteRecord
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    catalog = _catalog(include_four_way=False)
    state = LinearNetworkState(width=5, height=5, catalogs={ProcessKind.ROAD: catalog})
    assert state.reserve_path(_route(10, ((1, 2), (2, 2), (3, 2)))).success
    rejected = RouteRecord(
        edge_id=11,
        start_node_id=0,
        end_node_id=1,
        process=ProcessKind.ROAD,
        config_name="primary",
        priority=1,
        tile_cells=(GridCell(2, 1), GridCell(2, 2), GridCell(2, 3)),
    )

    route = NetworkRouter(grid_index=_grid(width=5, height=5))._reservation_failed_route(
        rejected,
        state.reserve_path(rejected),
    )

    assert not route.success
    assert route.diagnostics["failure_reason"] == "reservation_failed"
    assert route.diagnostics["linear_state_failures"][0]["failure_reason"] == "catalog_gap"


def test_four_way_anchor_routes_as_single_catalog_feasible_intersection() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    center = Point(20, 20)
    graph = _graph(
        (
            _edge(0, (0, center.coords[0]), (1, (20, 36))),
            _edge(1, (0, center.coords[0]), (2, (36, 20))),
            _edge(2, (3, (20, 4)), (0, center.coords[0])),
            _edge(3, (4, (4, 20)), (0, center.coords[0])),
        )
    )

    result = NetworkRouter(
        grid_index=_grid(width=5, height=5),
        catalogs={ProcessKind.ROAD: _catalog(include_four_way=True)},
        corridor_deviation_m=8.0,
    ).route(graph)

    assert result.failed_count == 0
    assert result.linear_state is not None
    assert result.linear_state.required_dirs(GridCell(2, 2)) == frozenset({"E", "N", "S", "W"})
    assert result.linear_state.intersection_kind_at(GridCell(2, 2)) == "four_way"


def test_split_anchor_plan_routes_incident_edges_to_split_cells() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    result = NetworkRouter(
        grid_index=_grid(width=7, height=7),
        catalogs={ProcessKind.ROAD: _catalog(include_four_way=False)},
        corridor_deviation_m=16.0,
    ).route(_cross_graph())
    plan = result.anchor_plans[0]
    split_cells = tuple(plan.split_anchor_cells)

    assert plan.plan_kind == "split"
    assert result.failed_count == 0
    assert len(split_cells) == 2
    routes = {route.edge_id: route for route in result.routes}
    assert routes[1].tile_cells[0] == GridCell(3, 3)
    assert routes[3].tile_cells[-1] == GridCell(3, 3)
    assert routes[0].tile_cells[0] == GridCell(3, 3)
    assert routes[2].tile_cells[-1] == GridCell(4, 3)
    assert plan.preserved_direction_set == ("E", "W")
    assert plan.attached_edge_ids == (0, 2)
    assert plan.dropped_edge_ids == ()
    assert result.linear_state is not None
    assert result.linear_state.required_dirs(GridCell(3, 3)) == frozenset({"E", "N", "W"})
    assert result.linear_state.required_dirs(GridCell(4, 3)) == frozenset({"E", "S", "W"})
    assert result.diagnostics["intersection_fallbacks"] == 1


def test_split_anchor_plan_drops_unattachable_arm_with_route_diagnostic() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    result = NetworkRouter(
        grid_index=_grid(width=7, height=7),
        catalogs={ProcessKind.ROAD: _ew_with_north_t_catalog()},
        corridor_deviation_m=16.0,
    ).route(_cross_graph())
    plan = result.anchor_plans[0]
    routes = {route.edge_id: route for route in result.routes}

    assert plan.preserved_direction_set == ("E", "W")
    assert plan.attached_edge_ids == (0,)
    assert plan.dropped_edge_ids == (2,)
    assert result.failed_count == 1
    assert routes[2].diagnostics["failure_reason"] == "anchor_fallback_drop"
    assert routes[2].diagnostics["intersection_fallback_decision"]["action"] == "drop"
    assert result.linear_state is not None
    assert result.linear_state.required_dirs(GridCell(3, 3)) == frozenset({"E", "N", "W"})
    assert result.diagnostics["intersection_fallback_dropped_arms"] == 1


def test_skewed_y_junction_fallback_is_deterministic_and_does_not_crash() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    kwargs = {
        "grid_index": _grid(width=7, height=7),
        "catalogs": {ProcessKind.ROAD: _catalog(include_four_way=False)},
        "corridor_deviation_m": 16.0,
    }
    first = NetworkRouter(**kwargs).route(_skewed_y_graph())
    second = NetworkRouter(**kwargs).route(_skewed_y_graph())

    assert first.routes
    assert first.failed_count == 0
    assert tuple(route.tile_cells for route in first.routes) == tuple(route.tile_cells for route in second.routes)
    assert first.diagnostics == second.diagnostics


def test_minor_corridor_retry_is_recorded_as_route_retry_mode() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    edge = _edge(0, (0, (4, 4)), (1, (36, 4)), config_name="track", priority=9)
    occupancy = OccupancyModel.from_grid_index(_grid(width=6, height=4))
    occupancy.reserve((GridCell(2, 0), GridCell(2, 1)), object_id="blocked", priority=0)

    route = NetworkRouter(
        grid_index=_grid(width=6, height=4),
        occupancy=occupancy,
        catalogs={ProcessKind.ROAD: _catalog()},
        corridor_deviation_m=8.0,
        minor_relaxation_m=24.0,
    ).route(_graph((edge,))).routes[0]

    assert route.success
    assert route.diagnostics["forced_relaxation"] == "minor_corridor"
    assert "corridor_widening" in route.diagnostics["retry_modes"]
