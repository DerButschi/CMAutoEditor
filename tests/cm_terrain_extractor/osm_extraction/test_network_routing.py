from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, Point

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid(width: int = 8, height: int = 6):
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


def test_routes_simple_topology_edge_on_integer_grid() -> None:
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    edge = _edge(0, (0, (0, 0)), (1, (32, 0)))
    result = NetworkRouter(grid_index=_grid()).route(_graph((edge,)))

    assert result.successful_count == 1
    assert result.failed_count == 0
    assert result.routes[0].nodes == (
        result.node_anchors[0],
        *result.routes[0].nodes[1:-1],
        result.node_anchors[1],
    )
    assert result.routes[0].cells == (
        result.routes[0].cells[0],
        *result.routes[0].cells[1:],
    )
    assert [node.xidx for node in result.routes[0].nodes] == [0, 1, 2, 3, 4]
    assert {cell.yidx for cell in result.routes[0].cells} == {0}
    assert result.routes[0].diagnostics["detour_ratio"] == pytest.approx(1.0)


def test_routes_around_blocked_occupancy_inside_corridor() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    edge = _edge(0, (0, (0, 0)), (1, (40, 0)))
    occupancy = OccupancyModel.from_grid_index(_grid())
    occupancy.reserve((GridCell(2, 0),), object_id="building", priority=0)

    route = NetworkRouter(grid_index=_grid(), occupancy=occupancy, corridor_deviation_m=16.0).route(_graph((edge,))).routes[0]

    assert route.success
    assert GridCell(2, 0) not in route.cells
    assert max(cell.yidx for cell in route.cells) > 0
    assert route.diagnostics["blocked_cells_considered"] >= 1


def test_incident_edges_share_one_integer_anchor() -> None:
    from terrain_extraction.osm_extraction.models import (
        ProcessKind,
        TopologyEdge,
        TopologyGraph,
        TopologyNode,
    )
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    graph = TopologyGraph(
        nodes=(
            TopologyNode(0, Point(0, 16)),
            TopologyNode(1, Point(16, 16)),
            TopologyNode(2, Point(16, 32)),
        ),
        edges=(
            TopologyEdge(0, 0, 1, LineString([(0, 16), (16, 16)]), ("a",), (0,), "primary", ProcessKind.ROAD, 1),
            TopologyEdge(1, 1, 2, LineString([(16, 16), (16, 32)]), ("b",), (1,), "primary", ProcessKind.ROAD, 1),
        ),
    )

    result = NetworkRouter(grid_index=_grid()).route(graph)

    assert result.node_anchors[1].xidx == 2
    assert result.node_anchors[1].yidx == 2
    assert result.routes[0].nodes[-1] == result.node_anchors[1]
    assert result.routes[1].nodes[0] == result.node_anchors[1]


def test_minor_route_gets_wider_retry_before_failure() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    edge = _edge(0, (0, (0, 0)), (1, (40, 0)), config_name="track", priority=9)
    occupancy = OccupancyModel.from_grid_index(_grid(height=4))
    occupancy.reserve((GridCell(2, 0), GridCell(2, 1)), object_id="blocked", priority=0)

    route = NetworkRouter(
        grid_index=_grid(height=4),
        occupancy=occupancy,
        corridor_deviation_m=8.0,
        minor_relaxation_m=24.0,
    ).route(_graph((edge,))).routes[0]

    assert route.success
    assert route.diagnostics["forced_relaxation"] == "minor_corridor"
    assert max(cell.yidx for cell in route.cells) == 2


def test_failed_route_is_explicit_and_counted() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    edge = _edge(0, (0, (0, 0)), (1, (32, 0)))
    occupancy = OccupancyModel.from_grid_index(_grid(height=2))
    occupancy.reserve((GridCell(xidx, yidx) for xidx in range(4) for yidx in range(2)), object_id="wall", priority=0)

    result = NetworkRouter(grid_index=_grid(height=2), occupancy=occupancy, corridor_deviation_m=8.0).route(
        _graph((edge,))
    )

    assert result.successful_count == 0
    assert result.failed_count == 1
    assert result.routes[0].success is False
    assert result.routes[0].diagnostics["failure_reason"] == "no_path"
    assert result.diagnostics["failed_routes"] == 1


def test_pipeline_runs_router_without_migration_flag() -> None:
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    edge = _edge(0, (0, (0, 0)), (1, (16, 0)))
    context = ExtractionContext(
        profile="cold_war",
        bbox=None,
        config_path="default_osm_config.json",
        seed=123,
        rng=np.random.default_rng(123),
    )

    result = ExtractionPipeline(context).run_network_router(topology=_graph((edge,)), grid_index=_grid())

    assert result.stats.counts["network_routes_succeeded"] == 1
    assert result.diagnostics["network_routes"].routes[0].edge_id == 0
    assert isinstance(result.diagnostics["network_router"], NetworkRouter)


def test_network_routing_module_does_not_import_networkx() -> None:
    import terrain_extraction.osm_extraction.network_routing as network_routing

    source = inspect.getsource(network_routing).lower()

    assert "networkx" not in source
    assert "nx." not in source
