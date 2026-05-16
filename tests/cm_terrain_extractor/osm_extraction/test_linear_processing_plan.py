from __future__ import annotations

import sys
from pathlib import Path

from shapely.geometry import LineString, Point

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid(width: int = 6, height: int = 6):
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


def _edge(edge_id, start, end, *, process, config_name="primary", priority=1):
    from terrain_extraction.osm_extraction.models import TopologyEdge

    return TopologyEdge(
        edge_id=edge_id,
        start_node_id=start[0],
        end_node_id=end[0],
        geometry=LineString([start[1], end[1]]),
        feature_ids=(f"{process.value}-{edge_id}",),
        source_indices=(edge_id,),
        config_name=config_name,
        process=process,
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


def _route(edge_id, cells, *, process, priority=1, config_name="primary"):
    from terrain_extraction.osm_extraction.models import GridCell, GridNode, RouteRecord

    tile_cells = tuple(GridCell(xidx, yidx) for xidx, yidx in cells)
    nodes = tuple(GridNode(cell.xidx, cell.yidx) for cell in tile_cells)
    return RouteRecord(
        edge_id=edge_id,
        start_node_id=edge_id * 2,
        end_node_id=edge_id * 2 + 1,
        process=process,
        config_name=config_name,
        priority=priority,
        nodes=nodes,
        tile_cells=tile_cells,
    )


def test_processing_plan_groups_edges_by_process_stage_priority_and_config_rank() -> None:
    from terrain_extraction.osm_extraction.linear_processing_plan import LinearProcessingPlan
    from terrain_extraction.osm_extraction.models import ProcessKind

    graph = _graph(
        (
            _edge(0, (0, (20, 4)), (1, (20, 36)), process=ProcessKind.FENCE, priority=0),
            _edge(1, (2, (4, 20)), (3, (36, 20)), process=ProcessKind.ROAD, config_name="track", priority=9),
            _edge(2, (4, (4, 28)), (5, (36, 28)), process=ProcessKind.ROAD, config_name="primary", priority=1),
            _edge(3, (6, (4, 12)), (7, (36, 12)), process=ProcessKind.STREAM, priority=0),
        )
    )

    plan = LinearProcessingPlan.from_topology(graph)

    assert [(group.process, group.config_name, group.priority, group.edge_ids) for group in plan.groups] == [
        (ProcessKind.ROAD, "primary", 1, (2,)),
        (ProcessKind.ROAD, "track", 9, (1,)),
        (ProcessKind.STREAM, "primary", 0, (3,)),
        (ProcessKind.FENCE, "primary", 0, (0,)),
    ]
    assert plan.diagnostics["linear_processing_groups"] == 4
    assert plan.diagnostics["linear_processing_stage_order"] == ("road", "stream", "fence")


def test_default_interaction_policy_allows_road_attachment_but_rejects_road_fence_merge() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.linear_processing_plan import (
        default_linear_interaction_policy,
    )
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind

    policy = default_linear_interaction_policy()
    assert policy.decision(ProcessKind.ROAD, ProcessKind.ROAD).interaction == "connect"
    assert policy.decision(ProcessKind.ROAD, ProcessKind.FENCE).interaction == "avoid"
    assert policy.decision(ProcessKind.ROAD, ProcessKind.STREAM).interaction == "avoid"

    state = LinearNetworkState(width=4, height=4, interaction_policy=policy)
    assert state.reserve_path(_route(1, ((0, 1), (1, 1), (2, 1)), process=ProcessKind.ROAD, priority=1)).success

    attachment = state.reserve_path(_route(2, ((1, 0), (1, 1)), process=ProcessKind.ROAD, priority=9))
    assert attachment.success
    assert state.required_dirs(GridCell(1, 1)) == frozenset({"E", "S", "W"})

    fence_merge = state.reserve_path(_route(3, ((2, 0), (2, 1)), process=ProcessKind.FENCE, priority=0))
    assert fence_merge.success is False
    assert fence_merge.failures[0]["failure_reason"] == "process_avoidance"

    stream_merge = state.reserve_path(_route(4, ((2, 0), (2, 1)), process=ProcessKind.STREAM, priority=0))
    assert stream_merge.success is False
    assert stream_merge.failures[0]["failure_reason"] == "process_avoidance"


def test_router_processes_roads_before_fences_even_when_fence_has_stronger_numeric_priority() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    graph = _graph(
        (
            _edge(0, (0, (20, 4)), (1, (20, 36)), process=ProcessKind.FENCE, priority=0),
            _edge(1, (2, (4, 20)), (3, (36, 20)), process=ProcessKind.ROAD, priority=5),
        )
    )

    result = NetworkRouter(grid_index=_grid(width=6, height=6), corridor_deviation_m=16.0).route(graph)

    assert [route.process for route in result.routes] == [ProcessKind.ROAD, ProcessKind.FENCE]
    assert result.routes[0].tile_cells == (
        GridCell(0, 2),
        GridCell(1, 2),
        GridCell(2, 2),
        GridCell(3, 2),
        GridCell(4, 2),
    )
    assert result.linear_state is not None
    assert result.linear_state.process_at_cell[GridCell(2, 2)] is ProcessKind.ROAD
    assert result.routes[1].diagnostics["processing_stage"] > result.routes[0].diagnostics["processing_stage"]
    assert result.diagnostics["linear_processing_groups"] == 2
    assert result.diagnostics["process_pair_policy"][(ProcessKind.ROAD.value, ProcessKind.FENCE.value)] == "avoid"
