from __future__ import annotations

import inspect
import sys
from dataclasses import replace
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


def _edge(edge_id, start, end, *, config_name="primary", priority=1, process=None):
    from terrain_extraction.osm_extraction.models import ProcessKind, TopologyEdge

    process = ProcessKind.ROAD if process is None else process
    return TopologyEdge(
        edge_id=edge_id,
        start_node_id=start[0],
        end_node_id=end[0],
        geometry=LineString([start[1], end[1]]),
        feature_ids=(f"road-{edge_id}",),
        source_indices=(edge_id,),
        config_name=config_name,
        process=process,
        priority=priority,
    )


def _with_authority(edge, *, cm_type_index=0, tag_rank=0, length=None, source_order=None, top_level_name=None):
    from terrain_extraction.osm_extraction.models import LinearFeatureAuthority

    authority = LinearFeatureAuthority(
        top_level_name=edge.config_name if top_level_name is None else top_level_name,
        process=edge.process,
        config_priority=edge.priority,
        cm_type_index=cm_type_index,
        first_matching_tag_index=tag_rank,
        source_feature_length_m=float(edge.geometry.length if length is None else length),
        logical_chain_length_m=float(edge.geometry.length if length is None else length),
        stable_source_order=edge.edge_id if source_order is None else source_order,
        source_feature_id=edge.feature_ids[0],
    )
    return replace(edge, linear_authority=authority)


def _compiled_catalog(process, rows):
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    return CompiledTileCatalog.from_records(rows, process=process)


def _cardinal_rows():
    return (
        {"direction": 0, "row": 0, "col": 0, "l": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 1, "col": 0, "l": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 1, "col": 1, "r": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 4, "row": 1, "col": 2, "l": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 5, "row": 1, "col": 3, "r": (2, 3), "d": (2, 3), "cost": 1.0},
    )


def _diagonal_rows():
    return (
        {"direction": 6, "row": 2, "col": 0, "ur": (2, 3), "dl": (2, 3), "cost": 0.2},
        {"direction": 7, "row": 2, "col": 1, "ul": (2, 3), "dr": (2, 3), "cost": 0.2},
    )


def _has_diagonal_step(cells) -> bool:
    return any(
        abs(first.xidx - second.xidx) == 1 and abs(first.yidx - second.yidx) == 1
        for first, second in zip(cells, cells[1:], strict=False)
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
    assert result.routes[0].tile_cells == (
        result.routes[0].tile_cells[0],
        *result.routes[0].tile_cells[1:],
    )
    assert [node.xidx for node in result.routes[0].nodes] == [0, 1, 2, 3, 4]
    assert {cell.yidx for cell in result.routes[0].tile_cells} == {0}
    assert result.routes[0].diagnostics["detour_ratio"] == pytest.approx(1.0)
    assert result.routes[0].diagnostics["elapsed_ms"] >= 0.0
    assert result.routes[0].diagnostics["attempt_count"] == 1
    assert result.routes[0].diagnostics["retry_count"] == 0
    assert result.routes[0].diagnostics["a_star_expansions"] > 0
    assert result.diagnostics["route_count"] == 1
    assert result.diagnostics["route_attempts"] == 1
    assert result.diagnostics["route_retries"] == 0
    assert result.diagnostics["total_a_star_expansions"] == result.routes[0].diagnostics["a_star_expansions"]
    assert result.diagnostics["total_tile_feasible_rejections"] == 0


def test_route_order_uses_authority_cm_type_tag_length_and_source_order() -> None:
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    edges = (
        _with_authority(_edge(0, (0, (4, 4)), (1, (28, 4))), cm_type_index=1, tag_rank=0),
        _with_authority(_edge(1, (2, (4, 12)), (3, (28, 12))), cm_type_index=0, tag_rank=0, length=24),
        _with_authority(_edge(2, (4, (4, 20)), (5, (28, 20))), cm_type_index=0, tag_rank=1, length=24),
        _with_authority(_edge(3, (6, (4, 28)), (7, (36, 28))), cm_type_index=0, tag_rank=0, length=32),
    )

    result = NetworkRouter(grid_index=_grid(width=6, height=5)).route(_graph(edges))

    assert [route.edge_id for route in result.routes] == [3, 1, 2, 0]
    assert [route.diagnostics["cm_type_index"] for route in result.routes] == [0, 0, 0, 1]
    assert [route.diagnostics["tag_rank"] for route in result.routes] == [0, 0, 1, 0]


def test_cross_family_route_failure_reports_policy_diagnostics_deterministically() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    from tests.cm_terrain_extractor.osm_extraction.test_network_topology import _feature

    topology = NetworkTopologyBuilder().build(
        (
            _feature("road-0", "road", ProcessKind.ROAD, LineString([(4, 20), (36, 20)]), priority=1),
            _feature("stream-1", "stream", ProcessKind.STREAM, LineString([(20, 4), (20, 36)]), priority=1),
        )
    )
    kwargs = {"grid_index": _grid(width=5, height=5), "corridor_deviation_m": 0.0}

    first = NetworkRouter(**kwargs).route(topology)
    second = NetworkRouter(**kwargs).route(topology)

    assert tuple(route.edge_id for route in first.routes) == tuple(route.edge_id for route in second.routes)
    assert tuple(route.tile_cells for route in first.routes) == tuple(route.tile_cells for route in second.routes)
    assert first.failed_count == 0
    stream = next(route for route in first.routes if route.process is ProcessKind.STREAM)
    assert stream.diagnostics["top_level_name"] == "stream"
    assert stream.diagnostics["conflict_family"] == "cross_family"
    assert stream.diagnostics["soft_avoid_cells"] >= 1
    assert stream.diagnostics["false_intersection_avoided"] is True
    assert stream.diagnostics["state_skipped_conflict_cells"]
    assert first.linear_state is not None
    assert all(first.linear_state.process_at_cell[cell] is not ProcessKind.STREAM for cell in stream.diagnostics["state_skipped_conflict_cells"])
    assert first.diagnostics["false_intersections_avoided"] >= 1


def test_lower_authority_same_family_overlap_shifts_without_intersection() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    primary = _with_authority(_edge(0, (0, (8, 4)), (1, (32, 4)), config_name="primary", priority=1))
    secondary = _with_authority(_edge(1, (2, (0, 6)), (3, (40, 6)), config_name="secondary", priority=6))

    result = NetworkRouter(grid_index=_grid(width=6, height=3), corridor_deviation_m=8.0).route(
        _graph((primary, secondary))
    )
    routes = {route.edge_id: route for route in result.routes}

    assert result.failed_count == 0
    assert set(routes[0].tile_cells).isdisjoint(routes[1].tile_cells)
    assert GridCell(2, 0) in routes[0].tile_cells
    assert GridCell(2, 1) in routes[1].tile_cells
    assert routes[1].diagnostics["conflict_family"] == "same_family"
    assert routes[1].diagnostics["forbidden_occupied_cells"]
    assert result.linear_state is not None
    assert result.linear_state.intersection_kind_at(GridCell(2, 0)) == "straight"


def test_same_family_true_topology_intersection_creates_junction() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    from tests.cm_terrain_extractor.osm_extraction.test_network_topology import _feature

    topology = NetworkTopologyBuilder().build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(4, 20), (36, 20)]), priority=1),
            _feature("road-1", "secondary", ProcessKind.ROAD, LineString([(20, 4), (20, 36)]), priority=6),
        )
    )

    result = NetworkRouter(grid_index=_grid(width=5, height=5), corridor_deviation_m=8.0).route(topology)

    assert result.failed_count == 0
    assert result.linear_state is not None
    assert result.linear_state.intersection_kind_at(GridCell(2, 2)) in {"t_junction", "four_way"}
    assert any(GridCell(2, 2) in route.diagnostics["planned_connect_cells"] for route in result.routes)


def test_endpoint_snap_connects_but_accidental_overlap_does_not() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    from tests.cm_terrain_extractor.osm_extraction.test_network_topology import _feature

    snapped_topology = NetworkTopologyBuilder(snap_tolerance_m=1.0).build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 4), (24, 4)]), priority=1),
            _feature("road-1", "secondary", ProcessKind.ROAD, LineString([(24.5, 4), (40, 4)]), priority=6),
        )
    )
    overlap_graph = _graph(
        (
            _with_authority(_edge(0, (0, (0, 4)), (1, (24, 4)), config_name="primary", priority=1)),
            _with_authority(_edge(1, (2, (24.5, 4)), (3, (40, 4)), config_name="secondary", priority=6)),
        )
    )

    snapped = NetworkRouter(grid_index=_grid(width=6, height=3), corridor_deviation_m=8.0).route(snapped_topology)
    overlapped = NetworkRouter(grid_index=_grid(width=6, height=3), corridor_deviation_m=8.0).route(overlap_graph)

    assert snapped.linear_state is not None
    assert snapped.linear_state.intersection_kind_at(GridCell(3, 0)) == "straight"
    assert any(GridCell(3, 0) in route.diagnostics["planned_connect_cells"] for route in snapped.routes)
    assert overlapped.linear_state is not None
    assert overlapped.linear_state.route_id_at_cell[GridCell(3, 0)] == (0,)
    assert any(GridCell(3, 0) in route.diagnostics["forbidden_occupied_cells"] for route in overlapped.routes)


def test_fence_near_road_prefers_adjacency_without_merging() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    road = _with_authority(_edge(0, (0, (0, 4)), (1, (40, 4)), config_name="primary", priority=1))
    fence = _with_authority(
        _edge(1, (2, (0, 12)), (3, (40, 12)), config_name="hedge", priority=8, process=ProcessKind.FENCE),
        top_level_name="hedge",
    )

    result = NetworkRouter(grid_index=_grid(width=6, height=3), corridor_deviation_m=8.0).route(_graph((road, fence)))
    routes = {route.edge_id: route for route in result.routes}

    assert result.failed_count == 0
    assert set(routes[0].tile_cells).isdisjoint(routes[1].tile_cells)
    assert {cell.yidx for cell in routes[1].tile_cells} == {1}
    assert GridCell(2, 1) in routes[1].diagnostics["preferred_adjacency_cells"]
    assert result.linear_state is not None
    assert all(result.linear_state.process_at_cell[cell] is ProcessKind.FENCE for cell in routes[1].tile_cells)


def test_route_cells_are_output_cells_not_grid_corner_edges() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    edge = _edge(0, (0, (4, 4)), (1, (28, 4)))
    route = NetworkRouter(grid_index=_grid()).route(_graph((edge,))).routes[0]

    assert route.success
    assert route.tile_cells == (GridCell(0, 0), GridCell(1, 0), GridCell(2, 0), GridCell(3, 0))


def test_route_follows_preserved_linestring_bend() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    edge = _edge(0, (0, (4, 4)), (1, (28, 28)))
    edge = type(edge)(
        edge.edge_id,
        edge.start_node_id,
        edge.end_node_id,
        LineString([(4, 4), (4, 28), (28, 28)]),
        edge.feature_ids,
        edge.source_indices,
        edge.config_name,
        edge.process,
        edge.priority,
    )

    route = NetworkRouter(grid_index=_grid(), corridor_deviation_m=8.0).route(_graph((edge,))).routes[0]

    assert route.success
    assert route.tile_cells == (
        GridCell(0, 0),
        GridCell(0, 1),
        GridCell(0, 2),
        GridCell(0, 3),
        GridCell(1, 3),
        GridCell(2, 3),
        GridCell(3, 3),
    )
    assert route.raster_spine is not None
    assert route.diagnostics["raster_spine_cell_count"] > 0
    assert route.diagnostics["skipped_spine_cells"] == ()
    assert route.diagnostics["extra_detour_cells"] == ()
    assert route.diagnostics["mean_spine_distance_m"] >= 0.0


def test_diagonalish_route_prefers_source_line_spine_support() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    edge = _edge(0, (0, (4, 4)), (1, (28, 20)))

    result = NetworkRouter(grid_index=_grid(), corridor_deviation_m=16.0).route(_graph((edge,)))
    route = result.routes[0]

    assert route.success
    assert route.raster_spine is not None
    assert route.raster_spine.cells[0] == GridCell(0, 0)
    assert route.raster_spine.cells[-1] == GridCell(3, 2)
    assert set(route.raster_spine.cells).issubset(set(route.tile_cells))
    assert result.diagnostics["raster_spines"] == 1


def test_diagonal_fence_routes_with_catalog_supported_diagonal_tiles() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _compiled_catalog(ProcessKind.FENCE, _diagonal_rows())
    edge = _edge(0, (0, (4, 4)), (1, (28, 28)), process=ProcessKind.FENCE)

    result = NetworkRouter(
        grid_index=_grid(),
        catalogs={ProcessKind.FENCE: catalog},
        corridor_deviation_m=8.0,
    ).route(_graph((edge,)))
    route = result.routes[0]
    assignment = TileAssigner({ProcessKind.FENCE: catalog}, rng=np.random.default_rng(12)).assign(
        result.routes,
        linear_state=result.linear_state,
    )

    assert route.success
    assert route.tile_cells == (GridCell(0, 0), GridCell(1, 1), GridCell(2, 2), GridCell(3, 3))
    assert assignment.success
    assert {placement.diagnostics["required_directions"] for placement in assignment.placements} == {("NE", "SW")}


def test_road_diagonal_source_remains_cardinal_without_diagonal_road_catalog() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    catalog = _compiled_catalog(ProcessKind.ROAD, _cardinal_rows())
    edge = _edge(0, (0, (4, 4)), (1, (28, 28)))

    route = NetworkRouter(
        grid_index=_grid(),
        catalogs={ProcessKind.ROAD: catalog},
        corridor_deviation_m=16.0,
    ).route(_graph((edge,))).routes[0]

    assert route.success
    assert not _has_diagonal_step(route.tile_cells)
    assert len(route.tile_cells) > 4


def test_road_diagonal_source_can_use_diagonal_when_road_catalog_supports_it() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    catalog = _compiled_catalog(ProcessKind.ROAD, _diagonal_rows())
    edge = _edge(0, (0, (4, 4)), (1, (28, 28)))

    route = NetworkRouter(
        grid_index=_grid(),
        catalogs={ProcessKind.ROAD: catalog},
        corridor_deviation_m=8.0,
    ).route(_graph((edge,))).routes[0]

    assert route.success
    assert route.tile_cells == (GridCell(0, 0), GridCell(1, 1), GridCell(2, 2), GridCell(3, 3))


def test_fence_without_diagonal_catalog_falls_back_to_cardinal_routing() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    catalog = _compiled_catalog(ProcessKind.FENCE, _cardinal_rows())
    edge = _edge(0, (0, (4, 4)), (1, (28, 28)), process=ProcessKind.FENCE)

    result = NetworkRouter(
        grid_index=_grid(),
        catalogs={ProcessKind.FENCE: catalog},
        corridor_deviation_m=16.0,
    ).route(_graph((edge,)))
    route = result.routes[0]

    assert route.success
    assert not _has_diagonal_step(route.tile_cells)
    assert all(
        not {"NE", "NW", "SE", "SW"}.intersection(result.linear_state.required_dirs(cell))
        for cell in route.tile_cells
    )


def test_routes_around_blocked_occupancy_inside_corridor() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    edge = _edge(0, (0, (0, 0)), (1, (40, 0)))
    occupancy = OccupancyModel.from_grid_index(_grid())
    occupancy.reserve((GridCell(2, 0),), object_id="building", priority=0)

    route = NetworkRouter(grid_index=_grid(), occupancy=occupancy, corridor_deviation_m=16.0).route(_graph((edge,))).routes[0]

    assert route.success
    assert GridCell(2, 0) not in route.tile_cells
    assert max(cell.yidx for cell in route.tile_cells) > 0
    assert route.diagnostics["blocked_cells_considered"] >= 1


def test_boundary_route_clamps_to_last_output_cell() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    edge = _edge(0, (0, (76 * 8, 175 * 8)), (1, (77 * 8, 176 * 8)))
    route = NetworkRouter(grid_index=_grid(width=100, height=176), corridor_deviation_m=16.0).route(_graph((edge,))).routes[0]

    assert route.success
    assert [node.yidx for node in route.nodes] == [175, 175]
    assert route.tile_cells == (GridCell(76, 175), GridCell(77, 175))


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
    assert max(cell.yidx for cell in route.tile_cells) == 2


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
