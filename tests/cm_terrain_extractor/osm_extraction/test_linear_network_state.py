from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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


def _catalog(*, include_four_way: bool = True):
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    rows = [
        {"direction": 0, "row": 0, "col": 0, "l": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 1, "col": 0, "l": (2, 3), "u": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 1, "col": 1, "r": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 4, "row": 1, "col": 2, "l": (2, 3), "r": (2, 3), "d": (2, 3), "cost": 1.0},
    ]
    if include_four_way:
        rows.append(
            {
                "direction": 5,
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


def _fence_diagonal_catalog():
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    rows = [
        {"direction": 0, "row": 0, "col": 0, "ur": (2, 3), "dl": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 1, "ul": (2, 3), "dr": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 1, "col": 0, "ur": (2, 3), "dl": (2, 3), "ul": (2, 3), "cost": 1.0},
        {
            "direction": 3,
            "row": 1,
            "col": 1,
            "ur": (2, 3),
            "dl": (2, 3),
            "ul": (2, 3),
            "dr": (2, 3),
            "cost": 1.0,
        },
    ]
    return CompiledTileCatalog.from_records(rows, process=ProcessKind.FENCE)


def _route(edge_id, cells, *, priority=1, process=None, config_name="primary"):
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    process = ProcessKind.ROAD if process is None else process
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


def test_reserves_straight_bend_t_and_four_way_connection_unions() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind

    state = LinearNetworkState(width=6, height=6, catalogs={ProcessKind.ROAD: _catalog()})

    assert state.reserve_path(_route(1, ((0, 2), (1, 2), (2, 2)))).success
    assert state.required_dirs(GridCell(1, 2)) == frozenset({"E", "W"})
    assert state.intersection_kind_at(GridCell(1, 2)) == "straight"

    assert state.reserve_path(_route(2, ((3, 1), (3, 2), (4, 2)))).success
    assert state.required_dirs(GridCell(3, 2)) == frozenset({"E", "S"})
    assert state.intersection_kind_at(GridCell(3, 2)) == "bend"

    assert state.reserve_path(_route(3, ((1, 1), (1, 2)))).success
    assert state.required_dirs(GridCell(1, 2)) == frozenset({"E", "S", "W"})
    assert state.intersection_kind_at(GridCell(1, 2)) == "t_junction"

    assert state.reserve_path(_route(4, ((1, 2), (1, 3)))).success
    assert state.required_dirs(GridCell(1, 2)) == frozenset({"E", "N", "S", "W"})
    assert state.intersection_kind_at(GridCell(1, 2)) == "four_way"


def test_reserves_diagonal_connection_bits_and_intersections_for_fences() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind

    state = LinearNetworkState(width=6, height=6, catalogs={ProcessKind.FENCE: _fence_diagonal_catalog()})

    assert state.reserve_path(_route(1, ((0, 0), (1, 1), (2, 2)), process=ProcessKind.FENCE)).success
    assert state.required_dirs(GridCell(1, 1)) == frozenset({"NE", "SW"})
    assert state.intersection_kind_at(GridCell(1, 1)) == "straight"

    assert state.reserve_path(_route(2, ((0, 2), (1, 1)), process=ProcessKind.FENCE)).success
    assert state.required_dirs(GridCell(1, 1)) == frozenset({"NE", "NW", "SW"})
    assert state.intersection_kind_at(GridCell(1, 1)) == "t_junction"

    assert state.reserve_path(_route(3, ((1, 1), (2, 0)), process=ProcessKind.FENCE)).success
    assert state.required_dirs(GridCell(1, 1)) == frozenset({"NE", "NW", "SE", "SW"})
    assert state.intersection_kind_at(GridCell(1, 1)) == "four_way"


def test_rejects_illegal_direction_union_without_mutating_state() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind

    center = GridCell(1, 2)
    state = LinearNetworkState(width=4, height=4, catalogs={ProcessKind.ROAD: _catalog(include_four_way=False)})

    assert state.reserve_path(_route(1, ((0, 2), (1, 2), (2, 2)))).success
    assert state.reserve_path(_route(2, ((1, 1), (1, 2)))).success
    rejected = state.reserve_path(_route(3, ((1, 2), (1, 3))))

    assert rejected.success is False
    assert rejected.failures[0]["failure_reason"] == "catalog_gap"
    assert rejected.failures[0]["required_directions"] == ("E", "N", "S", "W")
    assert state.required_dirs(center) == frozenset({"E", "S", "W"})


def test_rejects_lower_priority_overwrite_and_releases_tentative_routes() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind

    center = GridCell(1, 1)
    state = LinearNetworkState(width=4, height=4, catalogs={ProcessKind.ROAD: _catalog()})

    assert state.reserve_path(_route(1, ((0, 1), (1, 1), (2, 1)), priority=1)).success
    rejected = state.reserve_path(_route(2, ((1, 0), (1, 1), (1, 2)), priority=9))

    assert rejected.success is False
    assert rejected.failures[0]["failure_reason"] == "lower_priority_overwrite"
    assert state.required_dirs(center) == frozenset({"E", "W"})

    assert state.reserve_path(_route(99, ((3, 2), (3, 3)), priority=5)).success
    state.release_path(99)

    assert state.required_dirs(GridCell(3, 2)) == frozenset()
    assert state.required_dirs(GridCell(3, 3)) == frozenset()


def test_router_reserves_accepted_routes_into_linear_state() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        ProcessKind,
        TopologyEdge,
        TopologyGraph,
        TopologyNode,
    )
    from terrain_extraction.osm_extraction.network_routing import NetworkRouter

    graph = TopologyGraph(
        nodes=(
            TopologyNode(0, _grid().cell_center(GridCell(0, 1))),
            TopologyNode(1, _grid().cell_center(GridCell(3, 1))),
        ),
        edges=(
            TopologyEdge(0, 0, 1, LineString([(4, 12), (28, 12)]), ("road-0",), (0,), "primary", ProcessKind.ROAD, 1),
        ),
    )

    result = NetworkRouter(grid_index=_grid(), catalogs={ProcessKind.ROAD: _catalog()}).route(graph)

    assert result.successful_count == 1
    assert result.linear_state is not None
    assert result.linear_state.required_dirs(GridCell(1, 1)) == frozenset({"E", "W"})
    assert result.diagnostics["linear_state_cells"] >= 3


def test_tile_assignment_can_consume_state_derived_required_directions() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _catalog()
    state = LinearNetworkState(width=4, height=4, catalogs={ProcessKind.ROAD: catalog})
    horizontal = _route(1, ((0, 1), (1, 1), (2, 1)))
    assert state.reserve_path(horizontal).success
    assert state.reserve_path(_route(2, ((1, 0), (1, 1), (1, 2)))).success

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        (horizontal,),
        linear_state=state,
    )

    center = [placement for placement in result.placements if placement.cells == (GridCell(1, 1),)]
    assert len(center) == 1
    assert center[0].diagnostics["required_directions"] == ("E", "N", "S", "W")


def test_tile_assignment_finalizes_representable_diagonal_fence_intersection() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _fence_diagonal_catalog()
    state = LinearNetworkState(width=5, height=5, catalogs={ProcessKind.FENCE: catalog})
    routes = (
        _route(1, ((0, 0), (1, 1), (2, 2)), process=ProcessKind.FENCE, config_name="hedge"),
        _route(2, ((0, 2), (1, 1), (2, 0)), process=ProcessKind.FENCE, config_name="hedge"),
    )
    for route in routes:
        assert state.reserve_path(route).success

    result = TileAssigner({ProcessKind.FENCE: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    center = next(placement for placement in result.placements if placement.cells == (GridCell(1, 1),))
    assert result.success
    assert center.diagnostics["required_directions"] == ("NE", "NW", "SE", "SW")
    assert center.diagnostics["role"] == "intersection"
    assert center.diagnostics["intersection"] is True


def test_road_state_finalization_solves_compatible_side_signatures() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    catalog = CompiledTileCatalog.from_records(
        (
            {"direction": 1, "row": 0, "col": 0, "r": ("wide",), "l": ("narrow",), "cost": 0.1},
            {"direction": 1, "row": 1, "col": 0, "r": ("matched",), "l": ("matched",), "cost": 1.0},
        ),
        process=ProcessKind.ROAD,
    )
    state = LinearNetworkState(width=5, height=3, catalogs={ProcessKind.ROAD: catalog})
    route = _route(1, ((0, 1), (1, 1), (2, 1)))
    assert state.reserve_path(route).success

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        (route,),
        linear_state=state,
    )

    assert result.success
    assert [placement.diagnostics["tile_row"] for placement in result.placements] == [1, 1, 1]


@pytest.mark.parametrize(
    "process",
    [
        pytest.param("stream", id="stream"),
        pytest.param("fence", id="fence"),
        pytest.param("rail", id="rail"),
    ],
)
def test_linear_state_finalization_solves_exact_side_signatures_for_non_road_processes(process) -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    process_kind = ProcessKind(process)
    catalog = CompiledTileCatalog.from_records(
        (
            {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3, 4), "cost": 0.1},
            {"direction": 1, "row": 1, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        ),
        process=process_kind,
    )
    state = LinearNetworkState(width=5, height=3, catalogs={process_kind: catalog})
    route = _route(1, ((0, 1), (1, 1), (2, 1)), process=process_kind, config_name=process_kind.value)
    assert state.reserve_path(route).success

    result = TileAssigner({process_kind: catalog}, rng=np.random.default_rng(12)).assign(
        (route,),
        linear_state=state,
    )

    assert result.success, process
    assert [placement.diagnostics["tile_row"] for placement in result.placements] == [0, 1, 1]


def test_short_dead_end_road_finalizes_without_catalog_gap() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _catalog()
    state = LinearNetworkState(width=4, height=4, catalogs={ProcessKind.ROAD: catalog})
    route = _route(1, ((0, 1), (1, 1)))

    reservation = state.reserve_path(route)
    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        (route,),
        linear_state=state,
    )

    assert reservation.success
    assert result.success
    assert result.failures == ()
    assert {placement.cells[0] for placement in result.placements} == {GridCell(0, 1), GridCell(1, 1)}
    assert {placement.diagnostics["required_directions"] for placement in result.placements} == {("E", "W")}


def test_accepted_profile_catalog_endpoints_can_be_finalized() -> None:
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    from profiles.general import fence_tiles, rail_tiles, road_tiles, stream_tiles

    catalog_records = {
        ProcessKind.ROAD: road_tiles,
        ProcessKind.RAIL: rail_tiles,
        ProcessKind.STREAM: stream_tiles,
        ProcessKind.FENCE: fence_tiles,
    }

    for edge_id, (process, records) in enumerate(catalog_records.items(), start=1):
        catalog = CompiledTileCatalog.from_records(records, process=process)
        state = LinearNetworkState(width=4, height=4, catalogs={process: catalog})
        route = _route(edge_id, ((0, 1), (1, 1)), process=process, config_name=process.value)

        reservation = state.reserve_path(route)
        result = TileAssigner({process: catalog}, rng=np.random.default_rng(12)).assign(
            (route,),
            linear_state=state,
        )

        assert reservation.success, process.value
        assert result.success, process.value
        assert result.failures == (), process.value
        assert all(len(placement.diagnostics["required_directions"]) == 2 for placement in result.placements)


def test_debug_export_exposes_connection_bits_layer() -> None:
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import NetworkRoutingResult, ProcessKind

    state = LinearNetworkState(width=4, height=4, catalogs={ProcessKind.ROAD: _catalog()})
    assert state.reserve_path(_route(1, ((0, 1), (1, 1), (2, 1)))).success

    result = build_debug_layers(
        routing=NetworkRoutingResult(linear_state=state),
        grid_index=_grid(width=4, height=4),
    )

    assert "connection_bits" in result.layers
    assert result.layers["connection_bits"].loc[0, "required_directions"] == ("E",)
    assert {"xidx", "yidx", "process", "route_ids", "intersection_kind"} <= set(result.layers["connection_bits"].columns)
