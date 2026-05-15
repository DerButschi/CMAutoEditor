from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def _catalog_rows():
    return (
        {"direction": 0, "row": 0, "col": 0, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
    )


def _route_record(*, nodes, tile_cells):
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    return RouteRecord(
        edge_id=0,
        start_node_id=0,
        end_node_id=1,
        process=ProcessKind.ROAD,
        config_name="primary",
        priority=1,
        nodes=tuple(GridNode(xidx, yidx) for xidx, yidx in nodes),
        tile_cells=tuple(GridCell(xidx, yidx) for xidx, yidx in tile_cells),
    )


def test_route_record_exposes_explicit_tile_cells_not_ambiguous_cells() -> None:
    route = _route_record(nodes=((0, 0), (1, 0)), tile_cells=((0, 0), (1, 0)))

    assert route.tile_cells
    assert not hasattr(route, "cells")


def test_router_emits_direct_tile_cell_path_for_horizontal_road() -> None:
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
            TopologyNode(0, Point(4, 4)),
            TopologyNode(1, Point(28, 4)),
        ),
        edges=(
            TopologyEdge(
                0,
                0,
                1,
                LineString([(4, 4), (28, 4)]),
                ("road-0",),
                (0,),
                "primary",
                ProcessKind.ROAD,
                1,
            ),
        ),
    )

    route = NetworkRouter(grid_index=_grid()).route(graph).routes[0]

    assert route.success
    assert route.tile_cells == (GridCell(0, 0), GridCell(1, 0), GridCell(2, 0), GridCell(3, 0))
    assert not hasattr(route, "cells")


def test_tile_assignment_consumes_tile_cells_with_diagnostic_nodes() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    route = _route_record(nodes=((0, 0), (3, 0)), tile_cells=((0, 0), (1, 0), (2, 0), (3, 0)))
    catalog = CompiledTileCatalog.from_records(_catalog_rows(), process=ProcessKind.ROAD)

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign((route,))

    assert result.success
    assert tuple(placement.cells[0] for placement in result.placements) == (
        GridCell(0, 0),
        GridCell(1, 0),
        GridCell(2, 0),
        GridCell(3, 0),
    )


def test_diagonal_tile_cell_transition_requires_catalog_support() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    route = _route_record(nodes=((0, 0), (1, 1)), tile_cells=((0, 0), (1, 1)))
    catalog = CompiledTileCatalog.from_records(_catalog_rows(), process=ProcessKind.ROAD)

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign((route,))

    assert not result.success
    assert result.failures[0]["failure_reason"] == "catalog_gap"
    assert set(result.failures[0]["required_directions"]) == {"NE", "SW"}
