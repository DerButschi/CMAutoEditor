from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _catalog(*, include_four_way: bool = True):
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    rows = [
        {"direction": 6, "row": 0, "col": 2, "r": (2, 3), "cost": 1.0},
        {"direction": 7, "row": 0, "col": 2, "l": (2, 3), "cost": 1.0},
        {"direction": 8, "row": 0, "col": 2, "u": (2, 3), "cost": 1.0},
        {"direction": 9, "row": 0, "col": 2, "d": (2, 3), "cost": 1.0},
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


def _grid(width: int = 4, height: int = 4):
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


def _route(edge_id, cells, *, config_name="primary", priority=1):
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    tile_cells = tuple(GridCell(xidx, yidx) for xidx, yidx in cells)
    nodes = tuple(GridNode(cell.xidx, cell.yidx) for cell in tile_cells)
    return RouteRecord(
        edge_id=edge_id,
        start_node_id=edge_id * 2,
        end_node_id=edge_id * 2 + 1,
        process=ProcessKind.ROAD,
        config_name=config_name,
        priority=priority,
        nodes=nodes,
        tile_cells=tile_cells,
        diagnostics={"source_feature_ids": (f"osm-{edge_id}",)},
    )


def _state_with_four_way():
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import ProcessKind

    catalog = _catalog()
    state = LinearNetworkState(width=4, height=4, catalogs={ProcessKind.ROAD: catalog})
    routes = (
        _route(1, ((0, 1), (1, 1), (2, 1)), config_name="major"),
        _route(2, ((1, 0), (1, 1), (1, 2)), config_name="minor"),
    )
    for route in routes:
        assert state.reserve_path(route).success
    return state, routes


def test_state_finalizer_assigns_every_occupied_connection_cell() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _catalog()
    state, routes = _state_with_four_way()

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        (routes[0],),
        linear_state=state,
    )

    assert result.success
    assert {placement.cells[0] for placement in result.placements} == {
        GridCell(0, 1),
        GridCell(1, 0),
        GridCell(1, 1),
        GridCell(1, 2),
        GridCell(2, 1),
    }
    assert result.diagnostics["state_finalized_cells"] == 5


def test_state_finalizer_uses_route_metadata_without_synthetic_intersection_config() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _catalog()
    state, routes = _state_with_four_way()

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    center = next(placement for placement in result.placements if placement.cells == (GridCell(1, 1),))
    assert center.config_name == "major"
    assert center.config_name != "road_intersection"
    assert center.feature_id == 1
    assert center.diagnostics["source_process"] == "road"
    assert center.diagnostics["source_config"] == "major"
    assert center.diagnostics["source_feature_ids"] == ("osm-1", "osm-2")
    assert center.diagnostics["contributing_route_ids"] == (1, 2)
    assert center.diagnostics["connection_dirs"] == ("E", "N", "S", "W")
    assert center.diagnostics["selected_tile_id"] == center.cm_type.tile_id
    assert center.diagnostics["role"] == "intersection"


def test_state_finalizer_reports_missing_state_tile_as_hard_failure() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    state, routes = _state_with_four_way()
    catalog_without_four_way = _catalog(include_four_way=False)

    result = TileAssigner({ProcessKind.ROAD: catalog_without_four_way}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    assert not result.success
    assert result.failures == (
        {
            "process": "road",
            "cell": (1, 1),
            "required_directions": ("E", "N", "S", "W"),
            "failure_reason": "catalog_gap",
            "hard_failure": True,
            "route_ids": (1, 2),
        },
    )
    assert GridCell(1, 1) not in {placement.cells[0] for placement in result.placements}


def test_debug_export_exposes_tile_finalization_layers() -> None:
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
    from terrain_extraction.osm_extraction.models import NetworkRoutingResult, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    catalog = _catalog()
    state, routes = _state_with_four_way()
    assignment = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    result = build_debug_layers(
        routing=NetworkRoutingResult(linear_state=state),
        placements=assignment.placements,
        grid_index=_grid(),
    )

    assert {"tile_required_dirs", "selected_tiles"} <= set(result.layers)
    assert result.layers["tile_required_dirs"].required_directions.tolist() == [
        ("E",),
        ("N",),
        ("E", "N", "S", "W"),
        ("S",),
        ("W",),
    ]
    assert "selected_tile_id" in result.layers["selected_tiles"].columns
