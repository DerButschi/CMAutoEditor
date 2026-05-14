from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _route(edge_id, process, nodes, *, start_node_id=0, end_node_id=1, config_name="primary", cm_type=None):
    from terrain_extraction.osm_extraction.models import GridCell, GridNode, RouteRecord

    grid_nodes = tuple(GridNode(xidx, yidx) for xidx, yidx in nodes)
    cells = tuple(
        GridCell(min(a.xidx, b.xidx), min(a.yidx, b.yidx))
        for a, b in zip(grid_nodes, grid_nodes[1:], strict=False)
    )
    return RouteRecord(
        edge_id=edge_id,
        start_node_id=start_node_id,
        end_node_id=end_node_id,
        process=process,
        config_name=config_name,
        priority=1,
        nodes=grid_nodes,
        cells=cells,
        cm_type=cm_type,
    )


def _catalog_rows():
    return (
        {"direction": 0, "row": 0, "col": 0, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 2, "col": 2, "u": (2, 3), "d": (2, 3), "r": (2, 3), "l": (2, 3), "cost": 2.0},
    )


def test_catalog_compiles_direction_sets_and_required_lookup() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    catalog = CompiledTileCatalog.from_records(_catalog_rows(), process=ProcessKind.ROAD)

    assert catalog.variant_count == 3
    assert catalog.required_directions_for_nodes((0, 0), (1, 0)) == frozenset({"E", "W"})
    assert catalog.required_directions_for_nodes((3, 2), (3, 1)) == frozenset({"N", "S"})
    assert catalog.candidates_for(frozenset({"E", "W"}))[0].cm_type.cat2 == "Road Tile 1"


def test_intersection_anchor_uses_one_tile_with_unioned_directions() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    catalog = CompiledTileCatalog.from_records(_catalog_rows(), process=ProcessKind.ROAD)
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99),
        _route(1, ProcessKind.ROAD, ((1, 1), (2, 1), (3, 1)), start_node_id=99, end_node_id=1),
        _route(2, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=2, end_node_id=99),
        _route(3, ProcessKind.ROAD, ((1, 1), (1, 2), (1, 3)), start_node_id=99, end_node_id=3),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    intersection = [
        placement
        for placement in result.placements
        if placement.cells == (GridCell(1, 1),) and placement.diagnostics.get("intersection")
    ]
    assert len(intersection) == 1
    assert intersection[0].cm_type.cat2 == "Road Tile 9"
    assert intersection[0].diagnostics["required_directions"] == ("E", "N", "S", "W")


def test_boundary_intersection_uses_adjacent_route_cell() -> None:
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 0, "row": 0, "col": 0, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "l": (2, 3), "cost": 1.0},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    boundary_node = GridNode(3, 1)
    horizontal = RouteRecord(
        edge_id=0,
        start_node_id=0,
        end_node_id=99,
        process=ProcessKind.ROAD,
        config_name="primary",
        priority=1,
        nodes=(GridNode(1, 1), GridNode(2, 1), boundary_node),
        cells=(GridCell(1, 1), GridCell(2, 1)),
    )
    vertical_lower = RouteRecord(
        edge_id=1,
        start_node_id=1,
        end_node_id=99,
        process=ProcessKind.ROAD,
        config_name="primary",
        priority=1,
        nodes=(GridNode(3, 0), boundary_node),
        cells=(GridCell(2, 0),),
    )
    vertical_upper = RouteRecord(
        edge_id=2,
        start_node_id=99,
        end_node_id=2,
        process=ProcessKind.ROAD,
        config_name="primary",
        priority=1,
        nodes=(boundary_node, GridNode(3, 2), GridNode(3, 3)),
        cells=(GridCell(2, 1), GridCell(2, 2)),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        (horizontal, vertical_lower, vertical_upper)
    )

    intersections = [placement for placement in result.placements if placement.diagnostics.get("intersection")]
    assert len(intersections) == 1
    assert intersections[0].cells == (GridCell(2, 1),)
    assert all(cell.xidx <= 2 for placement in result.placements for cell in placement.cells)


def test_process_specific_tile_labels_are_generated_for_linear_catalogs() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    expected = {
        ProcessKind.ROAD: "Road Tile 1",
        ProcessKind.RAIL: "Rail Tile 1",
        ProcessKind.STREAM: "Stream Tile 1",
        ProcessKind.FENCE: "Fence Tile 1",
    }

    for process, cat2 in expected.items():
        catalog = CompiledTileCatalog.from_records((_catalog_rows()[0],), process=process)
        assert catalog.candidates_for(frozenset({"N", "S"}))[0].cm_type.cat2 == cat2


def test_route_cm_type_overrides_linear_catalog_base_type() -> None:
    from terrain_extraction.osm_extraction.models import CMType, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    base_cm_type = CMType(menu="Roads", cat1="Paved 1")
    route_cm_type = CMType(menu="Roads", cat1="Paved 2")
    catalog = CompiledTileCatalog.from_records(
        (_catalog_rows()[0],),
        process=ProcessKind.ROAD,
        base_cm_type=base_cm_type,
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        (_route(0, ProcessKind.ROAD, ((0, 0), (0, 1)), cm_type=route_cm_type),)
    )

    assert result.success
    assert result.placements[0].cm_type.menu == "Roads"
    assert result.placements[0].cm_type.cat1 == "Paved 2"
    assert result.placements[0].cm_type.cat2 == "Road Tile 1"


def test_same_type_intersection_uses_route_cm_type() -> None:
    from terrain_extraction.osm_extraction.models import CMType, GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    route_cm_type = CMType(menu="Roads", cat1="Paved 2")
    catalog = CompiledTileCatalog.from_records(
        _catalog_rows(),
        process=ProcessKind.ROAD,
        base_cm_type=CMType(menu="Roads", cat1="Paved 1"),
    )
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99, cm_type=route_cm_type),
        _route(1, ProcessKind.ROAD, ((1, 1), (2, 1), (3, 1)), start_node_id=99, end_node_id=1, cm_type=route_cm_type),
        _route(2, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=2, end_node_id=99, cm_type=route_cm_type),
        _route(3, ProcessKind.ROAD, ((1, 1), (1, 2), (1, 3)), start_node_id=99, end_node_id=3, cm_type=route_cm_type),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    intersection = [
        placement
        for placement in result.placements
        if placement.cells == (GridCell(1, 1),) and placement.diagnostics.get("intersection")
    ]
    assert len(intersection) == 1
    assert intersection[0].cm_type.cat1 == "Paved 2"


def test_mixed_intersection_uses_dominant_route_cm_type() -> None:
    from terrain_extraction.osm_extraction.models import CMType, GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    paved = CMType(menu="Roads", cat1="Paved 2")
    dirt = CMType(menu="Roads", cat1="Dirt Road")
    catalog = CompiledTileCatalog.from_records(
        _catalog_rows(),
        process=ProcessKind.ROAD,
        base_cm_type=CMType(menu="Roads", cat1="Paved 1"),
    )
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99, cm_type=paved),
        _route(1, ProcessKind.ROAD, ((1, 1), (2, 1), (3, 1)), start_node_id=99, end_node_id=1, cm_type=paved),
        _route(2, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=2, end_node_id=99, cm_type=dirt),
        _route(3, ProcessKind.ROAD, ((1, 1), (1, 2), (1, 3)), start_node_id=99, end_node_id=3, cm_type=paved),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    intersection = [
        placement
        for placement in result.placements
        if placement.cells == (GridCell(1, 1),) and placement.diagnostics.get("intersection")
    ]
    assert len(intersection) == 1
    assert intersection[0].cm_type.cat1 == "Paved 2"


def test_fixed_intersection_connection_variant_does_not_drop_route_arm() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": ("narrow",), "l": ("narrow",), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "u": ("narrow",), "d": ("narrow",), "cost": 1.0},
        {
            "direction": 0,
            "row": 2,
            "col": 2,
            "u": ("wide",),
            "d": ("wide",),
            "r": ("wide",),
            "l": ("wide",),
            "cost": 0.1,
        },
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99),
        _route(1, ProcessKind.ROAD, ((1, 1), (2, 1), (3, 1)), start_node_id=99, end_node_id=1),
        _route(2, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=2, end_node_id=99),
        _route(3, ProcessKind.ROAD, ((1, 1), (1, 2), (1, 3)), start_node_id=99, end_node_id=3),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    assert result.success
    assert result.diagnostics["intersection_assignments"] == 1


def test_impossible_intersection_records_missing_direction_diagnostic() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    catalog = CompiledTileCatalog.from_records(_catalog_rows()[:2], process=ProcessKind.ROAD)
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99),
        _route(1, ProcessKind.ROAD, ((1, 1), (2, 1), (3, 1)), start_node_id=99, end_node_id=1),
        _route(2, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=2, end_node_id=99),
        _route(3, ProcessKind.ROAD, ((1, 1), (1, 2), (1, 3)), start_node_id=99, end_node_id=3),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    assert not result.success
    assert result.diagnostics["failed_assignments"] == 1
    assert result.failures[0]["failure_reason"] == "catalog_gap"
    assert result.failures[0]["required_directions"] == ("E", "N", "S", "W")


def test_intersection_ignores_arm_that_does_not_continue_into_next_square() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 2, "col": 2, "u": (2, 3), "d": (2, 3), "l": (2, 3), "cost": 0.1},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99),
        _route(1, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=1, end_node_id=99),
        _route(2, ProcessKind.ROAD, ((1, 1), (1, 2)), start_node_id=99, end_node_id=2),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    assert result.success
    assert result.diagnostics["intersection_assignments"] == 0
    assert all(not placement.diagnostics.get("intersection") for placement in result.placements)


def test_intersection_counts_arm_that_continues_into_next_square() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 0, "row": 0, "col": 0, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 1, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 2, "col": 2, "u": (2, 3), "d": (2, 3), "l": (2, 3), "cost": 0.1},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    routes = (
        _route(0, ProcessKind.ROAD, ((0, 1), (1, 1)), start_node_id=0, end_node_id=99),
        _route(1, ProcessKind.ROAD, ((1, 0), (1, 1)), start_node_id=1, end_node_id=99),
        _route(2, ProcessKind.ROAD, ((1, 1), (1, 2), (1, 3)), start_node_id=99, end_node_id=2),
    )

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(routes)

    intersections = [placement for placement in result.placements if placement.diagnostics.get("intersection")]
    assert len(intersections) == 1
    assert intersections[0].cells == (GridCell(1, 1),)
    assert intersections[0].diagnostics["required_directions"] == ("N", "S", "W")


def test_candidate_choice_is_deterministic_for_equal_cost_variants() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0, "variant": 0},
        {"direction": 1, "row": 0, "col": 1, "r": (2, 3), "l": (2, 3), "cost": 1.0, "variant": 1},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 0), (1, 0), (2, 0)))

    first = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(7)).assign((route,))
    second = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(7)).assign((route,))

    assert [placement.cm_type.cat2 for placement in first.placements] == [
        placement.cm_type.cat2 for placement in second.placements
    ]
    assert [placement.diagnostics["variant"] for placement in first.placements] == [
        placement.diagnostics["variant"] for placement in second.placements
    ]


def test_route_turn_requires_corner_tile_not_straight_tile() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 0.1},
        {"direction": 0, "row": 0, "col": 2, "u": (2, 3), "l": (2, 3), "cost": 2.0},
        {"direction": 0, "row": 1, "col": 0, "u": (2, 3), "l": (2, 3), "cost": 0.5},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 0), (1, 0), (1, 1)))

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(11)).assign((route,))

    placements_by_cell = {placement.cells[0]: placement for placement in result.placements}
    turn = placements_by_cell[GridCell(1, 0)]
    assert turn.diagnostics["required_directions"] == ("N", "W")
    assert turn.diagnostics["tile_row"] == 1
    assert turn.diagnostics["tile_col"] == 0


def test_route_south_turn_does_not_require_diagonal_road_tile() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 1, "col": 0, "l": (2, 3), "d": (2, 3), "cost": 1.0},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 1), (1, 1), (1, 0)))

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(11)).assign((route,))

    assert result.success
    assert all("SE" not in failure["required_directions"] for failure in result.failures)


def test_stair_step_route_uses_curves_not_t_intersections() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 0, "row": 0, "col": 0, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "d": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 1, "col": 1, "u": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 2, "row": 1, "col": 0, "u": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 3, "row": 1, "col": 0, "d": (2, 3), "l": (2, 3), "cost": 1.0},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 3), (1, 3), (1, 2), (1, 1), (2, 1), (3, 1)))

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(11)).assign((route,))

    assert result.success
    assert all(len(placement.diagnostics["required_directions"]) == 2 for placement in result.placements)


def test_route_bend_does_not_emit_intersection_tile() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": (2, 3), "l": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 1, "col": 0, "u": (2, 3), "l": (2, 3), "cost": 1.0},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 0), (1, 0), (1, 1)))

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(11)).assign((route,))

    assert result.success
    assert result.diagnostics["intersection_assignments"] == 0
    assert all(not placement.diagnostics.get("intersection") for placement in result.placements)


def test_cardinal_catalog_rejects_diagonal_route_step() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    catalog = CompiledTileCatalog.from_records(_catalog_rows(), process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 0), (1, 1)))

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(11)).assign((route,))

    assert not result.success
    assert result.failures[0]["failure_reason"] == "catalog_gap"
    assert result.failures[0]["required_directions"] == ("NE", "SW")


def test_diagonal_catalog_can_assign_diagonal_route_step() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = ({"direction": 0, "row": 0, "col": 0, "ur": (2, 3), "dl": (2, 3), "cost": 1.0},)
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.FENCE)
    route = _route(0, ProcessKind.FENCE, ((0, 0), (1, 1)))

    result = TileAssigner({ProcessKind.FENCE: catalog}, rng=np.random.default_rng(11)).assign((route,))

    assert result.success
    assert result.placements[0].diagnostics["required_directions"] == ("NE", "SW")


def test_adjacent_route_tiles_must_have_matching_catalog_connections() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    records = (
        {"direction": 1, "row": 0, "col": 0, "r": ("wide",), "l": ("narrow",), "cost": 0.1},
        {"direction": 1, "row": 1, "col": 0, "r": ("matched",), "l": ("matched",), "cost": 1.0},
    )
    catalog = CompiledTileCatalog.from_records(records, process=ProcessKind.ROAD)
    route = _route(0, ProcessKind.ROAD, ((0, 0), (1, 0), (2, 0)))

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(11)).assign((route,))

    assert result.success
    assert [placement.diagnostics["tile_row"] for placement in result.placements] == [1, 1]


def test_pipeline_runs_tile_assignment_without_migration_flag() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    context = ExtractionContext(
        profile="cold_war",
        bbox=None,
        config_path="default_osm_config.json",
        seed=123,
        rng=np.random.default_rng(123),
    )
    catalog = CompiledTileCatalog.from_records(_catalog_rows(), process=ProcessKind.ROAD)

    result = ExtractionPipeline(context).run_tile_assignment(
        routes=(_route(0, ProcessKind.ROAD, ((0, 0), (1, 0))),),
        catalogs={ProcessKind.ROAD: catalog},
    )

    assert result.stats.counts["tile_assignments_succeeded"] == 1
    assert result.placements[0].cm_type.cat2 == "Road Tile 1"
    assert result.diagnostics["tile_assignment"].success
