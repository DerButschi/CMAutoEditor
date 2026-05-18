from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
ROOT_DIR = Path(__file__).parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


_DIRECTION_COLUMNS = {
    "N": "u",
    "S": "d",
    "E": "r",
    "W": "l",
    "NE": "ur",
    "NW": "ul",
    "SE": "dr",
    "SW": "dl",
}
_OPPOSITE = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
    "NE": "SW",
    "SW": "NE",
    "NW": "SE",
    "SE": "NW",
}


def test_tree_dp_solves_250_cell_branched_component_quickly() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    routes = _comb_tree_routes(ProcessKind.ROAD, spine_cells=125)
    catalog = _catalog_for_all_direction_sets(ProcessKind.ROAD, variants=5, directions=("N", "E", "S", "W"))
    state = _linear_state(routes, catalog, ProcessKind.ROAD, width=130, height=3)

    started_at = time.perf_counter()
    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )
    elapsed = time.perf_counter() - started_at

    assert result.success
    assert len(result.placements) == 250
    assert elapsed < 2.0
    diagnostics = result.diagnostics["state_component_diagnostics"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["component_size"] == 250
    assert diagnostics[0]["edge_count"] == 249
    assert diagnostics[0]["max_degree"] == 3
    assert diagnostics[0]["is_tree"] is True
    assert diagnostics[0]["solver_used"] == "tree_dp"
    assert diagnostics[0]["candidate_product_log10"] > 170
    _assert_placement_edges_compatible(result.placements, state, catalog)


def test_huge_tree_candidate_product_uses_tree_dp_not_tiny_exact() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    routes = _comb_tree_routes(ProcessKind.STREAM, spine_cells=30)
    catalog = _catalog_for_all_direction_sets(ProcessKind.STREAM, variants=8, directions=("N", "E", "S", "W"))
    state = _linear_state(routes, catalog, ProcessKind.STREAM, width=35, height=3)

    result = TileAssigner({ProcessKind.STREAM: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    assert result.success
    diagnostics = result.diagnostics["state_component_diagnostics"][0]
    assert diagnostics["solver_used"] == "tree_dp"
    assert diagnostics["candidate_product_log10"] > 50
    _assert_placement_edges_compatible(result.placements, state, catalog)


def test_path_and_cycle_components_keep_dp_compatibility() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    path_routes = _edge_routes(ProcessKind.FENCE, [((0, 0), (1, 0)), ((1, 0), (2, 0))])
    cycle_routes = _edge_routes(
        ProcessKind.FENCE,
        [((5, 0), (6, 0)), ((6, 0), (6, 1)), ((6, 1), (5, 1)), ((5, 1), (5, 0))],
        start_edge_id=20,
    )
    routes = path_routes + cycle_routes
    catalog = _catalog_for_all_direction_sets(ProcessKind.FENCE, variants=3, directions=("N", "E", "S", "W"))
    state = _linear_state(routes, catalog, ProcessKind.FENCE, width=8, height=3)

    result = TileAssigner({ProcessKind.FENCE: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    assert result.success
    solver_by_size = {
        diagnostic["component_size"]: diagnostic["solver_used"]
        for diagnostic in result.diagnostics["state_component_diagnostics"]
    }
    assert solver_by_size[3] == "path_dp"
    assert solver_by_size[4] == "cycle_dp"
    _assert_placement_edges_compatible(result.placements, state, catalog)


def test_small_loopy_component_uses_bounded_cutset_dp() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    routes = _edge_routes(
        ProcessKind.RAIL,
        [
            ((0, 0), (1, 0)),
            ((1, 0), (1, 1)),
            ((1, 1), (0, 1)),
            ((0, 1), (0, 0)),
            ((0, 0), (1, 1)),
        ],
    )
    catalog = _catalog_for_all_direction_sets(
        ProcessKind.RAIL,
        variants=2,
        directions=("N", "E", "S", "W", "NE", "SW"),
    )
    state = _linear_state(routes, catalog, ProcessKind.RAIL, width=3, height=3)

    result = TileAssigner({ProcessKind.RAIL: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    assert result.success
    diagnostics = result.diagnostics["state_component_diagnostics"][0]
    assert diagnostics["is_tree"] is False
    assert diagnostics["is_cycle"] is False
    assert diagnostics["cycle_count"] == 2
    assert diagnostics["solver_used"] == "cutset_dp"
    _assert_placement_edges_compatible(result.placements, state, catalog)


def test_large_cycle_rank_component_fails_with_structured_limit_diagnostic() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

    routes = _grid_routes(ProcessKind.ROAD, width=3, height=3)
    catalog = _catalog_for_all_direction_sets(ProcessKind.ROAD, variants=5, directions=("N", "E", "S", "W"))
    state = _linear_state(routes, catalog, ProcessKind.ROAD, width=4, height=4)

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    assert not result.success
    assert result.failures[0]["failure_reason"] == "component_solver_limit_exceeded"
    diagnostics = result.diagnostics["state_component_diagnostics"][0]
    assert diagnostics["cycle_count"] == 4
    assert diagnostics["solver_used"] == "unresolved"
    assert diagnostics["candidate_product_log10"] > 5
    assert result.failures[0]["component_diagnostics"]["solver_used"] == "unresolved"


def test_pipeline_warn_and_strict_modes_handle_solver_limit_failures() -> None:
    import pytest
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, TileAssignmentError

    routes = _grid_routes(ProcessKind.ROAD, width=3, height=3)
    catalog = _catalog_for_all_direction_sets(ProcessKind.ROAD, variants=5, directions=("N", "E", "S", "W"))
    state = _linear_state(routes, catalog, ProcessKind.ROAD, width=4, height=4)
    pipeline = _GraphSolverPipelineHarness(routes, state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )
    feature = FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)]))
    grid_index = GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=4,
        height=4,
    )

    warn_result = pipeline.run(
        features=(feature,),
        config=ExtractionConfig.from_mapping({"road_validation_mode": "warn"}),
        grid_index=grid_index,
        bounds=(0, 0, 3, 3),
        linear_catalog_provider=lambda _features: {ProcessKind.ROAD: catalog},
        road_validation_mode="warn",
    )

    assert warn_result.placements == ()
    assert warn_result.diagnostics["tile_assignment_failures"][0]["failure_reason"] == "component_solver_limit_exceeded"
    assert warn_result.diagnostics["tile_assignment_suppressed_placements"] == ()

    with pytest.raises(TileAssignmentError, match="component_solver_limit_exceeded"):
        pipeline.run(
            features=(feature,),
            config=ExtractionConfig.from_mapping({"road_validation_mode": "strict"}),
            grid_index=grid_index,
            bounds=(0, 0, 3, 3),
            linear_catalog_provider=lambda _features: {ProcessKind.ROAD: catalog},
            road_validation_mode="strict",
        )


def test_impossible_component_fails_without_per_cell_fallback() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog, TileAssigner

    catalog = CompiledTileCatalog.from_records(
        ({"direction": 1, "row": 0, "col": 0, "r": ("right-only",), "l": ("left-only",), "cost": 1.0},),
        process=ProcessKind.ROAD,
    )
    routes = _edge_routes(ProcessKind.ROAD, [((0, 0), (1, 0))])
    state = _linear_state(routes, catalog, ProcessKind.ROAD, width=3, height=1)

    result = TileAssigner({ProcessKind.ROAD: catalog}, rng=np.random.default_rng(12)).assign(
        routes,
        linear_state=state,
    )

    assert not result.success
    assert not result.placements
    assert result.failures[0]["failure_reason"] == "no_compatible_tile_component"
    assert result.failures[0]["component_diagnostics"]["solver_used"] == "path_dp"


def _route(edge_id, process, cells):
    from terrain_extraction.osm_extraction.models import GridCell, GridNode, RouteRecord

    grid_nodes = tuple(GridNode(xidx, yidx) for xidx, yidx in cells)
    return RouteRecord(
        edge_id=edge_id,
        start_node_id=edge_id * 2,
        end_node_id=edge_id * 2 + 1,
        process=process,
        config_name=process.value,
        priority=1,
        nodes=grid_nodes,
        tile_cells=tuple(GridCell(xidx, yidx) for xidx, yidx in cells),
    )


def _edge_routes(process, edges, *, start_edge_id=0):
    return tuple(_route(edge_id, process, cells) for edge_id, cells in enumerate(edges, start=start_edge_id))


def _comb_tree_routes(process, *, spine_cells):
    edges = [((xidx, 0), (xidx + 1, 0)) for xidx in range(spine_cells - 1)]
    edges.extend(((xidx, 0), (xidx, 1)) for xidx in range(spine_cells))
    return _edge_routes(process, edges)


def _grid_routes(process, *, width, height):
    edges = []
    for yidx in range(height):
        edges.extend(((xidx, yidx), (xidx + 1, yidx)) for xidx in range(width - 1))
    for xidx in range(width):
        edges.extend(((xidx, yidx), (xidx, yidx + 1)) for yidx in range(height - 1))
    return _edge_routes(process, edges)


def _linear_state(routes, catalog, process, *, width, height):
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState

    state = LinearNetworkState(width=width, height=height, catalogs={process: catalog})
    for route in routes:
        result = state.reserve_path(route)
        assert result.success, result.failures
    return state


def _catalog_for_all_direction_sets(process, *, variants, directions):
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    records = []
    for size in range(2, min(4, len(directions)) + 1):
        for direction_set in itertools.combinations(directions, size):
            for variant in range(variants):
                token = (f"sig-{variant}",)
                record = {
                    "direction": variant,
                    "row": len(records),
                    "col": variant,
                    "variant": variant,
                    "cost": float(variant + 1),
                }
                for direction in direction_set:
                    record[_DIRECTION_COLUMNS[direction]] = token
                records.append(record)
    return CompiledTileCatalog.from_records(records, process=process)


def _assert_placement_edges_compatible(placements, state, catalog) -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.tile_assignment import compatible_neighbor

    variants_by_identity = {
        (variant.row, variant.col, variant.variant): variant
        for variant in catalog.variants
    }
    variants_by_cell = {}
    for placement in placements:
        variants_by_cell[placement.cells[0]] = variants_by_identity[
            (
                placement.diagnostics["tile_row"],
                placement.diagnostics["tile_col"],
                placement.diagnostics["variant"],
            )
        ]
    for row in state.as_debug_layer():
        cell = GridCell(int(row["xidx"]), int(row["yidx"]))
        for direction in row["required_directions"]:
            neighbor = _neighbor(cell, direction)
            if neighbor not in variants_by_cell or (neighbor.yidx, neighbor.xidx) < (cell.yidx, cell.xidx):
                continue
            assert compatible_neighbor(variants_by_cell[cell], direction, variants_by_cell[neighbor])
            assert compatible_neighbor(variants_by_cell[neighbor], _OPPOSITE[direction], variants_by_cell[cell])


def _neighbor(cell, direction):
    from terrain_extraction.osm_extraction.models import GridCell

    offsets = {
        "N": (0, 1),
        "S": (0, -1),
        "E": (1, 0),
        "W": (-1, 0),
        "NE": (1, 1),
        "NW": (-1, 1),
        "SE": (1, -1),
        "SW": (-1, -1),
    }
    dx, dy = offsets[direction]
    return GridCell(cell.xidx + dx, cell.yidx + dy)


class _GraphSolverPipelineHarness:
    def __init__(self, routes, linear_state) -> None:
        from terrain_extraction.osm_extraction.pipeline import ExtractionContext

        self.context = ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=0,
        )
        self.routes = routes
        self.linear_state = linear_state

    from terrain_extraction.osm_extraction.pipeline import ExtractionPipeline

    run = ExtractionPipeline.run
    run_tile_assignment = ExtractionPipeline.run_tile_assignment

    def run_network_topology(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        return ExtractionResult(diagnostics={"network_topology": object()})

    def run_network_router(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult, NetworkRoutingResult

        return ExtractionResult(
            diagnostics={
                "network_routes": NetworkRoutingResult(
                    routes=self.routes,
                    linear_state=self.linear_state,
                )
            }
        )

    def run_area_rasterizer(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        return ExtractionResult()

    def run_output_rows(self, *, placements, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult
        from terrain_extraction.osm_extraction.stats import ExtractionStats

        return ExtractionResult(
            placements=placements,
            output_rows=(),
            stats=ExtractionStats(),
            diagnostics={"road_validation": object()},
        )
