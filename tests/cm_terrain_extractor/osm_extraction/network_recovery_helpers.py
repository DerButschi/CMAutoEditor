from __future__ import annotations

# ruff: noqa: E402, I001

import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from cm_terrain_extractor_app.terrain_extraction.osm_extraction_benchmark import (  # noqa: E402
    _FeatureCollection,
    _bbox_from_fixture,
    _legacy_seed,
)
from profiles.general import road_tiles  # noqa: E402
from terrain_extraction.osm_extraction.models import CMType, GridCell, ProcessKind  # noqa: E402
from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog  # noqa: E402

NETWORK_RECOVERY_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "network_recovery"

_NEIGHBOR_DIRECTIONS = {
    (0, 1): "N",
    (1, 0): "E",
    (0, -1): "S",
    (-1, 0): "W",
}


@dataclass(frozen=True, slots=True)
class RoadGraphIssue:
    stage: str
    cell: GridCell | None
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoadGraph:
    cells: frozenset[GridCell]
    adjacency: Mapping[GridCell, frozenset[GridCell]]
    direction_sets: Mapping[GridCell, frozenset[str]]
    components: tuple[frozenset[GridCell], ...]
    illegal_direction_sets: tuple[RoadGraphIssue, ...]

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def max_degree(self) -> int:
        if not self.adjacency:
            return 0
        return max(len(neighbors) for neighbors in self.adjacency.values())


@dataclass(frozen=True, slots=True)
class ExtractionTestResult:
    fixture_name: str
    fixture_path: Path
    profile: str
    config_name: str
    seed: int | None
    output_rows: tuple[Mapping[str, Any], ...]
    post_process_rows: tuple[Mapping[str, Any], ...]
    stats: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    road_graph: RoadGraph
    debug_layers: Mapping[str, Any] = field(default_factory=dict)
    topology: Any = None
    routing: Any = None
    placements: tuple[Any, ...] = ()

    @property
    def road_component_count(self) -> int:
        return self.road_graph.component_count

    @property
    def illegal_direction_sets(self) -> tuple[RoadGraphIssue, ...]:
        return self.road_graph.illegal_direction_sets

    def ascii_grid(self, *, max_size: int = 40) -> str:
        return road_graph_ascii(self.road_graph, max_size=max_size)


def run_osm_extraction_fixture(
    fixture_name: str | Path,
    profile: str,
    config_name: str | Path,
    bbox: Any | None,
    *,
    seed: int | None = 0,
    debug: bool = True,
) -> ExtractionTestResult:
    fixture_path = _resolve_network_recovery_fixture(fixture_name)
    fixture_data = _load_json(fixture_path)
    bbox = _bbox_from_fixture(fixture_data) if bbox is None else bbox
    config_path = Path(config_name)

    from terrain_extraction import osm_processor as osm_processor_module
    from terrain_extraction.osm_processor import OSMProcessor

    osm_processor_module.st.progress = lambda *args, **kwargs: _NullProgress()
    processor = OSMProcessor(profile=profile, bbox=bbox, path_to_config=str(config_path))
    if seed is not None:
        processor.pipeline.context.seed = seed
        processor.pipeline.context.rng = np.random.default_rng(seed)

    timings: dict[str, float] = {}
    with _legacy_seed(seed):
        _time_stage(timings, "preprocess_osm_data", lambda: processor.preprocess_osm_data(_FeatureCollection(fixture_data)))
        _time_stage(timings, "run_processors", processor.run_processors)
        _time_stage(timings, "post_process", processor.post_process)
        output_df = _time_stage(timings, "output_assembly", processor.get_output)

    post_process_df = processor.df if processor.df is not None else pd.DataFrame()
    output_rows = _records(output_df)
    post_process_rows = _records(post_process_df)
    config = _load_json(config_path)
    road_config_names = _road_config_names(config)
    road_graph = reconstruct_road_graph(post_process_rows, road_config_names=road_config_names)
    debug_layers = _debug_layers(processor) if debug else {}
    diagnostics = _diagnostics(
        processor=processor,
        road_graph=road_graph,
        output_rows=output_rows,
        post_process_rows=post_process_rows,
    )

    return ExtractionTestResult(
        fixture_name=fixture_path.stem,
        fixture_path=fixture_path,
        profile=profile,
        config_name=str(config_path),
        seed=seed,
        output_rows=output_rows,
        post_process_rows=post_process_rows,
        stats={
            "timings": timings,
            "counts": {
                "fixture_features": len(fixture_data.get("features", [])),
                "matched_elements": len(getattr(processor, "matched_elements", ())),
                "output_rows": len(output_rows),
                "post_process_rows": len(post_process_rows),
                "road_cells": len(road_graph.cells),
            },
            "quality": {
                "road_components": road_graph.component_count,
                "illegal_direction_sets": len(road_graph.illegal_direction_sets),
                "max_road_degree": road_graph.max_degree,
            },
        },
        diagnostics=diagnostics,
        road_graph=road_graph,
        debug_layers=debug_layers,
        topology=getattr(processor, "topology", None),
        routing=getattr(processor, "routing", None),
        placements=tuple(getattr(processor, "placements", ())),
    )


def reconstruct_road_graph(
    rows: tuple[Mapping[str, Any], ...],
    *,
    road_config_names: frozenset[str],
) -> RoadGraph:
    cells = frozenset(
        cell
        for row in rows
        if _is_road_row(row, road_config_names=road_config_names)
        for cell in [_cell_from_row(row)]
        if cell is not None
    )
    adjacency = {
        cell: frozenset(
            GridCell(cell.xidx + dx, cell.yidx + dy)
            for dx, dy in _NEIGHBOR_DIRECTIONS
            if GridCell(cell.xidx + dx, cell.yidx + dy) in cells
        )
        for cell in cells
    }
    direction_sets = {
        cell: frozenset(_direction_to(cell, neighbor) for neighbor in neighbors)
        for cell, neighbors in adjacency.items()
    }
    components = _components(cells, adjacency)
    illegal_direction_sets = _illegal_direction_set_issues(cells, direction_sets)
    return RoadGraph(
        cells=cells,
        adjacency=adjacency,
        direction_sets=direction_sets,
        components=components,
        illegal_direction_sets=illegal_direction_sets,
    )


def road_graph_ascii(graph: RoadGraph, *, max_size: int = 40) -> str:
    if not graph.cells:
        return "<empty road graph>"

    min_x = min(cell.xidx for cell in graph.cells)
    max_x = max(cell.xidx for cell in graph.cells)
    min_y = min(cell.yidx for cell in graph.cells)
    max_y = max(cell.yidx for cell in graph.cells)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width > max_size or height > max_size:
        return f"<road graph {width}x{height} omitted; exceeds {max_size}x{max_size}>"

    component_index = {
        cell: index
        for index, component in enumerate(graph.components)
        for cell in component
    }
    rows = []
    for yidx in range(max_y, min_y - 1, -1):
        chars = []
        for xidx in range(min_x, max_x + 1):
            cell = GridCell(xidx, yidx)
            if cell not in graph.cells:
                chars.append(".")
            elif component_index.get(cell, 0) == 0:
                chars.append("R")
            else:
                chars.append(chr(ord("A") + min(component_index[cell], 25)))
        rows.append("".join(chars))
    return "\n".join(rows)


class _NullProgress:
    def progress(self, *args: object, **kwargs: object) -> _NullProgress:
        return self


def _resolve_network_recovery_fixture(fixture_name: str | Path) -> Path:
    fixture_path = Path(fixture_name)
    if fixture_path.suffix:
        return fixture_path
    return NETWORK_RECOVERY_FIXTURE_DIR / f"{fixture_path}.geojson"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _time_stage(timings: dict[str, float], stage: str, callback: Any) -> Any:
    started_at = time.perf_counter()
    result = callback()
    timings[stage] = time.perf_counter() - started_at
    return result


def _records(df: pd.DataFrame) -> tuple[Mapping[str, Any], ...]:
    if len(df) == 0:
        return ()
    return tuple(json.loads(df.to_json(orient="records")))


def _road_config_names(config: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        name
        for name, entry in config.items()
        if "road_tiles" in tuple(entry.get("process", ()))
    )


def _is_road_row(row: Mapping[str, Any], *, road_config_names: frozenset[str]) -> bool:
    return str(row.get("name")) in road_config_names


def _cell_from_row(row: Mapping[str, Any]) -> GridCell | None:
    x_value = row.get("xidx", row.get("x"))
    y_value = row.get("yidx", row.get("y"))
    if x_value is None or y_value is None:
        return None
    return GridCell(int(round(float(x_value))), int(round(float(y_value))))


def _direction_to(cell: GridCell, neighbor: GridCell) -> str:
    return _NEIGHBOR_DIRECTIONS[(neighbor.xidx - cell.xidx, neighbor.yidx - cell.yidx)]


def _components(
    cells: frozenset[GridCell],
    adjacency: Mapping[GridCell, frozenset[GridCell]],
) -> tuple[frozenset[GridCell], ...]:
    remaining = set(cells)
    components = []
    while remaining:
        root = remaining.pop()
        component = {root}
        stack = [root]
        while stack:
            cell = stack.pop()
            for neighbor in adjacency[cell]:
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda component: (-len(component), min((cell.xidx, cell.yidx) for cell in component))))


def _illegal_direction_set_issues(
    cells: frozenset[GridCell],
    direction_sets: Mapping[GridCell, frozenset[str]],
) -> tuple[RoadGraphIssue, ...]:
    catalog = CompiledTileCatalog.from_records(
        road_tiles,
        process=ProcessKind.ROAD,
        base_cm_type=CMType(menu="Roads", cat1="Road"),
    )
    issues = []
    for cell, directions in sorted(direction_sets.items(), key=lambda entry: (entry[0].xidx, entry[0].yidx)):
        if len(cells) > 1 and len(directions) == 0:
            issues.append(
                RoadGraphIssue(
                    stage="output",
                    cell=cell,
                    message="isolated road cell",
                    details={"directions": ()},
                )
            )
        if len(directions) >= 2 and not catalog.candidates_for(directions):
            issues.append(
                RoadGraphIssue(
                    stage="tile",
                    cell=cell,
                    message="road direction set has no catalog tile",
                    details={"directions": tuple(sorted(directions))},
                )
            )
    return tuple(issues)


def _debug_layers(processor: Any) -> Mapping[str, Any]:
    try:
        return processor.get_debug_layers(crs=getattr(processor.bbox, "crs_projected", None))
    except Exception as exc:
        return {"debug_layer_error": f"{type(exc).__name__}: {exc}"}


def _diagnostics(
    *,
    processor: Any,
    road_graph: RoadGraph,
    output_rows: tuple[Mapping[str, Any], ...],
    post_process_rows: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    topology = getattr(processor, "topology", None)
    routing = getattr(processor, "routing", None)
    routes = tuple(getattr(routing, "routes", ()) or ())
    failed_routes = tuple(route for route in routes if not route.success)
    return {
        "topology": {
            "node_count": len(getattr(topology, "nodes", ()) or ()),
            "edge_count": len(getattr(topology, "edges", ()) or ()),
            "components": dict(getattr(topology, "diagnostics", {}) or {}).get("components"),
        },
        "anchor": {
            "anchor_count": len(getattr(routing, "node_anchors", {}) or {}),
            "anchor_node_ids": tuple(sorted((getattr(routing, "node_anchors", {}) or {}).keys())),
        },
        "route": {
            "route_count": len(routes),
            "successful_routes": sum(1 for route in routes if route.success),
            "failed_routes": len(failed_routes),
            "failure_reasons": tuple(route.diagnostics.get("failure_reason") for route in failed_routes),
            "raster_spine_count": sum(1 for route in routes if getattr(route, "raster_spine", None) is not None),
            "raster_spine_cells": sum(
                len(getattr(getattr(route, "raster_spine", None), "cells", ()) or ()) for route in routes
            ),
        },
        "step_cell": {
            "route_lengths": tuple(
                {
                    "edge_id": route.edge_id,
                    "node_count": len(route.nodes),
                    "cell_count": len(route.tile_cells),
                }
                for route in routes
            ),
        },
        "tile": {
            "illegal_direction_sets": tuple(
                {
                    "stage": issue.stage,
                    "cell": None if issue.cell is None else (issue.cell.xidx, issue.cell.yidx),
                    "message": issue.message,
                    "details": dict(issue.details),
                }
                for issue in road_graph.illegal_direction_sets
            ),
        },
        "output": {
            "output_rows": len(output_rows),
            "post_process_rows": len(post_process_rows),
            "road_cells": len(road_graph.cells),
            "road_components": road_graph.component_count,
            "max_road_degree": road_graph.max_degree,
        },
    }
