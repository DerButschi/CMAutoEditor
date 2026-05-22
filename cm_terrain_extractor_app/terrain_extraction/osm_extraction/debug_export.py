from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import geopandas
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from terrain_extraction.osm_extraction.models import (
    GridCell,
    GridKind,
    GridNode,
    LayerKind,
    PlacementRecord,
)
from terrain_extraction.osm_extraction.road_output_validation import validate_road_output_rows


@dataclass(frozen=True, slots=True)
class DebugExportResult:
    layers: Mapping[str, geopandas.GeoDataFrame] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def build_debug_layers(
    *,
    features: Iterable[Any] = (),
    topology: Any = None,
    routing: Any = None,
    occupancy: Any = None,
    placements: Iterable[PlacementRecord] = (),
    tile_assignment: Any = None,
    output_rows: Iterable[Mapping[str, Any]] = (),
    grid_index: Any = None,
    stats: Any = None,
    bounds: tuple[int | float, int | float, int | float, int | float] | None = None,
) -> DebugExportResult:
    layer_errors: list[dict[str, Any]] = []
    placement_tuple = tuple(placements)
    output_row_tuple = tuple(output_rows)
    road_validation = validate_road_output_rows(output_row_tuple, profile=None)
    _validate_building_road_footprints(placement_tuple, grid_index, layer_errors)
    layers = {
        **_layer("source_features", lambda: _source_features_layer(features, grid_index, layer_errors), layer_errors),
        **_layer("topology_nodes", lambda: _topology_nodes_layer(topology, grid_index), layer_errors),
        **_layer("topology_edges", lambda: _topology_edges_layer(topology, grid_index), layer_errors),
        **_layer("routed_paths", lambda: _routed_paths_layer(routing, grid_index), layer_errors),
        **_layer("raster_spines", lambda: _raster_spines_layer(routing, grid_index), layer_errors),
        **_layer("route_anchors", lambda: _route_anchors_layer(routing, grid_index), layer_errors),
        **_layer("anchor_candidates", lambda: _anchor_candidates_layer(routing, grid_index), layer_errors),
        **_layer("selected_anchor_plans", lambda: _selected_anchor_plans_layer(routing, grid_index), layer_errors),
        **_layer("connection_bits", lambda: _connection_bits_layer(routing, grid_index), layer_errors),
        **_layer("tile_required_dirs", lambda: _tile_required_dirs_layer(placement_tuple, grid_index), layer_errors),
        **_layer("selected_tiles", lambda: _selected_tiles_layer(placement_tuple, grid_index), layer_errors),
        **_layer("tile_failures", lambda: _tile_failures_layer(tile_assignment, grid_index), layer_errors),
        **_occupancy_layers(occupancy, grid_index, layer_errors),
        **_layer("building_footprints", lambda: _building_footprints_layer(placement_tuple, grid_index, layer_errors), layer_errors),
        **_layer("road_validation", lambda: _road_validation_layer(road_validation, grid_index), layer_errors),
        **_layer("final_rows", lambda: _final_rows_layer(output_row_tuple, grid_index, bounds=bounds, layer_errors=layer_errors), layer_errors),
    }
    diagnostics = {
        "layer_errors": tuple(layer_errors),
        "topology": dict(getattr(topology, "diagnostics", {}) or {}),
        "routing": dict(getattr(routing, "diagnostics", {}) or {}),
        "road_validation": {
            "is_valid": road_validation.is_valid,
            "summary": road_validation.issue_summary(),
            "ascii_grid": road_validation.ascii_grid(),
        },
        "stats": _stats_dict(stats),
    }
    return DebugExportResult(layers={name: layer for name, layer in layers.items() if not layer.empty}, diagnostics=diagnostics)


def write_debug_geojson(
    result: DebugExportResult,
    output_dir: str | Path,
    *,
    layers: Iterable[str] | None = None,
) -> Mapping[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    selected = set(result.layers) if layers is None else set(layers)
    written: dict[str, Path] = {}
    for name, layer in result.layers.items():
        if name not in selected or layer.empty:
            continue
        path = output_path / f"{name}.geojson"
        layer.to_file(path, driver="GeoJSON")
        written[name] = path
    return MappingProxyType(written)


def _layer(
    name: str,
    build: Any,
    layer_errors: list[dict[str, Any]],
) -> dict[str, geopandas.GeoDataFrame]:
    try:
        return {name: build()}
    except Exception as exc:
        layer_errors.append({"layer": name, "error": f"{type(exc).__name__}: {exc}"})
        return {name: _empty_layer()}


def _source_features_layer(
    features: Iterable[Any],
    grid_index: Any,
    layer_errors: list[dict[str, Any]],
) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for index, feature in enumerate(features):
        geometry = getattr(feature, "geometry", None)
        if not isinstance(geometry, BaseGeometry):
            layer_errors.append({"layer": "source_features", "index": index, "error": "missing shapely geometry"})
            continue
        rows.append(
            {
                "feature_id": _scalar(getattr(feature, "feature_id", None)),
                "source_index": getattr(feature, "source_index", index),
                "config_name": getattr(feature, "config_name", None),
                "process": _process_value(getattr(feature, "process", None), "source_features", index, layer_errors),
                "priority": getattr(feature, "priority", None),
                "source_tags": _json_value(getattr(feature, "source_tags", {})),
                "source_properties": _json_value(getattr(feature, "source_properties", {})),
            }
        )
        geometries.append(geometry)
    return _gdf(rows, geometries, grid_index)


def _topology_nodes_layer(topology: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for node in getattr(topology, "nodes", ()) or ():
        rows.append(
            {
                "node_id": node.node_id,
                "source_point_count": node.source_point_count,
                "diagnostics": _json_value(node.diagnostics),
            }
        )
        geometries.append(node.point)
    return _gdf(rows, geometries, grid_index)


def _topology_edges_layer(topology: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for edge in getattr(topology, "edges", ()) or ():
        rows.append(
            {
                "edge_id": edge.edge_id,
                "start_node_id": edge.start_node_id,
                "end_node_id": edge.end_node_id,
                "feature_ids": _json_value(edge.feature_ids),
                "source_indices": _json_value(edge.source_indices),
                "config_name": edge.config_name,
                "process": edge.process.value,
                "priority": edge.priority,
                "diagnostics": _json_value(edge.diagnostics),
            }
        )
        geometries.append(edge.geometry)
    return _gdf(rows, geometries, grid_index)


def _routed_paths_layer(routing: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for route in getattr(routing, "routes", ()) or ():
        rows.append(
            {
                "edge_id": route.edge_id,
                "start_node_id": route.start_node_id,
                "end_node_id": route.end_node_id,
                "process": route.process.value,
                "config_name": route.config_name,
                "priority": route.priority,
                "success": route.success,
                "detour_ratio": route.diagnostics.get("detour_ratio"),
                "failure_reason": route.diagnostics.get("failure_reason"),
                "diagnostics": _json_value(route.diagnostics),
            }
        )
        geometries.append(_route_geometry(route, grid_index))
    return _gdf(rows, geometries, grid_index)


def _route_anchors_layer(routing: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for node_id, anchor in sorted((getattr(routing, "node_anchors", {}) or {}).items()):
        rows.append({"node_id": node_id, "xidx": anchor.xidx, "yidx": anchor.yidx})
        geometries.append(_node_point(anchor, grid_index))
    return _gdf(rows, geometries, grid_index)


def _anchor_candidates_layer(routing: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    for plan in (getattr(routing, "anchor_plans", {}) or {}).values():
        for candidate in getattr(plan, "candidates", ()) or ():
            rows.append(
                {
                    "node_id": candidate.topology_node_id,
                    "xidx": candidate.cell.xidx,
                    "yidx": candidate.cell.yidx,
                    "score": candidate.score,
                    "search_radius": candidate.search_radius,
                    "required_dirs": _json_value(getattr(candidate, "required_dirs", ())),
                    "required_dirs_estimate": _json_value(candidate.required_dirs_estimate),
                    "tile_feasible": candidate.tile_feasible,
                    "occupancy_feasible": candidate.occupancy_feasible,
                    "impossible_arm_count": getattr(candidate, "impossible_arm_count", 0),
                    "impossible_arm_severity": getattr(candidate, "impossible_arm_severity", 0.0),
                    "reasons": _json_value(candidate.reasons),
                }
            )
            geometries.append(grid_index.cell_polygon(candidate.cell))
    return _gdf(rows, geometries, grid_index)


def _selected_anchor_plans_layer(routing: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    for node_id, plan in sorted((getattr(routing, "anchor_plans", {}) or {}).items()):
        for index, cell in enumerate(_plan_cells(plan)):
            rows.append(
                {
                    "node_id": node_id,
                    "plan_kind": getattr(plan, "plan_kind", None),
                    "xidx": cell.xidx,
                    "yidx": cell.yidx,
                    "anchor_index": index,
                    "reason": getattr(plan, "reason", None),
                    "required_dirs_estimate": _json_value(getattr(plan, "required_dirs_estimate", ())),
                    "split_direction_sets": _json_value(getattr(plan, "split_direction_sets", ())),
                    "edge_anchor_cells": _json_value(_edge_anchor_cells_for_debug(plan)),
                    "preserved_direction_set": _json_value(getattr(plan, "preserved_direction_set", ())),
                    "attached_edge_ids": _json_value(getattr(plan, "attached_edge_ids", ())),
                    "dropped_edge_ids": _json_value(getattr(plan, "dropped_edge_ids", ())),
                    "fallback_decisions": _json_value(getattr(plan, "fallback_decisions", ())),
                }
            )
            geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _raster_spines_layer(routing: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    for route in getattr(routing, "routes", ()) or ():
        spine = getattr(route, "raster_spine", None)
        if spine is None:
            continue
        for cell, progress, distance_m in zip(spine.cells, spine.progress, spine.distance_m, strict=True):
            rows.append(
                {
                    "edge_id": spine.topology_edge_id,
                    "route_success": route.success,
                    "xidx": cell.xidx,
                    "yidx": cell.yidx,
                    "progress": progress,
                    "distance_m": distance_m,
                    "source_length_m": spine.source_length_m,
                }
            )
            geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _plan_cells(plan: Any) -> tuple[GridCell, ...]:
    anchor_cell = getattr(plan, "anchor_cell", None)
    if anchor_cell is not None:
        return (anchor_cell,)
    split_anchor_cells = getattr(plan, "split_anchor_cells", None)
    if split_anchor_cells:
        return tuple(split_anchor_cells)
    fallback_cell = getattr(plan, "fallback_cell", None)
    return () if fallback_cell is None else (fallback_cell,)


def _edge_anchor_cells_for_debug(plan: Any) -> tuple[Mapping[str, Any], ...]:
    edge_anchor_cells = getattr(plan, "edge_anchor_cells", None) or {}
    return tuple(
        {
            "edge_id": edge_id,
            "cell": (cell.xidx, cell.yidx),
        }
        for edge_id, cell in sorted(edge_anchor_cells.items())
    )


def _connection_bits_layer(routing: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    linear_state = getattr(routing, "linear_state", None)
    if linear_state is None:
        return _empty_layer()
    for state_row in linear_state.as_debug_layer():
        row = dict(state_row)
        cell = GridCell(row["xidx"], row["yidx"])
        rows.append(row)
        geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _tile_required_dirs_layer(placements: Iterable[PlacementRecord], grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    for placement in _linear_tile_placements(placements):
        cell = placement.cells[0]
        rows.append(
            {
                "xidx": cell.xidx,
                "yidx": cell.yidx,
                "config_name": placement.config_name,
                "feature_id": placement.feature_id,
                "process": placement.diagnostics.get("source_process") or placement.diagnostics.get("process"),
                "required_directions": tuple(placement.diagnostics.get("required_directions", ())),
                "connection_dirs": tuple(placement.diagnostics.get("connection_dirs", ())),
                "role": placement.diagnostics.get("role"),
            }
        )
        geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _selected_tiles_layer(placements: Iterable[PlacementRecord], grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    for placement in _linear_tile_placements(placements):
        cell = placement.cells[0]
        rows.append(
            {
                "xidx": cell.xidx,
                "yidx": cell.yidx,
                "config_name": placement.config_name,
                "feature_id": placement.feature_id,
                "selected_tile_id": placement.diagnostics.get("selected_tile_id"),
                "catalog_direction": placement.diagnostics.get("catalog_direction"),
                "tile_row": placement.diagnostics.get("tile_row"),
                "tile_col": placement.diagnostics.get("tile_col"),
                "variant": placement.diagnostics.get("variant"),
                "role": placement.diagnostics.get("role"),
            }
        )
        geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _tile_failures_layer(tile_assignment: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if tile_assignment is None or grid_index is None:
        return _empty_layer()
    for failure in getattr(tile_assignment, "failures", ()) or ():
        cell_value = failure.get("cell")
        if cell_value is None:
            continue
        cell = GridCell(int(cell_value[0]), int(cell_value[1]))
        rows.append(
            {
                "xidx": cell.xidx,
                "yidx": cell.yidx,
                "process": failure.get("process"),
                "required_directions": tuple(failure.get("required_directions", ())),
                "failure_reason": failure.get("failure_reason"),
                "hard_failure": bool(failure.get("hard_failure")),
            }
        )
        geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _linear_tile_placements(placements: Iterable[PlacementRecord]) -> tuple[PlacementRecord, ...]:
    return tuple(
        sorted(
            (
                placement
                for placement in placements
                if placement.cells
                and placement.layer in {LayerKind.LINEAR_SURFACE, LayerKind.LINEAR_OBJECT}
                and placement.diagnostics.get("required_directions")
            ),
            key=lambda placement: (placement.cells[0].xidx, placement.cells[0].yidx, str(placement.feature_id)),
        )
    )


def _occupancy_layers(
    occupancy: Any,
    grid_index: Any,
    layer_errors: list[dict[str, Any]],
) -> dict[str, geopandas.GeoDataFrame]:
    if occupancy is None or grid_index is None:
        return {}
    layers = {}
    for layer_kind in LayerKind:
        layers.update(
            _layer(
                f"occupancy_{layer_kind.value}",
                lambda layer_kind=layer_kind: _occupancy_layer(occupancy, grid_index, layer_kind),
                layer_errors,
            )
        )
    return layers


def _occupancy_layer(occupancy: Any, grid_index: Any, layer_kind: LayerKind) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    occupied = occupancy.occupied[layer_kind]
    for xidx in range(occupancy.width):
        for yidx in range(occupancy.height):
            if int(occupied[xidx, yidx]) < 0:
                continue
            cell = GridCell(xidx, yidx)
            object_id = occupancy.object_id_at(layer_kind, cell)
            rows.append(
                {
                    "object_id": _scalar(object_id),
                    "layer": layer_kind.value,
                    "xidx": xidx,
                    "yidx": yidx,
                    "rank": occupancy.rank_at(layer_kind, cell),
                    "metadata": _json_value(occupancy.metadata.get(object_id, {})),
                }
            )
            geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _building_footprints_layer(
    placements: tuple[PlacementRecord, ...],
    grid_index: Any,
    layer_errors: list[dict[str, Any]],
) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for index, placement in enumerate(placements):
        if placement.layer is not LayerKind.BUILDING:
            continue
        geometry = placement.diagnostics.get("selected_footprint_polygon")
        if not isinstance(geometry, BaseGeometry):
            layer_errors.append(
                {
                    "layer": "building_footprints",
                    "index": index,
                    "error": "ValueError: building placement is missing selected_footprint_polygon",
                }
            )
            continue
        rows.append(_placement_row(placement, include_score=True))
        geometries.append(geometry)
    return _gdf(rows, geometries, grid_index)


def _validate_building_road_footprints(
    placements: tuple[PlacementRecord, ...],
    grid_index: Any,
    layer_errors: list[dict[str, Any]],
) -> None:
    if grid_index is None:
        return
    linear_geometries = []
    for linear_index, placement in enumerate(placements):
        if placement.layer not in {LayerKind.LINEAR_SURFACE, LayerKind.LINEAR_OBJECT}:
            continue
        for cell in placement.cells:
            linear_geometries.append((linear_index, cell, grid_index.cell_polygon(cell)))
    if not linear_geometries:
        return
    for building_index, placement in enumerate(placements):
        if placement.layer is not LayerKind.BUILDING:
            continue
        geometry = placement.diagnostics.get("selected_footprint_polygon")
        if not isinstance(geometry, BaseGeometry):
            continue
        for linear_index, cell, linear_geometry in linear_geometries:
            if geometry.intersection(linear_geometry).area <= 1e-9:
                continue
            layer_errors.append(
                {
                    "layer": "building_road_validation",
                    "index": building_index,
                    "error": "ValueError: building footprint overlaps linear placement",
                    "linear_index": linear_index,
                    "cell": (cell.xidx, cell.yidx),
                }
            )


def _final_rows_layer(
    output_rows: Iterable[Mapping[str, Any]],
    grid_index: Any,
    *,
    bounds: tuple[int | float, int | float, int | float, int | float] | None,
    layer_errors: list[dict[str, Any]],
) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    for index, row in enumerate(output_rows):
        if row.get("name") in ("extent_marker", "default_ground", "default_foliage"):
            continue
        try:
            xidx, yidx = _row_coordinates(row, bounds)
            grid_kind = _grid_kind(row)
            geometry = _row_geometry(grid_index, xidx, yidx, grid_kind)
        except Exception as exc:
            layer_errors.append({"layer": "final_rows", "index": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({key: _scalar(value) for key, value in row.items() if not str(key).startswith("_")})
        rows[-1].update({"grid_kind": grid_kind.value, "layer": row.get("_layer")})
        geometries.append(geometry)
    return _gdf(rows, geometries, grid_index)


def _road_validation_layer(report: Any, grid_index: Any) -> geopandas.GeoDataFrame:
    rows = []
    geometries = []
    if grid_index is None:
        return _empty_layer()
    for issue in report.hard_issues:
        cell = getattr(issue, "cell", None)
        if cell is None:
            gap_cell = issue.details.get("gap_cell")
            if gap_cell is None:
                continue
            cell = GridCell(int(gap_cell[0]), int(gap_cell[1]))
        rows.append(
            {
                "stage": issue.stage,
                "reason": issue.reason,
                "message": issue.message,
                "xidx": cell.xidx,
                "yidx": cell.yidx,
                "details": _json_value(issue.details),
            }
        )
        geometries.append(grid_index.cell_polygon(cell))
    return _gdf(rows, geometries, grid_index)


def _placement_row(placement: PlacementRecord, *, include_score: bool = False) -> dict[str, Any]:
    row = {
        "config_name": placement.config_name,
        "feature_id": _scalar(placement.feature_id),
        "layer": placement.layer.value,
        "grid_kind": placement.grid_kind.value,
        "priority": placement.priority,
        "menu": placement.cm_type.menu,
        "cat1": placement.cm_type.cat1,
        "cat2": _scalar(placement.cm_type.cat2),
        "direction": _scalar(placement.cm_type.direction),
        "tile_id": _scalar(placement.cm_type.tile_id),
        "diagnostics": _json_value(placement.diagnostics),
    }
    if include_score:
        row["score"] = placement.score
    return row


def _route_geometry(route: Any, grid_index: Any) -> BaseGeometry:
    if grid_index is None:
        return LineString()
    points = [_node_point(node, grid_index) for node in route.nodes]
    if len(points) < 2:
        points = [grid_index.cell_center(cell) for cell in route.tile_cells]
    if len(points) < 2:
        return points[0] if points else LineString()
    return LineString([(point.x, point.y) for point in points])


def _node_point(node: GridNode, grid_index: Any) -> Point:
    return grid_index.cell_center(GridCell(node.xidx, node.yidx))


def _row_geometry(grid_index: Any, xidx: float, yidx: float, grid_kind: GridKind) -> BaseGeometry:
    if grid_kind is GridKind.SUB_SQUARE:
        return grid_index.sub_square_polygon(xidx, yidx)
    if grid_kind is GridKind.DIAGONAL:
        return grid_index.diagonal_polygon(xidx, yidx)
    return grid_index.cell_polygon(GridCell(int(xidx), int(yidx)))


def _row_coordinates(
    row: Mapping[str, Any],
    bounds: tuple[int | float, int | float, int | float, int | float] | None,
) -> tuple[float, float]:
    if "xidx" in row and "yidx" in row:
        return float(row["xidx"]), float(row["yidx"])
    if "x" not in row or "y" not in row:
        raise ValueError("final row is missing xidx/yidx or x/y coordinates")
    x_offset = 0 if bounds is None else bounds[0]
    y_offset = 0 if bounds is None else bounds[1]
    return float(row["x"]) + x_offset, float(row["y"]) + y_offset


def _grid_kind(row: Mapping[str, Any]) -> GridKind:
    value = row.get("_grid_kind", row.get("grid_kind", GridKind.NORMAL.value))
    try:
        return GridKind(value)
    except ValueError:
        return GridKind.NORMAL


def _gdf(rows: list[Mapping[str, Any]], geometries: list[BaseGeometry], grid_index: Any) -> geopandas.GeoDataFrame:
    gdf = geopandas.GeoDataFrame(rows, geometry=geometries)
    crs_epsg = getattr(grid_index, "crs_epsg", None)
    if crs_epsg is not None and not gdf.empty:
        gdf = gdf.set_crs(epsg=crs_epsg)
    return gdf


def _empty_layer() -> geopandas.GeoDataFrame:
    return geopandas.GeoDataFrame(geometry=[])


def _process_value(
    process: Any,
    layer: str,
    index: int,
    layer_errors: list[dict[str, Any]],
) -> str | None:
    if process is None:
        return None
    value = getattr(process, "value", None)
    if value is not None:
        return str(value)
    layer_errors.append({"layer": layer, "index": index, "error": f"unknown process value {process!r}"})
    return str(process)


def _json_value(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, BaseGeometry):
        return value.wkt
    return _scalar(value)


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _stats_dict(stats: Any) -> Mapping[str, Any]:
    if stats is None:
        return {}
    if hasattr(stats, "to_dict"):
        return stats.to_dict()
    if isinstance(stats, Mapping):
        return dict(stats)
    return {"value": str(stats)}
