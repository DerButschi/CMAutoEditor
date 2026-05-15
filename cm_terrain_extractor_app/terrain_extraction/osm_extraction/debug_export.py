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
from shapely.ops import unary_union
from terrain_extraction.osm_extraction.models import (
    GridCell,
    GridKind,
    GridNode,
    LayerKind,
    PlacementRecord,
)


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
    output_rows: Iterable[Mapping[str, Any]] = (),
    grid_index: Any = None,
    stats: Any = None,
    bounds: tuple[int | float, int | float, int | float, int | float] | None = None,
) -> DebugExportResult:
    layer_errors: list[dict[str, Any]] = []
    placement_tuple = tuple(placements)
    layers = {
        **_layer("source_features", lambda: _source_features_layer(features, grid_index, layer_errors), layer_errors),
        **_layer("topology_nodes", lambda: _topology_nodes_layer(topology, grid_index), layer_errors),
        **_layer("topology_edges", lambda: _topology_edges_layer(topology, grid_index), layer_errors),
        **_layer("routed_paths", lambda: _routed_paths_layer(routing, grid_index), layer_errors),
        **_layer("route_anchors", lambda: _route_anchors_layer(routing, grid_index), layer_errors),
        **_occupancy_layers(occupancy, grid_index, layer_errors),
        **_layer("building_footprints", lambda: _building_footprints_layer(placement_tuple, grid_index, layer_errors), layer_errors),
        **_layer("final_rows", lambda: _final_rows_layer(output_rows, grid_index, bounds=bounds, layer_errors=layer_errors), layer_errors),
    }
    diagnostics = {
        "layer_errors": tuple(layer_errors),
        "topology": dict(getattr(topology, "diagnostics", {}) or {}),
        "routing": dict(getattr(routing, "diagnostics", {}) or {}),
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
            try:
                geometry = unary_union([grid_index.cell_polygon(cell) for cell in placement.cells])
            except Exception as exc:
                layer_errors.append({"layer": "building_footprints", "index": index, "error": f"{type(exc).__name__}: {exc}"})
                continue
        rows.append(_placement_row(placement, include_score=True))
        geometries.append(geometry)
    return _gdf(rows, geometries, grid_index)


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
        points = [grid_index.cell_center(cell) for cell in route.cells]
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
