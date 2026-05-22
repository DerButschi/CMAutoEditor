from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from shapely.geometry.base import BaseGeometry
from terrain_extraction.osm_extraction.building_geometry import (
    EPSILON_M2,
    footprint_spec_from_diagnostics,
    reconstruct_building_footprint_polygon,
)
from terrain_extraction.osm_extraction.final_geometry import validate_final_output_geometry
from terrain_extraction.osm_extraction.models import GridCell, GridKind, LayerKind, PlacementRecord

OUTPUT_ROW_COLUMNS = ("xidx", "yidx", "z", "menu", "cat1", "cat2", "direction", "id", "name", "priority")
NORMALIZED_OUTPUT_ROW_COLUMNS = ("x", "y", "z", "menu", "cat1", "cat2", "direction", "id", "name", "priority")

_DEFAULT_PRIORITY = -999
_EMPTY_VALUE = -1
_LAYER_ORDER = {
    LayerKind.GROUND: 0,
    LayerKind.FOLIAGE: 1,
    LayerKind.LINEAR_SURFACE: 2,
    LayerKind.LINEAR_OBJECT: 3,
    LayerKind.BUILDING: 4,
    LayerKind.POINT_OBJECT: 5,
    LayerKind.RESERVED: 6,
}
_LINEAR_LAYERS = frozenset({LayerKind.LINEAR_SURFACE, LayerKind.LINEAR_OBJECT})
_INTERNAL_KEYS = frozenset(
    {"_layer", "_grid_kind", "_feature_id", "_source_order", "_cell_xidx", "_cell_yidx", "_building_type"}
)


class OutputRowValidationError(ValueError):
    """Raised when typed placements would produce invalid output rows."""


def placements_to_output_rows(
    placements: Sequence[PlacementRecord],
    *,
    include_internal: bool = False,
    grid_index: Any | None = None,
    profile: str = "cold_war",
) -> tuple[Mapping[str, Any], ...]:
    placement_tuple = tuple(placements)
    _validate_building_placements(placement_tuple, grid_index=grid_index)
    rows = _rows_from_placements(placement_tuple)
    if grid_index is not None:
        final_geometry = validate_final_output_geometry(rows, grid_index, profile=profile)
        if not final_geometry.is_valid:
            raise OutputRowValidationError(final_geometry.issue_summary())
    validate_output_rows(rows)
    if include_internal:
        return tuple(rows)
    return tuple(_strip_internal(row) for row in rows)


def append_extent_marker(
    rows: Iterable[Mapping[str, Any]],
    *,
    bounds: Sequence[int | float],
    include_internal: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    out_rows = [dict(row) for row in rows]
    xmax, ymax = bounds[2], bounds[3]
    if not any(row.get("name") == "extent_marker" and row.get("xidx") == xmax and row.get("yidx") == ymax for row in out_rows):
        marker = {
            "xidx": _clean_number(xmax),
            "yidx": _clean_number(ymax),
            "z": _EMPTY_VALUE,
            "menu": _EMPTY_VALUE,
            "cat1": _EMPTY_VALUE,
            "cat2": _EMPTY_VALUE,
            "direction": _EMPTY_VALUE,
            "id": _EMPTY_VALUE,
            "name": "extent_marker",
            "priority": _DEFAULT_PRIORITY,
        }
        if include_internal:
            marker.update(
                {
                    "_layer": "extent",
                    "_grid_kind": GridKind.NORMAL.value,
                    "_feature_id": None,
                    "_source_order": len(out_rows),
                    "_cell_xidx": _clean_number(xmax),
                    "_cell_yidx": _clean_number(ymax),
                }
            )
        out_rows.append(marker)
    return tuple(out_rows)


def clip_output_rows_to_bounds(
    rows: Iterable[Mapping[str, Any]],
    *,
    bounds: Sequence[int | float],
) -> tuple[Mapping[str, Any], ...]:
    xmin, ymin, xmax, ymax = bounds
    clipped: list[Mapping[str, Any]] = []
    for row in rows:
        xidx = _coordinate(row, "xidx", "x")
        yidx = _coordinate(row, "yidx", "y")
        if xidx is None or yidx is None:
            raise OutputRowValidationError(f"output row is missing coordinates: {row}")
        if xmin <= xidx <= xmax and ymin <= yidx <= ymax:
            clipped.append(row)
    return tuple(clipped)


def normalize_output_coordinates(
    rows: Iterable[Mapping[str, Any]],
    *,
    bounds: Sequence[int | float],
) -> tuple[Mapping[str, Any], ...]:
    xmin, ymin, _, _ = bounds
    normalized: list[Mapping[str, Any]] = []
    for row in clip_output_rows_to_bounds(rows, bounds=bounds):
        xidx = _coordinate(row, "xidx", "x")
        yidx = _coordinate(row, "yidx", "y")
        if xidx is None or yidx is None:
            raise OutputRowValidationError(f"output row is missing coordinates: {row}")

        normalized.append(
            {
                "x": _clean_number(xidx - xmin),
                "y": _clean_number(yidx - ymin),
                "z": row.get("z", _EMPTY_VALUE),
                "menu": row.get("menu", _EMPTY_VALUE),
                "cat1": row.get("cat1", _EMPTY_VALUE),
                "cat2": row.get("cat2", _EMPTY_VALUE),
                "direction": row.get("direction", _EMPTY_VALUE),
                "id": row.get("id", _EMPTY_VALUE),
                "name": row.get("name", _EMPTY_VALUE),
                "priority": row.get("priority", _EMPTY_VALUE),
            }
        )
    return tuple(normalized)


def validate_output_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    bounds: Sequence[int | float] | None = None,
) -> None:
    row_tuple = tuple(rows)
    _validate_profile_labels(row_tuple)
    _validate_coordinates(row_tuple, bounds=bounds)
    _validate_layer_conflicts(row_tuple)


def _rows_from_placements(placements: tuple[PlacementRecord, ...]) -> tuple[dict[str, Any], ...]:
    non_default_cells = {
        (placement.layer, cell)
        for placement in placements
        if not _is_default_placement(placement)
        for cell in placement.cells
    }
    rows: list[dict[str, Any]] = []
    for source_order, placement in enumerate(placements):
        for cell, xidx, yidx in _row_entries_for_placement(placement):
            if _is_default_placement(placement) and (placement.layer, cell) in non_default_cells:
                continue
            row = {
                "xidx": xidx,
                "yidx": yidx,
                "z": _EMPTY_VALUE,
                "menu": placement.cm_type.menu,
                "cat1": placement.cm_type.cat1,
                "cat2": _output_value(placement.cm_type.cat2),
                "direction": _output_value(placement.cm_type.direction),
                "id": _output_value(placement.cm_type.tile_id),
                "name": placement.config_name,
                "priority": placement.priority,
                "_layer": placement.layer.value,
                "_grid_kind": placement.grid_kind.value,
                "_feature_id": placement.feature_id,
                "_source_order": source_order,
                "_cell_xidx": cell.xidx,
                "_cell_yidx": cell.yidx,
            }
            building_type = placement.diagnostics.get("building_type")
            if building_type is not None:
                row["_building_type"] = building_type
            rows.append(row)
    return tuple(sorted(rows, key=_row_sort_key))


def _row_entries_for_placement(placement: PlacementRecord) -> tuple[tuple[GridCell, int | float, int | float], ...]:
    if placement.layer is LayerKind.BUILDING:
        output_xidx = placement.diagnostics.get("output_xidx")
        output_yidx = placement.diagnostics.get("output_yidx")
        if output_xidx is None or output_yidx is None:
            raise OutputRowValidationError(f"building placement is missing explicit output coordinates: {placement}")
        if not placement.cells:
            raise OutputRowValidationError(f"building placement has no blocked normal cells: {placement}")
        return ((placement.cells[0], _clean_number(float(output_xidx)), _clean_number(float(output_yidx))),)
    return tuple((cell, *_row_coordinates(cell, placement.grid_kind)) for cell in placement.cells)


def _validate_building_placements(placements: tuple[PlacementRecord, ...], *, grid_index: Any | None = None) -> None:
    linear_cells = {
        cell
        for placement in placements
        if placement.layer in _LINEAR_LAYERS
        for cell in placement.cells
    }
    for placement in placements:
        if placement.layer is not LayerKind.BUILDING:
            continue
        if placement.diagnostics.get("output_xidx") is None or placement.diagnostics.get("output_yidx") is None:
            raise OutputRowValidationError(f"building placement is missing explicit output coordinates: {placement}")
        for cell in placement.cells:
            if cell in linear_cells:
                raise OutputRowValidationError(f"building-road collision at cell ({cell.xidx}, {cell.yidx})")


def _validate_building_footprint(
    placement: PlacementRecord,
    placements: tuple[PlacementRecord, ...],
    grid_index: Any,
) -> None:
    geometry = placement.diagnostics.get("selected_footprint_polygon")
    if not isinstance(geometry, BaseGeometry):
        raise OutputRowValidationError(f"building placement is missing selected_footprint_polygon: {placement}")
    if not placement.diagnostics.get("selected_is_modular", False):
        reconstructed = reconstruct_building_footprint_polygon(
            grid_index,
            placement.diagnostics["output_xidx"],
            placement.diagnostics["output_yidx"],
            footprint_spec_from_diagnostics(placement.diagnostics),
        )
        if geometry.symmetric_difference(reconstructed).area > EPSILON_M2:
            raise OutputRowValidationError("building selected footprint is inconsistent with output coordinates")
    for linear in placements:
        if linear.layer not in _LINEAR_LAYERS:
            continue
        for cell in linear.cells:
            linear_geometry = linear.diagnostics.get("selected_footprint_polygon", grid_index.cell_polygon(cell))
            if not isinstance(linear_geometry, BaseGeometry):
                linear_geometry = grid_index.cell_polygon(cell)
            if geometry.intersection(linear_geometry).area > EPSILON_M2:
                raise OutputRowValidationError(f"building-road footprint collision at cell ({cell.xidx}, {cell.yidx})")


def _validate_profile_labels(rows: tuple[Mapping[str, Any], ...]) -> None:
    for row in rows:
        if row.get("name") == "extent_marker":
            continue
        if row.get("menu") in (None, "", _EMPTY_VALUE) or row.get("cat1") in (None, "", _EMPTY_VALUE):
            raise OutputRowValidationError(f"output row has invalid profile labels: {row}")


def _validate_coordinates(rows: tuple[Mapping[str, Any], ...], *, bounds: Sequence[int | float] | None) -> None:
    for row in rows:
        xidx = _coordinate(row, "xidx", "x")
        yidx = _coordinate(row, "yidx", "y")
        if xidx is None or yidx is None:
            raise OutputRowValidationError(f"output row is missing coordinates: {row}")
        if bounds is not None and not (bounds[0] <= xidx <= bounds[2] and bounds[1] <= yidx <= bounds[3]):
            raise OutputRowValidationError(f"output row has coordinates outside bounds: {row}")


def _validate_layer_conflicts(rows: tuple[Mapping[str, Any], ...]) -> None:
    layer_cells: dict[tuple[str, GridCell], Mapping[str, Any]] = {}
    layers_by_cell: dict[GridCell, set[str]] = {}
    for row in rows:
        layer = row.get("_layer")
        if not isinstance(layer, str) or layer == "extent":
            continue
        cell = GridCell(int(row.get("_cell_xidx", _coordinate(row, "xidx", "x"))), int(row.get("_cell_yidx", _coordinate(row, "yidx", "y"))))
        key = (layer, cell)
        if key in layer_cells:
            raise OutputRowValidationError(
                f"duplicate mutually exclusive output row for layer {layer} at cell ({cell.xidx}, {cell.yidx})"
            )
        layer_cells[key] = row
        layers_by_cell.setdefault(cell, set()).add(layer)

    for cell, layers in layers_by_cell.items():
        if LayerKind.BUILDING.value in layers and any(layer.value in layers for layer in _LINEAR_LAYERS):
            raise OutputRowValidationError(f"building-road collision at cell ({cell.xidx}, {cell.yidx})")


def _row_coordinates(cell: GridCell, grid_kind: GridKind) -> tuple[int | float, int | float]:
    if grid_kind is GridKind.DIAGONAL:
        return _clean_number(cell.xidx - 0.25), _clean_number(cell.yidx + 0.25)
    return cell.xidx, cell.yidx


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    layer = _coerce_layer(row.get("_layer"))
    return (
        row["xidx"],
        row["yidx"],
        _LAYER_ORDER.get(layer, 99),
        _is_default_row(row),
        row["priority"],
        row["name"],
        "" if row.get("_feature_id") is None else str(row["_feature_id"]),
        row.get("menu"),
        row.get("cat1"),
        row.get("cat2"),
        row.get("direction"),
        row.get("_source_order", 0),
    )


def _is_default_placement(placement: PlacementRecord) -> bool:
    return placement.priority <= _DEFAULT_PRIORITY or placement.config_name.startswith("default_")


def _is_default_row(row: Mapping[str, Any]) -> bool:
    return row.get("priority", 0) <= _DEFAULT_PRIORITY or str(row.get("name", "")).startswith("default_")


def _coerce_layer(value: Any) -> LayerKind | None:
    try:
        return LayerKind(value)
    except ValueError:
        return None


def _coordinate(row: Mapping[str, Any], primary: str, fallback: str) -> float | None:
    value = row.get(primary, row.get(fallback))
    if value is None:
        return None
    return float(value)


def _output_value(value: Any) -> Any:
    return _EMPTY_VALUE if value is None else value


def _clean_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _strip_internal(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return {column: row[column] for column in OUTPUT_ROW_COLUMNS if column in row and column not in _INTERNAL_KEYS}
