from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from terrain_extraction.osm_extraction.models import (
    CMType,
    GridCell,
    GridKind,
    GridNode,
    LayerKind,
    PlacementRecord,
    ProcessKind,
    RouteRecord,
    TileAssignmentResult,
)

_CATALOG_DIRECTION_COLUMNS = {
    "u": "N",
    "d": "S",
    "r": "E",
    "l": "W",
    "ur": "NE",
    "ul": "NW",
    "dr": "SE",
    "dl": "SW",
}
_DIRECTION_ORDER = {"E": 0, "N": 1, "S": 2, "W": 3, "NE": 4, "NW": 5, "SE": 6, "SW": 7}
_OPPOSITE_DIRECTIONS = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
    "NE": "SW",
    "NW": "SE",
    "SE": "NW",
    "SW": "NE",
}
_LABEL_PREFIX = {
    ProcessKind.ROAD: "Road",
    ProcessKind.RAIL: "Rail",
    ProcessKind.STREAM: "Stream",
    ProcessKind.FENCE: "Fence",
}
_DEFAULT_CM_TYPES = {
    ProcessKind.ROAD: ("Roads", "Road"),
    ProcessKind.RAIL: ("Roads", "Railroad"),
    ProcessKind.STREAM: ("Roads", "Stream"),
    ProcessKind.FENCE: ("Walls/Fences", "Fence"),
}


@dataclass(frozen=True, slots=True)
class TileVariant:
    variant_id: str
    directions: frozenset[str]
    cm_type: CMType
    cost: float
    catalog_direction: int | None
    row: int
    col: int
    variant: int
    connections: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CompiledTileCatalog:
    process: ProcessKind
    variants: tuple[TileVariant, ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]] | Any,
        *,
        process: ProcessKind,
        base_cm_type: CMType | None = None,
    ) -> CompiledTileCatalog:
        normalized_records = _records_from_any(records)
        variants = tuple(
            _variant_from_record(record, process=process, base_cm_type=base_cm_type)
            for record in normalized_records
            if _directions_from_record(record)
        )
        return cls(process=process, variants=variants)

    @property
    def variant_count(self) -> int:
        return len(self.variants)

    def required_directions_for_nodes(
        self,
        first: GridNode | tuple[int, int],
        second: GridNode | tuple[int, int],
    ) -> frozenset[str]:
        direction = _direction_between_nodes(first, second)
        return frozenset((direction, _OPPOSITE_DIRECTIONS[direction]))

    def candidates_for(self, required_directions: frozenset[str]) -> tuple[TileVariant, ...]:
        return tuple(
            sorted(
                (variant for variant in self.variants if variant.directions == required_directions),
                key=_variant_sort_key,
            )
        )


class TileAssigner:
    def __init__(
        self,
        catalogs: Mapping[ProcessKind, CompiledTileCatalog],
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.catalogs = dict(catalogs)
        self.rng = rng or np.random.default_rng(0)

    def assign(self, routes: Sequence[RouteRecord]) -> TileAssignmentResult:
        successful_routes = tuple(route for route in routes if route.success and route.nodes)
        node_directions = _incident_directions_by_process(successful_routes)
        intersection_cells = self._intersection_cells(node_directions)

        placements: list[PlacementRecord] = []
        failures: list[Mapping[str, Any]] = []
        used_cells: set[tuple[ProcessKind, GridCell]] = set()

        for (process, node), required_directions in sorted(node_directions.items(), key=_node_direction_sort_key):
            if len(required_directions) < 3:
                continue
            cell = GridCell(node.xidx, node.yidx)
            placement = self._placement_for(
                process=process,
                cell=cell,
                required_directions=frozenset(required_directions),
                route=None,
                intersection=True,
                failures=failures,
            )
            if placement is not None:
                placements.append(placement)
                used_cells.add((process, cell))

        for route in successful_routes:
            catalog = self.catalogs.get(route.process)
            for step_index, (start, end) in enumerate(zip(route.nodes, route.nodes[1:], strict=False)):
                cell = route.cells[step_index] if step_index < len(route.cells) else _cell_for_step(start, end)
                if (route.process, cell) in intersection_cells or (route.process, cell) in used_cells:
                    continue
                required_directions = (
                    catalog.required_directions_for_nodes(start, end)
                    if catalog is not None
                    else _required_directions_for_nodes(start, end)
                )
                placement = self._placement_for(
                    process=route.process,
                    cell=cell,
                    required_directions=required_directions,
                    route=route,
                    intersection=False,
                    failures=failures,
                )
                if placement is not None:
                    placements.append(placement)
                    used_cells.add((route.process, cell))

        return TileAssignmentResult(
            placements=tuple(placements),
            failures=tuple(failures),
            diagnostics={
                "tile_assignments": len(placements),
                "failed_assignments": len(failures),
                "intersection_assignments": sum(
                    1 for placement in placements if placement.diagnostics.get("intersection")
                ),
            },
        )

    def _intersection_cells(
        self,
        node_directions: Mapping[tuple[ProcessKind, GridNode], set[str]],
    ) -> set[tuple[ProcessKind, GridCell]]:
        return {
            (process, GridCell(node.xidx, node.yidx))
            for (process, node), directions in node_directions.items()
            if len(directions) >= 3
        }

    def _placement_for(
        self,
        *,
        process: ProcessKind,
        cell: GridCell,
        required_directions: frozenset[str],
        route: RouteRecord | None,
        intersection: bool,
        failures: list[Mapping[str, Any]],
    ) -> PlacementRecord | None:
        catalog = self.catalogs.get(process)
        if catalog is None:
            failures.append(_failure(process, cell, required_directions, "missing_catalog"))
            return None

        candidates = catalog.candidates_for(required_directions)
        if not candidates:
            failures.append(_failure(process, cell, required_directions, "catalog_gap"))
            return None

        variant = self._choose_candidate(candidates)
        return PlacementRecord(
            layer=_layer_for_process(process),
            grid_kind=GridKind.NORMAL,
            cells=(cell,),
            config_name=route.config_name if route is not None else f"{process.value}_intersection",
            feature_id=route.edge_id if route is not None else f"intersection:{process.value}:{cell.xidx}:{cell.yidx}",
            priority=route.priority if route is not None else 0,
            cm_type=variant.cm_type,
            score=-variant.cost,
            diagnostics={
                "required_directions": _ordered_directions(required_directions),
                "intersection": intersection,
                "catalog_direction": variant.catalog_direction,
                "tile_row": variant.row,
                "tile_col": variant.col,
                "variant": variant.variant,
            },
        )

    def _choose_candidate(self, candidates: tuple[TileVariant, ...]) -> TileVariant:
        min_cost = min(candidate.cost for candidate in candidates)
        best = tuple(candidate for candidate in candidates if math.isclose(candidate.cost, min_cost))
        if len(best) == 1:
            return best[0]
        return best[int(self.rng.integers(0, len(best)))]


def _records_from_any(records: Iterable[Mapping[str, Any]] | Any) -> tuple[Mapping[str, Any], ...]:
    if hasattr(records, "to_dict"):
        return tuple(records.to_dict("records"))
    return tuple(records)


def _variant_from_record(
    record: Mapping[str, Any],
    *,
    process: ProcessKind,
    base_cm_type: CMType | None,
) -> TileVariant:
    directions = _directions_from_record(record)
    row = int(record.get("row", 0))
    col = int(record.get("col", 0))
    variant = int(record.get("variant", 0))
    catalog_direction = _optional_int(record.get("direction"))
    tile_number = row * 3 + col + 1
    direction_label = None if catalog_direction is None else f"Direction {catalog_direction + 1}"
    menu, cat1 = _base_menu_cat(process, base_cm_type)
    cm_type = CMType(
        menu=menu,
        cat1=cat1,
        cat2=f"{_LABEL_PREFIX.get(process, 'Linear')} Tile {tile_number}",
        direction=direction_label,
        tile_id=record.get("id", f"{process.value}:{catalog_direction}:{row}:{col}:{variant}"),
        modifiers={"connections": dict(_connections_from_record(record))},
    )
    return TileVariant(
        variant_id=str(cm_type.tile_id),
        directions=directions,
        cm_type=cm_type,
        cost=float(record.get("cost", 1.0)),
        catalog_direction=catalog_direction,
        row=row,
        col=col,
        variant=variant,
        connections=_connections_from_record(record),
    )


def _directions_from_record(record: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        direction
        for column, direction in _CATALOG_DIRECTION_COLUMNS.items()
        if column in record and not _is_missing(record[column])
    )


def _connections_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        _CATALOG_DIRECTION_COLUMNS[column]: record[column]
        for column in _CATALOG_DIRECTION_COLUMNS
        if column in record and not _is_missing(record[column])
    }


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _base_menu_cat(process: ProcessKind, base_cm_type: CMType | None) -> tuple[str, str]:
    if base_cm_type is not None:
        return base_cm_type.menu, base_cm_type.cat1
    return _DEFAULT_CM_TYPES.get(process, ("Linear", "Linear"))


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _direction_between_nodes(first: GridNode | tuple[int, int], second: GridNode | tuple[int, int]) -> str:
    first_x, first_y = _node_xy(first)
    second_x, second_y = _node_xy(second)
    dx = second_x - first_x
    dy = second_y - first_y
    if abs(dx) + abs(dy) != 1:
        raise ValueError(f"Only adjacent cardinal route nodes can be assigned tiles: {(first_x, first_y)} -> {(second_x, second_y)}")
    if dx == 1:
        return "E"
    if dx == -1:
        return "W"
    if dy == 1:
        return "N"
    return "S"


def _required_directions_for_nodes(first: GridNode, second: GridNode) -> frozenset[str]:
    direction = _direction_between_nodes(first, second)
    return frozenset((direction, _OPPOSITE_DIRECTIONS[direction]))


def _node_xy(node: GridNode | tuple[int, int]) -> tuple[int, int]:
    if isinstance(node, GridNode):
        return node.xidx, node.yidx
    return node


def _incident_directions_by_process(routes: Sequence[RouteRecord]) -> dict[tuple[ProcessKind, GridNode], set[str]]:
    directions: dict[tuple[ProcessKind, GridNode], set[str]] = {}
    for route in routes:
        for start, end in zip(route.nodes, route.nodes[1:], strict=False):
            direction = _direction_between_nodes(start, end)
            directions.setdefault((route.process, start), set()).add(direction)
            directions.setdefault((route.process, end), set()).add(_OPPOSITE_DIRECTIONS[direction])
    return directions


def _cell_for_step(start: GridNode, end: GridNode) -> GridCell:
    return GridCell(min(start.xidx, end.xidx), min(start.yidx, end.yidx))


def _layer_for_process(process: ProcessKind) -> LayerKind:
    if process in {ProcessKind.FENCE, ProcessKind.LINEAR, ProcessKind.RAIL}:
        return LayerKind.LINEAR_OBJECT
    return LayerKind.LINEAR_SURFACE


def _failure(
    process: ProcessKind,
    cell: GridCell,
    required_directions: frozenset[str],
    reason: str,
) -> Mapping[str, Any]:
    return {
        "process": process.value,
        "cell": (cell.xidx, cell.yidx),
        "required_directions": _ordered_directions(required_directions),
        "failure_reason": reason,
    }


def _ordered_directions(directions: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(directions, key=lambda direction: _DIRECTION_ORDER.get(direction, 99)))


def _variant_sort_key(variant: TileVariant) -> tuple[float, int, int, int, int]:
    return (
        variant.cost,
        -1 if variant.catalog_direction is None else variant.catalog_direction,
        variant.row,
        variant.col,
        variant.variant,
    )


def _node_direction_sort_key(item: tuple[tuple[ProcessKind, GridNode], set[str]]) -> tuple[str, int, int]:
    (process, node), _directions = item
    return process.value, node.xidx, node.yidx
