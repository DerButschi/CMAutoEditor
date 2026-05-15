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


@dataclass(frozen=True, slots=True)
class _RouteCellSpec:
    cell: GridCell
    required_directions: frozenset[str]


@dataclass(slots=True)
class _IntersectionSpec:
    process: ProcessKind
    node_id: int
    node: GridNode
    directions: set[str]
    cells: set[GridCell]
    cm_type: CMType | None = None


@dataclass(frozen=True, slots=True)
class _IntersectionEndpoint:
    process: ProcessKind
    node_id: int
    node: GridNode
    direction: str
    cells_from_node: tuple[GridCell, ...]
    cm_type: CMType | None = None


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
        intersections = _intersection_specs_by_process(successful_routes)

        placements: list[PlacementRecord] = []
        failures: list[Mapping[str, Any]] = []
        used_cells: set[tuple[ProcessKind, GridCell]] = set()
        fixed_variants: dict[tuple[ProcessKind, GridCell], TileVariant] = {}

        for intersection in sorted(intersections.values(), key=_intersection_sort_key):
            if len(intersection.directions) < 3:
                continue
            cell = _intersection_cell_for_node(intersection.node, intersection.cells)
            required = frozenset(intersection.directions)
            variant = self._variant_for(intersection.process, cell, required, failures)
            if variant is None:
                continue
            placements.append(
                self._placement_from_variant(
                    process=intersection.process,
                    cell=cell,
                    required_directions=required,
                    route=None,
                    intersection=True,
                    variant=variant,
                    cm_type_override=intersection.cm_type,
                )
            )
            used_cells.add((intersection.process, cell))
            fixed_variants[(intersection.process, cell)] = variant

        for route in successful_routes:
            catalog = self.catalogs.get(route.process)
            if catalog is None:
                for spec in _route_cell_specs(route):
                    if (route.process, spec.cell) not in used_cells:
                        failures.append(_failure(route.process, spec.cell, spec.required_directions, "missing_catalog"))
                continue
            route_placements = self._placements_for_route(
                route=route,
                catalog=catalog,
                fixed_variants=fixed_variants,
                used_cells=used_cells,
                failures=failures,
            )
            placements.extend(route_placements)

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

    def _variant_for(
        self,
        process: ProcessKind,
        cell: GridCell,
        required_directions: frozenset[str],
        failures: list[Mapping[str, Any]],
    ) -> TileVariant | None:
        catalog = self.catalogs.get(process)
        if catalog is None:
            failures.append(_failure(process, cell, required_directions, "missing_catalog"))
            return None

        candidates = catalog.candidates_for(required_directions)
        if not candidates:
            failures.append(_failure(process, cell, required_directions, "catalog_gap"))
            return None

        return self._choose_candidate(candidates)

    def _placement_from_variant(
        self,
        *,
        process: ProcessKind,
        cell: GridCell,
        required_directions: frozenset[str],
        route: RouteRecord | None,
        intersection: bool,
        variant: TileVariant,
        cm_type_override: CMType | None = None,
    ) -> PlacementRecord:
        cm_type = _variant_cm_type(variant, cm_type_override or (route.cm_type if route is not None else None))
        return PlacementRecord(
            layer=_layer_for_process(process),
            grid_kind=GridKind.NORMAL,
            cells=(cell,),
            config_name=route.config_name if route is not None else f"{process.value}_intersection",
            feature_id=route.edge_id if route is not None else f"intersection:{process.value}:{cell.xidx}:{cell.yidx}",
            priority=route.priority if route is not None else 0,
            cm_type=cm_type,
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

    def _placements_for_route(
        self,
        *,
        route: RouteRecord,
        catalog: CompiledTileCatalog,
        fixed_variants: Mapping[tuple[ProcessKind, GridCell], TileVariant],
        used_cells: set[tuple[ProcessKind, GridCell]],
        failures: list[Mapping[str, Any]],
    ) -> tuple[PlacementRecord, ...]:
        specs = _route_cell_specs(route)
        if not specs:
            return ()

        candidate_columns: list[tuple[TileVariant, ...]] = []
        fixed_columns: set[int] = set()
        for spec in specs:
            fixed_variant = fixed_variants.get((route.process, spec.cell))
            if fixed_variant is not None:
                if not spec.required_directions.issubset(fixed_variant.directions):
                    failures.append(_failure(route.process, spec.cell, spec.required_directions, "fixed_tile_mismatch"))
                    return ()
                fixed_columns.add(len(candidate_columns))
                candidate_columns.append((fixed_variant,))
                continue

            candidates = catalog.candidates_for(spec.required_directions)
            if not candidates:
                failures.append(_failure(route.process, spec.cell, spec.required_directions, "catalog_gap"))
                return ()
            candidate_columns.append(candidates)

        selected = self._least_cost_compatible_path(specs, candidate_columns, fixed_columns=fixed_columns)
        if selected is None:
            failures.append(_failure(route.process, specs[0].cell, specs[0].required_directions, "no_compatible_tile_path"))
            return ()

        placements = []
        for spec, variant in zip(specs, selected, strict=True):
            cell_key = (route.process, spec.cell)
            if cell_key in fixed_variants or cell_key in used_cells:
                continue
            placements.append(
                self._placement_from_variant(
                    process=route.process,
                    cell=spec.cell,
                    required_directions=spec.required_directions,
                    route=route,
                    intersection=False,
                    variant=variant,
                )
            )
            used_cells.add(cell_key)
        return tuple(placements)

    def _least_cost_compatible_path(
        self,
        specs: tuple[_RouteCellSpec, ...],
        candidate_columns: Sequence[tuple[TileVariant, ...]],
        *,
        fixed_columns: set[int] | None = None,
    ) -> tuple[TileVariant, ...] | None:
        fixed_columns = fixed_columns or set()
        costs: dict[tuple[int, int], float] = {}
        previous: dict[tuple[int, int], tuple[int, int] | None] = {}
        for variant_index, variant in enumerate(candidate_columns[0]):
            costs[(0, variant_index)] = variant.cost
            previous[(0, variant_index)] = None

        for column_index in range(1, len(candidate_columns)):
            step_direction = _direction_between_cells(specs[column_index - 1].cell, specs[column_index].cell)
            for variant_index, variant in enumerate(candidate_columns[column_index]):
                best: tuple[float, tuple[int, int]] | None = None
                for prev_index, prev_variant in enumerate(candidate_columns[column_index - 1]):
                    prev_key = (column_index - 1, prev_index)
                    if prev_key not in costs:
                        continue
                    crosses_fixed_tile = column_index in fixed_columns or column_index - 1 in fixed_columns
                    if (
                        step_direction is not None
                        and not crosses_fixed_tile
                        and not _variants_connect(prev_variant, variant, step_direction)
                    ):
                        continue
                    path_cost = costs[prev_key] + variant.cost
                    if best is None or path_cost < best[0]:
                        best = (path_cost, prev_key)
                if best is None:
                    continue
                costs[(column_index, variant_index)] = best[0]
                previous[(column_index, variant_index)] = best[1]

        final_column = len(candidate_columns) - 1
        final_keys = [key for key in costs if key[0] == final_column]
        if not final_keys:
            return None
        min_cost = min(costs[key] for key in final_keys)
        best_keys = sorted(key for key in final_keys if math.isclose(costs[key], min_cost))
        key = best_keys[int(self.rng.integers(0, len(best_keys)))] if len(best_keys) > 1 else best_keys[0]
        selected: list[TileVariant] = []
        while key is not None:
            selected.append(candidate_columns[key[0]][key[1]])
            key = previous[key]
        selected.reverse()
        return tuple(selected)

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


def _variant_cm_type(variant: TileVariant, base_cm_type: CMType | None) -> CMType:
    if base_cm_type is None:
        return variant.cm_type
    return CMType(
        menu=base_cm_type.menu,
        cat1=base_cm_type.cat1,
        cat2=variant.cm_type.cat2,
        direction=variant.cm_type.direction,
        tile_id=variant.cm_type.tile_id,
        modifiers=variant.cm_type.modifiers,
    )


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _direction_between_nodes(first: GridNode | tuple[int, int], second: GridNode | tuple[int, int]) -> str:
    first_x, first_y = _node_xy(first)
    second_x, second_y = _node_xy(second)
    dx = second_x - first_x
    dy = second_y - first_y
    if max(abs(dx), abs(dy)) != 1 or (dx == 0 and dy == 0):
        raise ValueError(f"Only adjacent route nodes can be assigned tiles: {(first_x, first_y)} -> {(second_x, second_y)}")
    if dx == 1:
        return "NE" if dy == 1 else "SE" if dy == -1 else "E"
    if dx == -1:
        return "NW" if dy == 1 else "SW" if dy == -1 else "W"
    if dy == 1:
        return "N"
    return "S"


def _required_directions_for_nodes(first: GridNode, second: GridNode) -> frozenset[str]:
    direction = _direction_between_nodes(first, second)
    return frozenset((direction, _OPPOSITE_DIRECTIONS[direction]))


def _route_cell_specs(route: RouteRecord) -> tuple[_RouteCellSpec, ...]:
    route_cells = _normalized_route_cells(route)
    if len(route_cells) == len(route.nodes):
        step_cells = list(route_cells)
    else:
        step_cells = []
        for step_index, (start, end) in enumerate(zip(route.nodes, route.nodes[1:], strict=False)):
            cell = route_cells[step_index] if step_index < len(route_cells) else _cell_for_step(start, end)
            if not step_cells or step_cells[-1] != cell:
                step_cells.append(cell)

    if not step_cells:
        return ()

    first_step_direction = _direction_between_nodes(route.nodes[0], route.nodes[1])
    last_step_direction = _direction_between_nodes(route.nodes[-2], route.nodes[-1])
    specs = []
    for cell_index, cell in enumerate(step_cells):
        directions = set()
        if cell_index > 0:
            direction = _direction_between_cells(cell, step_cells[cell_index - 1])
            if direction is not None:
                directions.add(direction)
        if cell_index < len(step_cells) - 1:
            direction = _direction_between_cells(cell, step_cells[cell_index + 1])
            if direction is not None:
                directions.add(direction)
        if cell_index == 0:
            directions.add(_OPPOSITE_DIRECTIONS[first_step_direction])
        if cell_index == len(step_cells) - 1:
            directions.add(last_step_direction)
        if len(directions) == 1:
            directions.add(_OPPOSITE_DIRECTIONS[next(iter(directions))])
        specs.append(_RouteCellSpec(cell=cell, required_directions=frozenset(directions)))
    return tuple(specs)


def _normalized_route_cells(route: RouteRecord) -> tuple[GridCell, ...]:
    if len(route.nodes) < 2:
        return ()
    if len(route.cells) == len(route.nodes):
        return route.cells
    candidate_columns = [
        _candidate_cells_for_step(start, end, route.cells[step_index] if step_index < len(route.cells) else None)
        for step_index, (start, end) in enumerate(zip(route.nodes, route.nodes[1:], strict=False))
    ]
    selected = _least_diagonal_cell_path(candidate_columns)
    if selected is None:
        return route.cells
    return selected


def _candidate_cells_for_step(start: GridNode, end: GridNode, default_cell: GridCell | None) -> tuple[GridCell, ...]:
    default = default_cell or _cell_for_step(start, end)
    try:
        direction = _direction_between_nodes(start, end)
    except ValueError:
        return (default,)

    if direction in {"E", "W"}:
        xidx = min(start.xidx, end.xidx)
        candidates = (GridCell(xidx, start.yidx), GridCell(xidx, start.yidx - 1))
    elif direction in {"N", "S"}:
        yidx = min(start.yidx, end.yidx)
        candidates = (GridCell(start.xidx, yidx), GridCell(start.xidx - 1, yidx))
    else:
        candidates = (default,)

    valid = tuple(dict.fromkeys(cell for cell in (default, *candidates) if cell.xidx >= 0 and cell.yidx >= 0))
    return valid or (default,)


def _least_diagonal_cell_path(candidate_columns: Sequence[tuple[GridCell, ...]]) -> tuple[GridCell, ...] | None:
    if not candidate_columns:
        return ()

    costs: dict[tuple[int, int], float] = {}
    previous: dict[tuple[int, int], tuple[int, int] | None] = {}
    for candidate_index, _candidate in enumerate(candidate_columns[0]):
        costs[(0, candidate_index)] = 0.0 if candidate_index == 0 else 0.01
        previous[(0, candidate_index)] = None

    for column_index in range(1, len(candidate_columns)):
        for candidate_index, candidate in enumerate(candidate_columns[column_index]):
            best: tuple[float, tuple[int, int]] | None = None
            for previous_index, previous_candidate in enumerate(candidate_columns[column_index - 1]):
                previous_key = (column_index - 1, previous_index)
                if previous_key not in costs:
                    continue
                transition_cost = _cell_transition_cost(previous_candidate, candidate)
                default_cost = 0.0 if candidate_index == 0 else 0.01
                path_cost = costs[previous_key] + transition_cost + default_cost
                if best is None or path_cost < best[0]:
                    best = (path_cost, previous_key)
            if best is None:
                continue
            costs[(column_index, candidate_index)] = best[0]
            previous[(column_index, candidate_index)] = best[1]

    final_column = len(candidate_columns) - 1
    final_keys = [key for key in costs if key[0] == final_column]
    if not final_keys:
        return None
    key = min(final_keys, key=lambda item: (costs[item], item[1]))
    selected: list[GridCell] = []
    while key is not None:
        selected.append(candidate_columns[key[0]][key[1]])
        key = previous[key]
    selected.reverse()
    return tuple(selected)


def _cell_transition_cost(first: GridCell, second: GridCell) -> float:
    dx = abs(second.xidx - first.xidx)
    dy = abs(second.yidx - first.yidx)
    if dx == 0 and dy == 0:
        return 0.0
    if dx + dy == 1:
        return 0.0
    return 100.0


def _node_xy(node: GridNode | tuple[int, int]) -> tuple[int, int]:
    if isinstance(node, GridNode):
        return node.xidx, node.yidx
    return node


def _intersection_specs_by_process(routes: Sequence[RouteRecord]) -> dict[tuple[ProcessKind, int], _IntersectionSpec]:
    endpoints_by_key: dict[tuple[ProcessKind, int], list[_IntersectionEndpoint]] = {}
    for route in routes:
        if len(route.nodes) < 2:
            continue
        _record_route_endpoint_candidate(
            endpoints_by_key,
            process=route.process,
            node_id=route.start_node_id,
            node=route.nodes[0],
            direction=_direction_between_nodes(route.nodes[0], route.nodes[1]),
            cells_from_node=route.cells or (_cell_for_step(route.nodes[0], route.nodes[1]),),
            cm_type=route.cm_type,
        )
        _record_route_endpoint_candidate(
            endpoints_by_key,
            process=route.process,
            node_id=route.end_node_id,
            node=route.nodes[-1],
            direction=_direction_between_nodes(route.nodes[-1], route.nodes[-2]),
            cells_from_node=tuple(reversed(route.cells or (_cell_for_step(route.nodes[-2], route.nodes[-1]),))),
            cm_type=route.cm_type,
        )

    intersections: dict[tuple[ProcessKind, int], _IntersectionSpec] = {}
    for key, endpoints in endpoints_by_key.items():
        if not endpoints:
            continue
        intersection_cell = _intersection_cell_for_endpoints(endpoints)
        valid_endpoints = tuple(
            endpoint
            for endpoint in endpoints
            if _arm_continues_in_next_square(intersection_cell, endpoint.direction, endpoint.cells_from_node)
        )
        if not valid_endpoints:
            continue
        spec = _IntersectionSpec(
            process=key[0],
            node_id=key[1],
            node=valid_endpoints[0].node,
            directions=set(),
            cells={intersection_cell},
            cm_type=_intersection_cm_type(valid_endpoints),
        )
        for endpoint in valid_endpoints:
            spec.directions.add(endpoint.direction)
        intersections[key] = spec
    return intersections


def _record_route_endpoint_candidate(
    endpoints_by_key: dict[tuple[ProcessKind, int], list[_IntersectionEndpoint]],
    *,
    process: ProcessKind,
    node_id: int,
    node: GridNode,
    direction: str,
    cells_from_node: tuple[GridCell, ...],
    cm_type: CMType | None,
) -> None:
    endpoints_by_key.setdefault((process, node_id), []).append(
        _IntersectionEndpoint(
            process=process,
            node_id=node_id,
            node=node,
            direction=direction,
            cells_from_node=cells_from_node,
            cm_type=cm_type,
        )
    )


def _intersection_cm_type(endpoints: Sequence[_IntersectionEndpoint]) -> CMType | None:
    cm_types = [endpoint.cm_type for endpoint in endpoints if endpoint.cm_type is not None]
    if not cm_types:
        return None
    counts: dict[tuple[str, str, str | None, str | int | None], int] = {}
    by_key = {}
    for cm_type in cm_types:
        key = _cm_type_key(cm_type)
        counts[key] = counts.get(key, 0) + 1
        by_key.setdefault(key, cm_type)
    best_key = min(
        counts,
        key=lambda key: (
            -counts[key],
            _ROAD_SURFACE_RANK.get(key[1], 99),
            tuple("" if value is None else str(value) for value in key),
        ),
    )
    return by_key[best_key]


_ROAD_SURFACE_RANK = {
    "Paved 1": 0,
    "Paved 2": 1,
    "Gravel Road": 2,
    "Dirt Road": 3,
}


def _arm_continues_in_next_square(
    intersection_cell: GridCell,
    direction: str,
    cells_from_node: Iterable[GridCell],
) -> bool:
    return _next_cell(intersection_cell, direction) in set(cells_from_node)


def _intersection_cell_for_endpoints(endpoints: Sequence[_IntersectionEndpoint]) -> GridCell:
    node = endpoints[0].node
    incident_cells = {cell for endpoint in endpoints for cell in endpoint.cells_from_node}
    preferred = GridCell(node.xidx, node.yidx)
    if preferred in incident_cells:
        return preferred

    cells = tuple(sorted(incident_cells, key=lambda cell: (cell.xidx, cell.yidx)))
    if not cells:
        return preferred

    return max(
        cells,
        key=lambda cell: (
            sum(
                1
                for endpoint in endpoints
                if _arm_continues_in_next_square(cell, endpoint.direction, endpoint.cells_from_node)
            ),
            -abs(cell.xidx - node.xidx),
            -abs(cell.yidx - node.yidx),
            cell.xidx,
            cell.yidx,
        ),
    )


def _intersection_cell_for_node(node: GridNode, incident_cells: Iterable[GridCell]) -> GridCell:
    preferred = GridCell(node.xidx, node.yidx)
    cells = tuple(sorted(incident_cells, key=lambda cell: (cell.xidx, cell.yidx)))
    return preferred if preferred in cells or not cells else cells[-1]


def _cell_for_step(start: GridNode, end: GridNode) -> GridCell:
    return GridCell(min(start.xidx, end.xidx), min(start.yidx, end.yidx))


def _direction_between_cells(first: GridCell, second: GridCell) -> str | None:
    dx = second.xidx - first.xidx
    dy = second.yidx - first.yidx
    if max(abs(dx), abs(dy)) != 1 or (dx == 0 and dy == 0):
        return None
    if dx == 1:
        return "NE" if dy == 1 else "SE" if dy == -1 else "E"
    if dx == -1:
        return "NW" if dy == 1 else "SW" if dy == -1 else "W"
    if dy == 1:
        return "N"
    return "S"


def _next_cell(cell: GridCell, direction: str) -> GridCell:
    dx = 1 if "E" in direction else -1 if "W" in direction else 0
    dy = 1 if "N" in direction else -1 if "S" in direction else 0
    return GridCell(cell.xidx + dx, cell.yidx + dy)


def _variants_connect(first: TileVariant, second: TileVariant, direction: str) -> bool:
    return first.connections.get(direction) == second.connections.get(_OPPOSITE_DIRECTIONS[direction])


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


def _intersection_sort_key(intersection: _IntersectionSpec) -> tuple[str, int, int, int]:
    return intersection.process.value, intersection.node.xidx, intersection.node.yidx, intersection.node_id


def _cm_type_key(cm_type: CMType) -> tuple[str, str, str | None, str | int | None]:
    return cm_type.menu, cm_type.cat1, cm_type.cat2, cm_type.direction
