from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from terrain_extraction.osm_extraction.direction_resolution import (
    DIRECTION_ORDER,
    OPPOSITE_DIRECTIONS,
    direction_between_cells,
    directions_are_opposite,
    ordered_directions,
    resolve_required_directions,
)
from terrain_extraction.osm_extraction.linear_processing_plan import (
    LinearInteractionPolicy,
    default_linear_interaction_policy,
)
from terrain_extraction.osm_extraction.models import GridCell, ProcessKind, RouteRecord

_DIRECTION_BITS = {"N": 1, "E": 2, "S": 4, "W": 8, "NE": 16, "NW": 32, "SE": 64, "SW": 128}
_DIRECTION_ORDER = DIRECTION_ORDER
_OPPOSITE_DIRECTIONS = OPPOSITE_DIRECTIONS


@dataclass(frozen=True, slots=True)
class LinearCellDecision:
    allowed: bool
    failures: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", tuple(MappingProxyType(dict(failure)) for failure in self.failures))


@dataclass(frozen=True, slots=True)
class LinearReservationResult:
    success: bool
    route_id: int | str | None
    failures: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", tuple(MappingProxyType(dict(failure)) for failure in self.failures))


@dataclass(slots=True)
class LinearNetworkState:
    width: int
    height: int
    catalogs: Mapping[ProcessKind, Any] = field(default_factory=dict)
    interaction_policy: LinearInteractionPolicy = field(default_factory=default_linear_interaction_policy)
    occupied: set[GridCell] = field(init=False, default_factory=set)
    connection_bits: dict[GridCell, int] = field(init=False, default_factory=dict)
    route_id_at_cell: dict[GridCell, tuple[int | str, ...]] = field(init=False, default_factory=dict)
    priority_at_cell: dict[GridCell, int] = field(init=False, default_factory=dict)
    process_at_cell: dict[GridCell, ProcessKind] = field(init=False, default_factory=dict)
    top_level_at_cell: dict[GridCell, str] = field(init=False, default_factory=dict)
    intersection_kind_at_cell: dict[GridCell, str] = field(init=False, default_factory=dict)
    _direction_route_ids: dict[GridCell, dict[str, set[int | str]]] = field(init=False, default_factory=dict)
    _route_cells: dict[int | str, set[GridCell]] = field(init=False, default_factory=dict)
    _route_priorities: dict[int | str, int] = field(init=False, default_factory=dict)
    _route_processes: dict[int | str, ProcessKind] = field(init=False, default_factory=dict)
    _route_top_levels: dict[int | str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("LinearNetworkState width and height must be positive")
        self.catalogs = MappingProxyType(dict(self.catalogs))

    def can_enter_cell(
        self,
        cell: GridCell,
        incoming_dir: str | None,
        outgoing_dir: str | None,
        process: ProcessKind,
        priority: int,
        *,
        top_level_name: str | None = None,
        allow_lower_priority_connection: bool = False,
    ) -> LinearCellDecision:
        top_level = process.value if top_level_name is None else top_level_name
        dirs = {direction for direction in (incoming_dir, outgoing_dir) if direction}
        failures = self._reservation_failures(
            additions={cell: dirs},
            process=process,
            top_level_name=top_level,
            priority=priority,
            route_id=None,
            lower_priority_connection_cells=frozenset((cell,)) if allow_lower_priority_connection else frozenset(),
        )
        return LinearCellDecision(allowed=not failures, failures=failures)

    def reserve_path(
        self,
        route: RouteRecord,
        *,
        route_id: int | str | None = None,
        planned_connect_cells: Iterable[GridCell] = (),
        skip_cells: Iterable[GridCell] = (),
    ) -> LinearReservationResult:
        return self.reserve_path_cells(
            route,
            route_id=route_id,
            planned_connect_cells=planned_connect_cells,
            skip_cells=skip_cells,
        )

    def reserve_path_cells(
        self,
        route: RouteRecord,
        *,
        route_id: int | str | None = None,
        planned_connect_cells: Iterable[GridCell] = (),
        skip_cells: Iterable[GridCell] = (),
    ) -> LinearReservationResult:
        route_key = route.edge_id if route_id is None else route_id
        top_level_name = _route_top_level_name(route)
        skip_cell_set = frozenset(skip_cells)
        additions = {
            cell: directions
            for cell, directions in _route_direction_additions(route.tile_cells).items()
            if cell not in skip_cell_set
        }
        failures = self._reservation_failures(
            additions=additions,
            process=route.process,
            top_level_name=top_level_name,
            priority=route.priority,
            route_id=route_key,
            lower_priority_connection_cells=_endpoint_cells(route.tile_cells) | frozenset(planned_connect_cells),
        )
        if failures:
            return LinearReservationResult(success=False, route_id=route_key, failures=failures)

        touched_cells: set[GridCell] = set()
        for cell, directions in additions.items():
            self._direction_route_ids.setdefault(cell, {})
            for direction in directions:
                self._direction_route_ids[cell].setdefault(direction, set()).add(route_key)
            touched_cells.add(cell)

        self._route_cells[route_key] = touched_cells
        self._route_priorities[route_key] = route.priority
        self._route_processes[route_key] = route.process
        self._route_top_levels[route_key] = top_level_name
        for cell in touched_cells:
            self._refresh_cell(cell)
        return LinearReservationResult(success=True, route_id=route_key)

    def release_path(self, route_id: int | str) -> None:
        cells = self._route_cells.pop(route_id, set())
        self._route_priorities.pop(route_id, None)
        self._route_processes.pop(route_id, None)
        self._route_top_levels.pop(route_id, None)
        for cell in cells:
            direction_routes = self._direction_route_ids.get(cell, {})
            for direction in tuple(direction_routes):
                direction_routes[direction].discard(route_id)
                if not direction_routes[direction]:
                    del direction_routes[direction]
            if not direction_routes:
                self._direction_route_ids.pop(cell, None)
            self._refresh_cell(cell)

    def required_dirs(self, cell: GridCell) -> frozenset[str]:
        return frozenset(_dirs_from_bits(self.connection_bits.get(cell, 0)))

    def intersection_kind_at(self, cell: GridCell) -> str | None:
        return self.intersection_kind_at_cell.get(cell)

    def cell_snapshot(self, cell: GridCell) -> Mapping[str, Any] | None:
        if cell not in self.occupied:
            return None
        return MappingProxyType(
            {
                "cell": cell,
                "process": self.process_at_cell[cell],
                "top_level_name": self.top_level_at_cell[cell],
                "priority": self.priority_at_cell[cell],
                "required_directions": self.required_dirs(cell),
                "route_ids": self.route_id_at_cell[cell],
                "intersection_kind": self.intersection_kind_at_cell[cell],
            }
        )

    def cell_snapshots(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            self.cell_snapshot(cell)
            for cell in sorted(self.occupied, key=lambda item: (item.yidx, item.xidx))
            if self.cell_snapshot(cell) is not None
        )

    def as_debug_layer(self) -> tuple[Mapping[str, Any], ...]:
        rows = []
        for cell in sorted(self.occupied, key=lambda item: (item.yidx, item.xidx)):
            directions = self.required_dirs(cell)
            rows.append(
                MappingProxyType(
                    {
                        "xidx": cell.xidx,
                        "yidx": cell.yidx,
                        "process": self.process_at_cell[cell].value,
                        "top_level_name": self.top_level_at_cell[cell],
                        "priority": self.priority_at_cell[cell],
                        "connection_bits": self.connection_bits[cell],
                        "required_directions": _ordered_directions(directions),
                        "route_ids": self.route_id_at_cell[cell],
                        "intersection_kind": self.intersection_kind_at_cell[cell],
                    }
                )
            )
        return tuple(rows)

    def diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "linear_state_cells": len(self.occupied),
                "linear_state_routes": len(self._route_cells),
                "linear_state_intersections": sum(
                    1 for kind in self.intersection_kind_at_cell.values() if kind in {"t_junction", "four_way"}
                ),
                "process_pair_policy": self.interaction_policy.as_diagnostics(),
            }
        )

    def _reservation_failures(
        self,
        *,
        additions: Mapping[GridCell, Iterable[str]],
        process: ProcessKind,
        top_level_name: str,
        priority: int,
        route_id: int | str | None,
        lower_priority_connection_cells: frozenset[GridCell],
    ) -> tuple[Mapping[str, Any], ...]:
        failures = []
        for cell, directions in additions.items():
            normalized_dirs = frozenset(direction.upper() for direction in directions)
            if not self._cell_in_bounds(cell):
                failures.append(_failure(process, cell, normalized_dirs, "out_of_bounds"))
                continue
            unsupported = normalized_dirs.difference(_DIRECTION_BITS)
            if unsupported:
                failures.append(_failure(process, cell, normalized_dirs, "unsupported_direction"))
                continue

            existing_dirs = self.required_dirs(cell)
            new_dirs = normalized_dirs.difference(existing_dirs)
            existing_process = self.process_at_cell.get(cell)
            existing_top_level = self.top_level_at_cell.get(cell)
            existing_priority = self.priority_at_cell.get(cell)
            if existing_top_level is not None and existing_top_level != top_level_name and new_dirs:
                decision = self.interaction_policy.decision(existing_process, process)
                if decision.interaction != "connect":
                    reason = "process_avoidance" if decision.interaction == "avoid" else "process_conflict"
                    failures.append(
                        _failure(
                            process,
                            cell,
                            existing_dirs | normalized_dirs,
                            reason,
                            existing_process=existing_process,
                            existing_top_level_name=existing_top_level,
                            incoming_top_level_name=top_level_name,
                            interaction=decision.interaction,
                        )
                    )
                    continue

            allows_lower_priority_connection = False
            if existing_process is not None and len(new_dirs) == 1 and cell in lower_priority_connection_cells:
                if existing_top_level == top_level_name:
                    allows_lower_priority_connection = True
                else:
                    allows_lower_priority_connection = (
                        self.interaction_policy.decision(existing_process, process).interaction == "connect"
                    )
            if (
                existing_priority is not None
                and priority > existing_priority
                and new_dirs
                and route_id not in self.route_id_at_cell.get(cell, ())
                and not allows_lower_priority_connection
            ):
                failures.append(_failure(process, cell, existing_dirs | normalized_dirs, "lower_priority_overwrite"))
                continue

            union_dirs = existing_dirs | normalized_dirs
            if not self._catalog_allows(process, union_dirs):
                failures.append(_failure(process, cell, union_dirs, "catalog_gap"))
        return tuple(failures)

    def _catalog_allows(self, process: ProcessKind, directions: frozenset[str]) -> bool:
        catalog = self.catalogs.get(process)
        if catalog is None:
            return True
        if hasattr(catalog, "resolved_required_directions"):
            return catalog.resolved_required_directions(directions) is not None
        return resolve_required_directions(directions, lambda required: bool(catalog.has_tile(required))) is not None

    def _cell_in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell.xidx < self.width and 0 <= cell.yidx < self.height

    def _refresh_cell(self, cell: GridCell) -> None:
        direction_routes = self._direction_route_ids.get(cell, {})
        route_ids = tuple(sorted({route_id for route_ids in direction_routes.values() for route_id in route_ids}))
        if not route_ids:
            self.occupied.discard(cell)
            self.connection_bits.pop(cell, None)
            self.route_id_at_cell.pop(cell, None)
            self.priority_at_cell.pop(cell, None)
            self.process_at_cell.pop(cell, None)
            self.top_level_at_cell.pop(cell, None)
            self.intersection_kind_at_cell.pop(cell, None)
            return

        directions = frozenset(direction_routes)
        self.occupied.add(cell)
        self.connection_bits[cell] = _bits_from_dirs(directions)
        self.route_id_at_cell[cell] = route_ids
        self.priority_at_cell[cell] = min(self._route_priorities[route_id] for route_id in route_ids)
        self.process_at_cell[cell] = self._route_processes[route_ids[0]]
        self.top_level_at_cell[cell] = self._route_top_levels[route_ids[0]]
        self.intersection_kind_at_cell[cell] = _intersection_kind(directions)


def _route_direction_additions(tile_cells: Iterable[GridCell]) -> dict[GridCell, set[str]]:
    cells = tuple(tile_cells)
    additions: dict[GridCell, set[str]] = {cell: set() for cell in cells}
    for first, second in zip(cells, cells[1:], strict=False):
        direction = _direction_between_cells(first, second)
        if direction is None or direction not in _OPPOSITE_DIRECTIONS:
            additions.setdefault(first, set()).add(direction or "")
            additions.setdefault(second, set()).add(direction or "")
            continue
        additions[first].add(direction)
        additions[second].add(_OPPOSITE_DIRECTIONS[direction])
    return additions


def _endpoint_cells(tile_cells: Iterable[GridCell]) -> frozenset[GridCell]:
    cells = tuple(tile_cells)
    if not cells:
        return frozenset()
    return frozenset((cells[0], cells[-1]))


def _route_top_level_name(route: RouteRecord) -> str:
    if route.linear_authority is not None:
        return route.linear_authority.top_level_name
    return route.process.value


def _direction_between_cells(first: GridCell, second: GridCell) -> str | None:
    return direction_between_cells(first, second)


def _bits_from_dirs(directions: Iterable[str]) -> int:
    bits = 0
    for direction in directions:
        bits |= _DIRECTION_BITS[direction]
    return bits


def _dirs_from_bits(bits: int) -> tuple[str, ...]:
    return tuple(direction for direction, bit in _DIRECTION_BITS.items() if bits & bit)


def _intersection_kind(directions: frozenset[str]) -> str:
    if len(directions) <= 1:
        return "endpoint"
    if len(directions) == 2:
        return "straight" if directions_are_opposite(directions) else "bend"
    if len(directions) == 3:
        return "t_junction"
    return "four_way"


def _failure(
    process: ProcessKind,
    cell: GridCell,
    required_directions: Iterable[str],
    reason: str,
    **extra: Any,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "process": process.value,
            "cell": (cell.xidx, cell.yidx),
            "required_directions": _ordered_directions(required_directions),
            "failure_reason": reason,
            **{
                key: value.value if isinstance(value, ProcessKind) else value
                for key, value in extra.items()
            },
        }
    )


def _ordered_directions(directions: Iterable[str]) -> tuple[str, ...]:
    return ordered_directions(direction for direction in directions if direction)
