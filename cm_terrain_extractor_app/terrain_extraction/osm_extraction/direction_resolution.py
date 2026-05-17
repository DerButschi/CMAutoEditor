from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

DIRECTION_ORDER = {"E": 0, "N": 1, "S": 2, "W": 3, "NE": 4, "NW": 5, "SE": 6, "SW": 7}
CARDINAL_DIRECTIONS = frozenset({"N", "E", "S", "W"})
DIAGONAL_DIRECTIONS = frozenset({"NE", "NW", "SE", "SW"})
ALL_DIRECTIONS = CARDINAL_DIRECTIONS | DIAGONAL_DIRECTIONS
DIRECTION_STEPS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
    "SW": (-1, -1),
}
OPPOSITE_DIRECTIONS = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
    "NE": "SW",
    "NW": "SE",
    "SE": "NW",
    "SW": "NE",
}


def normalize_direction_set(directions: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(direction.upper() for direction in directions)
    unsupported = normalized.difference(OPPOSITE_DIRECTIONS)
    if unsupported:
        raise ValueError(f"Unsupported tile direction(s): {ordered_directions(unsupported)}")
    return normalized


def resolve_required_directions(
    directions: Iterable[str],
    exact_tile_exists: Callable[[frozenset[str]], bool],
) -> frozenset[str] | None:
    normalized = normalize_direction_set(directions)
    if exact_tile_exists(normalized):
        return normalized
    if len(normalized) == 1:
        direction = next(iter(normalized))
        endpoint_pair = frozenset((direction, OPPOSITE_DIRECTIONS[direction]))
        if exact_tile_exists(endpoint_pair):
            return endpoint_pair
    return None


def ordered_directions(directions: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(directions, key=lambda direction: DIRECTION_ORDER.get(direction, 99)))


def direction_between_cells(first: Any, second: Any, *, cardinal_only: bool = False) -> str | None:
    dx = second.xidx - first.xidx
    dy = second.yidx - first.yidx
    if cardinal_only:
        if abs(dx) + abs(dy) != 1:
            return None
    elif max(abs(dx), abs(dy)) != 1 or (dx == 0 and dy == 0):
        return None
    if dx == 1:
        return "NE" if dy == 1 else "SE" if dy == -1 else "E"
    if dx == -1:
        return "NW" if dy == 1 else "SW" if dy == -1 else "W"
    if dy == 1:
        return "N"
    return "S"


def next_cell(cell: Any, direction: str) -> Any:
    dx, dy = DIRECTION_STEPS[direction]
    return type(cell)(cell.xidx + dx, cell.yidx + dy)


def directions_are_opposite(directions: Iterable[str]) -> bool:
    normalized = frozenset(directions)
    if len(normalized) != 2:
        return False
    first, second = tuple(normalized)
    return OPPOSITE_DIRECTIONS.get(first) == second


def supports_diagonal_directions(directions: Iterable[str]) -> bool:
    return bool(frozenset(directions).intersection(DIAGONAL_DIRECTIONS))
