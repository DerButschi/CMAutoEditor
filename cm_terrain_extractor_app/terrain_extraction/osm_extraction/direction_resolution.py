from __future__ import annotations

from collections.abc import Callable, Iterable

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


def normalize_direction_set(directions: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(direction.upper() for direction in directions)
    unsupported = normalized.difference(_OPPOSITE_DIRECTIONS)
    if unsupported:
        raise ValueError(f"Unsupported tile direction(s): {_ordered_directions(unsupported)}")
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
        endpoint_pair = frozenset((direction, _OPPOSITE_DIRECTIONS[direction]))
        if exact_tile_exists(endpoint_pair):
            return endpoint_pair
    return None


def _ordered_directions(directions: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(directions, key=lambda direction: _DIRECTION_ORDER.get(direction, 99)))
