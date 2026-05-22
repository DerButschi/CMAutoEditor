from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from terrain_extraction.osm_extraction.models import GridCell

EPSILON_M2 = 1e-9


def reconstruct_building_footprint_polygon(
    grid_index: Any,
    output_xidx: int | float,
    output_yidx: int | float,
    footprint: Any,
    swapped: bool = False,
) -> Polygon:
    """Reconstruct the CM building footprint consumed by output rows."""

    width_units = _footprint_int(footprint, "width_cells", "width", "selected_width_units", "selected_width_cells")
    height_units = _footprint_int(footprint, "height_cells", "height", "selected_height_units", "selected_height_cells")
    is_diagonal = _footprint_bool(footprint, "is_diagonal", "selected_is_diagonal")
    if swapped:
        width_units, height_units = height_units, width_units

    half_cell_size = grid_index.cell_size_m / 2.0
    origin_local_x = (float(output_xidx) + 0.5) * grid_index.cell_size_m
    origin_local_y = (float(output_yidx) + 0.5) * grid_index.cell_size_m
    p0 = (origin_local_x, origin_local_y)
    if is_diagonal:
        p1 = (p0[0] + half_cell_size * width_units, p0[1] - half_cell_size * width_units)
        p2 = (p1[0] + half_cell_size * height_units, p1[1] + half_cell_size * height_units)
        p3 = (p2[0] - half_cell_size * width_units, p2[1] + half_cell_size * width_units)
    else:
        p1 = (p0[0] + half_cell_size * width_units, p0[1])
        p2 = (p1[0], p1[1] + half_cell_size * height_units)
        p3 = (p2[0] - half_cell_size * width_units, p2[1])
    return _polygon_from_local_offsets(grid_index, (p0, p1, p2, p3))


def output_index_from_local(grid_index: Any, local_value: float) -> float:
    return _clean_float(local_value / grid_index.cell_size_m - 0.5)


def normal_cells_overlapped_by_polygon(
    grid_index: Any,
    polygon: BaseGeometry,
    *,
    min_overlap_area_m2: float = EPSILON_M2,
) -> tuple[GridCell, ...]:
    window = cell_window_for_geometry(grid_index, polygon)
    if window is None:
        return ()
    cells = []
    min_xidx, min_yidx, max_xidx, max_yidx = window
    for xidx in range(min_xidx, max_xidx + 1):
        for yidx in range(min_yidx, max_yidx + 1):
            cell = GridCell(xidx, yidx)
            if polygon.intersection(grid_index.cell_polygon(cell)).area > min_overlap_area_m2:
                cells.append(cell)
    return tuple(sorted(cells))


def cell_window_for_geometry(
    grid_index: Any,
    geometry: BaseGeometry,
    *,
    margin_cells: int = 0,
) -> tuple[int, int, int, int] | None:
    min_local_x, min_local_y, max_local_x, max_local_y = local_bounds(grid_index, geometry)
    return grid_index.clipped_cell_window(
        math.floor(min_local_x / grid_index.cell_size_m) - margin_cells,
        math.floor(min_local_y / grid_index.cell_size_m) - margin_cells,
        math.floor((max_local_x - EPSILON_M2) / grid_index.cell_size_m) + margin_cells,
        math.floor((max_local_y - EPSILON_M2) / grid_index.cell_size_m) + margin_cells,
    )


def local_bounds(grid_index: Any, geometry: BaseGeometry) -> tuple[float, float, float, float]:
    coordinates = []
    if isinstance(geometry, Polygon):
        coordinates.extend(geometry.exterior.coords)
    else:
        min_x, min_y, max_x, max_y = geometry.bounds
        coordinates.extend(((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)))
    local_points = [grid_index.local_from_projected(x, y) for x, y in coordinates]
    x_values = [point[0] for point in local_points]
    y_values = [point[1] for point in local_points]
    return min(x_values), min(y_values), max(x_values), max(y_values)


def footprint_spec_from_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "width": diagnostics.get("selected_width_units", diagnostics.get("selected_width_cells")),
        "height": diagnostics.get("selected_height_units", diagnostics.get("selected_height_cells")),
        "is_diagonal": diagnostics.get("selected_is_diagonal", False),
    }


def _footprint_int(footprint: Any, *names: str) -> int:
    for name in names:
        value = _footprint_value(footprint, name)
        if value is not None:
            return max(1, int(value))
    raise ValueError(f"building footprint is missing one of {names!r}")


def _footprint_bool(footprint: Any, *names: str) -> bool:
    for name in names:
        value = _footprint_value(footprint, name)
        if value is not None:
            return bool(value)
    return False


def _footprint_value(footprint: Any, name: str) -> Any:
    if isinstance(footprint, Mapping):
        return footprint.get(name)
    return getattr(footprint, name, None)


def _polygon_from_local_offsets(grid_index: Any, points: tuple[tuple[float, float], ...]) -> Polygon:
    projected = [grid_index.projected_from_local(local_x, local_y) for local_x, local_y in points]
    return Polygon([(point.x, point.y) for point in projected])


def _clean_float(value: float) -> float:
    rounded = round(value, 6)
    return int(rounded) if float(rounded).is_integer() else rounded
