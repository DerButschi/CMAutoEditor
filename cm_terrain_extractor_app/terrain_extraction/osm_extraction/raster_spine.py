from __future__ import annotations

import math

from shapely.geometry import LineString
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import GridCell, RasterSpine


def build_raster_spine(
    *,
    topology_edge_id: int,
    line: LineString,
    grid_index: GridIndex,
) -> RasterSpine:
    if line.is_empty:
        return RasterSpine(topology_edge_id, (), (), (), 0.0)

    window = _candidate_window(line, grid_index)
    if window is None:
        return RasterSpine(topology_edge_id, (), (), (), line.length)

    min_xidx, min_yidx, max_xidx, max_yidx = window
    entries: list[tuple[float, int, int, GridCell, float]] = []
    for xidx in range(min_xidx, max_xidx + 1):
        for yidx in range(min_yidx, max_yidx + 1):
            cell = GridCell(xidx, yidx)
            if not grid_index.cell_polygon(cell).intersects(line):
                continue
            progress, distance_m = _cell_metrics(cell, line, grid_index)
            entries.append((progress, xidx, yidx, cell, distance_m))

    if not entries:
        endpoint_cells = _endpoint_cells(line, grid_index)
        entries = [
            (*_cell_metrics(cell, line, grid_index), cell.xidx, cell.yidx, cell)
            for cell in endpoint_cells
            if _cell_in_bounds(cell, grid_index)
        ]
        entries = [(progress, xidx, yidx, cell, distance_m) for progress, distance_m, xidx, yidx, cell in entries]

    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    cells = tuple(entry[3] for entry in entries)
    progress = tuple(entry[0] for entry in entries)
    distance_m = tuple(entry[4] for entry in entries)
    progress = _normalize_endpoint_progress(progress)
    return RasterSpine(
        topology_edge_id=topology_edge_id,
        cells=cells,
        progress=progress,
        distance_m=distance_m,
        source_length_m=line.length,
    )


def _candidate_window(line: LineString, grid_index: GridIndex) -> tuple[int, int, int, int] | None:
    local_points = [grid_index.local_from_projected(x, y) for x, y in line.coords]
    cell_size = grid_index.cell_size_m
    min_xidx = math.floor(min(point[0] for point in local_points) / cell_size) - 1
    min_yidx = math.floor(min(point[1] for point in local_points) / cell_size) - 1
    max_xidx = math.ceil(max(point[0] for point in local_points) / cell_size) + 1
    max_yidx = math.ceil(max(point[1] for point in local_points) / cell_size) + 1
    return grid_index.clipped_cell_window(min_xidx, min_yidx, max_xidx, max_yidx)


def _cell_metrics(cell: GridCell, line: LineString, grid_index: GridIndex) -> tuple[float, float]:
    center = grid_index.cell_center(cell)
    progress = 0.0 if line.length <= 0 else line.project(center) / line.length
    return min(max(progress, 0.0), 1.0), center.distance(line)


def _normalize_endpoint_progress(progress: tuple[float, ...]) -> tuple[float, ...]:
    if not progress:
        return ()
    values = list(progress)
    values[0] = 0.0
    if len(values) > 1:
        values[-1] = 1.0
    return tuple(values)


def _endpoint_cells(line: LineString, grid_index: GridIndex) -> tuple[GridCell, ...]:
    coords = tuple(line.coords)
    cells = tuple(grid_index.projected_to_cell(x, y) for x, y in (coords[0], coords[-1]))
    return tuple(dict.fromkeys(cells))


def _cell_in_bounds(cell: GridCell, grid_index: GridIndex) -> bool:
    return 0 <= cell.xidx < grid_index.width and 0 <= cell.yidx < grid_index.height
