from __future__ import annotations

import math

from shapely.geometry import LineString, Point
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import GridCell, GridNode, RasterSpine


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


def derive_spine_guide_waypoints(
    *,
    line: LineString,
    raster_spine: RasterSpine,
    grid_index: GridIndex,
    start: GridNode,
    goal: GridNode,
    sample_every_cells: int = 4,
    max_waypoints: int = 64,
) -> tuple[tuple[GridCell, float], ...]:
    """Return sparse ordered route checkpoints along the rasterized source line."""
    start_cell = _clamp_cell(GridCell(start.xidx, start.yidx), grid_index)
    goal_cell = _clamp_cell(GridCell(goal.xidx, goal.yidx), grid_index)
    max_waypoints = max(2, max_waypoints)
    stride = max(1, sample_every_cells)
    candidates: list[tuple[float, int, bool, GridCell]] = [
        (0.0, 0, True, start_cell),
        (1.0, 3, True, goal_cell),
    ]

    for progress, cell in _bend_waypoints(line, raster_spine, grid_index):
        candidates.append((progress, 1, True, cell))

    if len(raster_spine.cells) > 2:
        sample_count = max(0, (len(raster_spine.cells) - 1) // stride)
        for index in range(1, sample_count + 1):
            progress = index / (sample_count + 1)
            point = line.interpolate(progress, normalized=True) if line.length > 0 else None
            cell = (
                grid_index.projected_to_cell(point.x, point.y)
                if point is not None
                else raster_spine.cells[min(index * stride, len(raster_spine.cells) - 1)]
            )
            candidates.append(
                (
                    _clamp_progress(progress),
                    2,
                    False,
                    _clamp_cell(cell, grid_index),
                )
            )

    ordered = _ordered_unique_waypoints(candidates)
    if len(ordered) > max_waypoints:
        ordered = _thin_waypoints(ordered, max_waypoints)
    return tuple((cell, progress) for progress, _order, _required, cell in ordered)


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


def _bend_waypoints(
    line: LineString,
    raster_spine: RasterSpine,
    grid_index: GridIndex,
) -> tuple[tuple[float, GridCell], ...]:
    if line.length <= 0:
        return ()
    coords = tuple(line.coords)
    waypoints: list[tuple[float, GridCell]] = []
    for index in range(1, len(coords) - 1):
        previous = coords[index - 1]
        current = coords[index]
        following = coords[index + 1]
        if not _is_significant_bend(previous, current, following):
            continue
        progress = _clamp_progress(line.project(Point(current)) / line.length)
        waypoints.append((progress, _spine_cell_at_progress(progress, raster_spine, line, grid_index)))
    return tuple(waypoints)


def _is_significant_bend(
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
) -> bool:
    first = (current[0] - previous[0], current[1] - previous[1])
    second = (following[0] - current[0], following[1] - current[1])
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length <= 1e-9 or second_length <= 1e-9:
        return False
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)))
    return math.degrees(math.acos(cosine)) >= 15.0


def _spine_cell_at_progress(
    progress: float,
    raster_spine: RasterSpine,
    line: LineString,
    grid_index: GridIndex,
) -> GridCell:
    if raster_spine.cells:
        index = min(
            range(len(raster_spine.cells)),
            key=lambda candidate: (
                abs(raster_spine.progress[candidate] - progress),
                raster_spine.distance_m[candidate],
                raster_spine.cells[candidate].xidx,
                raster_spine.cells[candidate].yidx,
            ),
        )
        return _clamp_cell(raster_spine.cells[index], grid_index)
    point = line.interpolate(progress, normalized=True)
    return _clamp_cell(grid_index.projected_to_cell(point.x, point.y), grid_index)


def _ordered_unique_waypoints(
    candidates: list[tuple[float, int, bool, GridCell]]
) -> list[tuple[float, int, bool, GridCell]]:
    candidates.sort(key=lambda entry: (entry[0], entry[1], entry[3].xidx, entry[3].yidx))
    deduped: list[tuple[float, int, bool, GridCell]] = []
    for progress, order, required, cell in candidates:
        progress = _clamp_progress(progress)
        if deduped and deduped[-1][3] == cell:
            old_progress, old_order, old_required, _old_cell = deduped[-1]
            deduped[-1] = (
                min(old_progress, progress) if old_order <= order else progress,
                min(old_order, order),
                old_required or required,
                cell,
            )
            continue
        deduped.append((progress, order, required, cell))
    if deduped:
        deduped[0] = (0.0, deduped[0][1], True, deduped[0][3])
        deduped[-1] = (1.0, deduped[-1][1], True, deduped[-1][3])
    return deduped


def _thin_waypoints(
    waypoints: list[tuple[float, int, bool, GridCell]],
    max_waypoints: int,
) -> list[tuple[float, int, bool, GridCell]]:
    required = [waypoint for waypoint in waypoints if waypoint[2]]
    optional = [waypoint for waypoint in waypoints if not waypoint[2]]
    slots = max(0, max_waypoints - len(required))
    if len(optional) > slots:
        if slots <= 0:
            optional = []
        else:
            step = (len(optional) - 1) / max(1, slots - 1)
            selected_indices = {round(index * step) for index in range(slots)}
            optional = [waypoint for index, waypoint in enumerate(optional) if index in selected_indices]
    return _ordered_unique_waypoints(required + optional)


def _clamp_cell(cell: GridCell, grid_index: GridIndex) -> GridCell:
    return GridCell(
        min(max(cell.xidx, 0), grid_index.width - 1),
        min(max(cell.yidx, 0), grid_index.height - 1),
    )


def _clamp_progress(progress: float) -> float:
    return min(max(float(progress), 0.0), 1.0)
