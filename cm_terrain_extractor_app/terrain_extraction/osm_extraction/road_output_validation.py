from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from terrain_extraction.osm_extraction.models import CMType, GridCell, ProcessKind
from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

from profiles.general import road_tiles

_CARDINAL_STEPS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}
_OPPOSITE_DIRECTIONS = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _label_value(value: Any) -> str:
    if value is None or value == -1:
        return ""
    return str(value)


_ROAD_CATALOG = CompiledTileCatalog.from_records(
    road_tiles,
    process=ProcessKind.ROAD,
    base_cm_type=CMType(menu="Roads", cat1="Road"),
)
_ROAD_LABEL_DIRECTIONS = {
    (_label_value(variant.cm_type.cat2), _label_value(variant.cm_type.direction)): variant.directions
    for variant in _ROAD_CATALOG.variants
}


@dataclass(frozen=True, slots=True)
class RoadValidationIssue:
    stage: str
    reason: str
    cell: GridCell | None
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True)
class RoadValidationReport:
    road_cells: frozenset[GridCell]
    direction_sets: Mapping[GridCell, frozenset[str]]
    adjacency: Mapping[GridCell, frozenset[GridCell]]
    disconnected_components: tuple[frozenset[GridCell], ...]
    illegal_tile_labels: tuple[RoadValidationIssue, ...] = ()
    cells_without_valid_connection_interpretation: tuple[RoadValidationIssue, ...] = ()
    one_cell_junction_gaps: tuple[RoadValidationIssue, ...] = ()
    dangling_arms: tuple[RoadValidationIssue, ...] = ()
    duplicate_mutually_exclusive_cells: tuple[RoadValidationIssue, ...] = ()
    unsupported_diagonal_continuations: tuple[RoadValidationIssue, ...] = ()
    invalid_intersections: tuple[RoadValidationIssue, ...] = ()
    road_building_overlaps: tuple[RoadValidationIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "road_cells", frozenset(self.road_cells))
        object.__setattr__(self, "direction_sets", MappingProxyType(dict(self.direction_sets)))
        object.__setattr__(self, "adjacency", MappingProxyType(dict(self.adjacency)))
        object.__setattr__(self, "disconnected_components", tuple(self.disconnected_components))

    @property
    def road_cell_count(self) -> int:
        return len(self.road_cells)

    @property
    def hard_issues(self) -> tuple[RoadValidationIssue, ...]:
        return (
            self.illegal_tile_labels
            + self.cells_without_valid_connection_interpretation
            + self.one_cell_junction_gaps
            + self.duplicate_mutually_exclusive_cells
            + self.unsupported_diagonal_continuations
            + self.invalid_intersections
            + self.road_building_overlaps
        )

    @property
    def is_valid(self) -> bool:
        return not self.hard_issues

    def issue_summary(self) -> str:
        counts = {
            "illegal_tile_labels": len(self.illegal_tile_labels),
            "cells_without_valid_connection_interpretation": len(self.cells_without_valid_connection_interpretation),
            "disconnected_components": max(0, len(self.disconnected_components) - 1),
            "one_cell_junction_gaps": len(self.one_cell_junction_gaps),
            "dangling_arms": len(self.dangling_arms),
            "duplicate_mutually_exclusive_cells": len(self.duplicate_mutually_exclusive_cells),
            "unsupported_diagonal_continuations": len(self.unsupported_diagonal_continuations),
            "invalid_intersections": len(self.invalid_intersections),
            "road_building_overlaps": len(self.road_building_overlaps),
        }
        active = [f"{name}={count}" for name, count in counts.items() if count]
        return "road output valid" if not active else "road output issues: " + ", ".join(active)

    def ascii_grid(self, *, max_size: int = 40) -> str:
        return render_road_validation_ascii(self, max_size=max_size)


def validate_road_output_rows(rows: Iterable[Mapping[str, Any]], profile: object | None = None) -> RoadValidationReport:
    del profile
    row_tuple = tuple(rows)
    road_entries = tuple(_road_entries(row_tuple))
    illegal_tile_labels = _illegal_tile_label_issues(road_entries)
    unsupported_diagonal = _unsupported_diagonal_issues(road_entries)
    valid_entries = tuple(entry for entry in road_entries if entry.directions and entry.cell is not None)
    road_cells = frozenset(entry.cell for entry in valid_entries if entry.cell is not None)
    directions_by_cell = _directions_by_cell(valid_entries)
    adjacency = _mutual_adjacency(road_cells, directions_by_cell)
    components = _components(road_cells, adjacency)
    duplicate_cells = _duplicate_cell_issues(road_entries)
    no_interpretation = _connection_interpretation_issues(road_cells, adjacency, directions_by_cell)
    one_cell_gaps = _one_cell_gap_issues(road_cells, directions_by_cell)
    dangling_arms = _dangling_arm_issues(road_cells, directions_by_cell)
    invalid_intersections = _invalid_intersection_issues(road_cells, directions_by_cell)
    road_building_overlaps = _road_building_overlap_issues(row_tuple, road_cells)
    return RoadValidationReport(
        road_cells=road_cells,
        direction_sets=directions_by_cell,
        adjacency=adjacency,
        disconnected_components=components,
        illegal_tile_labels=illegal_tile_labels,
        cells_without_valid_connection_interpretation=no_interpretation,
        one_cell_junction_gaps=one_cell_gaps,
        dangling_arms=dangling_arms,
        duplicate_mutually_exclusive_cells=duplicate_cells,
        unsupported_diagonal_continuations=unsupported_diagonal,
        invalid_intersections=invalid_intersections,
        road_building_overlaps=road_building_overlaps,
    )


def render_road_validation_ascii(report: RoadValidationReport, *, max_size: int = 40) -> str:
    cells = set(report.road_cells)
    gap_cells = {
        GridCell(int(issue.details["gap_cell"][0]), int(issue.details["gap_cell"][1]))
        for issue in report.one_cell_junction_gaps
        if "gap_cell" in issue.details
    }
    marked_cells = cells | gap_cells
    if not marked_cells:
        return "<empty road graph>"
    min_x = min(cell.xidx for cell in marked_cells)
    max_x = max(cell.xidx for cell in marked_cells)
    min_y = min(cell.yidx for cell in marked_cells)
    max_y = max(cell.yidx for cell in marked_cells)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    if width > max_size or height > max_size:
        return f"<road graph {width}x{height} omitted; exceeds {max_size}x{max_size}>"

    issue_marks = _issue_marks(report)
    component_index = {
        cell: index
        for index, component in enumerate(report.disconnected_components)
        for cell in component
    }
    rows = []
    for yidx in range(max_y, min_y - 1, -1):
        chars = []
        for xidx in range(min_x, max_x + 1):
            cell = GridCell(xidx, yidx)
            if cell in gap_cells:
                chars.append("G")
            elif cell in issue_marks:
                chars.append(issue_marks[cell])
            elif cell not in cells:
                chars.append(".")
            elif component_index.get(cell, 0) == 0:
                chars.append("R")
            else:
                chars.append(chr(ord("A") + min(component_index[cell], 25)))
        rows.append("".join(chars))
    return "\n".join(rows)


@dataclass(frozen=True, slots=True)
class _RoadEntry:
    row_index: int
    row: Mapping[str, Any]
    cell: GridCell | None
    directions: frozenset[str]
    label: tuple[str, str]


def _road_entries(rows: tuple[Mapping[str, Any], ...]) -> Iterable[_RoadEntry]:
    for index, row in enumerate(rows):
        if not _is_road_row(row):
            continue
        label = (_label_value(row.get("cat2")), _label_value(row.get("direction")))
        yield _RoadEntry(
            row_index=index,
            row=row,
            cell=_integer_cell(row),
            directions=_ROAD_LABEL_DIRECTIONS.get(label, frozenset()),
            label=label,
        )


def _is_road_row(row: Mapping[str, Any]) -> bool:
    cat2 = _label_value(row.get("cat2"))
    return row.get("name") == "road" or cat2.startswith("Road Tile ")


def _integer_cell(row: Mapping[str, Any]) -> GridCell | None:
    x_value = _coordinate_value(row, "xidx", "x")
    y_value = _coordinate_value(row, "yidx", "y")
    if x_value is None or y_value is None:
        return None
    x_float = float(x_value)
    y_float = float(y_value)
    if not x_float.is_integer() or not y_float.is_integer():
        return None
    return GridCell(int(x_float), int(y_float))


def _coordinate_value(row: Mapping[str, Any], primary: str, fallback: str) -> Any:
    value = row.get(primary)
    return row.get(fallback) if value is None else value


def _illegal_tile_label_issues(entries: tuple[_RoadEntry, ...]) -> tuple[RoadValidationIssue, ...]:
    return tuple(
        RoadValidationIssue(
            stage="output",
            reason="unknown_road_tile_label",
            cell=entry.cell,
            message="road row label does not map to a catalog road tile",
            details={"row_index": entry.row_index, "cat2": entry.label[0], "direction": entry.label[1]},
        )
        for entry in entries
        if not entry.directions
    )


def _unsupported_diagonal_issues(entries: tuple[_RoadEntry, ...]) -> tuple[RoadValidationIssue, ...]:
    issues = []
    for entry in entries:
        if entry.cell is None:
            issues.append(
                RoadValidationIssue(
                    stage="output",
                    reason="non_integer_road_coordinate",
                    cell=None,
                    message="road row uses a non-integer coordinate that cannot represent a cardinal road continuation",
                    details={"row_index": entry.row_index},
                )
            )
            continue
        diagonal_dirs = tuple(sorted(set(entry.directions).difference(_CARDINAL_STEPS)))
        if diagonal_dirs:
            issues.append(
                RoadValidationIssue(
                    stage="output",
                    reason="diagonal_road_direction",
                    cell=entry.cell,
                    message="road tile uses unsupported diagonal continuation directions",
                    details={"row_index": entry.row_index, "directions": diagonal_dirs},
                )
            )
    return tuple(issues)


def _directions_by_cell(entries: tuple[_RoadEntry, ...]) -> Mapping[GridCell, frozenset[str]]:
    by_cell: dict[GridCell, set[str]] = {}
    for entry in entries:
        if entry.cell is None:
            continue
        by_cell.setdefault(entry.cell, set()).update(entry.directions)
    return {cell: frozenset(directions) for cell, directions in by_cell.items()}


def _mutual_adjacency(
    cells: frozenset[GridCell],
    direction_sets: Mapping[GridCell, frozenset[str]],
) -> Mapping[GridCell, frozenset[GridCell]]:
    adjacency: dict[GridCell, set[GridCell]] = {cell: set() for cell in cells}
    for cell in cells:
        for direction in direction_sets.get(cell, frozenset()):
            if direction not in _CARDINAL_STEPS:
                continue
            neighbor = _next_cell(cell, direction)
            if neighbor in cells and _OPPOSITE_DIRECTIONS[direction] in direction_sets.get(neighbor, frozenset()):
                adjacency[cell].add(neighbor)
                adjacency[neighbor].add(cell)
    return {cell: frozenset(neighbors) for cell, neighbors in adjacency.items()}


def _duplicate_cell_issues(entries: tuple[_RoadEntry, ...]) -> tuple[RoadValidationIssue, ...]:
    counts = Counter(entry.cell for entry in entries if entry.cell is not None)
    return tuple(
        RoadValidationIssue(
            stage="output",
            reason="duplicate_road_cell",
            cell=cell,
            message="multiple mutually exclusive road rows occupy one cell",
            details={"count": count},
        )
        for cell, count in sorted(counts.items(), key=lambda item: (item[0].xidx, item[0].yidx))
        if count > 1
    )


def _connection_interpretation_issues(
    cells: frozenset[GridCell],
    adjacency: Mapping[GridCell, frozenset[GridCell]],
    direction_sets: Mapping[GridCell, frozenset[str]],
) -> tuple[RoadValidationIssue, ...]:
    if len(cells) <= 1:
        return tuple(
            RoadValidationIssue(
                stage="output",
                reason="isolated_road_cell",
                cell=cell,
                message="road cell has no valid road connection",
                details={"directions": tuple(sorted(direction_sets.get(cell, ())))},
            )
            for cell in cells
        )
    return tuple(
        RoadValidationIssue(
            stage="output",
            reason="isolated_road_cell",
            cell=cell,
            message="road cell has no valid road connection",
            details={"directions": tuple(sorted(direction_sets.get(cell, ())))},
        )
        for cell in sorted(cells, key=lambda item: (item.xidx, item.yidx))
        if not adjacency.get(cell)
    )


def _one_cell_gap_issues(
    cells: frozenset[GridCell],
    direction_sets: Mapping[GridCell, frozenset[str]],
) -> tuple[RoadValidationIssue, ...]:
    gaps: dict[GridCell, RoadValidationIssue] = {}
    for cell in cells:
        for direction in direction_sets.get(cell, frozenset()):
            if direction not in _CARDINAL_STEPS:
                continue
            gap_cell = _next_cell(cell, direction)
            if gap_cell in cells:
                continue
            beyond_cell = _next_cell(gap_cell, direction)
            if beyond_cell in cells and _OPPOSITE_DIRECTIONS[direction] in direction_sets.get(beyond_cell, frozenset()):
                gaps.setdefault(
                    gap_cell,
                    RoadValidationIssue(
                        stage="output",
                        reason="one_cell_gap",
                        cell=cell,
                        message="two road arms face each other across a one-cell gap",
                        details={
                            "gap_cell": (gap_cell.xidx, gap_cell.yidx),
                            "from_cell": (cell.xidx, cell.yidx),
                            "to_cell": (beyond_cell.xidx, beyond_cell.yidx),
                            "direction": direction,
                        },
                    ),
                )
    return tuple(gaps[cell] for cell in sorted(gaps, key=lambda item: (item.xidx, item.yidx)))


def _dangling_arm_issues(
    cells: frozenset[GridCell],
    direction_sets: Mapping[GridCell, frozenset[str]],
) -> tuple[RoadValidationIssue, ...]:
    issues = []
    for cell in sorted(cells, key=lambda item: (item.xidx, item.yidx)):
        missing = tuple(
            direction
            for direction in sorted(direction_sets.get(cell, frozenset()))
            if direction in _CARDINAL_STEPS and _next_cell(cell, direction) not in cells
        )
        if len(missing) > 1:
            issues.append(
                RoadValidationIssue(
                    stage="output",
                    reason="multiple_dangling_arms",
                    cell=cell,
                    message="road cell has multiple unconnected tile arms",
                    details={"directions": missing},
                )
            )
    return tuple(issues)


def _invalid_intersection_issues(
    cells: frozenset[GridCell],
    direction_sets: Mapping[GridCell, frozenset[str]],
) -> tuple[RoadValidationIssue, ...]:
    issues = []
    seen: set[tuple[GridCell, GridCell]] = set()
    for cell in sorted(cells, key=lambda item: (item.xidx, item.yidx)):
        for direction, (dx, dy) in _CARDINAL_STEPS.items():
            neighbor = GridCell(cell.xidx + dx, cell.yidx + dy)
            if neighbor not in cells:
                continue
            edge_key = tuple(sorted((cell, neighbor), key=lambda item: (item.xidx, item.yidx)))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            if direction in direction_sets.get(cell, frozenset()) and _OPPOSITE_DIRECTIONS[direction] in direction_sets.get(
                neighbor, frozenset()
            ):
                continue
            issues.append(
                RoadValidationIssue(
                    stage="output",
                    reason="unrepresented_adjacent_road",
                    cell=cell,
                    message="adjacent road cells touch without a mutual tile connection",
                    details={"neighbor": (neighbor.xidx, neighbor.yidx), "direction": direction},
                )
            )
    return tuple(issues)


def _road_building_overlap_issues(
    rows: tuple[Mapping[str, Any], ...],
    road_cells: frozenset[GridCell],
) -> tuple[RoadValidationIssue, ...]:
    building_cells = {
        cell
        for row in rows
        if _is_building_row(row)
        for cell in [_integer_cell(row)]
        if cell is not None
    }
    return tuple(
        RoadValidationIssue(
            stage="output",
            reason="road_building_overlap",
            cell=cell,
            message="road and building rows occupy the same final output cell",
            details={},
        )
        for cell in sorted(road_cells.intersection(building_cells), key=lambda item: (item.xidx, item.yidx))
    )


def _components(
    cells: frozenset[GridCell],
    adjacency: Mapping[GridCell, frozenset[GridCell]],
) -> tuple[frozenset[GridCell], ...]:
    remaining = set(cells)
    components: list[frozenset[GridCell]] = []
    while remaining:
        root = remaining.pop()
        component = {root}
        stack = [root]
        while stack:
            cell = stack.pop()
            for neighbor in adjacency[cell]:
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                component.add(neighbor)
                stack.append(neighbor)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda component: (-len(component), min((cell.xidx, cell.yidx) for cell in component))))


def _issue_marks(report: RoadValidationReport) -> Mapping[GridCell, str]:
    marks: dict[GridCell, str] = {}
    for issue in report.road_building_overlaps:
        if issue.cell is not None:
            marks[issue.cell] = "B"
    for issue in report.invalid_intersections:
        if issue.cell is not None:
            marks.setdefault(issue.cell, "X")
    for issue in report.duplicate_mutually_exclusive_cells:
        if issue.cell is not None:
            marks.setdefault(issue.cell, "D")
    for issue in report.cells_without_valid_connection_interpretation:
        if issue.cell is not None:
            marks.setdefault(issue.cell, "I")
    return marks


def _next_cell(cell: GridCell, direction: str) -> GridCell:
    dx, dy = _CARDINAL_STEPS[direction]
    return GridCell(cell.xidx + dx, cell.yidx + dy)


def _is_building_row(row: Mapping[str, Any]) -> bool:
    return row.get("_layer") == "building" or _label_value(row.get("menu")) == "Buildings"
