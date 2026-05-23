from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from terrain_extraction.osm_extraction.building_geometry import (
    EPSILON_M2,
    building_anchor_local_from_selected_grid_index,
    selected_grid_index_from_building_output,
)
from terrain_extraction.osm_extraction.models import GridCell, LayerKind

_LINEAR_LAYERS = frozenset({LayerKind.LINEAR_SURFACE.value, LayerKind.LINEAR_OBJECT.value})
_LINEAR_NAMES = frozenset({"road", "rail", "stream", "fence"})
_BUILDING_TYPES = ("residential_buildings", "churches", "barns")
_EMPTY_VALUES = {None, "", -1, "-1"}


@dataclass(frozen=True, slots=True)
class ReconstructedBuildingFootprint:
    geometry: BaseGeometry
    building_type: str
    is_diagonal: bool


@dataclass(frozen=True, slots=True)
class FinalGeometryOverlap:
    row_index: int
    xidx: float
    yidx: float
    name: str
    overlap_area_m2: float


@dataclass(frozen=True, slots=True)
class FinalGeometryIssue:
    building_row_index: int
    output_xidx: float
    output_yidx: float
    cat2: Any
    direction: Any
    footprint_bounds: tuple[float, float, float, float] | None
    footprint_area_m2: float
    overlapping_linear_cells: tuple[FinalGeometryOverlap, ...]
    overlap_area_m2: float
    reason: str


@dataclass(frozen=True, slots=True)
class FinalGeometryValidation:
    issues: tuple[FinalGeometryIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def issue_summary(self) -> str:
        if not self.issues:
            return "final output geometry is valid"
        first = self.issues[0]
        return (
            "final building-linear geometry collision: "
            f"row={first.building_row_index} x={first.output_xidx} y={first.output_yidx} "
            f"cat2={first.cat2!r} direction={first.direction!r} "
            f"overlap_area_m2={first.overlap_area_m2:.6f}"
        )


def reconstruct_linear_row_geometry(grid_index: Any, row: Mapping[str, Any]) -> BaseGeometry:
    xidx = int(_coordinate(row, "xidx", "x"))
    yidx = int(_coordinate(row, "yidx", "y"))
    return grid_index.cell_polygon(GridCell(xidx, yidx))


def reconstruct_building_row_geometry(
    grid_index: Any,
    row: Mapping[str, Any],
    *,
    profile: str = "cold_war",
    building_type: str | None = None,
) -> ReconstructedBuildingFootprint:
    from profiles import get_building_outline_by_df_entry

    xidx = _coordinate(row, "xidx", "x")
    yidx = _coordinate(row, "yidx", "y")
    menu = _required_value(row, "menu")
    cat1 = _required_value(row, "cat1")
    cat2 = _required_value(row, "cat2")
    direction = _direction_value(_required_value(row, "direction"))
    candidates = _building_type_candidates(row, building_type)

    for candidate_type in candidates:
        try:
            outline_result = get_building_outline_by_df_entry(
                candidate_type,
                menu,
                cat1,
                cat2,
                direction,
                profile=profile,
            )
        except (IndexError, TypeError, ValueError):
            continue
        if outline_result is None:
            continue
        outline, is_diagonal = outline_result
        return ReconstructedBuildingFootprint(
            geometry=_profile_outline_at_output_anchor(grid_index, outline, xidx, yidx),
            building_type=candidate_type,
            is_diagonal=bool(is_diagonal),
        )
    raise ValueError(f"building row could not be resolved through profile {profile!r}: {row}")


def validate_final_output_geometry(
    rows: Iterable[Mapping[str, Any]],
    grid_index: Any,
    *,
    profile: str = "cold_war",
    building_type: str | None = None,
    min_overlap_area_m2: float = EPSILON_M2,
) -> FinalGeometryValidation:
    row_tuple = tuple(rows)
    linear_geometries = tuple(
        (row_index, row, reconstruct_linear_row_geometry(grid_index, row))
        for row_index, row in enumerate(row_tuple)
        if is_linear_output_row(row)
    )
    issues: list[FinalGeometryIssue] = []
    for row_index, row in enumerate(row_tuple):
        if not _is_potential_building_row(row):
            continue
        try:
            reconstructed = reconstruct_building_row_geometry(
                grid_index,
                row,
                profile=profile,
                building_type=building_type,
            )
        except ValueError:
            if _is_explicit_building_row(row) and ("_building_type" in row or building_type is not None):
                issues.append(_empty_issue(row_index, row, reason="unresolved_building_profile_row"))
            continue

        footprint = reconstructed.geometry
        if footprint.is_empty or footprint.area <= min_overlap_area_m2:
            issues.append(_empty_issue(row_index, row, geometry=footprint, reason="empty_building_footprint"))
            continue

        overlaps: list[FinalGeometryOverlap] = []
        for linear_index, linear_row, linear_geometry in linear_geometries:
            overlap_area = footprint.intersection(linear_geometry).area
            if overlap_area <= min_overlap_area_m2:
                continue
            overlaps.append(
                FinalGeometryOverlap(
                    row_index=linear_index,
                    xidx=_coordinate(linear_row, "xidx", "x"),
                    yidx=_coordinate(linear_row, "yidx", "y"),
                    name=str(linear_row.get("name", "")),
                    overlap_area_m2=round(overlap_area, 6),
                )
            )
        if overlaps:
            issues.append(
                FinalGeometryIssue(
                    building_row_index=row_index,
                    output_xidx=_coordinate(row, "xidx", "x"),
                    output_yidx=_coordinate(row, "yidx", "y"),
                    cat2=row.get("cat2"),
                    direction=row.get("direction"),
                    footprint_bounds=tuple(round(value, 6) for value in footprint.bounds),
                    footprint_area_m2=round(footprint.area, 6),
                    overlapping_linear_cells=tuple(overlaps),
                    overlap_area_m2=round(sum(overlap.overlap_area_m2 for overlap in overlaps), 6),
                    reason="building_linear_overlap",
                )
            )
    return FinalGeometryValidation(issues=tuple(issues))


def is_linear_output_row(row: Mapping[str, Any]) -> bool:
    layer = row.get("_layer")
    if layer in _LINEAR_LAYERS:
        return True
    return str(row.get("name", "")).lower() in _LINEAR_NAMES


def _is_potential_building_row(row: Mapping[str, Any]) -> bool:
    return _is_explicit_building_row(row) or (
        row.get("name") not in _EMPTY_VALUES
        and row.get("direction") not in _EMPTY_VALUES
        and row.get("cat2") not in _EMPTY_VALUES
        and not is_linear_output_row(row)
        and str(row.get("name", "")) not in {"extent_marker", "default_ground", "default_foliage"}
    )


def _is_explicit_building_row(row: Mapping[str, Any]) -> bool:
    return row.get("_layer") == LayerKind.BUILDING.value


def _building_type_candidates(row: Mapping[str, Any], building_type: str | None) -> tuple[str, ...]:
    candidates = []
    for value in (
        building_type,
        row.get("_building_type"),
        row.get("building_type"),
        row.get("profile_building_type"),
        row.get("name"),
    ):
        if isinstance(value, str) and value in _BUILDING_TYPES and value not in candidates:
            candidates.append(value)
    for value in _BUILDING_TYPES:
        if value not in candidates:
            candidates.append(value)
    return tuple(candidates)


def _profile_outline_at_output_anchor(
    grid_index: Any,
    outline: Polygon,
    output_xidx: float,
    output_yidx: float,
) -> Polygon:
    selected_xidx, selected_yidx = selected_grid_index_from_building_output(output_xidx, output_yidx)
    anchor_local_x, anchor_local_y = building_anchor_local_from_selected_grid_index(
        grid_index,
        selected_xidx,
        selected_yidx,
    )
    points = []
    for local_x, local_y in list(outline.exterior.coords)[:-1]:
        projected = grid_index.projected_from_local(anchor_local_x + local_x, anchor_local_y + local_y)
        points.append((projected.x, projected.y))
    return Polygon(points)


def _empty_issue(
    row_index: int,
    row: Mapping[str, Any],
    *,
    geometry: BaseGeometry | None = None,
    reason: str,
) -> FinalGeometryIssue:
    return FinalGeometryIssue(
        building_row_index=row_index,
        output_xidx=_coordinate(row, "xidx", "x"),
        output_yidx=_coordinate(row, "yidx", "y"),
        cat2=row.get("cat2"),
        direction=row.get("direction"),
        footprint_bounds=None if geometry is None or geometry.is_empty else tuple(round(value, 6) for value in geometry.bounds),
        footprint_area_m2=0.0 if geometry is None else round(geometry.area, 6),
        overlapping_linear_cells=(),
        overlap_area_m2=0.0,
        reason=reason,
    )


def _coordinate(row: Mapping[str, Any], primary: str, fallback: str) -> float:
    value = row.get(primary, row.get(fallback))
    if value in _EMPTY_VALUES:
        raise ValueError(f"row is missing coordinate {primary!r}: {row}")
    return float(value)


def _required_value(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value in _EMPTY_VALUES:
        raise ValueError(f"row is missing {key!r}: {row}")
    return value


def _direction_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f"Direction {int(value) + 1}"
