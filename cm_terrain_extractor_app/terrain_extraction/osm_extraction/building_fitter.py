from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from terrain_extraction.osm_extraction.models import (
    BuildingFittingResult,
    CMType,
    FeatureRecord,
    GridCell,
    GridKind,
    LayerKind,
    PlacementRecord,
    ProcessKind,
)
from terrain_extraction.osm_extraction.occupancy import OccupancyModel

_LINEAR_LAYERS = frozenset({LayerKind.LINEAR_SURFACE, LayerKind.LINEAR_OBJECT})
_DEFAULT_MAX_CANDIDATES = 256


@dataclass(frozen=True, slots=True)
class BuildingFootprint:
    footprint_id: str
    width_cells: int
    height_cells: int
    cm_type: CMType
    direction: int | None
    row: int
    col: int
    is_diagonal: bool = False
    is_modular: bool = False
    weight: float = 1.0

    @property
    def area_cells(self) -> int:
        return self.width_cells * self.height_cells


@dataclass(frozen=True, slots=True)
class BuildingDescriptor:
    area: float
    oriented_rectangle: Polygon
    rectangularity: float
    elongation: float
    orientation_radians: float
    nearest_road_distance_m: float | None


@dataclass(frozen=True, slots=True)
class BuildingCandidate:
    footprint: BuildingFootprint
    cells: tuple[GridCell, ...]
    footprint_polygon: BaseGeometry
    iou: float
    centroid_shift_m: float
    angle_error_radians: float
    area_error_ratio: float
    road_overlap_cells: int
    collision_cells: int
    score: float
    limit_reached: bool


@dataclass(frozen=True, slots=True)
class _PreparedBuilding:
    feature: FeatureRecord
    geometry: Polygon
    descriptor: BuildingDescriptor
    candidates: tuple[BuildingCandidate, ...]
    candidate_limit_reached: bool
    diagnostics: Mapping[str, Any]

    @property
    def feature_key(self) -> str | int:
        if self.feature.feature_id is not None:
            return self.feature.feature_id
        return self.feature.source_index

    @property
    def available_count(self) -> int:
        return sum(1 for candidate in self.candidates if candidate.collision_cells == 0)


class BuildingCatalog:
    def __init__(self, footprints: Iterable[BuildingFootprint]) -> None:
        self.footprints = tuple(footprints)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]] | Any) -> BuildingCatalog:
        return cls(_footprint_from_record(record) for record in _records_from_any(records))

    def normalized(self) -> tuple[BuildingFootprint, ...]:
        return tuple(
            sorted(
                self.footprints,
                key=lambda footprint: (
                    footprint.is_modular,
                    footprint.is_diagonal,
                    footprint.area_cells,
                    footprint.row,
                    footprint.col,
                    footprint.footprint_id,
                ),
            )
        )


class BuildingFitter:
    def __init__(
        self,
        grid_index: Any,
        *,
        occupancy: OccupancyModel | None = None,
        rng: np.random.Generator | None = None,
        max_candidates_per_building: int = _DEFAULT_MAX_CANDIDATES,
        local_shift_cells: int = 1,
    ) -> None:
        self.grid_index = grid_index
        self.occupancy = occupancy or OccupancyModel.from_grid_index(grid_index)
        self.rng = rng or np.random.default_rng(0)
        self.max_candidates_per_building = max_candidates_per_building
        self.local_shift_cells = local_shift_cells

    def fit(
        self,
        features: Sequence[FeatureRecord],
        *,
        catalogs: Mapping[str, Iterable[Mapping[str, Any]] | BuildingCatalog | Any],
    ) -> BuildingFittingResult:
        compiled_catalogs = _compile_catalogs(catalogs)
        prepared: list[_PreparedBuilding] = []
        failures: list[Mapping[str, Any]] = []
        diagnostics_by_feature: dict[str | int, Mapping[str, Any]] = {}

        for feature in _building_features(features):
            feature_key = _feature_key(feature)
            polygon = _clean_building_polygon(feature.geometry)
            if polygon is None:
                failure = _failure(feature, "invalid_or_empty_geometry", diagnostics={})
                failures.append(failure)
                diagnostics_by_feature[feature_key] = failure
                continue

            catalog = compiled_catalogs.get(feature.config_name)
            if catalog is None or not catalog.footprints:
                failure = _failure(feature, "missing_building_catalog", diagnostics={})
                failures.append(failure)
                diagnostics_by_feature[feature_key] = failure
                continue

            descriptor = self._descriptor(polygon)
            candidates, candidate_limit_reached = self._candidates_for(polygon, descriptor, catalog.normalized())
            diagnostics = _building_diagnostics(descriptor, candidates, candidate_limit_reached)
            diagnostics_by_feature[feature_key] = diagnostics
            prepared.append(
                _PreparedBuilding(
                    feature=feature,
                    geometry=polygon,
                    descriptor=descriptor,
                    candidates=candidates,
                    candidate_limit_reached=candidate_limit_reached,
                    diagnostics=diagnostics,
                )
            )

        placements: list[PlacementRecord] = []
        cluster_order = []
        for building in sorted(prepared, key=_cluster_sort_key):
            cluster_order.append(building.feature_key)
            placement = self._place_building(building)
            if placement is None:
                failure = _failure(
                    building.feature,
                    "no_non_colliding_candidate",
                    diagnostics=building.diagnostics,
                )
                failures.append(failure)
                diagnostics_by_feature[building.feature_key] = failure
                continue
            placements.append(placement)
            diagnostics_by_feature[building.feature_key] = {
                **dict(building.diagnostics),
                **dict(placement.diagnostics),
            }

        return BuildingFittingResult(
            placements=tuple(placements),
            failures=tuple(failures),
            diagnostics={
                "buildings_placed": len(placements),
                "buildings_dropped": len(failures),
                "cluster_order": tuple(cluster_order),
            },
            diagnostics_by_feature=diagnostics_by_feature,
        )

    def _descriptor(self, polygon: Polygon) -> BuildingDescriptor:
        rectangle = polygon.minimum_rotated_rectangle
        rectangle_area = rectangle.area
        side_lengths, orientation = _rectangle_sides_and_orientation(rectangle)
        short_side = max(min(side_lengths), 1e-9)
        nearest_road_distance = self._nearest_linear_distance_m(polygon)
        return BuildingDescriptor(
            area=polygon.area,
            oriented_rectangle=rectangle,
            rectangularity=polygon.area / rectangle_area if rectangle_area > 0 else 0.0,
            elongation=max(side_lengths) / short_side,
            orientation_radians=orientation,
            nearest_road_distance_m=nearest_road_distance,
        )

    def _candidates_for(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprints: tuple[BuildingFootprint, ...],
    ) -> tuple[tuple[BuildingCandidate, ...], bool]:
        candidates: list[BuildingCandidate] = []
        seen: set[tuple[str, tuple[GridCell, ...]]] = set()
        limit_reached = False

        for footprint in footprints:
            for cells, orientation in self._candidate_cells(polygon, footprint):
                key = (footprint.footprint_id, cells)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(self._score_candidate(polygon, descriptor, footprint, cells, orientation))
                if len(candidates) >= self.max_candidates_per_building:
                    limit_reached = True
                    break
            if limit_reached:
                break

        candidates.sort(key=_candidate_sort_key)
        return tuple(candidates), limit_reached

    def _candidate_cells(
        self,
        polygon: Polygon,
        footprint: BuildingFootprint,
    ) -> Iterable[tuple[tuple[GridCell, ...], float]]:
        if footprint.is_modular:
            modular_cells = self._modular_cells(polygon)
            if modular_cells:
                yield modular_cells, 0.0
            return

        min_local_x, min_local_y = self.grid_index.local_from_projected(*polygon.bounds[:2])
        base_xidx = math.floor(min_local_x / self.grid_index.cell_size_m)
        base_yidx = math.floor(min_local_y / self.grid_index.cell_size_m)
        orientations = _footprint_orientations(footprint)

        for width, height, orientation in orientations:
            for x_shift in range(-self.local_shift_cells, self.local_shift_cells + 1):
                for y_shift in range(-self.local_shift_cells, self.local_shift_cells + 1):
                    xidx = base_xidx + x_shift
                    yidx = base_yidx + y_shift
                    cells = _rect_cells(xidx, yidx, width, height)
                    if all(_contains_cell(self.grid_index, cell) for cell in cells):
                        yield cells, orientation

    def _modular_cells(self, polygon: Polygon) -> tuple[GridCell, ...]:
        min_x, min_y, max_x, max_y = polygon.bounds
        min_local_x, min_local_y = self.grid_index.local_from_projected(min_x, min_y)
        max_local_x, max_local_y = self.grid_index.local_from_projected(max_x, max_y)
        window = self.grid_index.clipped_cell_window(
            math.floor(min_local_x / self.grid_index.cell_size_m),
            math.floor(min_local_y / self.grid_index.cell_size_m),
            math.floor(max_local_x / self.grid_index.cell_size_m),
            math.floor(max_local_y / self.grid_index.cell_size_m),
        )
        if window is None:
            return ()

        cells = []
        min_xidx, min_yidx, max_xidx, max_yidx = window
        cell_area = self.grid_index.cell_size_m * self.grid_index.cell_size_m
        for xidx in range(min_xidx, max_xidx + 1):
            for yidx in range(min_yidx, max_yidx + 1):
                cell = GridCell(xidx, yidx)
                overlap_ratio = self.grid_index.cell_polygon(cell).intersection(polygon).area / cell_area
                if overlap_ratio > 0.5:
                    cells.append(cell)
        return tuple(sorted(cells))

    def _score_candidate(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint: BuildingFootprint,
        cells: tuple[GridCell, ...],
        orientation: float,
    ) -> BuildingCandidate:
        footprint_polygon = unary_union([self.grid_index.cell_polygon(cell) for cell in cells])
        union_area = footprint_polygon.union(polygon).area
        iou = footprint_polygon.intersection(polygon).area / union_area if union_area > 0 else 0.0
        centroid_shift = footprint_polygon.centroid.distance(polygon.centroid)
        area_error = abs(footprint_polygon.area - polygon.area) / polygon.area if polygon.area > 0 else 1.0
        angle_error = _angle_error(descriptor.orientation_radians, orientation)
        placement = _placement_from_candidate(footprint, cells, polygon, iou, centroid_shift, angle_error, area_error)
        decision = self.occupancy.can_place(placement)
        road_overlap = sum(1 for conflict in decision.conflicts if conflict.blocking_layer in _LINEAR_LAYERS)
        collision_count = len(decision.conflicts)
        score = (
            iou * 100.0
            - min(centroid_shift, 128.0) * 0.2
            - area_error * 25.0
            - angle_error * 4.0
            - road_overlap * 250.0
            - (collision_count - road_overlap) * 100.0
            - max(0, len(cells) - 1) * (0.05 if footprint.is_modular else 0.0)
        )
        return BuildingCandidate(
            footprint=footprint,
            cells=cells,
            footprint_polygon=footprint_polygon,
            iou=round(iou, 6),
            centroid_shift_m=centroid_shift,
            angle_error_radians=angle_error,
            area_error_ratio=area_error,
            road_overlap_cells=road_overlap,
            collision_cells=collision_count,
            score=score,
            limit_reached=False,
        )

    def _place_building(self, building: _PreparedBuilding) -> PlacementRecord | None:
        ordered_candidates = sorted(building.candidates, key=_candidate_sort_key)
        for candidate in ordered_candidates:
            placement = _placement_from_candidate(
                candidate.footprint,
                candidate.cells,
                building.geometry,
                candidate.iou,
                candidate.centroid_shift_m,
                candidate.angle_error_radians,
                candidate.area_error_ratio,
                config_name=building.feature.config_name,
                feature_id=building.feature.feature_id,
                priority=building.feature.priority,
                road_overlap_cells=candidate.road_overlap_cells,
                score=candidate.score,
                candidate_limit_reached=building.candidate_limit_reached,
            )
            decision = self.occupancy.place(
                placement,
                object_id=building.feature_key,
                metadata={"source_feature_id": building.feature.feature_id},
            )
            if decision.allowed:
                return placement
        return None

    def _nearest_linear_distance_m(self, polygon: Polygon) -> float | None:
        distances = []
        for layer in _LINEAR_LAYERS:
            occupied = self.occupancy.occupied[layer]
            x_indices, y_indices = np.where(occupied >= 0)
            for xidx, yidx in zip(x_indices, y_indices, strict=False):
                center = self.grid_index.cell_center(GridCell(int(xidx), int(yidx)))
                distances.append(center.distance(polygon))
        if not distances:
            return None
        return min(distances)


def _compile_catalogs(
    catalogs: Mapping[str, Iterable[Mapping[str, Any]] | BuildingCatalog | Any],
) -> dict[str, BuildingCatalog]:
    compiled = {}
    for name, catalog in catalogs.items():
        compiled[name] = catalog if isinstance(catalog, BuildingCatalog) else BuildingCatalog.from_records(catalog)
    return compiled


def _records_from_any(records: Iterable[Mapping[str, Any]] | Any) -> tuple[Mapping[str, Any], ...]:
    if hasattr(records, "to_dict"):
        return tuple(records.to_dict("records"))
    return tuple(records)


def _footprint_from_record(record: Mapping[str, Any]) -> BuildingFootprint:
    width = max(1, int(record.get("width", 1)))
    height = max(1, int(record.get("height", 1)))
    row = int(record.get("row", 0))
    col = int(record.get("col", 0))
    direction = _optional_int(record.get("direction"))
    cm_type = CMType(
        menu=str(record.get("menu", "Buildings")),
        cat1=str(record.get("cat1", "Building")),
        cat2=str(record.get("cat2", f"Building {row}-{col}")),
        direction=None if direction is None else f"Direction {direction + 1}",
        tile_id=record.get("id"),
        modifiers={key: record[key] for key in ("stories", "weight") if key in record},
    )
    return BuildingFootprint(
        footprint_id=str(record.get("id", f"{row}:{col}:{direction}:{width}x{height}")),
        width_cells=width,
        height_cells=height,
        cm_type=cm_type,
        direction=direction,
        row=row,
        col=col,
        is_diagonal=bool(record.get("is_diagonal", False)),
        is_modular=bool(record.get("is_modular", False)),
        weight=float(record.get("weight", 1.0)),
    )


def _building_features(features: Sequence[FeatureRecord]) -> tuple[FeatureRecord, ...]:
    return tuple(feature for feature in features if feature.process is ProcessKind.BUILDING_OUTLINE)


def _clean_building_polygon(geometry: BaseGeometry) -> Polygon | None:
    if geometry.is_empty:
        return None
    cleaned = geometry.buffer(0) if not geometry.is_valid else geometry
    polygons: list[Polygon] = []
    if isinstance(cleaned, Polygon):
        polygons.append(cleaned)
    elif isinstance(cleaned, MultiPolygon):
        polygons.extend(polygon for polygon in cleaned.geoms if polygon.area > 0)
    if not polygons:
        return None
    largest = max(polygons, key=lambda polygon: polygon.area)
    return largest if largest.area > 0 else None


def _rectangle_sides_and_orientation(rectangle: Polygon) -> tuple[tuple[float, ...], float]:
    coords = list(rectangle.exterior.coords)
    sides = []
    for start, end in zip(coords, coords[1:], strict=False):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        sides.append((math.hypot(dx, dy), math.atan2(dy, dx)))
    longest = max(sides, key=lambda side: side[0])
    return tuple(side[0] for side in sides), longest[1]


def _footprint_orientations(footprint: BuildingFootprint) -> tuple[tuple[int, int, float], ...]:
    orientations = [(footprint.width_cells, footprint.height_cells, 0.0)]
    if footprint.width_cells != footprint.height_cells:
        orientations.append((footprint.height_cells, footprint.width_cells, math.pi / 2))
    return tuple(orientations)


def _rect_cells(xidx: int, yidx: int, width: int, height: int) -> tuple[GridCell, ...]:
    return tuple(GridCell(x, y) for x in range(xidx, xidx + width) for y in range(yidx, yidx + height))


def _contains_cell(grid_index: Any, cell: GridCell) -> bool:
    return 0 <= cell.xidx < grid_index.width and 0 <= cell.yidx < grid_index.height


def _placement_from_candidate(
    footprint: BuildingFootprint,
    cells: tuple[GridCell, ...],
    polygon: Polygon,
    iou: float,
    centroid_shift: float,
    angle_error: float,
    area_error: float,
    *,
    config_name: str = "building",
    feature_id: str | int | None = None,
    priority: int = 0,
    road_overlap_cells: int = 0,
    score: float = 0.0,
    candidate_limit_reached: bool = False,
) -> PlacementRecord:
    return PlacementRecord(
        layer=LayerKind.BUILDING,
        grid_kind=GridKind.DIAGONAL if footprint.is_diagonal else GridKind.NORMAL,
        cells=tuple(sorted(cells)),
        config_name=config_name,
        feature_id=feature_id,
        priority=priority,
        cm_type=footprint.cm_type,
        score=score,
        diagnostics={
            "iou": round(iou, 6),
            "centroid_shift_m": round(centroid_shift, 6),
            "angle_error_radians": round(angle_error, 6),
            "area_error_ratio": round(area_error, 6),
            "road_overlap_cells": road_overlap_cells,
            "module_count": len(cells),
            "source_area_m2": round(polygon.area, 6),
            "candidate_limit_reached": candidate_limit_reached,
        },
    )


def _building_diagnostics(
    descriptor: BuildingDescriptor,
    candidates: tuple[BuildingCandidate, ...],
    candidate_limit_reached: bool,
) -> Mapping[str, Any]:
    return {
        "area_m2": round(descriptor.area, 6),
        "rectangularity": round(descriptor.rectangularity, 6),
        "elongation": round(descriptor.elongation, 6),
        "orientation_radians": round(descriptor.orientation_radians, 6),
        "nearest_road_distance_m": (
            None if descriptor.nearest_road_distance_m is None else round(descriptor.nearest_road_distance_m, 6)
        ),
        "candidates_scored": len(candidates),
        "available_candidates": sum(1 for candidate in candidates if candidate.collision_cells == 0),
        "candidate_limit_reached": candidate_limit_reached,
        "best_iou": None if not candidates else max(candidate.iou for candidate in candidates),
    }


def _cluster_sort_key(building: _PreparedBuilding) -> tuple[int, int, float, str]:
    return (
        building.available_count,
        len(building.candidates),
        building.descriptor.area,
        str(building.feature_key),
    )


def _candidate_sort_key(candidate: BuildingCandidate) -> tuple[float, float, int, tuple[GridCell, ...]]:
    return (-candidate.score, -candidate.iou, candidate.collision_cells, candidate.cells)


def _angle_error(source_angle: float, candidate_angle: float) -> float:
    diff = abs((source_angle - candidate_angle + math.pi / 2) % math.pi - math.pi / 2)
    return min(diff, abs(math.pi / 2 - diff))


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def _feature_key(feature: FeatureRecord) -> str | int:
    if feature.feature_id is not None:
        return feature.feature_id
    return feature.source_index


def _failure(
    feature: FeatureRecord,
    reason: str,
    *,
    diagnostics: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "feature_id": feature.feature_id,
        "source_index": feature.source_index,
        "config_name": feature.config_name,
        "failure_reason": reason,
        **dict(diagnostics),
    }
