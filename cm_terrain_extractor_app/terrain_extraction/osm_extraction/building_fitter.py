from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from terrain_extraction.osm_extraction.building_geometry import (
    EPSILON_M2,
    building_output_from_selected_grid_index,
    cell_window_for_geometry,
    local_bounds,
    normal_cells_overlapped_by_polygon,
    reconstruct_building_footprint_polygon,
    selected_grid_index_from_building_output,
)
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
_DEFAULT_MAX_FOOTPRINT_TYPES = 24
_DEFAULT_MAX_SHIFTED_CANDIDATES_PER_FOOTPRINT = 9
_DEFAULT_MAX_SCORED_CANDIDATES = 64
_DEFAULT_SINGLE_RECT_TOP_K = 5
_DEFAULT_SINGLE_RECT_SHIFT_STEPS = 1
_DEFAULT_MODULAR_AREA_THRESHOLD_M2 = 320.0
_DEFAULT_MODULAR_RECTANGULARITY_THRESHOLD = 0.80
_DEFAULT_MODULAR_SINGLE_SHAPE_ERROR_THRESHOLD = 0.55
_DEFAULT_MAX_MODULAR_PIECES = 12
_DEFAULT_MAX_MODULAR_STATES = 128
_DEFAULT_MAX_MODULAR_TIME_MS = 30.0
_SPECIAL_BUILDING_TAGS = frozenset(
    {
        "barn",
        "church",
        "civic",
        "college",
        "commercial",
        "farm",
        "farm_auxiliary",
        "industrial",
        "retail",
        "school",
        "university",
        "warehouse",
    }
)


@dataclass(frozen=True, slots=True)
class BuildingFootprint:
    footprint_id: str
    width_cells: int
    height_cells: int
    cm_type: CMType
    direction: int | None
    row: int
    col: int
    building_type: str | None = None
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
    mrr_length_m: float
    mrr_width_m: float
    aspect_ratio: float
    orientation_radians: float
    preferred_orientation: str
    nearest_road_distance_m: float | None


@dataclass(frozen=True, slots=True)
class BuildingCandidate:
    footprint: BuildingFootprint
    cells: tuple[GridCell, ...]
    footprint_polygon: BaseGeometry
    selected_grid_xidx: float | None
    selected_grid_yidx: float | None
    output_xidx: float | None
    output_yidx: float | None
    reconstructed_selected_grid_xidx: float | None
    reconstructed_selected_grid_yidx: float | None
    roundtrip_error: float
    footprint_width_units: int
    footprint_height_units: int
    footprint_is_diagonal: bool
    footprint_row: int
    footprint_col: int
    footprint_direction: int | None
    orientation_class: str
    iou: float
    centroid_shift_m: float
    angle_error_radians: float
    area_error_ratio: float
    road_overlap_area_m2: float
    road_overlap_ratio: float
    road_overlap_cells: int
    collision_cells: int
    score: float
    limit_reached: bool


@dataclass(frozen=True, slots=True)
class BuildingPlacementGeometry:
    selected_grid_xidx: float | None
    selected_grid_yidx: float | None
    output_xidx: float | None
    output_yidx: float | None
    reconstructed_selected_grid_xidx: float | None
    reconstructed_selected_grid_yidx: float | None
    roundtrip_error: float
    output_grid_kind: GridKind
    anchor_cell: GridCell | None
    blocked_normal_cells: tuple[GridCell, ...]
    footprint_polygon: BaseGeometry
    footprint_width_units: int
    footprint_height_units: int
    footprint_is_diagonal: bool
    footprint_row: int
    footprint_col: int
    footprint_direction: int | None
    orientation: float
    orientation_class: str

    @property
    def cells(self) -> tuple[GridCell, ...]:
        return self.blocked_normal_cells


_CandidateGeometry = BuildingPlacementGeometry


@dataclass(frozen=True, slots=True)
class _FootprintProfile:
    footprint: BuildingFootprint
    area_m2: float
    area_error_ratio: float
    best_angle_error: float
    best_swapped: bool
    length_error_ratio: float
    width_error_ratio: float
    aspect_error: float
    shape_error: float
    combined_error: float
    best_orientation_class: str
    tie_key: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _CheapCandidate:
    geometry: _CandidateGeometry
    footprint_profile: _FootprintProfile
    area_error_ratio: float
    centroid_shift_m: float
    angle_error_radians: float
    orientation_mismatch: int
    linear_overlap_cells: int
    sort_key: tuple[Any, ...]


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
        max_candidates_per_building: int | None = None,
        max_footprint_types_per_building: int = _DEFAULT_MAX_FOOTPRINT_TYPES,
        max_shifted_candidates_per_footprint: int = _DEFAULT_MAX_SHIFTED_CANDIDATES_PER_FOOTPRINT,
        max_scored_candidates_per_building: int = _DEFAULT_MAX_SCORED_CANDIDATES,
        single_rect_top_k: int = _DEFAULT_SINGLE_RECT_TOP_K,
        single_rect_shift_steps: int = _DEFAULT_SINGLE_RECT_SHIFT_STEPS,
        local_shift_cells: int = 1,
        modular_area_threshold: float = 1.0,
        modular_area_threshold_m2: float = _DEFAULT_MODULAR_AREA_THRESHOLD_M2,
        modular_rectangularity_threshold: float = _DEFAULT_MODULAR_RECTANGULARITY_THRESHOLD,
        modular_single_shape_error_threshold: float = _DEFAULT_MODULAR_SINGLE_SHAPE_ERROR_THRESHOLD,
        max_modular_pieces: int = _DEFAULT_MAX_MODULAR_PIECES,
        max_modular_states: int = _DEFAULT_MAX_MODULAR_STATES,
        max_modular_time_ms: float = _DEFAULT_MAX_MODULAR_TIME_MS,
        debug_geometry: bool = False,
        profile: str = "cold_war",
    ) -> None:
        self.grid_index = grid_index
        self.occupancy = occupancy or OccupancyModel.from_grid_index(grid_index)
        self.rng = rng or np.random.default_rng(0)
        if max_candidates_per_building is not None:
            max_scored_candidates_per_building = max_candidates_per_building
        self.max_footprint_types_per_building = max(1, max_footprint_types_per_building)
        self.max_shifted_candidates_per_footprint = max(1, max_shifted_candidates_per_footprint)
        self.max_scored_candidates_per_building = max(1, max_scored_candidates_per_building)
        self.max_candidates_per_building = self.max_scored_candidates_per_building
        self.single_rect_top_k = max(1, single_rect_top_k)
        self.single_rect_shift_steps = max(0, single_rect_shift_steps)
        self.local_shift_cells = local_shift_cells
        self.modular_area_threshold = modular_area_threshold
        self.modular_area_threshold_m2 = modular_area_threshold_m2
        self.modular_rectangularity_threshold = modular_rectangularity_threshold
        self.modular_single_shape_error_threshold = modular_single_shape_error_threshold
        self.max_modular_pieces = max(1, max_modular_pieces)
        self.max_modular_states = max(1, max_modular_states)
        self.max_modular_time_ms = max(0.0, max_modular_time_ms)
        self.debug_geometry = debug_geometry
        self.profile = profile
        self._shapely_score_evaluations = 0
        self._shapely_overlap_evaluations = 0

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
        self._shapely_score_evaluations = 0
        self._shapely_overlap_evaluations = 0

        for feature in _building_features(features):
            feature_started_at = time.perf_counter()
            feature_key = _feature_key(feature)
            polygon = _clean_building_polygon(feature.geometry)
            if polygon is None:
                failure = _failure(
                    feature,
                    "invalid_or_empty_geometry",
                    diagnostics=_timed_building_diagnostics(feature_started_at),
                )
                failures.append(failure)
                diagnostics_by_feature[feature_key] = failure
                continue

            catalog = compiled_catalogs.get(feature.config_name)
            if catalog is None or not catalog.footprints:
                failure = _failure(
                    feature,
                    "missing_building_catalog",
                    diagnostics=_timed_building_diagnostics(feature_started_at),
                )
                failures.append(failure)
                diagnostics_by_feature[feature_key] = failure
                continue

            descriptor = self._descriptor(polygon)
            candidate_started_at = time.perf_counter()
            candidates, candidate_limit_reached, candidate_diagnostics = self._candidates_for(
                feature,
                polygon,
                descriptor,
                catalog.normalized(),
            )
            candidate_generation_ms = round((time.perf_counter() - candidate_started_at) * 1000.0, 3)
            diagnostics = {
                **_building_diagnostics(descriptor, candidates, candidate_limit_reached),
                **candidate_diagnostics,
                **_timed_building_diagnostics(feature_started_at),
                "candidate_generation_ms": candidate_generation_ms,
                "placement_ms": 0.0,
                "feature_id": feature.feature_id,
                "config_name": feature.config_name,
            }
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

        placements, placement_failures, cluster_order = self._place_prepared_buildings(prepared, diagnostics_by_feature)
        failures.extend(placement_failures)
        candidates_scored_per_building = {
            str(feature_key): int(diagnostics.get("candidates_scored", 0))
            for feature_key, diagnostics in diagnostics_by_feature.items()
        }

        return BuildingFittingResult(
            placements=tuple(placements),
            failures=tuple(failures),
            diagnostics={
                "building_feature_count": len(diagnostics_by_feature),
                "buildings_placed": len(placements),
                "buildings_dropped": len(failures),
                "total_candidates_scored": sum(candidates_scored_per_building.values()),
                "candidates_scored_per_building": candidates_scored_per_building,
                "candidate_limit_reached_count": sum(
                    1
                    for diagnostics in diagnostics_by_feature.values()
                    if diagnostics.get("candidate_limit_reached")
                ),
                "shapely_score_evaluations": self._shapely_score_evaluations,
                "shapely_overlap_evaluations": self._shapely_overlap_evaluations,
                "cluster_order": tuple(cluster_order),
            },
            diagnostics_by_feature=diagnostics_by_feature,
        )

    def _place_prepared_buildings(
        self,
        prepared: list[_PreparedBuilding],
        diagnostics_by_feature: dict[str | int, Mapping[str, Any]],
    ) -> tuple[list[PlacementRecord], list[Mapping[str, Any]], list[str | int]]:
        placements: list[PlacementRecord] = []
        failures: list[Mapping[str, Any]] = []
        placed_by_key: dict[str | int, PlacementRecord] = {}
        placed_index_by_key: dict[str | int, int] = {}
        prepared_by_key = {building.feature_key: building for building in prepared}
        cluster_order = []
        for building in sorted(prepared, key=_cluster_sort_key):
            cluster_order.append(building.feature_key)
            placement_started_at = time.perf_counter()
            placement = self._place_building(building)
            if placement is None:
                repair = self._repair_with_neighbor(building, placed_by_key, prepared_by_key)
                if repair is not None:
                    neighbor_key, repaired_neighbor, placement = repair
                    placements[placed_index_by_key[neighbor_key]] = repaired_neighbor
                    placed_by_key[neighbor_key] = repaired_neighbor
            placement_ms = round((time.perf_counter() - placement_started_at) * 1000.0, 3)
            elapsed_ms = round(float(building.diagnostics.get("elapsed_ms", 0.0)) + placement_ms, 3)
            timed_diagnostics = {
                **dict(building.diagnostics),
                "placement_ms": placement_ms,
                "elapsed_ms": elapsed_ms,
            }
            if placement is None:
                failure = _failure(
                    building.feature,
                    "no_non_colliding_candidate",
                    diagnostics=timed_diagnostics,
                )
                failures.append(failure)
                diagnostics_by_feature[building.feature_key] = failure
                continue
            placed_index_by_key[building.feature_key] = len(placements)
            placed_by_key[building.feature_key] = placement
            placements.append(placement)
            diagnostics_by_feature[building.feature_key] = {
                **timed_diagnostics,
                **dict(placement.diagnostics),
            }
        return placements, failures, cluster_order

    def _descriptor(self, polygon: Polygon) -> BuildingDescriptor:
        rectangle = polygon.minimum_rotated_rectangle
        rectangle_area = rectangle.area
        side_lengths, orientation = _rectangle_sides_and_orientation(rectangle)
        mrr_length = max(side_lengths)
        mrr_width = max(min(side_lengths), 1e-9)
        nearest_road_distance = self._nearest_linear_distance_m(polygon)
        return BuildingDescriptor(
            area=polygon.area,
            oriented_rectangle=rectangle,
            rectangularity=polygon.area / rectangle_area if rectangle_area > 0 else 0.0,
            elongation=mrr_length / mrr_width,
            mrr_length_m=mrr_length,
            mrr_width_m=mrr_width,
            aspect_ratio=mrr_length / mrr_width,
            orientation_radians=orientation,
            preferred_orientation=_orientation_class(orientation),
            nearest_road_distance_m=nearest_road_distance,
        )

    def _candidates_for(
        self,
        feature: FeatureRecord,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprints: tuple[BuildingFootprint, ...],
    ) -> tuple[tuple[BuildingCandidate, ...], bool, Mapping[str, Any]]:
        single_profiles = self._single_rect_footprint_profiles(descriptor, footprints)
        best_single = single_profiles[0] if single_profiles else None
        fit_mode, fallback_reason = self._fit_mode(feature, descriptor, best_single)
        candidates: list[BuildingCandidate] = []
        selected_single_profiles: tuple[_FootprintProfile, ...]
        single_generated = 0
        modular_attempted = False
        modular_piece_count = 0
        modular_limit_reached = False
        modular_elapsed_ms = 0.0
        modular_profiles_considered = 0

        if fit_mode == "modular_cover":
            selected_single_profiles = single_profiles[:1]
            modular_attempted = True
            modular_started_at = time.perf_counter()
            modular_candidates, modular_limit_reached, modular_diagnostics = self._modular_cover_candidates_for(
                polygon,
                descriptor,
                footprints,
                feature.config_name,
            )
            modular_elapsed_ms = round((time.perf_counter() - modular_started_at) * 1000.0, 3)
            modular_piece_count = int(modular_diagnostics.get("modular_piece_count", 0))
            modular_profiles_considered = int(modular_diagnostics.get("modular_profiles_considered", 0))
            candidates.extend(modular_candidates)
            fallback_limit = max(0, self.max_scored_candidates_per_building - len(candidates))
            single_candidates, single_generated = self._single_rect_candidates_for(
                polygon,
                descriptor,
                selected_single_profiles,
                feature.config_name,
                placement_limit=fallback_limit,
            )
            candidates.extend(single_candidates)
        else:
            single_budget = min(self.single_rect_top_k, self.max_scored_candidates_per_building)
            selected_single_profiles = single_profiles[:single_budget]
            single_candidates, single_generated = self._single_rect_candidates_for(
                polygon,
                descriptor,
                selected_single_profiles,
                feature.config_name,
            )
            candidates.extend(single_candidates)

        candidates.sort(key=_candidate_sort_key)

        best_single_shape_error = None if best_single is None else round(best_single.shape_error, 6)
        best_single_footprint = None if best_single is None else best_single.footprint.footprint_id
        selected_best = candidates[0] if candidates else None
        diagnostics = {
            "fit_mode": fit_mode,
            "preferred_orientation": descriptor.preferred_orientation,
            "best_single_rect_footprint": best_single_footprint,
            "best_single_rect_shape_error": best_single_shape_error,
            "footprint_types_considered": len(selected_single_profiles) + modular_profiles_considered,
            "shifted_candidates_generated": single_generated,
            "expensive_candidates_scored": len(candidates),
            "placements_tested": len(candidates),
            "modular_attempted": modular_attempted,
            "modular_piece_count": modular_piece_count,
            "modular_elapsed_ms": modular_elapsed_ms,
            "fallback_reason": fallback_reason,
            "selected_width_units": None if selected_best is None else selected_best.footprint.width_cells,
            "selected_height_units": None if selected_best is None else selected_best.footprint.height_cells,
            "selected_is_diagonal": None if selected_best is None else selected_best.footprint.is_diagonal,
            "selected_centroid_shift_m": None if selected_best is None else round(selected_best.centroid_shift_m, 6),
            "selected_area_error": None if selected_best is None else round(selected_best.area_error_ratio, 6),
            "selected_road_overlap": None if selected_best is None else selected_best.road_overlap_cells,
        }
        return tuple(candidates), modular_limit_reached, diagnostics

    def _single_rect_footprint_profiles(
        self,
        descriptor: BuildingDescriptor,
        footprints: tuple[BuildingFootprint, ...],
    ) -> tuple[_FootprintProfile, ...]:
        profiles = tuple(
            self._footprint_profile(footprint, descriptor, descriptor.preferred_orientation)
            for footprint in footprints
            if not footprint.is_modular
        )
        return tuple(sorted(profiles, key=_footprint_combined_sort_key))

    def _fit_mode(
        self,
        feature: FeatureRecord,
        descriptor: BuildingDescriptor,
        best_single: _FootprintProfile | None,
    ) -> tuple[str, str | None]:
        if best_single is None:
            return "modular_cover", "missing_single_rect_footprint"
        if _is_special_building(feature):
            return "modular_cover", "special_building_type"
        if descriptor.rectangularity < self.modular_rectangularity_threshold:
            return "modular_cover", "low_rectangularity"
        if (
            not _is_residential_like_building(feature)
            and best_single.shape_error > self.modular_single_shape_error_threshold
        ):
            return "modular_cover", "poor_single_rect_shape_fit"
        if (
            descriptor.area >= self.modular_area_threshold_m2 * 2.0
            and best_single.shape_error > self.modular_single_shape_error_threshold * 0.5
        ):
            return "modular_cover", "large_area"
        return "single_rect", None

    def _single_rect_candidates_for(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint_profiles: tuple[_FootprintProfile, ...],
        config_name: str,
        placement_limit: int | None = None,
    ) -> tuple[tuple[BuildingCandidate, ...], int]:
        candidates: list[BuildingCandidate] = []
        seen: set[tuple[str, tuple[GridCell, ...], float | None, float | None]] = set()
        generated = 0
        for profile in footprint_profiles:
            for geometry in self._single_rect_candidate_geometries(polygon, descriptor, profile):
                key = _candidate_geometry_key(profile.footprint, geometry)
                if key in seen:
                    continue
                seen.add(key)
                if placement_limit is not None and len(candidates) >= placement_limit:
                    candidates.sort(key=_candidate_sort_key)
                    return tuple(candidates), generated
                generated += 1
                candidates.append(self._score_candidate(polygon, descriptor, profile.footprint, geometry, config_name))
        candidates.sort(key=_candidate_sort_key)
        return tuple(candidates), generated

    def _single_rect_candidate_geometries(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint_profile: _FootprintProfile,
    ) -> Iterable[_CandidateGeometry]:
        footprint = footprint_profile.footprint
        half_cell_size = self.grid_index.cell_size_m / 2.0
        source_centroid_local = self.grid_index.local_from_projected(polygon.centroid.x, polygon.centroid.y)
        swapped = footprint_profile.best_swapped
        orientation = _orientation_for_swapped(footprint, swapped)
        origin_x_base, origin_y_base = _centered_origin_local(
            source_centroid_local,
            footprint,
            self.grid_index.cell_size_m,
            swapped=swapped,
        )
        snapped_origin_x = round(origin_x_base / half_cell_size) * half_cell_size
        snapped_origin_y = round(origin_y_base / half_cell_size) * half_cell_size
        shift_steps = self.single_rect_shift_steps
        if descriptor.nearest_road_distance_m is not None and descriptor.nearest_road_distance_m <= self.grid_index.cell_size_m:
            shift_steps += 1
        for x_shift, y_shift in _placement_shift_offsets(shift_steps):
            origin_x = snapped_origin_x + x_shift * half_cell_size
            origin_y = snapped_origin_y + y_shift * half_cell_size
            geometry = _candidate_geometry_from_local_origin(
                self.grid_index,
                footprint,
                origin_x,
                origin_y,
                swapped=swapped,
                orientation=orientation,
                orientation_class="diagonal" if footprint.is_diagonal else "axis",
            )
            if geometry is None:
                continue
            yield geometry

    def _modular_cover_candidates_for(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprints: tuple[BuildingFootprint, ...],
        config_name: str,
    ) -> tuple[tuple[BuildingCandidate, ...], bool, Mapping[str, Any]]:
        modular_profiles = tuple(
            sorted(
                (
                    self._footprint_profile(footprint, descriptor, descriptor.preferred_orientation)
                    for footprint in footprints
                    if footprint.is_modular
                ),
                key=_footprint_combined_sort_key,
            )[:1]
        )
        if not modular_profiles:
            return (), False, {"modular_piece_count": 0, "modular_profiles_considered": 0}

        modular_cells, piece_limit_reached = self._bounded_modular_cells(polygon)
        if not modular_cells:
            return (), piece_limit_reached, {
                "modular_piece_count": 0,
                "modular_profiles_considered": len(modular_profiles),
            }
        footprint_polygon = unary_union([self.grid_index.cell_polygon(cell) for cell in modular_cells])
        modular_footprint = modular_profiles[0].footprint
        origin_x, origin_y = _polygon_origin_local(self.grid_index, footprint_polygon)
        geometry = _candidate_geometry_from_local_origin(
            self.grid_index,
            modular_footprint,
            origin_x,
            origin_y,
            swapped=False,
            orientation=0.0,
            orientation_class="diagonal" if modular_footprint.is_diagonal else "axis",
        )
        if geometry is None:
            return (), piece_limit_reached, {
                "modular_piece_count": len(modular_cells),
                "modular_profiles_considered": len(modular_profiles),
            }
        candidates = tuple(
            self._score_candidate(polygon, descriptor, profile.footprint, geometry, config_name)
            for profile in modular_profiles
        )
        return candidates, piece_limit_reached, {
            "modular_piece_count": len(modular_cells),
            "modular_profiles_considered": len(modular_profiles),
        }

    def _bounded_modular_cells(self, polygon: Polygon) -> tuple[tuple[GridCell, ...], bool]:
        started_at = time.perf_counter()
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
            return (), False

        min_xidx, min_yidx, max_xidx, max_yidx = window
        cell_area = self.grid_index.cell_size_m * self.grid_index.cell_size_m
        scored_cells: list[tuple[float, GridCell]] = []
        limit_reached = False
        states_seen = 0
        for xidx in range(min_xidx, max_xidx + 1):
            for yidx in range(min_yidx, max_yidx + 1):
                states_seen += 1
                if states_seen > self.max_modular_states:
                    limit_reached = True
                    break
                if self.max_modular_time_ms and (time.perf_counter() - started_at) * 1000.0 > self.max_modular_time_ms:
                    limit_reached = True
                    break
                cell = GridCell(xidx, yidx)
                overlap_ratio = self.grid_index.cell_polygon(cell).intersection(polygon).area / cell_area
                if overlap_ratio > 0.5:
                    scored_cells.append((overlap_ratio, cell))
            if limit_reached:
                break
        scored_cells.sort(key=lambda item: (-item[0], item[1]))
        if len(scored_cells) > self.max_modular_pieces:
            limit_reached = True
        return tuple(sorted(cell for _, cell in scored_cells[: self.max_modular_pieces])), limit_reached

    def _shortlisted_footprint_profiles(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprints: tuple[BuildingFootprint, ...],
    ) -> tuple[_FootprintProfile, ...]:
        source_orientation_class = _orientation_class(descriptor.orientation_radians)
        profiles = tuple(
            self._footprint_profile(footprint, descriptor, source_orientation_class)
            for footprint in footprints
        )
        if not profiles:
            return ()

        non_modular_profiles = tuple(profile for profile in profiles if not profile.footprint.is_modular)
        best_non_modular_area_error = min(
            (profile.area_error_ratio for profile in non_modular_profiles),
            default=math.inf,
        )
        max_non_modular_area = max((profile.area_m2 for profile in non_modular_profiles), default=0.0)
        allow_modular = (
            not non_modular_profiles
            or polygon.area > max_non_modular_area * self.modular_area_threshold
            or best_non_modular_area_error > 0.65
            or descriptor.rectangularity < 0.72
        )
        eligible = tuple(
            profile
            for profile in profiles
            if allow_modular or not profile.footprint.is_modular
        )
        if not eligible:
            return ()

        selected: dict[str, _FootprintProfile] = {}
        area_budget = min(8, max(1, self.max_footprint_types_per_building // 2))
        combined_budget = min(8, self.max_footprint_types_per_building)

        for profile in sorted(eligible, key=_footprint_area_sort_key)[:area_budget]:
            selected[profile.footprint.footprint_id] = profile
        for profile in sorted(eligible, key=_footprint_combined_sort_key)[:combined_budget]:
            selected[profile.footprint.footprint_id] = profile

        diagonal_profiles = tuple(profile for profile in eligible if profile.footprint.is_diagonal)
        if diagonal_profiles:
            best_combined = min(profile.combined_error for profile in eligible)
            best_diagonal = min(diagonal_profiles, key=_footprint_combined_sort_key)
            if source_orientation_class == "diagonal" or best_diagonal.combined_error <= best_combined + 0.35:
                selected[best_diagonal.footprint.footprint_id] = best_diagonal

        for profile in sorted(eligible, key=_footprint_combined_sort_key):
            if len(selected) >= self.max_footprint_types_per_building:
                break
            selected.setdefault(profile.footprint.footprint_id, profile)

        return tuple(sorted(selected.values(), key=_footprint_combined_sort_key))

    def _footprint_profile(
        self,
        footprint: BuildingFootprint,
        descriptor: BuildingDescriptor,
        source_orientation_class: str,
    ) -> _FootprintProfile:
        area_m2 = _footprint_area_m2(footprint, self.grid_index.cell_size_m)
        area_error = abs(area_m2 - descriptor.area) / descriptor.area if descriptor.area > 0 else 1.0
        variant_errors = []
        for swapped, orientation in _footprint_orientations(footprint):
            width_m, height_m = _footprint_dimensions_m(footprint, self.grid_index.cell_size_m, swapped=swapped)
            candidate_length = max(width_m, height_m)
            candidate_width = max(min(width_m, height_m), 1e-9)
            length_error = abs(candidate_length - descriptor.mrr_length_m) / max(descriptor.mrr_length_m, 1e-9)
            width_error = abs(candidate_width - descriptor.mrr_width_m) / max(descriptor.mrr_width_m, 1e-9)
            aspect = candidate_length / candidate_width
            aspect_error = abs(math.log(max(aspect, 1.0) / max(descriptor.aspect_ratio, 1.0)))
            angle_error = _angle_error(descriptor.orientation_radians, orientation)
            orientation_class = "diagonal" if footprint.is_diagonal else "axis"
            variant_errors.append((swapped, length_error, width_error, aspect_error, angle_error, orientation_class))
        best_swapped, length_error, width_error, aspect_error, best_angle_error, best_orientation_class = min(
            variant_errors,
            key=lambda item: (
                0 if item[5] == source_orientation_class else 1,
                item[1] + item[2] + item[3] + item[4],
                item[5],
            ),
        )
        orientation_mismatch = 0 if source_orientation_class == best_orientation_class else 1
        modular_penalty = 0.2 if footprint.is_modular else 0.0
        shape_error = (
            length_error * 0.45
            + width_error * 0.45
            + area_error * 0.65
            + aspect_error * 0.25
            + best_angle_error * 0.35
            + orientation_mismatch * 0.3
        )
        combined_error = (
            shape_error
            + modular_penalty
        )
        return _FootprintProfile(
            footprint=footprint,
            area_m2=area_m2,
            area_error_ratio=area_error,
            best_angle_error=best_angle_error,
            best_swapped=best_swapped,
            length_error_ratio=length_error,
            width_error_ratio=width_error,
            aspect_error=aspect_error,
            shape_error=shape_error,
            combined_error=combined_error,
            best_orientation_class=best_orientation_class,
            tie_key=_footprint_tie_key(footprint),
        )

    def _candidate_geometries(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint_profile: _FootprintProfile,
    ) -> Iterable[_CandidateGeometry]:
        footprint = footprint_profile.footprint
        if footprint.is_modular:
            modular_cells = self._modular_cells(polygon)
            if modular_cells:
                footprint_polygon = unary_union([self.grid_index.cell_polygon(cell) for cell in modular_cells])
                origin_x, origin_y = _polygon_origin_local(self.grid_index, footprint_polygon)
                geometry = _candidate_geometry_from_local_origin(
                    self.grid_index,
                    footprint,
                    origin_x,
                    origin_y,
                    swapped=False,
                    orientation=0.0,
                    orientation_class="diagonal" if footprint.is_diagonal else "axis",
                )
                if geometry is not None:
                    yield geometry
            return

        yield from self._profile_candidate_geometries(polygon, descriptor, footprint_profile)

    def _profile_candidate_geometries(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint_profile: _FootprintProfile,
    ) -> Iterable[_CandidateGeometry]:
        footprint = footprint_profile.footprint
        half_cell_size = self.grid_index.cell_size_m / 2.0
        variants = _footprint_orientations(footprint)
        geometries: list[_CheapCandidate] = []

        for swapped, orientation in variants:
            for origin_x_base, origin_y_base in self._placement_origin_indices(polygon, descriptor, footprint, swapped):
                for x_shift, y_shift in _shift_offsets(self.local_shift_cells):
                    origin_x = (origin_x_base + x_shift) * half_cell_size
                    origin_y = (origin_y_base + y_shift) * half_cell_size
                    geometry = _candidate_geometry_from_local_origin(
                        self.grid_index,
                        footprint,
                        origin_x,
                        origin_y,
                        swapped=swapped,
                        orientation=orientation,
                        orientation_class="diagonal" if footprint.is_diagonal else "axis",
                    )
                    if geometry is None:
                        continue
                    geometries.append(self._cheap_candidate(polygon, descriptor, footprint_profile, geometry))

        geometries.sort(key=lambda candidate: candidate.sort_key)
        for cheap_candidate in geometries[: self.max_shifted_candidates_per_footprint]:
            yield cheap_candidate.geometry

    def _placement_origin_indices(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint: BuildingFootprint,
        swapped: bool,
    ) -> tuple[tuple[int, int], ...]:
        half_cell_size = self.grid_index.cell_size_m / 2.0
        width_units = footprint.height_cells if swapped else footprint.width_cells
        height_units = footprint.width_cells if swapped else footprint.height_cells
        width_m = width_units * half_cell_size
        height_m = height_units * half_cell_size
        source_centroid_local = self.grid_index.local_from_projected(polygon.centroid.x, polygon.centroid.y)
        polygon_min_x, polygon_min_y, _, _ = _local_bounds(self.grid_index, polygon)
        rectangle_min_x, rectangle_min_y, _, _ = _local_bounds(self.grid_index, descriptor.oriented_rectangle)

        origins = []
        if footprint.is_diagonal:
            centroid_offset = ((width_m + height_m) / 2.0, (-width_m + height_m) / 2.0)
            origins.append((source_centroid_local[0] - centroid_offset[0], source_centroid_local[1] - centroid_offset[1]))
            origins.append((rectangle_min_x, rectangle_min_y + width_m))
            origins.append((polygon_min_x, polygon_min_y + width_m))
        else:
            origins.append((source_centroid_local[0] - width_m / 2.0, source_centroid_local[1] - height_m / 2.0))
            origins.append((rectangle_min_x, rectangle_min_y))
            origins.append((polygon_min_x, polygon_min_y))

        snapped = []
        seen = set()
        for origin_x, origin_y in origins:
            key = (round(origin_x / half_cell_size), round(origin_y / half_cell_size))
            if key in seen:
                continue
            seen.add(key)
            snapped.append((int(key[0]), int(key[1])))
        return tuple(snapped)

    def _cheap_candidate(
        self,
        polygon: Polygon,
        descriptor: BuildingDescriptor,
        footprint_profile: _FootprintProfile,
        geometry: _CandidateGeometry,
    ) -> _CheapCandidate:
        area_error = abs(geometry.footprint_polygon.area - polygon.area) / polygon.area if polygon.area > 0 else 1.0
        centroid_shift = geometry.footprint_polygon.centroid.distance(polygon.centroid)
        angle_error = _angle_error(descriptor.orientation_radians, geometry.orientation)
        source_orientation_class = _orientation_class(descriptor.orientation_radians)
        orientation_mismatch = 0 if source_orientation_class == geometry.orientation_class else 1
        linear_overlap = self._linear_cell_overlap_count(geometry.cells)
        max_side = max(math.sqrt(max(polygon.area, 1e-9)), self.grid_index.cell_size_m)
        sort_key = (
            area_error > 1.75,
            centroid_shift > max_side * 2.5,
            linear_overlap,
            orientation_mismatch,
            area_error,
            centroid_shift / max_side,
            angle_error,
            len(geometry.cells),
            footprint_profile.tie_key,
            geometry.cells,
            geometry.output_xidx if geometry.output_xidx is not None else -1.0,
            geometry.output_yidx if geometry.output_yidx is not None else -1.0,
        )
        return _CheapCandidate(
            geometry=geometry,
            footprint_profile=footprint_profile,
            area_error_ratio=area_error,
            centroid_shift_m=centroid_shift,
            angle_error_radians=angle_error,
            orientation_mismatch=orientation_mismatch,
            linear_overlap_cells=linear_overlap,
            sort_key=sort_key,
        )

    def _linear_cell_overlap_count(self, cells: tuple[GridCell, ...]) -> int:
        overlap = 0
        for cell in cells:
            if not _contains_cell(self.grid_index, cell):
                continue
            if any(int(self.occupancy.occupied[layer][cell.xidx, cell.yidx]) >= 0 for layer in _LINEAR_LAYERS):
                overlap += 1
        return overlap

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
        geometry: _CandidateGeometry,
        config_name: str,
    ) -> BuildingCandidate:
        footprint_polygon = self._final_candidate_footprint(footprint, geometry, config_name)
        self._shapely_score_evaluations += 1
        final_cells = _cells_overlapped_by_polygon(self.grid_index, footprint_polygon)
        if not final_cells:
            final_cells = geometry.cells
        union_area = footprint_polygon.union(polygon).area
        iou = footprint_polygon.intersection(polygon).area / union_area if union_area > 0 else 0.0
        centroid_shift = footprint_polygon.centroid.distance(polygon.centroid)
        area_error = abs(footprint_polygon.area - polygon.area) / polygon.area if polygon.area > 0 else 1.0
        angle_error = _angle_error(descriptor.orientation_radians, geometry.orientation)
        source_orientation_class = _orientation_class(descriptor.orientation_radians)
        orientation_mismatch = 0 if source_orientation_class == geometry.orientation_class else 1
        road_overlap_area, road_overlap_ratio, road_overlap_cells = self._linear_overlap_metrics(footprint_polygon)
        placement = _placement_from_candidate(
            footprint,
            final_cells,
            polygon,
            iou,
            centroid_shift,
            angle_error,
            area_error,
            selected_footprint_polygon=footprint_polygon,
            selected_grid_xidx=geometry.selected_grid_xidx,
            selected_grid_yidx=geometry.selected_grid_yidx,
            output_xidx=geometry.output_xidx,
            output_yidx=geometry.output_yidx,
            reconstructed_selected_grid_xidx=geometry.reconstructed_selected_grid_xidx,
            reconstructed_selected_grid_yidx=geometry.reconstructed_selected_grid_yidx,
            roundtrip_error=geometry.roundtrip_error,
            footprint_width_units=geometry.footprint_width_units,
            footprint_height_units=geometry.footprint_height_units,
            footprint_is_diagonal=geometry.footprint_is_diagonal,
            footprint_row=geometry.footprint_row,
            footprint_col=geometry.footprint_col,
            footprint_direction=geometry.footprint_direction,
            orientation_class=geometry.orientation_class,
            road_overlap_area_m2=road_overlap_area,
            road_overlap_ratio=road_overlap_ratio,
            road_overlap_cells=road_overlap_cells,
            debug_geometry=self.debug_geometry,
        )
        decision = self.occupancy.can_place(placement)
        road_overlap_conflicts = sum(1 for conflict in decision.conflicts if conflict.blocking_layer in _LINEAR_LAYERS)
        road_overlap = max(road_overlap_cells, road_overlap_conflicts)
        collision_count = len(decision.conflicts)
        non_road_collision_count = max(0, collision_count - road_overlap_conflicts)
        score = (
            iou * 100.0
            - min(centroid_shift, 128.0) * 0.2
            - area_error * 25.0
            - angle_error * 4.0
            - orientation_mismatch * 35.0
            - road_overlap_area * 8.0
            - road_overlap_ratio * 250.0
            - road_overlap * 250.0
            - non_road_collision_count * 100.0
            - max(0, len(final_cells) - 1) * (0.05 if footprint.is_modular else 0.0)
        )
        return BuildingCandidate(
            footprint=footprint,
            cells=final_cells,
            footprint_polygon=footprint_polygon,
            selected_grid_xidx=geometry.selected_grid_xidx,
            selected_grid_yidx=geometry.selected_grid_yidx,
            output_xidx=geometry.output_xidx,
            output_yidx=geometry.output_yidx,
            reconstructed_selected_grid_xidx=geometry.reconstructed_selected_grid_xidx,
            reconstructed_selected_grid_yidx=geometry.reconstructed_selected_grid_yidx,
            roundtrip_error=geometry.roundtrip_error,
            footprint_width_units=geometry.footprint_width_units,
            footprint_height_units=geometry.footprint_height_units,
            footprint_is_diagonal=geometry.footprint_is_diagonal,
            footprint_row=geometry.footprint_row,
            footprint_col=geometry.footprint_col,
            footprint_direction=geometry.footprint_direction,
            orientation_class=geometry.orientation_class,
            iou=round(iou, 6),
            centroid_shift_m=centroid_shift,
            angle_error_radians=angle_error,
            area_error_ratio=area_error,
            road_overlap_area_m2=road_overlap_area,
            road_overlap_ratio=road_overlap_ratio,
            road_overlap_cells=road_overlap,
            collision_cells=collision_count,
            score=score,
            limit_reached=False,
        )

    def _final_candidate_footprint(
        self,
        footprint: BuildingFootprint,
        geometry: _CandidateGeometry,
        config_name: str,
    ) -> BaseGeometry:
        return geometry.footprint_polygon

    def _place_building(self, building: _PreparedBuilding) -> PlacementRecord | None:
        ordered_candidates = self._ordered_candidates(building.candidates)
        for candidate in ordered_candidates:
            if candidate.road_overlap_area_m2 > EPSILON_M2 or candidate.road_overlap_cells > 0:
                continue
            placement = self._placement_for_candidate(building, candidate)
            decision = self.occupancy.place(
                placement,
                object_id=building.feature_key,
                metadata={"source_feature_id": building.feature.feature_id},
            )
            if decision.allowed:
                return placement
        return None

    def _placement_for_candidate(self, building: _PreparedBuilding, candidate: BuildingCandidate) -> PlacementRecord:
        return _placement_from_candidate(
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
            selected_footprint_polygon=candidate.footprint_polygon,
            selected_grid_xidx=candidate.selected_grid_xidx,
            selected_grid_yidx=candidate.selected_grid_yidx,
            output_xidx=candidate.output_xidx,
            output_yidx=candidate.output_yidx,
            reconstructed_selected_grid_xidx=candidate.reconstructed_selected_grid_xidx,
            reconstructed_selected_grid_yidx=candidate.reconstructed_selected_grid_yidx,
            roundtrip_error=candidate.roundtrip_error,
            footprint_width_units=candidate.footprint_width_units,
            footprint_height_units=candidate.footprint_height_units,
            footprint_is_diagonal=candidate.footprint_is_diagonal,
            footprint_row=candidate.footprint_row,
            footprint_col=candidate.footprint_col,
            footprint_direction=candidate.footprint_direction,
            orientation_class=candidate.orientation_class,
            road_overlap_area_m2=candidate.road_overlap_area_m2,
            road_overlap_ratio=candidate.road_overlap_ratio,
            debug_geometry=self.debug_geometry,
        )

    def _repair_with_neighbor(
        self,
        building: _PreparedBuilding,
        placed_by_key: Mapping[str | int, PlacementRecord],
        prepared_by_key: Mapping[str | int, _PreparedBuilding],
    ) -> tuple[str | int, PlacementRecord, PlacementRecord] | None:
        for neighbor_key in self._blocking_building_keys(building):
            old_neighbor = placed_by_key.get(neighbor_key)
            neighbor_building = prepared_by_key.get(neighbor_key)
            if old_neighbor is None or neighbor_building is None:
                continue
            if not self.occupancy.release(neighbor_key):
                continue
            blocked_placement = self._place_building(building)
            if blocked_placement is None:
                self.occupancy.place(
                    old_neighbor,
                    object_id=neighbor_key,
                    metadata={"source_feature_id": neighbor_building.feature.feature_id},
                )
                continue
            repaired_neighbor = self._place_building(neighbor_building)
            if repaired_neighbor is not None:
                return neighbor_key, repaired_neighbor, blocked_placement
            self.occupancy.release(building.feature_key)
            self.occupancy.place(
                old_neighbor,
                object_id=neighbor_key,
                metadata={"source_feature_id": neighbor_building.feature.feature_id},
            )
        return None

    def _blocking_building_keys(self, building: _PreparedBuilding) -> tuple[str | int, ...]:
        blocking_keys: list[str | int] = []
        seen = set()
        for candidate in self._ordered_candidates(building.candidates):
            decision = self.occupancy.can_place(self._placement_for_candidate(building, candidate))
            for conflict in decision.conflicts:
                if conflict.blocking_layer is not LayerKind.BUILDING:
                    continue
                if conflict.blocking_object_id in seen:
                    continue
                seen.add(conflict.blocking_object_id)
                blocking_keys.append(conflict.blocking_object_id)
            if blocking_keys:
                break
        return tuple(blocking_keys)

    def _ordered_candidates(self, candidates: tuple[BuildingCandidate, ...]) -> tuple[BuildingCandidate, ...]:
        ordered = sorted(candidates, key=_candidate_sort_key)
        if len(ordered) < 2:
            return tuple(ordered)

        top_score = ordered[0].score
        top = [candidate for candidate in ordered if math.isclose(candidate.score, top_score, rel_tol=1e-9, abs_tol=1e-9)]
        if len(top) < 2:
            return tuple(ordered)

        weights = np.array([max(0.0, candidate.footprint.weight) for candidate in top], dtype=float)
        if not np.any(weights > 0):
            weights = np.ones(len(top), dtype=float)
        chosen_index = int(self.rng.choice(len(top), p=weights / weights.sum()))
        chosen = top[chosen_index]
        return (chosen, *(candidate for candidate in ordered if candidate is not chosen))

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

    def _linear_overlap_metrics(self, polygon: BaseGeometry) -> tuple[float, float, int]:
        overlap_area = 0.0
        overlap_cells = 0
        seen_geometry_objects: set[str | int] = set()
        window = cell_window_for_geometry(self.grid_index, polygon, margin_cells=1)
        if window is None:
            return 0.0, 0.0, 0
        min_xidx, min_yidx, max_xidx, max_yidx = window
        for layer in _LINEAR_LAYERS:
            occupied = self.occupancy.occupied[layer]
            local_occupied = occupied[min_xidx : max_xidx + 1, min_yidx : max_yidx + 1]
            x_indices, y_indices = np.where(local_occupied >= 0)
            for xidx, yidx in zip(x_indices, y_indices, strict=False):
                cell = GridCell(int(xidx) + min_xidx, int(yidx) + min_yidx)
                object_id = self.occupancy.object_id_at(layer, cell)
                linear_polygon = self._occupied_linear_polygon(layer, cell, object_id)
                if object_id is not None and object_id in seen_geometry_objects:
                    continue
                if object_id is not None and _uses_diagnostic_polygon(self.occupancy.metadata.get(object_id, {})):
                    seen_geometry_objects.add(object_id)
                self._shapely_overlap_evaluations += 1
                intersection_area = polygon.intersection(linear_polygon).area
                if intersection_area <= 0:
                    continue
                overlap_area += intersection_area
                overlap_cells += 1
        ratio = overlap_area / polygon.area if polygon.area > 0 else 0.0
        return overlap_area, ratio, overlap_cells

    def _occupied_linear_polygon(self, layer: LayerKind, cell: GridCell, object_id: str | int | None) -> BaseGeometry:
        metadata = self.occupancy.metadata.get(object_id, {}) if object_id is not None else {}
        diagnostics = metadata.get("diagnostics", {}) if isinstance(metadata, Mapping) else {}
        geometry = diagnostics.get("selected_footprint_polygon") if isinstance(diagnostics, Mapping) else None
        if isinstance(geometry, BaseGeometry):
            return geometry
        return self.grid_index.cell_polygon(cell)


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
        building_type=record.get("building_type"),
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


def _footprint_orientations(footprint: BuildingFootprint) -> tuple[tuple[bool, float], ...]:
    if footprint.is_diagonal:
        return ((False, -math.pi / 4),)
    orientations = [(False, 0.0)]
    if footprint.width_cells != footprint.height_cells and _allows_synthetic_rotation(footprint):
        orientations.append((True, math.pi / 2))
    return tuple(orientations)


def _allows_synthetic_rotation(footprint: BuildingFootprint) -> bool:
    return footprint.building_type is None


def _orientation_for_swapped(footprint: BuildingFootprint, swapped: bool) -> float:
    for variant_swapped, orientation in _footprint_orientations(footprint):
        if variant_swapped == swapped:
            return orientation
    return -math.pi / 4 if footprint.is_diagonal else 0.0


def _candidate_geometry_from_local_origin(
    grid_index: Any,
    footprint: BuildingFootprint,
    origin_local_x: float,
    origin_local_y: float,
    *,
    swapped: bool,
    orientation: float,
    orientation_class: str,
) -> _CandidateGeometry | None:
    width_units = footprint.height_cells if swapped else footprint.width_cells
    height_units = footprint.width_cells if swapped else footprint.height_cells
    try:
        selected_xidx, selected_yidx = _nearest_selected_grid_index(
            grid_index,
            origin_local_x,
            origin_local_y,
            GridKind.DIAGONAL if footprint.is_diagonal else GridKind.SUB_SQUARE,
        )
    except ValueError:
        return None
    return _candidate_geometry_from_selected_grid_index(
        grid_index,
        footprint,
        selected_xidx,
        selected_yidx,
        swapped=swapped,
        width_units=width_units,
        height_units=height_units,
        orientation=orientation,
        orientation_class=orientation_class,
    )


def _candidate_geometry_from_selected_grid_index(
    grid_index: Any,
    footprint: BuildingFootprint,
    selected_grid_xidx: float,
    selected_grid_yidx: float,
    *,
    swapped: bool,
    width_units: int,
    height_units: int,
    orientation: float,
    orientation_class: str,
) -> _CandidateGeometry | None:
    output_xidx, output_yidx = building_output_from_selected_grid_index(selected_grid_xidx, selected_grid_yidx)
    reconstructed_xidx, reconstructed_yidx = selected_grid_index_from_building_output(output_xidx, output_yidx)
    roundtrip_error = math.hypot(reconstructed_xidx - selected_grid_xidx, reconstructed_yidx - selected_grid_yidx)
    footprint_polygon = reconstruct_building_footprint_polygon(
        grid_index,
        output_xidx,
        output_yidx,
        footprint,
        swapped=swapped,
    )
    if not _polygon_within_grid(grid_index, footprint_polygon):
        return None
    cells = _cells_overlapped_by_polygon(grid_index, footprint_polygon)
    if not cells:
        return None
    return _CandidateGeometry(
        selected_grid_xidx=selected_grid_xidx,
        selected_grid_yidx=selected_grid_yidx,
        output_xidx=output_xidx,
        output_yidx=output_yidx,
        reconstructed_selected_grid_xidx=reconstructed_xidx,
        reconstructed_selected_grid_yidx=reconstructed_yidx,
        roundtrip_error=roundtrip_error,
        output_grid_kind=GridKind.DIAGONAL if footprint.is_diagonal else GridKind.SUB_SQUARE,
        anchor_cell=cells[0],
        blocked_normal_cells=cells,
        footprint_polygon=footprint_polygon,
        footprint_width_units=width_units,
        footprint_height_units=height_units,
        footprint_is_diagonal=footprint.is_diagonal,
        footprint_row=footprint.row,
        footprint_col=footprint.col,
        footprint_direction=footprint.direction,
        orientation=orientation,
        orientation_class=orientation_class,
    )


def _nearest_selected_grid_index(
    grid_index: Any,
    origin_local_x: float,
    origin_local_y: float,
    grid_kind: GridKind,
) -> tuple[float, float]:
    center_cell = GridCell(
        math.floor(origin_local_x / grid_index.cell_size_m),
        math.floor(origin_local_y / grid_index.cell_size_m),
    )
    candidates: list[tuple[float, float, float]] = []
    for xidx in range(center_cell.xidx - 1, center_cell.xidx + 2):
        for yidx in range(center_cell.yidx - 1, center_cell.yidx + 2):
            cell = GridCell(xidx, yidx)
            if not _contains_cell(grid_index, cell):
                continue
            centers = grid_index.diagonal_centers(cell) if grid_kind is GridKind.DIAGONAL else grid_index.sub_square_centers(cell)
            for (selected_xidx, selected_yidx), point in centers.items():
                local_x, local_y = grid_index.local_from_projected(point.x, point.y)
                candidates.append(
                    (
                        math.hypot(local_x - origin_local_x, local_y - origin_local_y),
                        selected_xidx,
                        selected_yidx,
                    )
                )
    if not candidates:
        raise ValueError("no valid selected building placement grid coordinate near origin")
    _, selected_xidx, selected_yidx = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return _clean_float(selected_xidx), _clean_float(selected_yidx)


def _centered_origin_local(
    centroid_local: tuple[float, float],
    footprint: BuildingFootprint,
    cell_size_m: float,
    *,
    swapped: bool,
) -> tuple[float, float]:
    width_units = footprint.height_cells if swapped else footprint.width_cells
    height_units = footprint.width_cells if swapped else footprint.height_cells
    half_cell_size = cell_size_m / 2.0
    width_m = width_units * half_cell_size
    height_m = height_units * half_cell_size
    if footprint.is_diagonal:
        centroid_offset = ((width_m + height_m) / 2.0, (-width_m + height_m) / 2.0)
        return centroid_local[0] - centroid_offset[0], centroid_local[1] - centroid_offset[1]
    return centroid_local[0] - width_m / 2.0, centroid_local[1] - height_m / 2.0


def _cells_overlapped_by_polygon(grid_index: Any, polygon: BaseGeometry) -> tuple[GridCell, ...]:
    return normal_cells_overlapped_by_polygon(grid_index, polygon)


def _cell_window_for_geometry(
    grid_index: Any,
    geometry: BaseGeometry,
    *,
    margin_cells: int = 0,
) -> tuple[int, int, int, int] | None:
    return cell_window_for_geometry(grid_index, geometry, margin_cells=margin_cells)


def _local_bounds(grid_index: Any, geometry: BaseGeometry) -> tuple[float, float, float, float]:
    return local_bounds(grid_index, geometry)


def _polygon_within_grid(grid_index: Any, polygon: BaseGeometry) -> bool:
    min_local_x, min_local_y, max_local_x, max_local_y = _local_bounds(grid_index, polygon)
    epsilon = 1e-9
    return (
        min_local_x >= -epsilon
        and min_local_y >= -epsilon
        and max_local_x <= grid_index.width * grid_index.cell_size_m + epsilon
        and max_local_y <= grid_index.height * grid_index.cell_size_m + epsilon
    )


def _polygon_origin_local(grid_index: Any, polygon: BaseGeometry) -> tuple[float, float]:
    min_local_x, min_local_y, _, _ = _local_bounds(grid_index, polygon)
    return min_local_x, min_local_y


def _clean_float(value: float) -> float:
    rounded = round(value, 6)
    return int(rounded) if float(rounded).is_integer() else rounded


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
    selected_footprint_polygon: BaseGeometry | None = None,
    selected_grid_xidx: float | None = None,
    selected_grid_yidx: float | None = None,
    output_xidx: float | None = None,
    output_yidx: float | None = None,
    reconstructed_selected_grid_xidx: float | None = None,
    reconstructed_selected_grid_yidx: float | None = None,
    roundtrip_error: float = 0.0,
    footprint_width_units: int | None = None,
    footprint_height_units: int | None = None,
    footprint_is_diagonal: bool | None = None,
    footprint_row: int | None = None,
    footprint_col: int | None = None,
    footprint_direction: int | None = None,
    orientation_class: str | None = None,
    road_overlap_area_m2: float = 0.0,
    road_overlap_ratio: float = 0.0,
    debug_geometry: bool = False,
) -> PlacementRecord:
    grid_kind = GridKind.SUB_SQUARE
    if footprint.is_diagonal:
        grid_kind = GridKind.DIAGONAL
    selected_width_units = footprint.width_cells if footprint_width_units is None else footprint_width_units
    selected_height_units = footprint.height_cells if footprint_height_units is None else footprint_height_units
    selected_is_diagonal = footprint.is_diagonal if footprint_is_diagonal is None else footprint_is_diagonal
    selected_row = footprint.row if footprint_row is None else footprint_row
    selected_col = footprint.col if footprint_col is None else footprint_col
    selected_direction = footprint.direction if footprint_direction is None else footprint_direction
    diagnostics = {
        "iou": round(iou, 6),
        "centroid_shift_m": round(centroid_shift, 6),
        "angle_error_radians": round(angle_error, 6),
        "area_error_ratio": round(area_error, 6),
        "road_overlap_cells": road_overlap_cells,
        "road_overlap_area_m2": round(road_overlap_area_m2, 6),
        "road_overlap_ratio": round(road_overlap_ratio, 6),
        "module_count": len(cells),
        "source_area_m2": round(polygon.area, 6),
        "candidate_limit_reached": candidate_limit_reached,
        "selected_footprint_id": footprint.footprint_id,
        "selected_width_cells": footprint.width_cells,
        "selected_height_cells": footprint.height_cells,
        "selected_width_units": selected_width_units,
        "selected_height_units": selected_height_units,
        "selected_row": selected_row,
        "selected_col": selected_col,
        "selected_direction": selected_direction,
        "selected_is_diagonal": selected_is_diagonal,
        "selected_is_modular": footprint.is_modular,
        "building_type": footprint.building_type,
        "selected_iou": round(iou, 6),
        "selected_centroid_shift_m": round(centroid_shift, 6),
        "selected_area_error_ratio": round(area_error, 6),
        "selected_area_error": round(area_error, 6),
        "selected_road_overlap": road_overlap_cells,
        "output_grid_kind": grid_kind.value,
        "blocked_normal_cells": tuple((cell.xidx, cell.yidx) for cell in sorted(cells)),
        "selected_blocked_normal_cells": tuple((cell.xidx, cell.yidx) for cell in sorted(cells)),
        "selected_all_overlap_cells": tuple((cell.xidx, cell.yidx) for cell in sorted(cells)),
        "debug_geometry_enabled": debug_geometry,
    }
    if selected_footprint_polygon is not None:
        diagnostics["selected_footprint_polygon"] = selected_footprint_polygon
    if output_xidx is not None and output_yidx is not None:
        diagnostics["output_xidx"] = output_xidx
        diagnostics["output_yidx"] = output_yidx
        diagnostics["selected_output_xidx"] = output_xidx
        diagnostics["selected_output_yidx"] = output_yidx
        diagnostics["emitted_output_xidx"] = output_xidx
        diagnostics["emitted_output_yidx"] = output_yidx
    if selected_grid_xidx is not None and selected_grid_yidx is not None:
        diagnostics["selected_grid_xidx"] = selected_grid_xidx
        diagnostics["selected_grid_yidx"] = selected_grid_yidx
    if reconstructed_selected_grid_xidx is not None and reconstructed_selected_grid_yidx is not None:
        diagnostics["reconstructed_selected_grid_xidx"] = reconstructed_selected_grid_xidx
        diagnostics["reconstructed_selected_grid_yidx"] = reconstructed_selected_grid_yidx
        diagnostics["roundtrip_error"] = round(roundtrip_error, 9)
    if selected_footprint_polygon is not None:
        diagnostics["final_reconstructed_bounds"] = tuple(round(value, 6) for value in selected_footprint_polygon.bounds)
    if orientation_class is not None:
        diagnostics["footprint_orientation_class"] = orientation_class
    return PlacementRecord(
        layer=LayerKind.BUILDING,
        grid_kind=grid_kind,
        cells=tuple(sorted(cells)),
        config_name=config_name,
        feature_id=feature_id,
        priority=priority,
        cm_type=footprint.cm_type,
        score=score,
        diagnostics=diagnostics,
    )


def _building_diagnostics(
    descriptor: BuildingDescriptor,
    candidates: tuple[BuildingCandidate, ...],
    candidate_limit_reached: bool,
) -> Mapping[str, Any]:
    return {
        "area_m2": round(descriptor.area, 6),
        "source_area_m2": round(descriptor.area, 6),
        "source_orientation_class": _orientation_class(descriptor.orientation_radians),
        "preferred_orientation": descriptor.preferred_orientation,
        "rectangularity": round(descriptor.rectangularity, 6),
        "elongation": round(descriptor.elongation, 6),
        "mrr_length_m": round(descriptor.mrr_length_m, 6),
        "mrr_width_m": round(descriptor.mrr_width_m, 6),
        "aspect_ratio": round(descriptor.aspect_ratio, 6),
        "orientation_radians": round(descriptor.orientation_radians, 6),
        "nearest_road_distance_m": (
            None if descriptor.nearest_road_distance_m is None else round(descriptor.nearest_road_distance_m, 6)
        ),
        "candidates_scored": len(candidates),
        "available_candidates": sum(1 for candidate in candidates if candidate.collision_cells == 0),
        "candidate_limit_reached": candidate_limit_reached,
        "best_iou": None if not candidates else max(candidate.iou for candidate in candidates),
    }


def _timed_building_diagnostics(started_at: float) -> Mapping[str, Any]:
    return {
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
    }


def _cluster_sort_key(building: _PreparedBuilding) -> tuple[int, int, float, str]:
    nearest_linear = building.descriptor.nearest_road_distance_m
    return (
        building.available_count,
        math.inf if nearest_linear is None else nearest_linear,
        len(building.candidates),
        -building.descriptor.area,
        str(building.feature_key),
    )


def _candidate_sort_key(candidate: BuildingCandidate) -> tuple[float, float, int, tuple[GridCell, ...]]:
    return (-candidate.score, -candidate.iou, candidate.collision_cells, candidate.cells)


def _footprint_area_sort_key(profile: _FootprintProfile) -> tuple[Any, ...]:
    return (
        profile.area_error_ratio,
        profile.aspect_error,
        profile.best_angle_error,
        profile.footprint.is_modular,
        profile.footprint.is_diagonal,
        profile.tie_key,
    )


def _footprint_combined_sort_key(profile: _FootprintProfile) -> tuple[Any, ...]:
    return (
        profile.combined_error,
        profile.area_error_ratio,
        profile.best_angle_error,
        profile.aspect_error,
        profile.footprint.is_modular,
        profile.tie_key,
    )


def _footprint_tie_key(footprint: BuildingFootprint) -> tuple[Any, ...]:
    return (
        footprint.is_modular,
        footprint.is_diagonal,
        footprint.row,
        footprint.col,
        footprint.footprint_id,
    )


def _candidate_geometry_key(
    footprint: BuildingFootprint,
    geometry: _CandidateGeometry,
) -> tuple[str, tuple[GridCell, ...], float | None, float | None]:
    return (
        footprint.footprint_id,
        geometry.cells,
        None if geometry.output_xidx is None else round(geometry.output_xidx, 6),
        None if geometry.output_yidx is None else round(geometry.output_yidx, 6),
    )


def _shift_offsets(local_shift_cells: int) -> tuple[tuple[int, int], ...]:
    shift = max(1, local_shift_cells * 2)
    offsets = [(0, 0)]
    for x_shift in range(-shift, shift + 1):
        for y_shift in range(-shift, shift + 1):
            if x_shift == 0 and y_shift == 0:
                continue
            offsets.append((x_shift, y_shift))
    return tuple(sorted(offsets, key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[0]), abs(item[1]), item[0], item[1])))


def _placement_shift_offsets(steps: int) -> tuple[tuple[int, int], ...]:
    offsets = [(0, 0)]
    for x_shift in range(-steps, steps + 1):
        for y_shift in range(-steps, steps + 1):
            if x_shift == 0 and y_shift == 0:
                continue
            offsets.append((x_shift, y_shift))
    return tuple(
        sorted(
            offsets,
            key=lambda item: (
                abs(item[0]) + abs(item[1]),
                abs(item[0]),
                abs(item[1]),
                item[0],
                item[1],
            ),
        )
    )


def _allow_modular_footprints(
    polygon: Polygon,
    footprints: tuple[BuildingFootprint, ...],
    *,
    cell_size_m: float,
    threshold: float,
) -> bool:
    non_modular_areas = [
        _legacy_footprint_area_m2(footprint, cell_size_m)
        for footprint in footprints
        if not footprint.is_modular
    ]
    if not non_modular_areas:
        return True
    return polygon.area > max(non_modular_areas) * threshold


def _footprint_area_m2(footprint: BuildingFootprint, cell_size_m: float) -> float:
    half_cell_area = (cell_size_m / 2.0) ** 2
    diagonal_factor = 2.0 if footprint.is_diagonal else 1.0
    return footprint.area_cells * half_cell_area * diagonal_factor


def _footprint_dimensions_m(footprint: BuildingFootprint, cell_size_m: float, *, swapped: bool) -> tuple[float, float]:
    width_units = footprint.height_cells if swapped else footprint.width_cells
    height_units = footprint.width_cells if swapped else footprint.height_cells
    scale = cell_size_m / 2.0
    if footprint.is_diagonal:
        scale *= math.sqrt(2.0)
    return width_units * scale, height_units * scale


def _footprint_aspect_error(footprint: BuildingFootprint, source_elongation: float) -> float:
    source = max(source_elongation, 1.0)
    width = max(footprint.width_cells, 1)
    height = max(footprint.height_cells, 1)
    aspect = max(width, height) / max(min(width, height), 1)
    return abs(math.log(max(aspect, 1.0) / source))


def _legacy_footprint_area_m2(footprint: BuildingFootprint, cell_size_m: float) -> float:
    return _footprint_area_m2(footprint, cell_size_m)


def _angle_error(source_angle: float, candidate_angle: float) -> float:
    diff = abs((source_angle - candidate_angle + math.pi / 2) % math.pi - math.pi / 2)
    return min(diff, abs(math.pi / 2 - diff))


def _orientation_class(angle: float) -> str:
    axis_diff = abs((angle + math.pi / 4) % (math.pi / 2) - math.pi / 4)
    diagonal_diff = abs(axis_diff - math.pi / 4)
    return "diagonal" if diagonal_diff < axis_diff else "axis"


def _is_special_building(feature: FeatureRecord) -> bool:
    values = [
        feature.config_name,
        *(str(value) for value in feature.source_tags.values()),
        *(str(value) for value in feature.source_properties.values()),
    ]
    for value in values:
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if normalized in _SPECIAL_BUILDING_TAGS:
            return True
        if any(token in normalized for token in _SPECIAL_BUILDING_TAGS):
            return True
    return False


def _is_residential_like_building(feature: FeatureRecord) -> bool:
    values = [
        feature.config_name,
        *(str(value) for value in feature.source_tags.values()),
        *(str(value) for value in feature.source_properties.values()),
    ]
    residential_tokens = ("house", "houses", "residential", "yes", "detached", "apartments", "terrace")
    for value in values:
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in residential_tokens):
            return True
    return False


def _uses_diagnostic_polygon(metadata: Any) -> bool:
    diagnostics = metadata.get("diagnostics", {}) if isinstance(metadata, Mapping) else {}
    geometry = diagnostics.get("selected_footprint_polygon") if isinstance(diagnostics, Mapping) else None
    return isinstance(geometry, BaseGeometry)


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
