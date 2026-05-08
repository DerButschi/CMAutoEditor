from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from terrain_extraction.osm_extraction.config_schema import (
    ConfigEntry,
    ExtractionConfig,
    TagSelector,
)
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import (
    CMType,
    FeatureRecord,
    GridCell,
    GridKind,
    LayerKind,
    PlacementRecord,
    ProcessKind,
)
from terrain_extraction.osm_extraction.occupancy import OccupancyModel


@dataclass(slots=True)
class AreaRasterizer:
    grid_index: GridIndex
    occupancy: OccupancyModel
    rng: np.random.Generator
    coverage_threshold: float = 0.5
    cluster_cell_span: int = 3

    def rasterize(
        self,
        features: Iterable[FeatureRecord],
        config: ExtractionConfig,
    ) -> tuple[PlacementRecord, ...]:
        accepted: dict[str | int, PlacementRecord] = {}
        for feature in sorted(features, key=_feature_order):
            entry = config.entry_by_name(feature.config_name)
            if feature.process is ProcessKind.POINT:
                self._place_point(feature, entry, accepted)
            elif feature.process in {ProcessKind.AREA, ProcessKind.RANDOM}:
                self._place_area(feature, entry, accepted)

        self._emit_defaults(config, accepted)
        return tuple(accepted[object_id] for object_id in accepted)

    def _place_area(
        self,
        feature: FeatureRecord,
        entry: ConfigEntry,
        accepted: dict[str | int, PlacementRecord],
    ) -> None:
        geometry = self._geometry_for_entry(feature.geometry, entry)
        if geometry.is_empty:
            return

        cells, diagnostics = self._candidate_cells(geometry)
        cells = self._apply_cell_modifiers(cells, entry)
        if not cells:
            return

        if self._uses_individual_random(entry):
            for index, cell in enumerate(cells):
                cm_type = self.choose_weighted_cm_type(entry.cm_types)
                if cm_type is None:
                    continue
                object_id = f"{feature.feature_id}:{cell.xidx}:{cell.yidx}:{index}"
                self._accept_cells(feature, entry, cm_type, (cell,), accepted, object_id, diagnostics)
            return

        if self._uses_feature_random(entry):
            cm_type = self.choose_weighted_cm_type(entry.cm_types)
            if cm_type is None:
                return
            self._accept_cells(feature, entry, cm_type, tuple(cells), accepted, feature.feature_id, diagnostics)
            return

        if self._uses_clustered_random(entry):
            cluster_types = self._cluster_types(cells, entry.cm_types)
            for index, cell in enumerate(cells):
                cm_type = cluster_types[(cell.xidx // self.cluster_cell_span, cell.yidx // self.cluster_cell_span)]
                if cm_type is None:
                    continue
                object_id = f"{feature.feature_id}:cluster:{cell.xidx}:{cell.yidx}:{index}"
                self._accept_cells(feature, entry, cm_type, (cell,), accepted, object_id, diagnostics)
            return

        cm_type = self._matched_or_first_cm_type(entry, feature.source_tags)
        if cm_type is None:
            return
        self._accept_cells(feature, entry, cm_type, tuple(cells), accepted, feature.feature_id, diagnostics)

    def _place_point(
        self,
        feature: FeatureRecord,
        entry: ConfigEntry,
        accepted: dict[str | int, PlacementRecord],
    ) -> None:
        point = feature.geometry if isinstance(feature.geometry, Point) else feature.geometry.centroid
        cell = self.grid_index.projected_to_cell(point.x, point.y)
        cm_type = self.choose_weighted_cm_type(entry.cm_types)
        if cm_type is None:
            return
        self._accept_cells(feature, entry, cm_type, (cell,), accepted, feature.feature_id, {"point": (point.x, point.y)})

    def _emit_defaults(
        self,
        config: ExtractionConfig,
        accepted: dict[str | int, PlacementRecord],
    ) -> None:
        for entry in config.entries:
            if ProcessKind.DEFAULT not in entry.processes:
                continue
            layer = self._default_layer(entry)
            for xidx in range(self.grid_index.width):
                for yidx in range(self.grid_index.height):
                    cell = GridCell(xidx, yidx)
                    if self.occupancy.object_id_at(layer, cell) is not None:
                        continue
                    cm_type = self.choose_weighted_cm_type(entry.cm_types)
                    if cm_type is None:
                        continue
                    object_id = f"default:{entry.name}:{xidx}:{yidx}"
                    placement = PlacementRecord(
                        layer=layer,
                        grid_kind=GridKind.NORMAL,
                        cells=(cell,),
                        config_name=entry.name,
                        feature_id=None,
                        priority=entry.priority,
                        cm_type=cm_type,
                        score=1.0,
                        diagnostics={"default": True},
                    )
                    decision = self.occupancy.place(placement, object_id=object_id)
                    if decision.allowed:
                        accepted[object_id] = placement

    def _candidate_cells(self, geometry: BaseGeometry) -> tuple[list[GridCell], dict[str, Any]]:
        window = self._geometry_window(geometry)
        if window is None:
            return [], {"candidate_window": None, "candidate_cells": 0}

        min_xidx, min_yidx, max_xidx, max_yidx = window
        cells = []
        cell_area = self.grid_index.cell_size_m * self.grid_index.cell_size_m
        for xidx in range(min_xidx, max_xidx + 1):
            for yidx in range(min_yidx, max_yidx + 1):
                cell = GridCell(xidx, yidx)
                cell_polygon = self.grid_index.cell_polygon(cell)
                if cell_polygon.within(geometry) or cell_polygon.intersection(geometry).area / cell_area > self.coverage_threshold:
                    cells.append(cell)

        return cells, {"candidate_window": window, "candidate_cells": (max_xidx - min_xidx + 1) * (max_yidx - min_yidx + 1)}

    def _geometry_window(self, geometry: BaseGeometry) -> tuple[int, int, int, int] | None:
        min_x, min_y, max_x, max_y = geometry.bounds
        local_corners = [
            self.grid_index.local_from_projected(x, y)
            for x, y in ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))
        ]
        min_cell = GridCell(
            int(np.floor(min(local_x for local_x, _local_y in local_corners) / self.grid_index.cell_size_m + 1e-9)),
            int(np.floor(min(local_y for _local_x, local_y in local_corners) / self.grid_index.cell_size_m + 1e-9)),
        )
        max_cell = GridCell(
            int(np.floor(max(local_x for local_x, _local_y in local_corners) / self.grid_index.cell_size_m + 1e-9)),
            int(np.floor(max(local_y for _local_x, local_y in local_corners) / self.grid_index.cell_size_m + 1e-9)),
        )
        return self.grid_index.clipped_cell_window(
            min(min_cell.xidx, max_cell.xidx),
            min(min_cell.yidx, max_cell.yidx),
            max(min_cell.xidx, max_cell.xidx),
            max(min_cell.yidx, max_cell.yidx),
        )

    def _accept_cells(
        self,
        feature: FeatureRecord,
        entry: ConfigEntry,
        cm_type: CMType,
        cells: tuple[GridCell, ...],
        accepted: dict[str | int, PlacementRecord],
        object_id: str | int | None,
        diagnostics: dict[str, Any],
    ) -> None:
        layer = self._layer_for(feature.process, entry, cm_type)
        placeable_cells = self.occupancy.placeable_cells(
            cells,
            layer=layer,
            priority=entry.priority,
            allow_replace=True,
        )
        if not placeable_cells:
            return

        resolved_object_id = (
            object_id
            if object_id is not None
            else f"{entry.name}:{feature.source_index}:{placeable_cells[0].xidx}:{placeable_cells[0].yidx}"
        )
        placement = self._placement(feature, entry, cm_type, layer, placeable_cells, diagnostics)
        replaced_object_ids: set[str | int] = set()
        decision = self.occupancy.place_prechecked(
            placement,
            object_id=resolved_object_id,
            allow_replace=True,
            replaced_object_ids=replaced_object_ids,
        )
        if decision.allowed:
            accepted[resolved_object_id] = placement
            for replaced_object_id in replaced_object_ids:
                accepted.pop(replaced_object_id, None)

    def _placement(
        self,
        feature: FeatureRecord,
        entry: ConfigEntry,
        cm_type: CMType,
        layer: LayerKind,
        cells: tuple[GridCell, ...],
        diagnostics: dict[str, Any],
    ) -> PlacementRecord:
        return PlacementRecord(
            layer=layer,
            grid_kind=GridKind.NORMAL,
            cells=cells,
            config_name=entry.name,
            feature_id=feature.feature_id,
            priority=entry.priority,
            cm_type=cm_type,
            score=1.0,
            diagnostics=diagnostics,
        )

    def _cluster_types(
        self,
        cells: Iterable[GridCell],
        cm_types: tuple[CMType, ...],
    ) -> dict[tuple[int, int], CMType | None]:
        cluster_keys = sorted({(cell.xidx // self.cluster_cell_span, cell.yidx // self.cluster_cell_span) for cell in cells})
        return {cluster_key: self.choose_weighted_cm_type(cm_types) for cluster_key in cluster_keys}

    def _drop_replaced_placements(self, accepted: dict[str | int, PlacementRecord]) -> None:
        active_object_ids = set(self.occupancy.metadata)
        for object_id in tuple(accepted):
            if object_id not in active_object_ids:
                del accepted[object_id]

    def choose_weighted_cm_type(self, cm_types: tuple[CMType, ...]) -> CMType | None:
        if not cm_types:
            return None
        weights = np.array([float(cm_type.modifiers.get("weight", 1.0)) for cm_type in cm_types], dtype=float)
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("CM type weights must include at least one non-negative positive weight")
        probabilities = weights / weights.sum()
        selected = cm_types[int(self.rng.choice(len(cm_types), p=probabilities))]
        if selected.modifiers.get("dummy") is True:
            return None
        return selected

    def _geometry_for_entry(self, geometry: BaseGeometry, entry: ConfigEntry) -> BaseGeometry:
        if entry.modifiers.get("is_core") and "border_size" in entry.modifiers:
            return geometry.buffer(-float(entry.modifiers["border_size"]) * self.grid_index.cell_size_m)
        return geometry

    @staticmethod
    def _apply_cell_modifiers(cells: list[GridCell], entry: ConfigEntry) -> list[GridCell]:
        if not cells:
            return cells
        stride_x = entry.modifiers.get("stride_x")
        stride_y = entry.modifiers.get("stride_y")
        if stride_x is None and stride_y is None:
            return cells

        min_xidx = min(cell.xidx for cell in cells)
        min_yidx = min(cell.yidx for cell in cells)
        stride_x = 1 if stride_x is None else max(1, int(stride_x))
        stride_y = 1 if stride_y is None else max(1, int(stride_y))
        return [
            cell
            for cell in cells
            if (cell.xidx - min_xidx) % stride_x == 0 and (cell.yidx - min_yidx) % stride_y == 0
        ]

    def _matched_or_first_cm_type(self, entry: ConfigEntry, tags: dict[str, Any] | Any) -> CMType | None:
        for index, raw_cm_type in enumerate(entry.raw_cm_types):
            selector = TagSelector.from_raw(raw_cm_type.get("tags", ()), field_name=f"{entry.name}.cm_types.tags")
            if selector.matches(tags):
                cm_type = entry.cm_types[index]
                return None if cm_type.modifiers.get("dummy") is True else cm_type
        if not entry.cm_types:
            return None
        cm_type = entry.cm_types[0]
        return None if cm_type.modifiers.get("dummy") is True else cm_type

    def _layer_for(self, process: ProcessKind, entry: ConfigEntry, cm_type: CMType) -> LayerKind:
        if process is ProcessKind.POINT:
            return LayerKind.POINT_OBJECT
        if ProcessKind.DEFAULT in entry.processes:
            return self._default_layer(entry)
        menu = cm_type.menu.lower()
        if menu.startswith("foliage") or menu.startswith("brush"):
            return LayerKind.FOLIAGE
        return LayerKind.GROUND

    def _default_layer(self, entry: ConfigEntry) -> LayerKind:
        if "default_foliage" in entry.legacy_processes:
            return LayerKind.FOLIAGE
        return LayerKind.GROUND

    @staticmethod
    def _uses_individual_random(entry: ConfigEntry) -> bool:
        return "type_random_individual" in entry.legacy_processes

    @staticmethod
    def _uses_feature_random(entry: ConfigEntry) -> bool:
        return "type_random_area" in entry.legacy_processes

    @staticmethod
    def _uses_clustered_random(entry: ConfigEntry) -> bool:
        return "type_random_clusters" in entry.legacy_processes


def _feature_order(feature: FeatureRecord) -> tuple[int, int, str]:
    return feature.priority, feature.source_index, str(feature.feature_id)
