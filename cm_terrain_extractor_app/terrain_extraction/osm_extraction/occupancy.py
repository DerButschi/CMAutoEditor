from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from terrain_extraction.osm_extraction.models import (
    ConflictDecision,
    GridCell,
    LayerKind,
    OccupancyConflict,
    PlacementRecord,
)

_CONFLICTING_LAYERS = {
    LayerKind.GROUND: frozenset(),
    LayerKind.FOLIAGE: frozenset({LayerKind.BUILDING, LayerKind.POINT_OBJECT}),
    LayerKind.LINEAR_SURFACE: frozenset({LayerKind.BUILDING, LayerKind.POINT_OBJECT}),
    LayerKind.LINEAR_OBJECT: frozenset({LayerKind.BUILDING, LayerKind.POINT_OBJECT}),
    LayerKind.BUILDING: frozenset(
        {
            LayerKind.FOLIAGE,
            LayerKind.LINEAR_SURFACE,
            LayerKind.LINEAR_OBJECT,
            LayerKind.BUILDING,
            LayerKind.POINT_OBJECT,
        }
    ),
    LayerKind.POINT_OBJECT: frozenset(
        {
            LayerKind.FOLIAGE,
            LayerKind.LINEAR_SURFACE,
            LayerKind.LINEAR_OBJECT,
            LayerKind.BUILDING,
            LayerKind.POINT_OBJECT,
        }
    ),
    LayerKind.RESERVED: frozenset(set(LayerKind)),
}


@dataclass(slots=True)
class OccupancyModel:
    width: int
    height: int
    occupied: dict[LayerKind, np.ndarray] = field(init=False)
    ranks: dict[LayerKind, np.ndarray] = field(init=False)
    metadata: dict[str | int, dict[str, Any]] = field(default_factory=dict, init=False)
    _object_to_internal_id: dict[str | int, int] = field(default_factory=dict, init=False, repr=False)
    _internal_id_to_object: dict[int, str | int] = field(default_factory=dict, init=False, repr=False)
    _next_internal_id: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("OccupancyModel dimensions must be positive")
        shape = (self.width, self.height)
        self.occupied = {layer: np.full(shape, -1, dtype=np.int64) for layer in LayerKind}
        self.ranks = {layer: np.full(shape, np.iinfo(np.int64).max, dtype=np.int64) for layer in LayerKind}

    @classmethod
    def from_grid_index(cls, grid_index: Any) -> OccupancyModel:
        return cls(width=grid_index.width, height=grid_index.height)

    def reserve(
        self,
        cells: Iterable[GridCell],
        *,
        object_id: str | int,
        metadata: Mapping[str, Any] | None = None,
        priority: int = 0,
    ) -> ConflictDecision:
        placement = _reserved_placement(tuple(cells), priority)
        return self.place(placement, object_id=object_id, metadata=metadata)

    def can_place(self, placement: PlacementRecord, *, allow_replace: bool = False) -> ConflictDecision:
        conflicts = []
        reasons = []
        for cell in placement.cells:
            if not self._contains(cell):
                reasons.append(f"cell ({cell.xidx}, {cell.yidx}) is outside the grid")
                continue
            conflicts.extend(self._cell_conflicts(cell, placement.layer, placement.priority, allow_replace))

        if conflicts:
            reasons.extend(conflict.reason for conflict in conflicts)
        return ConflictDecision(allowed=not reasons and not conflicts, reasons=tuple(reasons), conflicts=tuple(conflicts))

    def placeable_cells(
        self,
        cells: Iterable[GridCell],
        *,
        layer: LayerKind,
        priority: int,
        allow_replace: bool = False,
    ) -> tuple[GridCell, ...]:
        return tuple(
            cell
            for cell in cells
            if self._contains(cell) and not self._cell_conflicts(cell, layer, priority, allow_replace)
        )

    def place(
        self,
        placement: PlacementRecord,
        *,
        object_id: str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_replace: bool = False,
    ) -> ConflictDecision:
        if object_id is None:
            object_id = placement.feature_id if placement.feature_id is not None else self._next_internal_id
        decision = self.can_place(placement, allow_replace=allow_replace)
        if not decision.allowed:
            if any("outside the grid" in reason for reason in decision.reasons):
                raise ValueError("; ".join(decision.reasons))
            return decision

        if allow_replace:
            self._release_replaceable_conflicts(placement)

        internal_id = self._ensure_internal_id(object_id)
        for cell in placement.cells:
            self.occupied[placement.layer][cell.xidx, cell.yidx] = internal_id
            self.ranks[placement.layer][cell.xidx, cell.yidx] = placement.priority
        self.metadata[object_id] = self._metadata_for(placement, metadata)
        return decision

    def place_prechecked(
        self,
        placement: PlacementRecord,
        *,
        object_id: str | int | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_replace: bool = False,
        replaced_object_ids: set[str | int] | None = None,
    ) -> ConflictDecision:
        if object_id is None:
            object_id = placement.feature_id if placement.feature_id is not None else self._next_internal_id
        if allow_replace:
            replaced = self._release_replaceable_conflicts(placement)
            if replaced_object_ids is not None:
                replaced_object_ids.update(replaced)

        internal_id = self._ensure_internal_id(object_id)
        for cell in placement.cells:
            self.occupied[placement.layer][cell.xidx, cell.yidx] = internal_id
            self.ranks[placement.layer][cell.xidx, cell.yidx] = placement.priority
        self.metadata[object_id] = self._metadata_for(placement, metadata)
        return ConflictDecision(allowed=True)

    def release(self, object_id: str | int) -> bool:
        internal_id = self._object_to_internal_id.get(object_id)
        if internal_id is None:
            return False

        for layer in LayerKind:
            mask = self.occupied[layer] == internal_id
            self.occupied[layer][mask] = -1
            self.ranks[layer][mask] = np.iinfo(np.int64).max

        del self._object_to_internal_id[object_id]
        del self._internal_id_to_object[internal_id]
        self.metadata.pop(object_id, None)
        return True

    def is_blocked(self, cell: GridCell, layer: LayerKind) -> bool:
        if not self._contains(cell):
            return True
        return bool(self._cell_conflicts(cell, layer, priority=np.iinfo(np.int64).max, allow_replace=False))

    def object_id_at(self, layer: LayerKind, cell: GridCell) -> str | int | None:
        if not self._contains(cell):
            return None
        internal_id = int(self.occupied[layer][cell.xidx, cell.yidx])
        if internal_id < 0:
            return None
        return self._internal_id_to_object[internal_id]

    def rank_at(self, layer: LayerKind, cell: GridCell) -> int | None:
        if not self._contains(cell) or self.occupied[layer][cell.xidx, cell.yidx] < 0:
            return None
        return int(self.ranks[layer][cell.xidx, cell.yidx])

    def _cell_conflicts(
        self,
        cell: GridCell,
        requested_layer: LayerKind,
        priority: int,
        allow_replace: bool,
    ) -> list[OccupancyConflict]:
        conflicts = []
        layers_to_check = {requested_layer, LayerKind.RESERVED, *_CONFLICTING_LAYERS[requested_layer]}
        for blocking_layer in layers_to_check:
            internal_id = int(self.occupied[blocking_layer][cell.xidx, cell.yidx])
            if internal_id < 0:
                continue
            if self._can_replace(blocking_layer, requested_layer, cell, priority, allow_replace):
                continue
            object_id = self._internal_id_to_object[internal_id]
            reason = (
                f"cell ({cell.xidx}, {cell.yidx}) is occupied by {blocking_layer.value} "
                f"object {object_id!r} and blocks {requested_layer.value}"
            )
            conflicts.append(
                OccupancyConflict(
                    cell=cell,
                    requested_layer=requested_layer,
                    blocking_layer=blocking_layer,
                    blocking_object_id=object_id,
                    reason=reason,
                )
            )
        return conflicts

    def _can_replace(
        self,
        blocking_layer: LayerKind,
        requested_layer: LayerKind,
        cell: GridCell,
        priority: int,
        allow_replace: bool,
    ) -> bool:
        if not allow_replace or blocking_layer is not requested_layer:
            return False
        return priority < self.ranks[blocking_layer][cell.xidx, cell.yidx]

    def _release_replaceable_conflicts(self, placement: PlacementRecord) -> set[str | int]:
        object_ids = set()
        for cell in placement.cells:
            internal_id = int(self.occupied[placement.layer][cell.xidx, cell.yidx])
            if internal_id >= 0 and placement.priority < self.ranks[placement.layer][cell.xidx, cell.yidx]:
                object_ids.add(self._internal_id_to_object[internal_id])
        for object_id in object_ids:
            self.release(object_id)
        return object_ids

    def _ensure_internal_id(self, object_id: str | int) -> int:
        if object_id in self._object_to_internal_id:
            return self._object_to_internal_id[object_id]
        internal_id = self._next_internal_id
        self._next_internal_id += 1
        self._object_to_internal_id[object_id] = internal_id
        self._internal_id_to_object[internal_id] = object_id
        return internal_id

    def _contains(self, cell: GridCell) -> bool:
        return 0 <= cell.xidx < self.width and 0 <= cell.yidx < self.height

    def _metadata_for(
        self,
        placement: PlacementRecord,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        result = {
            "layer": placement.layer.value,
            "grid_kind": placement.grid_kind.value,
            "cells": tuple((cell.xidx, cell.yidx) for cell in placement.cells),
            "config_name": placement.config_name,
            "feature_id": placement.feature_id,
            "priority": placement.priority,
            "diagnostics": dict(placement.diagnostics),
        }
        if metadata is not None:
            result.update(dict(metadata))
        return result


def _reserved_placement(cells: tuple[GridCell, ...], priority: int) -> PlacementRecord:
    from terrain_extraction.osm_extraction.models import CMType, GridKind

    return PlacementRecord(
        layer=LayerKind.RESERVED,
        grid_kind=GridKind.NORMAL,
        cells=cells,
        config_name="reserved",
        feature_id=None,
        priority=priority,
        cm_type=CMType(menu="Reserved", cat1="Reserved"),
        score=1.0,
    )
