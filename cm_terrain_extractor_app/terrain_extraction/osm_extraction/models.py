from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from shapely.geometry.base import BaseGeometry


class ProcessKind(StrEnum):
    AREA = "area"
    POINT = "point"
    RANDOM = "random"
    DEFAULT = "default"
    LINEAR = "linear"
    ROAD = "road"
    RAIL = "rail"
    STREAM = "stream"
    FENCE = "fence"
    BUILDING_OUTLINE = "building_outline"

    @classmethod
    def from_legacy(cls, process: str) -> ProcessKind:
        try:
            return _LEGACY_PROCESS_KINDS[process]
        except KeyError as exc:
            raise ValueError(f"Unknown OSM process kind: {process}") from exc


class GridKind(StrEnum):
    NORMAL = "normal"
    SUB_SQUARE = "sub_square"
    DIAGONAL = "diagonal"


class LayerKind(StrEnum):
    GROUND = "ground"
    FOLIAGE = "foliage"
    LINEAR_SURFACE = "linear_surface"
    LINEAR_OBJECT = "linear_object"
    BUILDING = "building"
    POINT_OBJECT = "point_object"
    RESERVED = "reserved"


@dataclass(frozen=True, slots=True)
class CMType:
    menu: str
    cat1: str
    cat2: str | None = None
    direction: str | int | None = None
    tile_id: str | int | None = None
    modifiers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modifiers", _frozen_mapping(self.modifiers))


@dataclass(frozen=True, order=True, slots=True)
class GridCell:
    xidx: int
    yidx: int


@dataclass(frozen=True, order=True, slots=True)
class GridNode:
    xidx: int
    yidx: int


@dataclass(frozen=True, slots=True)
class OccupancyConflict:
    cell: GridCell
    requested_layer: LayerKind
    blocking_layer: LayerKind
    blocking_object_id: str | int
    reason: str


@dataclass(frozen=True, slots=True)
class ConflictDecision:
    allowed: bool
    reasons: tuple[str, ...] = ()
    conflicts: tuple[OccupancyConflict, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    feature_id: str | int | None
    source_index: int
    config_name: str
    process: ProcessKind
    priority: int
    geometry: BaseGeometry
    source_tags: Mapping[str, Any] = field(default_factory=dict)
    source_properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_tags", _frozen_mapping(self.source_tags))
        object.__setattr__(self, "source_properties", _frozen_mapping(self.source_properties))


@dataclass(frozen=True, slots=True)
class PlacementRecord:
    layer: LayerKind
    grid_kind: GridKind
    cells: tuple[GridCell, ...]
    config_name: str
    feature_id: str | int | None
    priority: int
    cm_type: CMType
    score: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    features: tuple[FeatureRecord, ...] = ()
    placements: tuple[PlacementRecord, ...] = ()
    output_rows: tuple[Mapping[str, Any], ...] = ()
    stats: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "output_rows", tuple(self.output_rows))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))


def _frozen_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(mapping, MappingProxyType):
        return mapping
    return MappingProxyType(dict(mapping))


_LEGACY_PROCESS_KINDS = {
    "type_from_tag": ProcessKind.AREA,
    "type_random_area": ProcessKind.AREA,
    "type_random_individual": ProcessKind.RANDOM,
    "type_random_clusters": ProcessKind.RANDOM,
    "single_object_random": ProcessKind.POINT,
    "default_ground": ProcessKind.DEFAULT,
    "default_foliage": ProcessKind.DEFAULT,
    "road_tiles": ProcessKind.ROAD,
    "rail_tiles": ProcessKind.RAIL,
    "stream_tiles": ProcessKind.STREAM,
    "fence_tiles": ProcessKind.FENCE,
    "type_from_linear": ProcessKind.LINEAR,
    "type_from_residential_building_outline": ProcessKind.BUILDING_OUTLINE,
    "type_from_church_outline": ProcessKind.BUILDING_OUTLINE,
    "type_from_barn_outline": ProcessKind.BUILDING_OUTLINE,
    "type_from_barn_outlines": ProcessKind.BUILDING_OUTLINE,
}
