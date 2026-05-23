from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from shapely.geometry import LineString, Point
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


@dataclass(frozen=True, slots=True)
class LinearFeatureAuthority:
    top_level_name: str
    process: ProcessKind
    config_priority: int
    cm_type_index: int | None
    first_matching_tag_index: int | None
    source_feature_length_m: float
    logical_chain_length_m: float
    stable_source_order: int
    source_feature_id: str | int | None

    def as_diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "top_level_name": self.top_level_name,
                "process": self.process.value,
                "config_priority": self.config_priority,
                "cm_type_index": self.cm_type_index,
                "tag_rank": self.first_matching_tag_index,
                "source_length_m": self.source_feature_length_m,
                "logical_chain_length_m": self.logical_chain_length_m,
                "stable_source_order": self.stable_source_order,
                "source_feature_id": self.source_feature_id,
            }
        )


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
    cm_type: CMType | None = None
    linear_authority: LinearFeatureAuthority | None = None

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
class TopologyNode:
    node_id: int
    point: Point
    source_point_count: int = 1
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))


@dataclass(frozen=True, slots=True)
class TopologyEdge:
    edge_id: int
    start_node_id: int
    end_node_id: int
    geometry: LineString
    feature_ids: tuple[str | int | None, ...]
    source_indices: tuple[int, ...]
    config_name: str
    process: ProcessKind
    priority: int
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cm_type: CMType | None = None
    linear_authority: LinearFeatureAuthority | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        object.__setattr__(self, "source_indices", tuple(self.source_indices))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))


@dataclass(frozen=True, slots=True)
class TopologyGraph:
    nodes: tuple[TopologyNode, ...] = ()
    edges: tuple[TopologyEdge, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    def degree(self, node_id: int) -> int:
        return sum(1 for edge in self.edges if edge.start_node_id == node_id or edge.end_node_id == node_id)

    def incident_edges(self, node_id: int) -> tuple[TopologyEdge, ...]:
        return tuple(edge for edge in self.edges if edge.start_node_id == node_id or edge.end_node_id == node_id)

    def nearest_node(self, point: Point, *, tolerance: float = 1e-6) -> TopologyNode | None:
        if not self.nodes:
            return None
        node = min(self.nodes, key=lambda candidate: candidate.point.distance(point))
        if node.point.distance(point) <= tolerance:
            return node
        return None


@dataclass(frozen=True, slots=True)
class RasterSpine:
    topology_edge_id: int
    cells: tuple[GridCell, ...]
    progress: tuple[float, ...]
    distance_m: tuple[float, ...]
    source_length_m: float

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        progress = tuple(float(value) for value in self.progress)
        distance_m = tuple(float(value) for value in self.distance_m)
        if not (len(cells) == len(progress) == len(distance_m)):
            raise ValueError("RasterSpine cells, progress, and distance_m must have equal length")
        if any(progress[index] > progress[index + 1] for index in range(len(progress) - 1)):
            raise ValueError("RasterSpine progress must be ordered")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "distance_m", distance_m)


@dataclass(frozen=True, slots=True)
class RouteRecord:
    edge_id: int
    start_node_id: int
    end_node_id: int
    process: ProcessKind
    config_name: str
    priority: int
    nodes: tuple[GridNode, ...] = ()
    tile_cells: tuple[GridCell, ...] = ()
    raster_spine: RasterSpine | None = None
    success: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    cm_type: CMType | None = None
    linear_authority: LinearFeatureAuthority | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "tile_cells", tuple(self.tile_cells))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))


@dataclass(frozen=True, slots=True)
class NetworkRoutingResult:
    routes: tuple[RouteRecord, ...] = ()
    node_anchors: Mapping[int, GridNode] = field(default_factory=dict)
    anchor_plans: Mapping[int, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    linear_state: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", tuple(self.routes))
        object.__setattr__(self, "node_anchors", _frozen_mapping(self.node_anchors))
        object.__setattr__(self, "anchor_plans", _frozen_mapping(self.anchor_plans))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    @property
    def successful_count(self) -> int:
        return sum(1 for route in self.routes if route.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for route in self.routes if not route.success)


@dataclass(frozen=True, slots=True)
class TileAssignmentResult:
    placements: tuple[PlacementRecord, ...] = ()
    failures: tuple[Mapping[str, Any], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "failures", tuple(_frozen_mapping(failure) for failure in self.failures))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))

    @property
    def success(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class BuildingFittingResult:
    placements: tuple[PlacementRecord, ...] = ()
    failures: tuple[Mapping[str, Any], ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    diagnostics_by_feature: Mapping[str | int, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "placements", tuple(self.placements))
        object.__setattr__(self, "failures", tuple(_frozen_mapping(failure) for failure in self.failures))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))
        object.__setattr__(self, "diagnostics_by_feature", _frozen_mapping(self.diagnostics_by_feature))

    @property
    def placed_count(self) -> int:
        return len(self.placements)

    @property
    def dropped_count(self) -> int:
        return len(self.failures)

    @property
    def success(self) -> bool:
        return not self.failures


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
