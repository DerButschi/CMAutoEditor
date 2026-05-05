from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    timings: Mapping[str, float | None] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timings", MappingProxyType(dict(self.timings)))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))
        object.__setattr__(self, "quality", MappingProxyType(dict(self.quality)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timings": dict(self.timings),
            "counts": dict(self.counts),
            "quality": dict(self.quality),
            "diagnostics": dict(self.diagnostics),
        }


def stats_from_network_routing(routing_result: Any) -> ExtractionStats:
    diagnostics = dict(routing_result.diagnostics)
    return ExtractionStats(
        timings={"network_routing": None},
        counts={
            "network_routes_succeeded": routing_result.successful_count,
            "network_routes_failed": routing_result.failed_count,
        },
        quality={
            "network_route_mean_detour_ratio": diagnostics.get("mean_detour_ratio"),
            "network_route_max_source_line_distance_m": diagnostics.get("max_source_line_distance_m"),
        },
        diagnostics={"mode": "network_routing", **diagnostics},
    )


def stats_from_tile_assignment(tile_assignment_result: Any) -> ExtractionStats:
    diagnostics = dict(tile_assignment_result.diagnostics)
    return ExtractionStats(
        timings={"tile_assignment": None},
        counts={
            "tile_assignments_succeeded": len(tile_assignment_result.placements),
            "tile_assignments_failed": len(tile_assignment_result.failures),
        },
        diagnostics={"mode": "tile_assignment", **diagnostics},
    )


def stats_from_building_fitting(building_fitting_result: Any) -> ExtractionStats:
    diagnostics = dict(building_fitting_result.diagnostics)
    placed = list(building_fitting_result.placements)
    ious = [placement.diagnostics.get("iou") for placement in placed]
    centroid_shifts = [placement.diagnostics.get("centroid_shift_m") for placement in placed]
    numeric_ious = [float(value) for value in ious if value is not None]
    numeric_shifts = [float(value) for value in centroid_shifts if value is not None]
    return ExtractionStats(
        timings={"building_fitting": None},
        counts={
            "buildings_placed": building_fitting_result.placed_count,
            "buildings_dropped": building_fitting_result.dropped_count,
        },
        quality={
            "building_mean_iou": None if not numeric_ious else sum(numeric_ious) / len(numeric_ious),
            "building_max_centroid_shift_m": None if not numeric_shifts else max(numeric_shifts),
        },
        diagnostics={"mode": "building_fitting", **diagnostics},
    )


def stats_from_debug_export(debug_export_result: Any) -> ExtractionStats:
    layer_counts = {f"debug_{name}_features": len(layer) for name, layer in debug_export_result.layers.items()}
    return ExtractionStats(
        timings={"debug_export": None},
        counts={
            "debug_layers": len(debug_export_result.layers),
            "debug_features": sum(layer_counts.values()),
            **layer_counts,
        },
        diagnostics={"mode": "debug_export", **dict(debug_export_result.diagnostics)},
    )
