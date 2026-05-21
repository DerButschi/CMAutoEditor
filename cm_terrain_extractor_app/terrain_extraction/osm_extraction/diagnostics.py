from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry.base import BaseGeometry


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, BaseGeometry):
        return value.wkt
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: json_safe(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return str(value)


def write_diagnostics_sidecar(
    output_path: str | Path,
    diagnostics: Any,
    sidecar_path: str | Path | None = None,
) -> Path:
    resolved_output_path = Path(output_path)
    resolved_sidecar_path = (
        Path(sidecar_path)
        if sidecar_path is not None
        else resolved_output_path.with_name(f"{resolved_output_path.stem}_diagnostics.json")
    )
    resolved_sidecar_path.write_text(
        json.dumps(json_safe(diagnostics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_sidecar_path


def summarize_tile_assignment(tile_assignment: Any, *, top_n: int = 10) -> dict[str, Any]:
    diagnostics = dict(getattr(tile_assignment, "diagnostics", {}) or {})
    components = tuple(diagnostics.get("state_component_diagnostics", ()) or ())
    top_components = tuple(
        _component_summary(component)
        for component in sorted(
            components,
            key=lambda item: float(dict(item).get("elapsed_ms", 0.0) or 0.0),
            reverse=True,
        )[:top_n]
    )
    return {
        "state_component_diagnostics": components,
        "top_slowest_components": top_components,
    }


def summarize_building_fitting(fitting: Any, *, top_n: int = 10) -> dict[str, Any]:
    diagnostics = dict(getattr(fitting, "diagnostics", {}) or {})
    by_feature = dict(getattr(fitting, "diagnostics_by_feature", {}) or {})
    building_summaries = tuple(
        _building_summary(feature_key, feature_diagnostics)
        for feature_key, feature_diagnostics in by_feature.items()
    )
    return {
        "building_feature_count": int(diagnostics.get("building_feature_count", len(by_feature))),
        "total_candidates_scored": int(diagnostics.get("total_candidates_scored", 0)),
        "candidates_scored_per_building": dict(diagnostics.get("candidates_scored_per_building", {}) or {}),
        "candidate_limit_reached_count": int(diagnostics.get("candidate_limit_reached_count", 0)),
        "shapely_score_evaluations": int(diagnostics.get("shapely_score_evaluations", 0)),
        "shapely_overlap_evaluations": int(diagnostics.get("shapely_overlap_evaluations", 0)),
        "top_slowest_buildings": tuple(
            sorted(
                building_summaries,
                key=lambda item: float(item.get("elapsed_ms", 0.0) or 0.0),
                reverse=True,
            )[:top_n]
        ),
    }


def summarize_routing(routing: Any, *, top_n: int = 10) -> dict[str, Any]:
    diagnostics = dict(getattr(routing, "diagnostics", {}) or {})
    routes = tuple(getattr(routing, "routes", ()) or ())
    route_summaries = tuple(_route_summary(route) for route in routes)
    return {
        "route_count": int(diagnostics.get("route_count", len(routes))),
        "route_attempts": int(diagnostics.get("route_attempts", 0)),
        "route_retries": int(diagnostics.get("route_retries", 0)),
        "total_a_star_expansions": int(diagnostics.get("total_a_star_expansions", 0)),
        "total_tile_feasible_rejections": int(diagnostics.get("total_tile_feasible_rejections", 0)),
        "top_slowest_routes": tuple(
            sorted(
                route_summaries,
                key=lambda item: float(item.get("elapsed_ms", 0.0) or 0.0),
                reverse=True,
            )[:top_n]
        ),
    }


def timing_table_rows(timings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    total = _timing_seconds(timings.get("total")) or sum(
        seconds
        for stage, value in timings.items()
        if stage != "total"
        for seconds in (_timing_seconds(value),)
        if seconds is not None
    )
    rows = []
    for stage, value in timings.items():
        seconds = _timing_seconds(value)
        if seconds is None:
            continue
        rows.append(
            {
                "stage": stage,
                "elapsed_ms": round(seconds * 1000.0, 3),
                "percent_total": None if total <= 0 else round(seconds / total * 100.0, 1),
            }
        )
    return tuple(rows)


def _component_summary(component: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(component)
    return {
        "component_size": data.get("component_size"),
        "edge_count": data.get("edge_count"),
        "max_degree": data.get("max_degree"),
        "cycle_count": data.get("cycle_count"),
        "candidate_product_log10": data.get("candidate_product_log10"),
        "solver_used": data.get("solver_used"),
        "elapsed_ms": data.get("elapsed_ms"),
        "process": data.get("process"),
        "representative_cells": data.get("representative_cells", ()),
    }


def _building_summary(feature_key: str | int, diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(diagnostics)
    return {
        "feature_key": feature_key,
        "feature_id": data.get("feature_id", feature_key),
        "config_name": data.get("config_name"),
        "elapsed_ms": data.get("elapsed_ms", 0.0),
        "candidate_generation_ms": data.get("candidate_generation_ms", 0.0),
        "placement_ms": data.get("placement_ms", 0.0),
        "candidates_scored": data.get("candidates_scored", 0),
        "footprint_types_considered": data.get("footprint_types_considered", 0),
        "shifted_candidates_generated": data.get("shifted_candidates_generated", 0),
        "expensive_candidates_scored": data.get("expensive_candidates_scored", data.get("candidates_scored", 0)),
        "selected_footprint_id": data.get("selected_footprint_id"),
        "selected_iou": data.get("selected_iou", data.get("iou")),
        "selected_centroid_shift_m": data.get("selected_centroid_shift_m", data.get("centroid_shift_m")),
        "selected_area_error_ratio": data.get("selected_area_error_ratio", data.get("area_error_ratio")),
        "candidate_limit_reached": data.get("candidate_limit_reached", False),
        "failure_reason": data.get("failure_reason"),
    }


def _route_summary(route: Any) -> dict[str, Any]:
    diagnostics = dict(getattr(route, "diagnostics", {}) or {})
    return {
        "edge_id": getattr(route, "edge_id", None),
        "process": getattr(getattr(route, "process", None), "value", getattr(route, "process", None)),
        "config_name": getattr(route, "config_name", None),
        "success": bool(getattr(route, "success", False)),
        "elapsed_ms": diagnostics.get("elapsed_ms", 0.0),
        "attempt_count": diagnostics.get("attempt_count", 0),
        "retry_count": diagnostics.get("retry_count", 0),
        "a_star_expansions": diagnostics.get("a_star_expansions", 0),
        "tile_feasible_rejections": diagnostics.get("tile_feasible_rejections", 0),
        "failure_reason": diagnostics.get("failure_reason"),
    }


def _timing_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("elapsed_s", value.get("seconds"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
