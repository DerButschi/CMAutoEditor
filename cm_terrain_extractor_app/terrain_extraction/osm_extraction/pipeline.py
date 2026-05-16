from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from terrain_extraction.osm_extraction.config_schema import ExtractionConfig, RoadValidationMode
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import (
    CMType,
    ExtractionResult,
    GridCell,
    GridKind,
    LayerKind,
    PlacementRecord,
    ProcessKind,
)
from terrain_extraction.osm_extraction.occupancy import OccupancyModel
from terrain_extraction.osm_extraction.stats import (
    ExtractionStats,
    stats_from_building_fitting,
    stats_from_debug_export,
    stats_from_network_routing,
    stats_from_tile_assignment,
)

ProgressCallback = Callable[[str, float, str | None], None]


class LegacyProcessor(Protocol):
    def preprocess_osm_data(self, osm_data: object) -> None: ...

    def run_processors(self) -> None: ...

    def post_process(self) -> None: ...

    def get_output(self) -> Any: ...


class TileAssignmentError(ValueError):
    def __init__(self, failures: tuple[Mapping[str, Any], ...]) -> None:
        self.failures = failures
        super().__init__(_tile_assignment_error_message(failures))


def noop_progress(stage: str, value: float, message: str | None = None) -> None:
    return None


@dataclass(slots=True)
class ExtractionContext:
    profile: str
    bbox: Any
    config_path: str
    seed: int | None
    rng: np.random.Generator
    progress: ProgressCallback = noop_progress
    feature_flags: Mapping[str, bool] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        profile: str,
        bbox: Any,
        config_path: str,
        seed: int | None = None,
        progress: ProgressCallback | None = None,
        feature_flags: Mapping[str, bool] | None = None,
    ) -> ExtractionContext:
        return cls(
            profile=profile,
            bbox=bbox,
            config_path=config_path,
            seed=seed,
            rng=np.random.default_rng(seed),
            progress=progress or noop_progress,
            feature_flags={} if feature_flags is None else dict(feature_flags),
        )


@dataclass(slots=True)
class ExtractionPipeline:
    context: ExtractionContext

    def run(
        self,
        *,
        features: tuple[Any, ...],
        config: ExtractionConfig,
        grid_index: GridIndex,
        bounds: tuple[int | float, int | float, int | float, int | float],
        clip_geometry: Any | None = None,
        occupancy: OccupancyModel | None = None,
        linear_catalog_provider: Callable[[tuple[Any, ...]], Mapping[Any, Any]] | None = None,
        building_catalog_provider: Callable[[tuple[Any, ...]], Mapping[str, Any]] | None = None,
        road_validation_mode: RoadValidationMode | None = None,
    ) -> ExtractionResult:
        occupancy_model = occupancy or OccupancyModel.from_grid_index(grid_index)
        placements: list[PlacementRecord] = []
        diagnostics: dict[str, Any] = {"occupancy": occupancy_model, "catalog_gaps": ()}
        resolved_road_validation_mode = road_validation_mode or getattr(config, "road_validation_mode", "warn")

        area_features = tuple(
            feature
            for feature in features
            if feature.process in {ProcessKind.AREA, ProcessKind.RANDOM, ProcessKind.POINT}
        )
        linear_features = tuple(
            feature
            for feature in features
            if feature.process in {ProcessKind.ROAD, ProcessKind.RAIL, ProcessKind.STREAM, ProcessKind.FENCE}
        )
        building_features = tuple(feature for feature in features if feature.process is ProcessKind.BUILDING_OUTLINE)

        if linear_features:
            linear_catalogs = (
                self._tile_catalogs_for(linear_features, config)
                if linear_catalog_provider is None
                else linear_catalog_provider(linear_features)
            )
            diagnostics["catalog_gaps"] = _catalog_gap_diagnostics(linear_catalogs)
            topology_result = self.run_network_topology(
                features=linear_features,
                clip_geometry=clip_geometry,
            )
            topology = topology_result.diagnostics["network_topology"]
            diagnostics["network_topology"] = topology
            routing_result = self.run_network_router(
                topology=topology,
                grid_index=grid_index,
                occupancy=occupancy_model,
                catalogs=linear_catalogs,
            )
            routing = routing_result.diagnostics["network_routes"]
            diagnostics["network_routes"] = routing
            tile_result = self.run_tile_assignment(
                routes=routing.routes,
                catalogs=linear_catalogs,
                linear_state=routing.linear_state,
            )
            tile_assignment = tile_result.diagnostics["tile_assignment"]
            tile_failures = _normalize_tile_assignment_failures(tile_assignment)
            diagnostics["tile_assignment"] = tile_assignment
            diagnostics["tile_assignment_failures"] = tile_failures
            if tile_failures and resolved_road_validation_mode == "strict":
                raise TileAssignmentError(tile_failures)
            tile_placements, suppressed_tile_placements = _suppress_failed_tile_placements(
                tile_result.placements,
                tile_failures,
            )
            diagnostics["tile_assignment_suppressed_placements"] = suppressed_tile_placements
            placements.extend(tile_placements)
            _reserve_output_placements(occupancy_model, tile_placements)

        linear_dependent_placements = _linear_feature_placements(
            config=config,
            source_placements=tuple(placements),
            rng=self.context.rng,
        )
        placements.extend(linear_dependent_placements)
        _reserve_output_placements(occupancy_model, linear_dependent_placements)

        if building_features:
            building_catalogs = (
                self._building_catalogs_for(building_features, config)
                if building_catalog_provider is None
                else building_catalog_provider(building_features)
            )
            building_result = self.run_building_fitter(
                features=building_features,
                catalogs=building_catalogs,
                grid_index=grid_index,
                occupancy=occupancy_model,
            )
            placements.extend(building_result.placements)
            diagnostics["building_fitting"] = building_result.diagnostics.get("building_fitting")

        area_result = self.run_area_rasterizer(
            features=area_features,
            config=config,
            grid_index=grid_index,
            occupancy=occupancy_model,
        )
        placements.extend(area_result.placements)

        resolved_placements = _resolve_output_layer_conflicts(tuple(placements))
        output_result = self.run_output_rows(
            placements=resolved_placements,
            bounds=bounds,
            road_validation_mode=resolved_road_validation_mode,
        )
        diagnostics.update(output_result.diagnostics)
        if "network_topology" in diagnostics and "road_validation" in diagnostics:
            diagnostics["source_aware_road_validation"] = _source_aware_road_validation_diagnostics(
                road_validation=diagnostics["road_validation"],
                topology=diagnostics["network_topology"],
                grid_index=grid_index,
            )
        return ExtractionResult(
            features=features,
            placements=resolved_placements,
            output_rows=output_result.output_rows,
            stats=output_result.stats,
            diagnostics=diagnostics,
        )

    def run_area_rasterizer(
        self,
        *,
        features: tuple[Any, ...],
        config: ExtractionConfig,
        grid_index: GridIndex,
        occupancy: OccupancyModel | None = None,
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.area_rasterizer import AreaRasterizer

        occupancy_model = occupancy or OccupancyModel.from_grid_index(grid_index)
        placements = AreaRasterizer(grid_index, occupancy_model, self.context.rng).rasterize(features, config)
        self.context.progress("area_rasterization", 1.0, "Area rasterization complete")
        return ExtractionResult(
            features=features,
            placements=placements,
            stats=ExtractionStats(
                timings={"area_rasterization": None},
                counts={"area_rasterizer_placements": len(placements)},
                diagnostics={"mode": "area_rasterizer"},
            ),
        )

    def run_network_topology(
        self,
        *,
        features: tuple[Any, ...],
        clip_geometry: Any | None = None,
        snap_tolerance_m: float = 1.0,
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

        topology = NetworkTopologyBuilder(
            clip_geometry=clip_geometry,
            snap_tolerance_m=snap_tolerance_m,
        ).build(features)
        self.context.progress("network_noding", 1.0, "Network topology complete")
        return ExtractionResult(
            features=features,
            stats=ExtractionStats(
                timings={"network_noding": None},
                counts={"topology_nodes": len(topology.nodes), "topology_edges": len(topology.edges)},
                quality={"topology_components": _topology_component_count(topology)},
                diagnostics={"mode": "network_topology", **dict(topology.diagnostics)},
            ),
            diagnostics={"network_topology": topology},
        )

    def run_network_router(
        self,
        *,
        topology: Any,
        grid_index: GridIndex,
        occupancy: OccupancyModel | None = None,
        catalogs: Mapping[Any, Any] | None = None,
        corridor_deviation_m: float = 32.0,
        minor_relaxation_m: float = 48.0,
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.network_routing import NetworkRouter

        router = NetworkRouter(
            grid_index=grid_index,
            occupancy=occupancy,
            catalogs=catalogs,
            corridor_deviation_m=corridor_deviation_m,
            minor_relaxation_m=minor_relaxation_m,
        )
        routes = router.route(topology)
        self.context.progress("network_routing", 1.0, "Network routing complete")
        return ExtractionResult(
            stats=stats_from_network_routing(routes),
            diagnostics={"network_router": router, "network_routes": routes},
        )

    def run_tile_assignment(
        self,
        *,
        routes: tuple[Any, ...],
        catalogs: Mapping[Any, Any],
        linear_state: Any = None,
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.tile_assignment import TileAssigner

        assignment = TileAssigner(catalogs, rng=self.context.rng).assign(routes, linear_state=linear_state)
        self.context.progress("tile_assignment", 1.0, "Tile assignment complete")
        return ExtractionResult(
            placements=assignment.placements,
            stats=stats_from_tile_assignment(assignment),
            diagnostics={"tile_assignment": assignment},
        )

    def run_building_fitter(
        self,
        *,
        features: tuple[Any, ...],
        catalogs: Mapping[str, Any],
        grid_index: GridIndex,
        occupancy: OccupancyModel | None = None,
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.building_fitter import BuildingFitter

        fitter = BuildingFitter(grid_index, occupancy=occupancy, rng=self.context.rng)
        fitting = fitter.fit(features, catalogs=catalogs)
        self.context.progress("building_fitting", 1.0, "Building fitting complete")
        return ExtractionResult(
            placements=fitting.placements,
            stats=stats_from_building_fitting(fitting),
            diagnostics={"building_fitting": fitting},
        )

    def run_output_rows(
        self,
        *,
        placements: tuple[Any, ...],
        bounds: tuple[int | float, int | float, int | float, int | float],
        road_validation_mode: RoadValidationMode = "strict",
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.output_rows import (
            OutputRowValidationError,
            append_extent_marker,
            clip_output_rows_to_bounds,
            normalize_output_coordinates,
            placements_to_output_rows,
            validate_output_rows,
        )
        from terrain_extraction.osm_extraction.road_output_validation import (
            validate_road_output_rows,
        )

        internal_rows = placements_to_output_rows(placements, include_internal=True)
        rows_with_extent = append_extent_marker(internal_rows, bounds=bounds, include_internal=True)
        clipped_rows = clip_output_rows_to_bounds(rows_with_extent, bounds=bounds)
        validate_output_rows(clipped_rows, bounds=bounds)
        road_validation = validate_road_output_rows(clipped_rows, profile=self.context.profile)
        road_validation_status = _road_validation_status(road_validation, mode=road_validation_mode)
        if road_validation_mode == "strict" and not road_validation.is_valid:
            raise OutputRowValidationError(road_validation.issue_summary())
        output_rows = normalize_output_coordinates(clipped_rows, bounds=bounds)
        self.context.progress("output_assembly", 1.0, "Layered output rows assembled")
        return ExtractionResult(
            placements=placements,
            output_rows=output_rows,
            stats=ExtractionStats(
                timings={"output_assembly": None},
                counts={"output_rows": len(output_rows)},
                diagnostics={
                    "mode": "layered_output",
                    "road_validation": road_validation.issue_summary(),
                    "road_validation_status": road_validation_status,
                },
            ),
            diagnostics={"road_validation": road_validation, "road_validation_status": road_validation_status},
        )

    def run_debug_export(
        self,
        *,
        features: tuple[Any, ...] = (),
        topology: Any = None,
        routing: Any = None,
        occupancy: Any = None,
        placements: tuple[Any, ...] = (),
        tile_assignment: Any = None,
        output_rows: tuple[Mapping[str, Any], ...] = (),
        grid_index: GridIndex | None = None,
        bounds: tuple[int | float, int | float, int | float, int | float] | None = None,
        stats: ExtractionStats | None = None,
    ) -> ExtractionResult:
        from terrain_extraction.osm_extraction.debug_export import build_debug_layers

        debug_export = build_debug_layers(
            features=features,
            topology=topology,
            routing=routing,
            occupancy=occupancy,
            placements=placements,
            tile_assignment=tile_assignment,
            output_rows=output_rows,
            grid_index=grid_index,
            bounds=bounds,
            stats=stats,
        )
        self.context.progress("debug_export", 1.0, "Debug export complete")
        return ExtractionResult(
            features=features,
            placements=placements,
            output_rows=output_rows,
            stats=stats_from_debug_export(debug_export),
            diagnostics={"debug_export": debug_export},
        )

    def run_legacy(self, processor: LegacyProcessor, osm_data: object) -> ExtractionResult:
        timings: dict[str, float] = {}

        self._run_legacy_stage(
            timings,
            "preprocess_osm_data",
            "Legacy preprocessing complete",
            lambda: processor.preprocess_osm_data(osm_data),
        )
        self._run_legacy_stage(
            timings,
            "run_processors",
            "Legacy processors complete",
            processor.run_processors,
        )
        self._run_legacy_stage(
            timings,
            "post_process",
            "Legacy post-processing complete",
            processor.post_process,
        )
        output_rows: tuple[Mapping[str, Any], ...] = ()

        def assemble_output() -> None:
            nonlocal output_rows
            output_rows = _records_from_output(processor.get_output())

        self._run_legacy_stage(
            timings,
            "output_assembly",
            "Legacy output rows assembled",
            assemble_output,
        )

        return ExtractionResult(
            output_rows=output_rows,
            stats=ExtractionStats(
                timings=timings,
                counts={"output_rows": len(output_rows)},
                diagnostics={"mode": "legacy_wrapper"},
            ),
        )

    def _run_legacy_stage(
        self,
        timings: dict[str, float],
        stage: str,
        message: str,
        action: Callable[[], None],
    ) -> None:
        start = time.perf_counter()
        action()
        timings[stage] = time.perf_counter() - start
        self.context.progress(stage, 1.0, message)

    def _tile_catalogs_for(self, features: tuple[Any, ...], config: ExtractionConfig) -> Mapping[Any, Any]:
        from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

        from profiles.general import fence_tiles, rail_tiles, road_tiles, stream_tiles

        tile_sources = {
            ProcessKind.ROAD: road_tiles,
            ProcessKind.RAIL: rail_tiles,
            ProcessKind.STREAM: stream_tiles,
            ProcessKind.FENCE: fence_tiles,
        }
        processes = {feature.process for feature in features}
        return {
            process: CompiledTileCatalog.from_records(
                tile_sources[process],
                process=process,
                base_cm_type=_base_cm_type_for_process(config, process),
            )
            for process in sorted(processes, key=lambda item: item.value)
        }

    def _building_catalogs_for(self, features: tuple[Any, ...], config: ExtractionConfig) -> Mapping[str, Any]:
        from profiles import get_building_tiles, process_to_building_type

        catalogs = {}
        for feature in features:
            if feature.config_name in catalogs:
                continue
            entry = config.entry_by_name(feature.config_name)
            building_type = None
            for legacy_process in entry.legacy_processes:
                if legacy_process in process_to_building_type:
                    building_type = process_to_building_type[legacy_process]
                    break
            if building_type is not None:
                catalogs[feature.config_name] = get_building_tiles(building_type, self.context.profile)
        return catalogs


def _road_validation_status(report: Any, *, mode: RoadValidationMode) -> dict[str, Any]:
    if mode not in {"strict", "warn"}:
        raise ValueError("road_validation_mode must be 'strict' or 'warn'")
    return {
        "mode": mode,
        "is_valid": bool(report.is_valid),
        "summary": report.issue_summary(),
        "hard_issues": len(report.hard_issues),
    }


def _normalize_tile_assignment_failures(tile_assignment: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(_normalize_tile_assignment_failure(failure) for failure in getattr(tile_assignment, "failures", ()) or ())


def _normalize_tile_assignment_failure(failure: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(failure)
    if "cell" in normalized:
        normalized["cell"] = _cell_tuple(normalized["cell"])
    if "required_directions" in normalized:
        normalized["required_directions"] = tuple(normalized["required_directions"])
    if "route_ids" in normalized:
        normalized["route_ids"] = tuple(normalized["route_ids"])
    return normalized


def _suppress_failed_tile_placements(
    placements: tuple[PlacementRecord, ...],
    failures: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[PlacementRecord, ...], tuple[Mapping[str, Any], ...]]:
    if not failures:
        return placements, ()

    failed_process_cells = {
        (failure.get("process"), failure.get("cell"))
        for failure in failures
        if failure.get("process") is not None and failure.get("cell") is not None
    }
    failed_route_ids = _failed_tile_route_ids(failures)
    kept = []
    suppressed = []
    for placement in placements:
        if _placement_matches_failed_tile_piece(
            placement,
            failed_process_cells=failed_process_cells,
            failed_route_ids=failed_route_ids,
        ):
            suppressed.append(_tile_suppression_diagnostic(placement))
            continue
        kept.append(placement)
    return tuple(kept), tuple(suppressed)


def _failed_tile_route_ids(failures: tuple[Mapping[str, Any], ...]) -> frozenset[Any]:
    route_ids = set()
    for failure in failures:
        if failure.get("route_id") is not None:
            route_ids.add(failure["route_id"])
        route_ids.update(failure.get("route_ids", ()) or ())
    return frozenset(route_ids)


def _placement_matches_failed_tile_piece(
    placement: PlacementRecord,
    *,
    failed_process_cells: set[tuple[Any, Any]],
    failed_route_ids: frozenset[Any],
) -> bool:
    placement_process = placement.diagnostics.get("source_process")
    if placement_process is not None:
        for cell in placement.cells:
            if (placement_process, _cell_tuple(cell)) in failed_process_cells:
                return True

    placement_route_ids = set(placement.diagnostics.get("contributing_route_ids", ()) or ())
    if placement.feature_id is not None:
        placement_route_ids.add(placement.feature_id)
    return bool(placement_route_ids.intersection(failed_route_ids))


def _tile_suppression_diagnostic(placement: PlacementRecord) -> Mapping[str, Any]:
    diagnostic: dict[str, Any] = {
        "process": placement.diagnostics.get("source_process"),
        "feature_id": placement.feature_id,
        "cells": tuple(_cell_tuple(cell) for cell in placement.cells),
        "reason": "tile_assignment_failure",
    }
    contributing_route_ids = placement.diagnostics.get("contributing_route_ids")
    if contributing_route_ids:
        diagnostic["route_ids"] = tuple(contributing_route_ids)
    return diagnostic


def _cell_tuple(cell: Any) -> tuple[int, int]:
    if isinstance(cell, GridCell):
        return cell.xidx, cell.yidx
    if isinstance(cell, tuple) and len(cell) == 2:
        return int(cell[0]), int(cell[1])
    return int(cell.xidx), int(cell.yidx)


def _tile_assignment_error_message(failures: tuple[Mapping[str, Any], ...]) -> str:
    details = "; ".join(_tile_failure_summary(failure) for failure in failures[:5])
    suffix = "" if len(failures) <= 5 else f"; +{len(failures) - 5} more"
    return f"tile assignment failures ({len(failures)}): {details}{suffix}"


def _tile_failure_summary(failure: Mapping[str, Any]) -> str:
    route_ids = failure.get("route_ids")
    route_label = f" route_ids={tuple(route_ids)}" if route_ids else ""
    if failure.get("route_id") is not None:
        route_label = f" route_id={failure['route_id']}"
    return (
        f"process={failure.get('process')}{route_label} cell={failure.get('cell')} "
        f"required_directions={failure.get('required_directions')} reason={failure.get('failure_reason')}"
    )


def _records_from_output(output: Any) -> tuple[Mapping[str, Any], ...]:
    if hasattr(output, "to_dict"):
        return tuple(output.to_dict("records"))
    if output is None:
        return ()
    return tuple(output)


def _topology_component_count(topology: Any) -> int:
    if not topology.nodes:
        return 0
    parent = {node.node_id: node.node_id for node in topology.nodes}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for edge in topology.edges:
        union(edge.start_node_id, edge.end_node_id)
    return len({find(node.node_id) for node in topology.nodes})


def _catalog_gap_diagnostics(catalogs: Mapping[Any, Any]) -> tuple[Mapping[str, Any], ...]:
    required_direction_sets = (
        ("N",),
        ("E",),
        ("S",),
        ("W",),
        ("N", "S"),
        ("E", "W"),
        ("N", "E"),
        ("E", "S"),
        ("S", "W"),
        ("N", "W"),
        ("N", "E", "S"),
        ("E", "S", "W"),
        ("N", "S", "W"),
        ("N", "E", "W"),
        ("N", "E", "S", "W"),
    )
    diagnostics: list[Mapping[str, Any]] = []
    for catalog in catalogs.values():
        if hasattr(catalog, "catalog_gap_diagnostics"):
            diagnostics.extend(catalog.catalog_gap_diagnostics(required_direction_sets))
    return tuple(diagnostics)


def _base_cm_type_for_process(config: ExtractionConfig, process: ProcessKind) -> CMType | None:
    for entry in config.entries:
        if process in entry.processes and entry.cm_types:
            return entry.cm_types[0]
    return None


def _linear_feature_placements(
    *,
    config: ExtractionConfig,
    source_placements: tuple[PlacementRecord, ...],
    rng: np.random.Generator,
) -> tuple[PlacementRecord, ...]:
    linear_placements = []
    cells_by_name: dict[str, set[GridCell]] = {}
    for placement in source_placements:
        for cell in placement.cells:
            cells_by_name.setdefault(placement.config_name, set()).add(cell)

    for entry in config.entries:
        if ProcessKind.LINEAR not in entry.processes:
            continue
        source_name = entry.modifiers.get("linear_name")
        if not source_name:
            continue
        for cell in sorted(cells_by_name.get(source_name, ()), key=lambda item: (item.xidx, item.yidx)):
            cm_type = _choose_cm_type(entry.cm_types, rng)
            if cm_type is None:
                continue
            linear_placements.append(
                PlacementRecord(
                    layer=_layer_for_cm_type(cm_type),
                    grid_kind=GridKind.NORMAL,
                    cells=(GridCell(cell.xidx, cell.yidx),),
                    config_name=entry.name,
                    feature_id=f"{entry.name}:{source_name}:{cell.xidx}:{cell.yidx}",
                    priority=entry.priority,
                    cm_type=cm_type,
                    score=1.0,
                    diagnostics={"derived_from_linear": source_name},
                )
            )
    return tuple(linear_placements)


def _choose_cm_type(cm_types: tuple[CMType, ...], rng: np.random.Generator) -> CMType | None:
    if not cm_types:
        return None
    weights = np.array([float(cm_type.modifiers.get("weight", 1.0)) for cm_type in cm_types], dtype=float)
    probabilities = weights / weights.sum()
    cm_type = cm_types[int(rng.choice(len(cm_types), p=probabilities))]
    return None if cm_type.modifiers.get("dummy") is True else cm_type


def _layer_for_cm_type(cm_type: CMType) -> LayerKind:
    menu = cm_type.menu.lower()
    if menu.startswith("foliage") or menu.startswith("brush"):
        return LayerKind.FOLIAGE
    if menu.startswith("flavor objects"):
        return LayerKind.POINT_OBJECT
    if menu.startswith("walls") or menu.startswith("fence"):
        return LayerKind.LINEAR_OBJECT
    if menu.startswith("roads"):
        return LayerKind.LINEAR_SURFACE
    if "building" in menu:
        return LayerKind.BUILDING
    return LayerKind.GROUND


def _reserve_output_placements(occupancy: OccupancyModel | None, placements: tuple[PlacementRecord, ...]) -> None:
    if occupancy is None:
        return
    for placement in placements:
        object_id = (
            placement.feature_id
            if placement.feature_id is not None
            else f"{placement.config_name}:{placement.cells[0].xidx}:{placement.cells[0].yidx}"
        )
        occupancy.place(placement, object_id=object_id, allow_replace=False)


def _resolve_output_layer_conflicts(placements: tuple[PlacementRecord, ...]) -> tuple[PlacementRecord, ...]:
    winners_by_cell = {}
    for placement_idx, placement in enumerate(placements):
        for cell in placement.cells:
            key = (placement.layer, cell)
            winner = winners_by_cell.get(key)
            candidate = (_output_priority_key(placement), placement_idx)
            if winner is None or candidate < winner[0]:
                winners_by_cell[key] = (candidate, placement)

    resolved = []
    for placement in placements:
        cells = tuple(
            cell
            for cell in placement.cells
            if winners_by_cell.get((placement.layer, cell), (None, None))[1] is placement
        )
        if not cells:
            continue
        if cells == placement.cells:
            resolved.append(placement)
            continue
        resolved.append(
            PlacementRecord(
                layer=placement.layer,
                grid_kind=placement.grid_kind,
                cells=cells,
                config_name=placement.config_name,
                feature_id=placement.feature_id,
                priority=placement.priority,
                cm_type=placement.cm_type,
                score=placement.score,
                diagnostics=placement.diagnostics,
            )
        )
    return tuple(resolved)


def _output_priority_key(placement: PlacementRecord) -> tuple[int, int]:
    if placement.priority > 0:
        return 0, placement.priority
    if placement.priority > -999:
        return 1, -placement.priority
    return 2, 0


def _source_aware_road_validation_diagnostics(
    *,
    road_validation: Any,
    topology: Any,
    grid_index: GridIndex,
) -> Mapping[str, Any]:
    endpoint_cells = _road_topology_endpoint_cells(topology, grid_index)
    dangling_cells = frozenset(
        issue.cell
        for issue in getattr(road_validation, "dangling_arms", ()) or ()
        if getattr(issue, "cell", None) is not None
    )
    unexplained = tuple(sorted(dangling_cells - endpoint_cells, key=lambda cell: (cell.xidx, cell.yidx)))
    return {
        "topology_endpoint_cells": tuple((cell.xidx, cell.yidx) for cell in sorted(endpoint_cells)),
        "dangling_arm_cells": tuple((cell.xidx, cell.yidx) for cell in sorted(dangling_cells)),
        "unexplained_dangling_arm_cells": tuple((cell.xidx, cell.yidx) for cell in unexplained),
    }


def _road_topology_endpoint_cells(topology: Any, grid_index: GridIndex) -> frozenset[GridCell]:
    nodes = tuple(getattr(topology, "nodes", ()) or ())
    edges = tuple(edge for edge in (getattr(topology, "edges", ()) or ()) if edge.process is ProcessKind.ROAD)
    if not nodes or not edges:
        return frozenset()
    degree_by_node = {node.node_id: 0 for node in nodes}
    for edge in edges:
        degree_by_node[edge.start_node_id] = degree_by_node.get(edge.start_node_id, 0) + 1
        degree_by_node[edge.end_node_id] = degree_by_node.get(edge.end_node_id, 0) + 1

    node_by_id = {node.node_id: node for node in nodes}
    return frozenset(
        grid_index.projected_to_cell(node.point.x, node.point.y)
        for node_id, degree in degree_by_node.items()
        if degree == 1
        for node in (node_by_id[node_id],)
    )
