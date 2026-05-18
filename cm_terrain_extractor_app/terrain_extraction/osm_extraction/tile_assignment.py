from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any

import numpy as np
from terrain_extraction.osm_extraction.config_schema import TileAssignmentSolverConfig
from terrain_extraction.osm_extraction.direction_resolution import (
    DIRECTION_ORDER,
    OPPOSITE_DIRECTIONS,
    normalize_direction_set,
    ordered_directions,
    resolve_required_directions,
)
from terrain_extraction.osm_extraction.direction_resolution import (
    direction_between_cells as shared_direction_between_cells,
)
from terrain_extraction.osm_extraction.direction_resolution import (
    next_cell as shared_next_cell,
)
from terrain_extraction.osm_extraction.models import (
    CMType,
    GridCell,
    GridKind,
    GridNode,
    LayerKind,
    PlacementRecord,
    ProcessKind,
    RouteRecord,
    TileAssignmentResult,
)

_CATALOG_DIRECTION_COLUMNS = {
    "u": "N",
    "d": "S",
    "r": "E",
    "l": "W",
    "ur": "NE",
    "ul": "NW",
    "dr": "SE",
    "dl": "SW",
}
_DIRECTION_ORDER = DIRECTION_ORDER
_OPPOSITE_DIRECTIONS = OPPOSITE_DIRECTIONS
_LABEL_PREFIX = {
    ProcessKind.ROAD: "Road",
    ProcessKind.RAIL: "Rail",
    ProcessKind.STREAM: "Stream",
    ProcessKind.FENCE: "Fence",
}
_DEFAULT_CM_TYPES = {
    ProcessKind.ROAD: ("Roads", "Road"),
    ProcessKind.RAIL: ("Roads", "Railroad"),
    ProcessKind.STREAM: ("Roads", "Stream"),
    ProcessKind.FENCE: ("Walls/Fences", "Fence"),
}


@dataclass(frozen=True, slots=True)
class TileVariant:
    variant_id: str
    process: ProcessKind
    directions: frozenset[str]
    side_signatures: Mapping[str, Any]
    cm_type: CMType
    cost: float
    catalog_direction: int | None
    row: int
    col: int
    variant: int
    connections: Mapping[str, Any]

    @property
    def open_directions(self) -> frozenset[str]:
        return self.directions


@dataclass(frozen=True, slots=True)
class CompiledTileCatalog:
    process: ProcessKind
    variants: tuple[TileVariant, ...]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]] | Any,
        *,
        process: ProcessKind,
        base_cm_type: CMType | None = None,
    ) -> CompiledTileCatalog:
        normalized_records = _records_from_any(records)
        variants = tuple(
            _variant_from_record(record, process=process, base_cm_type=base_cm_type)
            for record in normalized_records
            if _directions_from_record(record)
        )
        return cls(process=process, variants=variants)

    @property
    def variant_count(self) -> int:
        return len(self.variants)

    def required_directions_for_nodes(
        self,
        first: GridNode | tuple[int, int],
        second: GridNode | tuple[int, int],
    ) -> frozenset[str]:
        direction = _direction_between_nodes(first, second)
        return frozenset((direction, _OPPOSITE_DIRECTIONS[direction]))

    def has_tile(self, required_dirs: Iterable[str]) -> bool:
        return bool(self.candidates_for(required_dirs))

    def best_tile(self, required_dirs: Iterable[str], road_class: object | None = None) -> TileVariant | None:
        del road_class
        candidates = self.candidates_for(required_dirs)
        return None if not candidates else candidates[0]

    def allowed_step_dirs(self) -> frozenset[str]:
        return frozenset(direction for variant in self.variants for direction in variant.directions)

    def can_extend(self, existing_dirs: Iterable[str], new_dir: str) -> bool:
        return self.has_tile((*existing_dirs, new_dir))

    def missing_direction_sets(self, required_direction_sets: Iterable[Iterable[str]]) -> tuple[tuple[str, ...], ...]:
        return tuple(
            _ordered_directions(normalized)
            for normalized in (normalize_direction_set(direction_set) for direction_set in required_direction_sets)
            if self.resolved_required_directions(normalized) is None
        )

    def catalog_gap_diagnostics(self, required_direction_sets: Iterable[Iterable[str]]) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            {
                "process": self.process.value,
                "required_directions": missing,
                "failure_reason": "catalog_gap",
            }
            for missing in self.missing_direction_sets(required_direction_sets)
        )

    def candidates_for(self, required_directions: Iterable[str]) -> tuple[TileVariant, ...]:
        normalized = self.resolved_required_directions(required_directions)
        if normalized is None:
            return ()
        return self._exact_candidates_for(normalized)

    def resolved_required_directions(self, required_directions: Iterable[str]) -> frozenset[str] | None:
        return resolve_required_directions(required_directions, self._has_exact_tile)

    def _has_exact_tile(self, required_directions: frozenset[str]) -> bool:
        return any(variant.directions == required_directions for variant in self.variants)

    def _exact_candidates_for(self, required_directions: frozenset[str]) -> tuple[TileVariant, ...]:
        return tuple(
            sorted(
                (variant for variant in self.variants if variant.directions == required_directions),
                key=_variant_sort_key,
            )
        )


@dataclass(frozen=True, slots=True)
class _RouteCellSpec:
    cell: GridCell
    required_directions: frozenset[str]


@dataclass(frozen=True, slots=True)
class _StateCellSpec:
    process: ProcessKind
    cell: GridCell
    required_directions: frozenset[str]
    route_ids: tuple[int | str, ...]
    priority: int
    intersection_kind: str


@dataclass(frozen=True, slots=True)
class _StateCellOptions:
    spec: _StateCellSpec
    candidates: tuple[TileVariant, ...]


@dataclass(frozen=True, slots=True)
class _StateAdjacency:
    first: GridCell
    direction: str
    second: GridCell


@dataclass(frozen=True, slots=True)
class _StateComponentResult:
    selection: dict[GridCell, TileVariant] | None
    solver_used: str
    failure_reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _BranchedSearchFrame:
    action: str
    cost: float
    cell: GridCell | None = None
    candidate: TileVariant | None = None


@dataclass(slots=True)
class _BranchedStateSearch:
    component: tuple[GridCell, ...]
    options_by_cell: Mapping[GridCell, _StateCellOptions]
    neighbors: Mapping[GridCell, tuple[tuple[GridCell, _StateAdjacency], ...]]
    rng: np.random.Generator
    min_remaining_cost: dict[GridCell, float] = field(init=False)
    assigned: dict[GridCell, TileVariant] = field(init=False, default_factory=dict)
    best: dict[GridCell, TileVariant] | None = None
    best_cost: float = math.inf
    tie_count: int = 0

    def __post_init__(self) -> None:
        self.min_remaining_cost = {
            cell: min(candidate.cost for candidate in self.options_by_cell[cell].candidates)
            for cell in self.component
        }

    def run(self) -> dict[GridCell, TileVariant] | None:
        stack = [_BranchedSearchFrame("search", 0.0)]
        while stack:
            frame = stack.pop()
            if frame.action == "cleanup":
                self._cleanup(frame.cell)
            elif frame.action == "branch":
                self._enter_branch(frame, stack)
            else:
                self._search(frame.cost, stack)
        return self.best

    def viable_candidates(self, cell: GridCell) -> tuple[TileVariant, ...]:
        viable = []
        for candidate in self.options_by_cell[cell].candidates:
            if all(
                _state_edge_compatible(cell, candidate, edge, self.assigned[neighbor])
                for neighbor, edge in self.neighbors[cell]
                if neighbor in self.assigned
            ):
                viable.append(candidate)
        return tuple(viable)

    def has_forward_candidate(self, cell: GridCell, candidate: TileVariant) -> bool:
        for neighbor, edge in self.neighbors[cell]:
            if neighbor in self.assigned:
                continue
            if not self._has_compatible_neighbor_candidate(cell, candidate, neighbor, edge):
                return False
        return True

    def _has_compatible_neighbor_candidate(
        self,
        cell: GridCell,
        candidate: TileVariant,
        neighbor: GridCell,
        edge: _StateAdjacency,
    ) -> bool:
        neighbor_candidates = (
            neighbor_candidate
            for neighbor_candidate in self.options_by_cell[neighbor].candidates
            if _state_edge_compatible(cell, candidate, edge, neighbor_candidate)
        )
        return any(
            all(
                _state_edge_compatible(neighbor, neighbor_candidate, neighbor_edge, self.assigned[other])
                for other, neighbor_edge in self.neighbors[neighbor]
                if other in self.assigned and other != cell
            )
            for neighbor_candidate in neighbor_candidates
        )

    def _search(
        self,
        cost_so_far: float,
        stack: list[_BranchedSearchFrame],
    ) -> None:
        if len(self.assigned) == len(self.component):
            self._record_solution(cost_so_far)
            return

        selection = self._select_next_cell(cost_so_far)
        if selection is None:
            return

        cell, candidates = selection
        for candidate in reversed(candidates):
            self._push_branch(cell, candidate, cost_so_far, stack)

    def _select_next_cell(self, cost_so_far: float) -> tuple[GridCell, tuple[TileVariant, ...]] | None:
        unassigned = tuple(cell for cell in self.component if cell not in self.assigned)
        lower_bound = cost_so_far + sum(self.min_remaining_cost[cell] for cell in unassigned)
        if self._cost_exceeds_best(lower_bound):
            return None

        viable_by_cell = {cell: self.viable_candidates(cell) for cell in unassigned}
        cell = min(unassigned, key=lambda item: (len(viable_by_cell[item]), _cell_sort_key(item)))
        candidates = viable_by_cell[cell]
        if not candidates:
            return None
        return cell, candidates

    def _push_branch(
        self,
        cell: GridCell,
        candidate: TileVariant,
        cost_so_far: float,
        stack: list[_BranchedSearchFrame],
    ) -> None:
        next_cost = cost_so_far + candidate.cost
        if self._cost_exceeds_best(next_cost) or not self.has_forward_candidate(cell, candidate):
            return
        stack.append(_BranchedSearchFrame("branch", next_cost, cell, candidate))

    def _enter_branch(
        self,
        frame: _BranchedSearchFrame,
        stack: list[_BranchedSearchFrame],
    ) -> None:
        if frame.cell is None or frame.candidate is None or self._cost_exceeds_best(frame.cost):
            return
        self.assigned[frame.cell] = frame.candidate
        stack.append(_BranchedSearchFrame("cleanup", frame.cost, frame.cell))
        stack.append(_BranchedSearchFrame("search", frame.cost))

    def _cleanup(self, cell: GridCell | None) -> None:
        if cell is not None:
            del self.assigned[cell]

    def _record_solution(self, cost_so_far: float) -> None:
        if cost_so_far < self.best_cost and not math.isclose(cost_so_far, self.best_cost):
            self.best = dict(self.assigned)
            self.best_cost = cost_so_far
            self.tie_count = 1
            return
        if math.isclose(cost_so_far, self.best_cost):
            self.tie_count += 1
            if int(self.rng.integers(0, self.tie_count)) == 0:
                self.best = dict(self.assigned)

    def _cost_exceeds_best(self, cost: float) -> bool:
        return cost > self.best_cost and not math.isclose(cost, self.best_cost)


@dataclass(slots=True)
class _IntersectionSpec:
    process: ProcessKind
    node_id: int
    node: GridNode
    directions: set[str]
    cells: set[GridCell]
    cm_type: CMType | None = None


@dataclass(frozen=True, slots=True)
class _IntersectionEndpoint:
    process: ProcessKind
    node_id: int
    node: GridNode
    direction: str
    cells_from_node: tuple[GridCell, ...]
    cm_type: CMType | None = None


class TileAssigner:
    def __init__(
        self,
        catalogs: Mapping[ProcessKind, CompiledTileCatalog],
        *,
        rng: np.random.Generator | None = None,
        solver_config: TileAssignmentSolverConfig | None = None,
    ) -> None:
        self.catalogs = dict(catalogs)
        self.rng = rng or np.random.default_rng(0)
        self.solver_config = solver_config or TileAssignmentSolverConfig()

    def assign(self, routes: Sequence[RouteRecord], *, linear_state: Any = None) -> TileAssignmentResult:
        successful_routes = tuple(route for route in routes if route.success and route.nodes)
        if linear_state is not None:
            return self._assign_from_linear_state(successful_routes, linear_state)

        intersections = _intersection_specs_by_process(successful_routes)

        placements: list[PlacementRecord] = []
        failures: list[Mapping[str, Any]] = []
        used_cells: set[tuple[ProcessKind, GridCell]] = set()
        fixed_variants: dict[tuple[ProcessKind, GridCell], TileVariant] = {}

        for intersection in sorted(intersections.values(), key=_intersection_sort_key):
            if len(intersection.directions) < 3:
                continue
            cell = _intersection_cell_for_node(intersection.node, intersection.cells)
            required = frozenset(intersection.directions)
            variant = self._variant_for(intersection.process, cell, required, failures)
            if variant is None:
                continue
            placements.append(
                self._placement_from_variant(
                    process=intersection.process,
                    cell=cell,
                    required_directions=required,
                    route=None,
                    intersection=True,
                    variant=variant,
                    cm_type_override=intersection.cm_type,
                )
            )
            used_cells.add((intersection.process, cell))
            fixed_variants[(intersection.process, cell)] = variant

        for route in successful_routes:
            catalog = self.catalogs.get(route.process)
            if catalog is None:
                for spec in _route_cell_specs(route):
                    if (route.process, spec.cell) not in used_cells:
                        failures.append(
                            _failure(
                                route.process,
                                spec.cell,
                                spec.required_directions,
                                "missing_catalog",
                                route_id=route.edge_id,
                            )
                        )
                continue
            route_placements = self._placements_for_route(
                route=route,
                catalog=catalog,
                fixed_variants=fixed_variants,
                used_cells=used_cells,
                failures=failures,
                linear_state=linear_state,
            )
            placements.extend(route_placements)

        return TileAssignmentResult(
            placements=tuple(placements),
            failures=tuple(failures),
            diagnostics={
                "tile_assignments": len(placements),
                "failed_assignments": len(failures),
                "intersection_assignments": sum(
                    1 for placement in placements if placement.diagnostics.get("intersection")
                ),
            },
        )

    def _assign_from_linear_state(
        self,
        routes: Sequence[RouteRecord],
        linear_state: Any,
    ) -> TileAssignmentResult:
        route_by_id = _routes_by_state_id(routes)
        failures: list[Mapping[str, Any]] = []
        options_by_cell: dict[GridCell, _StateCellOptions] = {}
        state_required_dirs_by_cell: dict[GridCell, frozenset[str]] = {}

        for spec in _state_cell_specs(linear_state):
            state_required_dirs_by_cell[spec.cell] = spec.required_directions
            catalog = self.catalogs.get(spec.process)
            if catalog is None:
                failures.append(
                    _failure(
                        spec.process,
                        spec.cell,
                        spec.required_directions,
                        "missing_catalog",
                        hard_failure=True,
                        route_ids=spec.route_ids,
                    )
                )
                continue
            resolved_required_directions = catalog.resolved_required_directions(spec.required_directions)
            if resolved_required_directions is None:
                failures.append(
                    _failure(
                        spec.process,
                        spec.cell,
                        spec.required_directions,
                        "catalog_gap",
                        hard_failure=True,
                        route_ids=spec.route_ids,
                    )
                )
                continue
            candidates = catalog.candidates_for(resolved_required_directions)
            if not candidates:
                failures.append(
                    _failure(
                        spec.process,
                        spec.cell,
                        spec.required_directions,
                        "catalog_gap",
                        hard_failure=True,
                        route_ids=spec.route_ids,
                    )
                )
                continue
            resolved_spec = _StateCellSpec(
                process=spec.process,
                cell=spec.cell,
                required_directions=resolved_required_directions,
                route_ids=spec.route_ids,
                priority=spec.priority,
                intersection_kind=spec.intersection_kind,
            )
            options_by_cell[spec.cell] = _StateCellOptions(spec=resolved_spec, candidates=candidates)

        selected_variants, compatibility_failures, component_diagnostics = self._solve_state_components(
            options_by_cell,
            state_required_dirs_by_cell,
        )
        failures.extend(compatibility_failures)

        placements: list[PlacementRecord] = []
        for cell in sorted(selected_variants, key=_cell_sort_key):
            option = options_by_cell[cell]
            variant = selected_variants[cell]
            placements.append(
                self._state_placement_from_variant(
                    spec=option.spec,
                    variant=variant,
                    contributing_routes=tuple(
                        route_by_id[route_id] for route_id in option.spec.route_ids if route_id in route_by_id
                    ),
                )
            )

        return TileAssignmentResult(
            placements=tuple(placements),
            failures=tuple(failures),
            diagnostics={
                "tile_assignments": len(placements),
                "failed_assignments": len(failures),
                "intersection_assignments": sum(
                    1
                    for placement in placements
                    if placement.diagnostics.get("role") in {"intersection", "t_junction"}
                ),
                "state_finalized_cells": len(placements),
                "state_finalizer": True,
                "state_component_diagnostics": component_diagnostics,
            },
        )

    def _variant_for(
        self,
        process: ProcessKind,
        cell: GridCell,
        required_directions: frozenset[str],
        failures: list[Mapping[str, Any]],
    ) -> TileVariant | None:
        catalog = self.catalogs.get(process)
        if catalog is None:
            failures.append(_failure(process, cell, required_directions, "missing_catalog"))
            return None

        candidates = catalog.candidates_for(required_directions)
        if not candidates:
            failures.append(_failure(process, cell, required_directions, "catalog_gap"))
            return None

        return self._choose_candidate(candidates)

    def _placement_from_variant(
        self,
        *,
        process: ProcessKind,
        cell: GridCell,
        required_directions: frozenset[str],
        route: RouteRecord | None,
        intersection: bool,
        variant: TileVariant,
        cm_type_override: CMType | None = None,
    ) -> PlacementRecord:
        cm_type = _variant_cm_type(variant, cm_type_override or (route.cm_type if route is not None else None))
        return PlacementRecord(
            layer=_layer_for_process(process),
            grid_kind=GridKind.NORMAL,
            cells=(cell,),
            config_name=route.config_name if route is not None else f"{process.value}_intersection",
            feature_id=route.edge_id if route is not None else f"intersection:{process.value}:{cell.xidx}:{cell.yidx}",
            priority=route.priority if route is not None else 0,
            cm_type=cm_type,
            score=-variant.cost,
            diagnostics={
                "required_directions": _ordered_directions(required_directions),
                "intersection": intersection,
                "catalog_direction": variant.catalog_direction,
                "tile_row": variant.row,
                "tile_col": variant.col,
                "variant": variant.variant,
                "selected_tile_id": variant.cm_type.tile_id,
                "connection_dirs": _ordered_directions(required_directions),
                "source_process": process.value,
                "role": _placement_role(required_directions),
            },
        )

    def _state_placement_from_variant(
        self,
        *,
        spec: _StateCellSpec,
        variant: TileVariant,
        contributing_routes: tuple[RouteRecord, ...],
    ) -> PlacementRecord:
        primary_route = _primary_route(contributing_routes)
        cm_type = _variant_cm_type(variant, primary_route.cm_type if primary_route is not None else None)
        config_name = primary_route.config_name if primary_route is not None else spec.process.value
        feature_id = primary_route.edge_id if primary_route is not None else _first_or_none(spec.route_ids)
        source_feature_ids = _source_feature_ids(contributing_routes) or spec.route_ids
        role = _placement_role(spec.required_directions)
        return PlacementRecord(
            layer=_layer_for_process(spec.process),
            grid_kind=GridKind.NORMAL,
            cells=(spec.cell,),
            config_name=config_name,
            feature_id=feature_id,
            priority=spec.priority,
            cm_type=cm_type,
            score=-variant.cost,
            diagnostics={
                "required_directions": _ordered_directions(spec.required_directions),
                "connection_dirs": _ordered_directions(spec.required_directions),
                "intersection": role in {"t_junction", "intersection"},
                "catalog_direction": variant.catalog_direction,
                "tile_row": variant.row,
                "tile_col": variant.col,
                "variant": variant.variant,
                "selected_tile_id": variant.cm_type.tile_id,
                "source_process": spec.process.value,
                "source_config": config_name,
                "source_feature_ids": source_feature_ids,
                "contributing_route_ids": spec.route_ids,
                "intersection_kind": spec.intersection_kind,
                "role": role,
            },
        )

    def _placements_for_route(
        self,
        *,
        route: RouteRecord,
        catalog: CompiledTileCatalog,
        fixed_variants: Mapping[tuple[ProcessKind, GridCell], TileVariant],
        used_cells: set[tuple[ProcessKind, GridCell]],
        failures: list[Mapping[str, Any]],
        linear_state: Any = None,
    ) -> tuple[PlacementRecord, ...]:
        specs = _route_cell_specs(route, linear_state=linear_state)
        if not specs:
            return ()

        active_specs: list[_RouteCellSpec] = []
        candidate_columns: list[tuple[TileVariant, ...]] = []
        fixed_columns: set[int] = set()
        for spec in specs:
            fixed_variant = fixed_variants.get((route.process, spec.cell))
            if fixed_variant is not None:
                if not spec.required_directions.issubset(fixed_variant.directions):
                    if spec.cell in {specs[0].cell, specs[-1].cell}:
                        continue
                    failures.append(
                        _failure(
                            route.process,
                            spec.cell,
                            spec.required_directions,
                            "fixed_tile_mismatch",
                            route_id=route.edge_id,
                        )
                    )
                    return ()
                active_specs.append(spec)
                fixed_columns.add(len(candidate_columns))
                candidate_columns.append((fixed_variant,))
                continue

            candidates = catalog.candidates_for(spec.required_directions)
            if not candidates:
                failures.append(
                    _failure(
                        route.process,
                        spec.cell,
                        spec.required_directions,
                        "catalog_gap",
                        route_id=route.edge_id,
                    )
                )
                return ()
            active_specs.append(spec)
            candidate_columns.append(candidates)

        specs = tuple(active_specs)
        if not specs:
            return ()

        selected = self._least_cost_compatible_path(specs, candidate_columns, fixed_columns=fixed_columns)
        if selected is None:
            failures.append(
                _failure(
                    route.process,
                    specs[0].cell,
                    specs[0].required_directions,
                    "no_compatible_tile_path",
                    route_id=route.edge_id,
                )
            )
            return ()

        placements = []
        for spec, variant in zip(specs, selected, strict=True):
            cell_key = (route.process, spec.cell)
            if cell_key in fixed_variants or cell_key in used_cells:
                continue
            placements.append(
                self._placement_from_variant(
                    process=route.process,
                    cell=spec.cell,
                    required_directions=spec.required_directions,
                    route=route,
                    intersection=False,
                    variant=variant,
                )
            )
            used_cells.add(cell_key)
        return tuple(placements)

    def _least_cost_compatible_path(
        self,
        specs: tuple[_RouteCellSpec, ...],
        candidate_columns: Sequence[tuple[TileVariant, ...]],
        *,
        fixed_columns: set[int] | None = None,
    ) -> tuple[TileVariant, ...] | None:
        fixed_columns = fixed_columns or set()
        costs: dict[tuple[int, int], float] = {}
        previous: dict[tuple[int, int], tuple[int, int] | None] = {}
        for variant_index, variant in enumerate(candidate_columns[0]):
            costs[(0, variant_index)] = variant.cost
            previous[(0, variant_index)] = None

        for column_index in range(1, len(candidate_columns)):
            step_direction = _direction_between_cells(specs[column_index - 1].cell, specs[column_index].cell)
            for variant_index, variant in enumerate(candidate_columns[column_index]):
                best: tuple[float, tuple[int, int]] | None = None
                for prev_index, prev_variant in enumerate(candidate_columns[column_index - 1]):
                    prev_key = (column_index - 1, prev_index)
                    if prev_key not in costs:
                        continue
                    crosses_fixed_tile = column_index in fixed_columns or column_index - 1 in fixed_columns
                    if (
                        step_direction is not None
                        and not crosses_fixed_tile
                        and not _variants_connect(prev_variant, variant, step_direction)
                    ):
                        continue
                    path_cost = costs[prev_key] + variant.cost
                    if best is None or path_cost < best[0]:
                        best = (path_cost, prev_key)
                if best is None:
                    continue
                costs[(column_index, variant_index)] = best[0]
                previous[(column_index, variant_index)] = best[1]

        final_column = len(candidate_columns) - 1
        final_keys = [key for key in costs if key[0] == final_column]
        if not final_keys:
            return None
        key = min(
            final_keys,
            key=lambda item: (
                costs[item],
                _variant_sort_key(candidate_columns[item[0]][item[1]]),
                item,
            ),
        )
        selected: list[TileVariant] = []
        while key is not None:
            selected.append(candidate_columns[key[0]][key[1]])
            key = previous[key]
        selected.reverse()
        return tuple(selected)

    def _solve_state_components(
        self,
        options_by_cell: Mapping[GridCell, _StateCellOptions],
        state_required_dirs_by_cell: Mapping[GridCell, frozenset[str]],
    ) -> tuple[dict[GridCell, TileVariant], tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
        if not options_by_cell:
            return {}, (), ()

        edges = _state_adjacencies(options_by_cell, state_required_dirs_by_cell)
        components = _state_components(options_by_cell, edges)
        edges_by_component = _edges_by_component(components, edges)
        selected: dict[GridCell, TileVariant] = {}
        failures: list[Mapping[str, Any]] = []
        diagnostics: list[Mapping[str, Any]] = []

        for component in components:
            component_edges = edges_by_component[frozenset(component)]
            started_at = time.perf_counter()
            result = self._solve_state_component(component, component_edges, options_by_cell)
            component_diagnostics = {
                **_state_component_diagnostics(component, component_edges, options_by_cell),
                **dict(result.details),
                "solver_used": result.solver_used,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
            }
            diagnostics.append(component_diagnostics)
            if result.selection is None:
                failures.append(
                    _state_component_failure(
                        component,
                        component_edges,
                        options_by_cell,
                        reason=result.failure_reason or "no_compatible_tile_component",
                        diagnostics=component_diagnostics,
                    )
                )
                continue
            selected.update(result.selection)
        return selected, tuple(failures), tuple(diagnostics)

    def _solve_state_component(
        self,
        component: tuple[GridCell, ...],
        edges: tuple[_StateAdjacency, ...],
        options_by_cell: Mapping[GridCell, _StateCellOptions],
    ) -> _StateComponentResult:
        candidate_product_log10 = _candidate_product_log10(component, options_by_cell)
        if len(component) == 1:
            return _StateComponentResult(
                selection={component[0]: self._choose_candidate(options_by_cell[component[0]].candidates)},
                solver_used="single_cell",
            )

        ordered_path = _ordered_simple_component_path(component, edges)
        if ordered_path is not None:
            selection = self._solve_state_path_component(ordered_path, options_by_cell)
            return _StateComponentResult(
                selection=selection,
                solver_used="path_dp",
                failure_reason=None if selection is not None else "no_compatible_tile_component",
            )

        ordered_cycle = _ordered_simple_component_cycle(component, edges)
        if ordered_cycle is not None:
            selection = self._solve_state_cycle_component(ordered_cycle, options_by_cell)
            return _StateComponentResult(
                selection=selection,
                solver_used="cycle_dp",
                failure_reason=None if selection is not None else "no_compatible_tile_component",
            )

        if _is_tree_component(component, edges):
            selection = self._solve_state_tree_component(component, edges, options_by_cell)
            return _StateComponentResult(
                selection=selection,
                solver_used="tree_dp",
                failure_reason=None if selection is not None else "no_compatible_tile_component",
            )

        cycle_count = _cycle_count(component, edges)
        if cycle_count <= self.solver_config.max_cutset_cycle_rank:
            cutset = self._solve_state_cutset_component(component, edges, options_by_cell)
            if cutset.selection is not None or cutset.failure_reason == "no_compatible_tile_component":
                return cutset

        if (
            len(component) <= self.solver_config.tiny_exact_max_cells
            and candidate_product_log10 <= self.solver_config.tiny_exact_candidate_product_log10
        ):
            selection = self._solve_state_branched_component(component, edges, options_by_cell)
            return _StateComponentResult(
                selection=selection,
                solver_used="tiny_exact",
                failure_reason=None if selection is not None else "no_compatible_tile_component",
            )

        return _StateComponentResult(
            selection=None,
            solver_used="unresolved",
            failure_reason="component_solver_limit_exceeded",
            details={
                "solver_limit": {
                    "max_cutset_cycle_rank": self.solver_config.max_cutset_cycle_rank,
                    "max_cutset_vertices": self.solver_config.max_cutset_vertices,
                    "max_cutset_candidate_product_log10": self.solver_config.max_cutset_candidate_product_log10,
                    "tiny_exact_max_cells": self.solver_config.tiny_exact_max_cells,
                    "tiny_exact_candidate_product_log10": self.solver_config.tiny_exact_candidate_product_log10,
                }
            },
        )

    def _solve_state_path_component(
        self,
        ordered_cells: tuple[GridCell, ...],
        options_by_cell: Mapping[GridCell, _StateCellOptions],
    ) -> dict[GridCell, TileVariant] | None:
        specs = tuple(_RouteCellSpec(cell=cell, required_directions=options_by_cell[cell].spec.required_directions) for cell in ordered_cells)
        candidate_columns = tuple(options_by_cell[cell].candidates for cell in ordered_cells)
        selected = self._least_cost_compatible_path(specs, candidate_columns)
        if selected is None:
            return None
        return dict(zip(ordered_cells, selected, strict=True))

    def _solve_state_cycle_component(
        self,
        ordered_cells: tuple[GridCell, ...],
        options_by_cell: Mapping[GridCell, _StateCellOptions],
    ) -> dict[GridCell, TileVariant] | None:
        first_cell = ordered_cells[0]
        first_candidates = options_by_cell[first_cell].candidates
        best: dict[GridCell, TileVariant] | None = None
        best_cost = math.inf

        for first_variant in first_candidates:
            candidate_columns = [(first_variant,)]
            candidate_columns.extend(options_by_cell[cell].candidates for cell in ordered_cells[1:])
            selected = self._least_cost_compatible_path(
                tuple(
                    _RouteCellSpec(cell=cell, required_directions=options_by_cell[cell].spec.required_directions)
                    for cell in ordered_cells
                ),
                tuple(candidate_columns),
            )
            if selected is None:
                continue
            closing_direction = _direction_between_cells(ordered_cells[-1], ordered_cells[0])
            if closing_direction is None or not compatible_neighbor(selected[-1], closing_direction, selected[0]):
                continue
            path_cost = sum(variant.cost for variant in selected)
            if path_cost < best_cost and not math.isclose(path_cost, best_cost):
                best = dict(zip(ordered_cells, selected, strict=True))
                best_cost = path_cost
            elif math.isclose(path_cost, best_cost):
                candidate = dict(zip(ordered_cells, selected, strict=True))
                if best is None or _selection_sort_key(candidate) < _selection_sort_key(best):
                    best = candidate
        return best

    def _solve_state_tree_component(
        self,
        component: tuple[GridCell, ...],
        edges: tuple[_StateAdjacency, ...],
        options_by_cell: Mapping[GridCell, _StateCellOptions],
        *,
        fixed_variants: Mapping[GridCell, TileVariant] | None = None,
    ) -> dict[GridCell, TileVariant] | None:
        fixed_variants = fixed_variants or {}
        component_set = frozenset(component)
        tree_edges = tuple(edge for edge in edges if edge.first in component_set and edge.second in component_set)
        neighbors = _state_neighbor_edges(tree_edges)
        root = sorted(
            component,
            key=lambda cell: (-len(neighbors.get(cell, ())), _cell_sort_key(cell)),
        )[0]
        traversal = _tree_traversal(component, neighbors, root)
        if traversal is None:
            return None
        parent_edge, children, order = traversal

        fixed_edges_by_cell = _fixed_edges_by_cell(component_set, edges, fixed_variants)
        costs: dict[tuple[GridCell, int], float] = {}
        argmins: dict[tuple[GridCell, int, GridCell], int] = {}
        for cell in reversed(order):
            for variant_index, variant in enumerate(options_by_cell[cell].candidates):
                if not all(
                    _state_edge_compatible(cell, variant, edge, fixed_variant)
                    for edge, fixed_variant in fixed_edges_by_cell.get(cell, ())
                ):
                    continue
                total = variant.cost
                valid = True
                for child in sorted(children[cell], key=_cell_sort_key):
                    edge = parent_edge[child]
                    best_child = _best_child_state(cell, variant, child, edge, options_by_cell, costs)
                    if best_child is None:
                        valid = False
                        break
                    total += best_child[0]
                    argmins[(cell, variant_index, child)] = best_child[1]
                if valid:
                    costs[(cell, variant_index)] = total

        root_options = [
            (costs[(root, index)], index)
            for index in range(len(options_by_cell[root].candidates))
            if (root, index) in costs
        ]
        if not root_options:
            return None
        _root_cost, root_index = min(
            root_options,
            key=lambda item: (
                item[0],
                _variant_sort_key(options_by_cell[root].candidates[item[1]]),
                item[1],
            ),
        )
        selected: dict[GridCell, TileVariant] = {}
        stack = [(root, root_index)]
        while stack:
            cell, variant_index = stack.pop()
            selected[cell] = options_by_cell[cell].candidates[variant_index]
            for child in sorted(children[cell], key=_cell_sort_key, reverse=True):
                stack.append((child, argmins[(cell, variant_index, child)]))
        return selected

    def _solve_state_cutset_component(
        self,
        component: tuple[GridCell, ...],
        edges: tuple[_StateAdjacency, ...],
        options_by_cell: Mapping[GridCell, _StateCellOptions],
    ) -> _StateComponentResult:
        candidate_vertices = _cutset_candidate_vertices(component, edges)
        max_vertices = min(self.solver_config.max_cutset_vertices, len(candidate_vertices))
        best: dict[GridCell, TileVariant] | None = None
        best_cost = math.inf
        saw_bounded_cutset = False
        rejected_cutsets = 0
        for cutset_size in range(1, max_vertices + 1):
            for cutset in combinations(candidate_vertices, cutset_size):
                if not _removal_leaves_forest(component, edges, frozenset(cutset)):
                    continue
                cutset_product_log10 = _candidate_product_log10(cutset, options_by_cell)
                if cutset_product_log10 > self.solver_config.max_cutset_candidate_product_log10:
                    rejected_cutsets += 1
                    continue
                saw_bounded_cutset = True
                cutset_columns = tuple(options_by_cell[cell].candidates for cell in cutset)
                for variants in product(*cutset_columns):
                    fixed_variants = dict(zip(cutset, variants, strict=True))
                    if not _selection_edges_compatible(fixed_variants, edges):
                        continue
                    selected = dict(fixed_variants)
                    valid = True
                    for forest_component in _forest_components_after_cutset(component, edges, frozenset(cutset)):
                        tree_selection = self._solve_state_tree_component(
                            forest_component,
                            edges,
                            options_by_cell,
                            fixed_variants=fixed_variants,
                        )
                        if tree_selection is None:
                            valid = False
                            break
                        selected.update(tree_selection)
                    if not valid or not _selection_edges_compatible(selected, edges):
                        continue
                    total_cost = sum(variant.cost for variant in selected.values())
                    if best is None or (
                        total_cost,
                        _selection_sort_key(selected),
                    ) < (
                        best_cost,
                        _selection_sort_key(best),
                    ):
                        best = selected
                        best_cost = total_cost
        if best is not None:
            return _StateComponentResult(selection=best, solver_used="cutset_dp")
        if saw_bounded_cutset:
            return _StateComponentResult(
                selection=None,
                solver_used="cutset_dp",
                failure_reason="no_compatible_tile_component",
            )
        return _StateComponentResult(
            selection=None,
            solver_used="unresolved",
            failure_reason="component_solver_limit_exceeded",
            details={
                "cutset_candidate_vertices": tuple((cell.xidx, cell.yidx) for cell in candidate_vertices),
                "rejected_cutsets_over_candidate_limit": rejected_cutsets,
            },
        )

    def _solve_state_branched_component(
        self,
        component: tuple[GridCell, ...],
        edges: tuple[_StateAdjacency, ...],
        options_by_cell: Mapping[GridCell, _StateCellOptions],
    ) -> dict[GridCell, TileVariant] | None:
        neighbors = _state_neighbor_edges(edges)
        return _BranchedStateSearch(
            component=component,
            options_by_cell=options_by_cell,
            neighbors=neighbors,
            rng=self.rng,
        ).run()

    def _choose_candidate(self, candidates: tuple[TileVariant, ...]) -> TileVariant:
        min_cost = min(candidate.cost for candidate in candidates)
        best = tuple(candidate for candidate in candidates if math.isclose(candidate.cost, min_cost))
        return min(best, key=_variant_sort_key)


def _records_from_any(records: Iterable[Mapping[str, Any]] | Any) -> tuple[Mapping[str, Any], ...]:
    if hasattr(records, "to_dict"):
        return tuple(records.to_dict("records"))
    return tuple(records)


def _state_adjacencies(
    options_by_cell: Mapping[GridCell, _StateCellOptions],
    state_required_dirs_by_cell: Mapping[GridCell, frozenset[str]],
) -> tuple[_StateAdjacency, ...]:
    edges = []
    cells = frozenset(options_by_cell)
    for cell in sorted(cells, key=_cell_sort_key):
        process = options_by_cell[cell].spec.process
        for direction in _ordered_directions(state_required_dirs_by_cell.get(cell, ())):
            neighbor = _next_cell(cell, direction)
            if neighbor not in cells or _cell_sort_key(neighbor) <= _cell_sort_key(cell):
                continue
            if options_by_cell[neighbor].spec.process is not process:
                continue
            opposite = _OPPOSITE_DIRECTIONS.get(direction)
            if opposite in state_required_dirs_by_cell.get(neighbor, frozenset()):
                edges.append(_StateAdjacency(first=cell, direction=direction, second=neighbor))
    return tuple(edges)


def _state_components(
    options_by_cell: Mapping[GridCell, _StateCellOptions],
    edges: tuple[_StateAdjacency, ...],
) -> tuple[tuple[GridCell, ...], ...]:
    neighbors: dict[GridCell, set[GridCell]] = {cell: set() for cell in options_by_cell}
    for edge in edges:
        neighbors[edge.first].add(edge.second)
        neighbors[edge.second].add(edge.first)

    remaining = set(options_by_cell)
    components = []
    while remaining:
        start = min(remaining, key=_cell_sort_key)
        stack = [start]
        component = set()
        remaining.remove(start)
        while stack:
            cell = stack.pop()
            component.add(cell)
            for neighbor in neighbors[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component, key=_cell_sort_key)))
    return tuple(components)


def _edges_by_component(
    components: tuple[tuple[GridCell, ...], ...],
    edges: tuple[_StateAdjacency, ...],
) -> dict[frozenset[GridCell], tuple[_StateAdjacency, ...]]:
    result = {}
    for component in components:
        cells = frozenset(component)
        result[cells] = tuple(edge for edge in edges if edge.first in cells and edge.second in cells)
    return result


def _state_neighbor_edges(edges: tuple[_StateAdjacency, ...]) -> dict[GridCell, tuple[tuple[GridCell, _StateAdjacency], ...]]:
    neighbors: dict[GridCell, list[tuple[GridCell, _StateAdjacency]]] = {}
    for edge in edges:
        neighbors.setdefault(edge.first, []).append((edge.second, edge))
        neighbors.setdefault(edge.second, []).append((edge.first, edge))
    return {
        cell: tuple(sorted(cell_neighbors, key=lambda item: _cell_sort_key(item[0])))
        for cell, cell_neighbors in neighbors.items()
    }


def _state_component_diagnostics(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
    options_by_cell: Mapping[GridCell, _StateCellOptions],
) -> Mapping[str, Any]:
    neighbors = _state_neighbor_edges(edges)
    degrees = {cell: len(neighbors.get(cell, ())) for cell in component}
    return {
        "component_size": len(component),
        "edge_count": len(edges),
        "max_degree": max(degrees.values(), default=0),
        "is_path": _ordered_simple_component_path(component, edges) is not None,
        "is_cycle": _ordered_simple_component_cycle(component, edges) is not None,
        "is_tree": _is_tree_component(component, edges),
        "cycle_count": _cycle_count(component, edges),
        "candidate_product_log10": round(_candidate_product_log10(component, options_by_cell), 3),
    }


def _cycle_count(component: tuple[GridCell, ...], edges: tuple[_StateAdjacency, ...]) -> int:
    return max(0, len(edges) - len(component) + 1)


def _candidate_product_log10(
    component: Iterable[GridCell],
    options_by_cell: Mapping[GridCell, _StateCellOptions],
) -> float:
    return sum(math.log10(len(options_by_cell[cell].candidates)) for cell in component)


def _is_tree_component(component: tuple[GridCell, ...], edges: tuple[_StateAdjacency, ...]) -> bool:
    return len(component) > 1 and len(edges) == len(component) - 1


def _fixed_edges_by_cell(
    component: frozenset[GridCell],
    edges: tuple[_StateAdjacency, ...],
    fixed_variants: Mapping[GridCell, TileVariant],
) -> dict[GridCell, tuple[tuple[_StateAdjacency, TileVariant], ...]]:
    result: dict[GridCell, list[tuple[_StateAdjacency, TileVariant]]] = {}
    for edge in edges:
        if edge.first in component and edge.second in fixed_variants:
            result.setdefault(edge.first, []).append((edge, fixed_variants[edge.second]))
        elif edge.second in component and edge.first in fixed_variants:
            result.setdefault(edge.second, []).append((edge, fixed_variants[edge.first]))
    return {cell: tuple(items) for cell, items in result.items()}


def _tree_traversal(
    component: tuple[GridCell, ...],
    neighbors: Mapping[GridCell, tuple[tuple[GridCell, _StateAdjacency], ...]],
    root: GridCell,
) -> tuple[dict[GridCell, _StateAdjacency], dict[GridCell, list[GridCell]], list[GridCell]] | None:
    parent: dict[GridCell, GridCell | None] = {root: None}
    parent_edge: dict[GridCell, _StateAdjacency] = {}
    children: dict[GridCell, list[GridCell]] = {cell: [] for cell in component}
    order: list[GridCell] = []
    stack = [root]
    while stack:
        cell = stack.pop()
        order.append(cell)
        for neighbor, edge in reversed(neighbors.get(cell, ())):
            if neighbor == parent.get(cell):
                continue
            parent[neighbor] = cell
            parent_edge[neighbor] = edge
            children[cell].append(neighbor)
            stack.append(neighbor)
    if len(parent) != len(component):
        return None
    return parent_edge, children, order


def _best_child_state(
    cell: GridCell,
    variant: TileVariant,
    child: GridCell,
    edge: _StateAdjacency,
    options_by_cell: Mapping[GridCell, _StateCellOptions],
    costs: Mapping[tuple[GridCell, int], float],
) -> tuple[float, int] | None:
    best_child: tuple[float, int] | None = None
    for child_index, child_variant in enumerate(options_by_cell[child].candidates):
        child_key = (child, child_index)
        if child_key not in costs or not _state_edge_compatible(cell, variant, edge, child_variant):
            continue
        child_cost = costs[child_key]
        candidate_key = (child_cost, _variant_sort_key(child_variant), child_index)
        if best_child is None:
            best_child = (child_cost, child_index)
            continue
        best_variant = options_by_cell[child].candidates[best_child[1]]
        if candidate_key < (best_child[0], _variant_sort_key(best_variant), best_child[1]):
            best_child = (child_cost, child_index)
    return best_child


def _cutset_candidate_vertices(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
) -> tuple[GridCell, ...]:
    parents = {cell: cell for cell in component}
    non_tree_edges = []
    for edge in sorted(edges, key=_state_edge_sort_key):
        first_root = _find_parent(parents, edge.first)
        second_root = _find_parent(parents, edge.second)
        if first_root == second_root:
            non_tree_edges.append(edge)
            continue
        parents[second_root] = first_root
    return tuple(sorted({cell for edge in non_tree_edges for cell in (edge.first, edge.second)}, key=_cell_sort_key))


def _find_parent(parents: dict[GridCell, GridCell], cell: GridCell) -> GridCell:
    while parents[cell] != cell:
        parents[cell] = parents[parents[cell]]
        cell = parents[cell]
    return cell


def _removal_leaves_forest(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
    cutset: frozenset[GridCell],
) -> bool:
    remaining = tuple(cell for cell in component if cell not in cutset)
    if not remaining:
        return True
    remaining_edges = tuple(edge for edge in edges if edge.first not in cutset and edge.second not in cutset)
    return len(remaining_edges) == len(remaining) - len(_components_from_edges(remaining, remaining_edges))


def _forest_components_after_cutset(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
    cutset: frozenset[GridCell],
) -> tuple[tuple[GridCell, ...], ...]:
    remaining = tuple(cell for cell in component if cell not in cutset)
    remaining_edges = tuple(edge for edge in edges if edge.first not in cutset and edge.second not in cutset)
    return _components_from_edges(remaining, remaining_edges)


def _components_from_edges(
    cells: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
) -> tuple[tuple[GridCell, ...], ...]:
    neighbors: dict[GridCell, set[GridCell]] = {cell: set() for cell in cells}
    for edge in edges:
        if edge.first in neighbors and edge.second in neighbors:
            neighbors[edge.first].add(edge.second)
            neighbors[edge.second].add(edge.first)
    remaining = set(cells)
    components = []
    while remaining:
        start = min(remaining, key=_cell_sort_key)
        stack = [start]
        component = set()
        remaining.remove(start)
        while stack:
            cell = stack.pop()
            component.add(cell)
            for neighbor in sorted(neighbors[cell], key=_cell_sort_key, reverse=True):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component, key=_cell_sort_key)))
    return tuple(components)


def _selection_edges_compatible(
    selection: Mapping[GridCell, TileVariant],
    edges: tuple[_StateAdjacency, ...],
) -> bool:
    return all(
        edge.first not in selection
        or edge.second not in selection
        or compatible_neighbor(selection[edge.first], edge.direction, selection[edge.second])
        for edge in edges
    )


def _selection_sort_key(selection: Mapping[GridCell, TileVariant]) -> tuple[tuple[tuple[int, int], tuple[float, int, int, int, int]], ...]:
    return tuple(
        ((_cell_sort_key(cell)), _variant_sort_key(selection[cell]))
        for cell in sorted(selection, key=_cell_sort_key)
    )


def _state_edge_sort_key(edge: _StateAdjacency) -> tuple[tuple[int, int], tuple[int, int], str]:
    return _cell_sort_key(edge.first), _cell_sort_key(edge.second), edge.direction


def _ordered_simple_component_path(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
) -> tuple[GridCell, ...] | None:
    neighbors = _state_neighbor_edges(edges)
    degrees = {cell: len(neighbors.get(cell, ())) for cell in component}
    if any(degree > 2 for degree in degrees.values()):
        return None
    endpoints = tuple(sorted((cell for cell, degree in degrees.items() if degree == 1), key=_cell_sort_key))
    if len(endpoints) != 2:
        return None
    return _walk_simple_component(endpoints[0], component, neighbors)


def _ordered_simple_component_cycle(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
) -> tuple[GridCell, ...] | None:
    neighbors = _state_neighbor_edges(edges)
    if len(component) < 3 or any(len(neighbors.get(cell, ())) != 2 for cell in component):
        return None
    return _walk_simple_component(min(component, key=_cell_sort_key), component, neighbors)


def _walk_simple_component(
    start: GridCell,
    component: tuple[GridCell, ...],
    neighbors: Mapping[GridCell, tuple[tuple[GridCell, _StateAdjacency], ...]],
) -> tuple[GridCell, ...] | None:
    ordered = [start]
    previous: GridCell | None = None
    while len(ordered) < len(component):
        candidates = tuple(neighbor for neighbor, _edge in neighbors.get(ordered[-1], ()) if neighbor != previous)
        if not candidates:
            return None
        next_cell = min(candidates, key=_cell_sort_key)
        previous = ordered[-1]
        ordered.append(next_cell)
    return tuple(ordered)


def _state_edge_compatible(
    cell: GridCell,
    variant: TileVariant,
    edge: _StateAdjacency,
    neighbor_variant: TileVariant,
) -> bool:
    if cell == edge.first:
        return compatible_neighbor(variant, edge.direction, neighbor_variant)
    return compatible_neighbor(neighbor_variant, edge.direction, variant)


def _state_component_failure(
    component: tuple[GridCell, ...],
    edges: tuple[_StateAdjacency, ...],
    options_by_cell: Mapping[GridCell, _StateCellOptions],
    *,
    reason: str = "no_compatible_tile_component",
    diagnostics: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    representative = min(component, key=_cell_sort_key)
    spec = options_by_cell[representative].spec
    route_ids = tuple(
        sorted(
            {route_id for cell in component for route_id in options_by_cell[cell].spec.route_ids},
            key=str,
        )
    )
    return _failure(
        spec.process,
        representative,
        spec.required_directions,
        reason,
        hard_failure=True,
        route_ids=route_ids,
        component_cells=tuple((cell.xidx, cell.yidx) for cell in sorted(component, key=_cell_sort_key)),
        incompatible_edges=_incompatible_edge_diagnostics(edges, options_by_cell),
        component_diagnostics=dict(diagnostics or {}),
    )


def _incompatible_edge_diagnostics(
    edges: tuple[_StateAdjacency, ...],
    options_by_cell: Mapping[GridCell, _StateCellOptions],
) -> tuple[Mapping[str, Any], ...]:
    incompatible = []
    for edge in edges:
        if any(
            compatible_neighbor(first, edge.direction, second)
            for first in options_by_cell[edge.first].candidates
            for second in options_by_cell[edge.second].candidates
        ):
            continue
        incompatible.append(_edge_diagnostic(edge))
        if len(incompatible) >= 5:
            return tuple(incompatible)
    if incompatible:
        return tuple(incompatible)
    return tuple(_edge_diagnostic(edge) for edge in edges[:5])


def _edge_diagnostic(edge: _StateAdjacency) -> Mapping[str, Any]:
    return {
        "cell_a": (edge.first.xidx, edge.first.yidx),
        "direction": edge.direction,
        "cell_b": (edge.second.xidx, edge.second.yidx),
    }


def _cell_sort_key(cell: GridCell) -> tuple[int, int]:
    return cell.yidx, cell.xidx


def _variant_from_record(
    record: Mapping[str, Any],
    *,
    process: ProcessKind,
    base_cm_type: CMType | None,
) -> TileVariant:
    directions = _directions_from_record(record)
    row = int(record.get("row", 0))
    col = int(record.get("col", 0))
    variant = int(record.get("variant", 0))
    catalog_direction = _optional_int(record.get("direction"))
    tile_number = row * 3 + col + 1
    direction_label = None if catalog_direction is None else f"Direction {catalog_direction + 1}"
    menu, cat1 = _base_menu_cat(process, base_cm_type)
    cm_type = CMType(
        menu=menu,
        cat1=cat1,
        cat2=f"{_LABEL_PREFIX.get(process, 'Linear')} Tile {tile_number}",
        direction=direction_label,
        tile_id=record.get("id", f"{process.value}:{catalog_direction}:{row}:{col}:{variant}"),
        modifiers={"connections": dict(_connections_from_record(record))},
    )
    return TileVariant(
        variant_id=str(cm_type.tile_id),
        process=process,
        directions=directions,
        side_signatures=_side_signatures_from_record(record),
        cm_type=cm_type,
        cost=float(record.get("cost", 1.0)),
        catalog_direction=catalog_direction,
        row=row,
        col=col,
        variant=variant,
        connections=_connections_from_record(record),
    )


def _directions_from_record(record: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        direction
        for column, direction in _CATALOG_DIRECTION_COLUMNS.items()
        if column in record and not _is_missing(record[column])
    )


def _connections_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        _CATALOG_DIRECTION_COLUMNS[column]: record[column]
        for column in _CATALOG_DIRECTION_COLUMNS
        if column in record and not _is_missing(record[column])
    }


def _side_signatures_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        direction: _normalize_side_signature(signature)
        for direction, signature in _connections_from_record(record).items()
    }


def _normalize_side_signature(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return tuple(_normalize_side_signature(item) for item in value)
    if isinstance(value, frozenset | set):
        return tuple(sorted((_normalize_side_signature(item) for item in value), key=repr))
    if isinstance(value, dict):
        return tuple(
            sorted(
                ((_normalize_side_signature(key), _normalize_side_signature(item)) for key, item in value.items()),
                key=repr,
            )
        )
    return value


def _normalize_direction_set(directions: Iterable[str]) -> frozenset[str]:
    return normalize_direction_set(directions)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _base_menu_cat(process: ProcessKind, base_cm_type: CMType | None) -> tuple[str, str]:
    if base_cm_type is not None:
        return base_cm_type.menu, base_cm_type.cat1
    return _DEFAULT_CM_TYPES.get(process, ("Linear", "Linear"))


def _variant_cm_type(variant: TileVariant, base_cm_type: CMType | None) -> CMType:
    if base_cm_type is None:
        return variant.cm_type
    return CMType(
        menu=base_cm_type.menu,
        cat1=base_cm_type.cat1,
        cat2=variant.cm_type.cat2,
        direction=variant.cm_type.direction,
        tile_id=variant.cm_type.tile_id,
        modifiers=variant.cm_type.modifiers,
    )


def _optional_int(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _direction_between_nodes(first: GridNode | tuple[int, int], second: GridNode | tuple[int, int]) -> str:
    first_x, first_y = _node_xy(first)
    second_x, second_y = _node_xy(second)
    dx = second_x - first_x
    dy = second_y - first_y
    if max(abs(dx), abs(dy)) != 1 or (dx == 0 and dy == 0):
        raise ValueError(f"Only adjacent route nodes can be assigned tiles: {(first_x, first_y)} -> {(second_x, second_y)}")
    if dx == 1:
        return "NE" if dy == 1 else "SE" if dy == -1 else "E"
    if dx == -1:
        return "NW" if dy == 1 else "SW" if dy == -1 else "W"
    if dy == 1:
        return "N"
    return "S"


def _required_directions_for_nodes(first: GridNode, second: GridNode) -> frozenset[str]:
    direction = _direction_between_nodes(first, second)
    return frozenset((direction, _OPPOSITE_DIRECTIONS[direction]))


def _route_cell_specs(route: RouteRecord, *, linear_state: Any = None) -> tuple[_RouteCellSpec, ...]:
    tile_cells = route.tile_cells
    if not tile_cells:
        return ()

    specs = []
    for cell_index, cell in enumerate(tile_cells):
        directions = set()
        if cell_index > 0:
            direction = _direction_between_cells(cell, tile_cells[cell_index - 1])
            if direction is not None:
                directions.add(direction)
        if cell_index < len(tile_cells) - 1:
            direction = _direction_between_cells(cell, tile_cells[cell_index + 1])
            if direction is not None:
                directions.add(direction)
        if not directions:
            continue
        if len(directions) == 1:
            directions.add(_OPPOSITE_DIRECTIONS[next(iter(directions))])
        required_directions = _state_required_dirs(linear_state, cell, directions)
        specs.append(_RouteCellSpec(cell=cell, required_directions=required_directions))
    return tuple(specs)


def _state_required_dirs(linear_state: Any, cell: GridCell, route_directions: set[str]) -> frozenset[str]:
    if linear_state is None:
        return frozenset(route_directions)
    state_directions = frozenset(linear_state.required_dirs(cell))
    if len(state_directions) >= 2:
        return state_directions
    return frozenset(route_directions)


def _state_cell_specs(linear_state: Any) -> tuple[_StateCellSpec, ...]:
    specs = []
    for row in linear_state.as_debug_layer():
        process = ProcessKind(row["process"])
        cell = GridCell(int(row["xidx"]), int(row["yidx"]))
        specs.append(
            _StateCellSpec(
                process=process,
                cell=cell,
                required_directions=frozenset(row["required_directions"]),
                route_ids=tuple(row["route_ids"]),
                priority=int(row["priority"]),
                intersection_kind=str(row["intersection_kind"]),
            )
        )
    return tuple(specs)


def _routes_by_state_id(routes: Sequence[RouteRecord]) -> dict[int | str, RouteRecord]:
    return {route.edge_id: route for route in routes}


def _primary_route(routes: Sequence[RouteRecord]) -> RouteRecord | None:
    if not routes:
        return None
    return sorted(routes, key=lambda route: (route.priority, str(route.edge_id)))[0]


def _first_or_none(values: Sequence[int | str]) -> int | str | None:
    return values[0] if values else None


def _source_feature_ids(routes: Sequence[RouteRecord]) -> tuple[Any, ...]:
    feature_ids = []
    for route in sorted(routes, key=lambda item: (item.priority, str(item.edge_id))):
        route_feature_ids = route.diagnostics.get("source_feature_ids", ())
        if route_feature_ids:
            feature_ids.extend(route_feature_ids)
        else:
            feature_ids.append(route.edge_id)
    return tuple(dict.fromkeys(feature_ids))


def _node_xy(node: GridNode | tuple[int, int]) -> tuple[int, int]:
    if isinstance(node, GridNode):
        return node.xidx, node.yidx
    return node


def _intersection_specs_by_process(routes: Sequence[RouteRecord]) -> dict[tuple[ProcessKind, int], _IntersectionSpec]:
    endpoints_by_key: dict[tuple[ProcessKind, int], list[_IntersectionEndpoint]] = {}
    for route in routes:
        if len(route.nodes) < 2 or not route.tile_cells:
            continue
        _record_route_endpoint_candidate(
            endpoints_by_key,
            process=route.process,
            node_id=route.start_node_id,
            node=route.nodes[0],
            direction=_start_endpoint_direction(route),
            cells_from_node=route.tile_cells,
            cm_type=route.cm_type,
        )
        _record_route_endpoint_candidate(
            endpoints_by_key,
            process=route.process,
            node_id=route.end_node_id,
            node=route.nodes[-1],
            direction=_end_endpoint_direction(route),
            cells_from_node=tuple(reversed(route.tile_cells)),
            cm_type=route.cm_type,
        )

    intersections: dict[tuple[ProcessKind, int], _IntersectionSpec] = {}
    for key, endpoints in endpoints_by_key.items():
        if not endpoints:
            continue
        intersection_cell = _intersection_cell_for_endpoints(endpoints)
        valid_endpoints = tuple(
            endpoint
            for endpoint in endpoints
            if _arm_continues_in_next_square(intersection_cell, endpoint.direction, endpoint.cells_from_node)
        )
        if not valid_endpoints:
            continue
        spec = _IntersectionSpec(
            process=key[0],
            node_id=key[1],
            node=valid_endpoints[0].node,
            directions=set(),
            cells={intersection_cell},
            cm_type=_intersection_cm_type(valid_endpoints),
        )
        for endpoint in valid_endpoints:
            spec.directions.add(endpoint.direction)
        intersections[key] = spec
    return intersections


def _start_endpoint_direction(route: RouteRecord) -> str:
    if len(route.tile_cells) >= 2:
        direction = _direction_between_cells(route.tile_cells[0], route.tile_cells[1])
        if direction is not None:
            return direction
    return _direction_between_nodes(route.nodes[0], route.nodes[1])


def _end_endpoint_direction(route: RouteRecord) -> str:
    if len(route.tile_cells) >= 2:
        direction = _direction_between_cells(route.tile_cells[-1], route.tile_cells[-2])
        if direction is not None:
            return direction
    return _direction_between_nodes(route.nodes[-1], route.nodes[-2])


def _record_route_endpoint_candidate(
    endpoints_by_key: dict[tuple[ProcessKind, int], list[_IntersectionEndpoint]],
    *,
    process: ProcessKind,
    node_id: int,
    node: GridNode,
    direction: str,
    cells_from_node: tuple[GridCell, ...],
    cm_type: CMType | None,
) -> None:
    endpoints_by_key.setdefault((process, node_id), []).append(
        _IntersectionEndpoint(
            process=process,
            node_id=node_id,
            node=node,
            direction=direction,
            cells_from_node=cells_from_node,
            cm_type=cm_type,
        )
    )


def _intersection_cm_type(endpoints: Sequence[_IntersectionEndpoint]) -> CMType | None:
    cm_types = [endpoint.cm_type for endpoint in endpoints if endpoint.cm_type is not None]
    if not cm_types:
        return None
    counts: dict[tuple[str, str, str | None, str | int | None], int] = {}
    by_key = {}
    for cm_type in cm_types:
        key = _cm_type_key(cm_type)
        counts[key] = counts.get(key, 0) + 1
        by_key.setdefault(key, cm_type)
    best_key = min(
        counts,
        key=lambda key: (
            -counts[key],
            _ROAD_SURFACE_RANK.get(key[1], 99),
            tuple("" if value is None else str(value) for value in key),
        ),
    )
    return by_key[best_key]


_ROAD_SURFACE_RANK = {
    "Paved 1": 0,
    "Paved 2": 1,
    "Gravel Road": 2,
    "Dirt Road": 3,
}


def _arm_continues_in_next_square(
    intersection_cell: GridCell,
    direction: str,
    cells_from_node: Iterable[GridCell],
) -> bool:
    return _next_cell(intersection_cell, direction) in set(cells_from_node)


def _intersection_cell_for_endpoints(endpoints: Sequence[_IntersectionEndpoint]) -> GridCell:
    node = endpoints[0].node
    incident_cells = {cell for endpoint in endpoints for cell in endpoint.cells_from_node}
    preferred = GridCell(node.xidx, node.yidx)
    if preferred in incident_cells:
        return preferred

    cells = tuple(sorted(incident_cells, key=lambda cell: (cell.xidx, cell.yidx)))
    if not cells:
        return preferred

    return max(
        cells,
        key=lambda cell: (
            sum(
                1
                for endpoint in endpoints
                if _arm_continues_in_next_square(cell, endpoint.direction, endpoint.cells_from_node)
            ),
            -abs(cell.xidx - node.xidx),
            -abs(cell.yidx - node.yidx),
            cell.xidx,
            cell.yidx,
        ),
    )


def _intersection_cell_for_node(node: GridNode, incident_cells: Iterable[GridCell]) -> GridCell:
    preferred = GridCell(node.xidx, node.yidx)
    cells = tuple(sorted(incident_cells, key=lambda cell: (cell.xidx, cell.yidx)))
    return preferred if preferred in cells or not cells else cells[-1]


def _direction_between_cells(first: GridCell, second: GridCell) -> str | None:
    return shared_direction_between_cells(first, second)


def _next_cell(cell: GridCell, direction: str) -> GridCell:
    return shared_next_cell(cell, direction)


def compatible_neighbor(tile_a: TileVariant, dir_a_to_b: str, tile_b: TileVariant) -> bool:
    opposite = _OPPOSITE_DIRECTIONS.get(dir_a_to_b)
    if opposite is None:
        return False
    if dir_a_to_b not in tile_a.open_directions or opposite not in tile_b.open_directions:
        return False

    # Catalog tuple/list values are atomic connector identities, not sets of allowed
    # tokens: a connector like (2, 3) can connect only to the same normalized
    # signature (2, 3), not to (3, 2) or a wider tuple containing 2 or 3.
    return tile_a.side_signatures.get(dir_a_to_b) == tile_b.side_signatures.get(opposite)


def _variants_connect(first: TileVariant, second: TileVariant, direction: str) -> bool:
    return compatible_neighbor(first, direction, second)


def _layer_for_process(process: ProcessKind) -> LayerKind:
    if process in {ProcessKind.FENCE, ProcessKind.LINEAR, ProcessKind.RAIL}:
        return LayerKind.LINEAR_OBJECT
    return LayerKind.LINEAR_SURFACE


def _failure(
    process: ProcessKind,
    cell: GridCell,
    required_directions: Iterable[str],
    reason: str,
    *,
    hard_failure: bool = False,
    route_id: int | str | None = None,
    route_ids: Sequence[int | str] = (),
    **extra: Any,
) -> Mapping[str, Any]:
    failure = {
        "process": process.value,
        "cell": (cell.xidx, cell.yidx),
        "required_directions": _ordered_directions(required_directions),
        "failure_reason": reason,
    }
    if hard_failure:
        failure["hard_failure"] = True
    if route_id is not None:
        failure["route_id"] = route_id
    if route_ids:
        failure["route_ids"] = tuple(route_ids)
    failure.update(extra)
    return failure


def _ordered_directions(directions: Iterable[str]) -> tuple[str, ...]:
    return ordered_directions(directions)


def _placement_role(required_directions: Iterable[str]) -> str:
    directions = frozenset(required_directions)
    if len(directions) <= 1:
        return "dead_end"
    if len(directions) == 2:
        return "straight" if directions in {frozenset({"N", "S"}), frozenset({"E", "W"})} else "bend"
    if len(directions) == 3:
        return "t_junction"
    if len(directions) == 4:
        return "intersection"
    return "linear"


def _variant_sort_key(variant: TileVariant) -> tuple[float, int, int, int, int]:
    return (
        variant.cost,
        -1 if variant.catalog_direction is None else variant.catalog_direction,
        variant.row,
        variant.col,
        variant.variant,
    )


def _intersection_sort_key(intersection: _IntersectionSpec) -> tuple[str, int, int, int]:
    return intersection.process.value, intersection.node.xidx, intersection.node.yidx, intersection.node_id


def _cm_type_key(cm_type: CMType) -> tuple[str, str, str | None, str | int | None]:
    return cm_type.menu, cm_type.cat1, cm_type.cat2, cm_type.direction
