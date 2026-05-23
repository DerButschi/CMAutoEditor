from __future__ import annotations

import heapq
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shapely.geometry import LineString
from terrain_extraction.osm_extraction.anchor_selection import AnchorSelector, anchor_cell_for_plan
from terrain_extraction.osm_extraction.direction_resolution import (
    CARDINAL_DIRECTIONS,
    DIRECTION_STEPS,
    OPPOSITE_DIRECTIONS,
    supports_diagonal_directions,
)
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.linear_network_state import (
    LinearNetworkState,
    LinearReservationResult,
)
from terrain_extraction.osm_extraction.linear_processing_plan import (
    LinearInteractionPolicy,
    LinearProcessingGroup,
    LinearProcessingPlan,
    default_linear_interaction_policy,
)
from terrain_extraction.osm_extraction.models import (
    CMType,
    GridCell,
    GridKind,
    GridNode,
    LayerKind,
    NetworkRoutingResult,
    PlacementRecord,
    ProcessKind,
    RasterSpine,
    RouteRecord,
    TopologyEdge,
    TopologyGraph,
)
from terrain_extraction.osm_extraction.occupancy import OccupancyModel
from terrain_extraction.osm_extraction.raster_spine import build_raster_spine

_MOVE_ORDER = ("N", "E", "S", "W", "NE", "NW", "SE", "SW")
_DIRECTION_STEPS = DIRECTION_STEPS
_OPPOSITE_DIRECTIONS = OPPOSITE_DIRECTIONS
_NETWORK_CLASS_RANK = {
    "motorway": 0,
    "trunk": 0,
    "primary": 1,
    "secondary": 2,
    "tertiary": 3,
    "residential": 4,
    "unclassified": 4,
    "service": 5,
    "track": 6,
    "path": 7,
    "footway": 7,
}
_MINOR_CLASSES = frozenset({"service", "track", "path", "footway", "cycleway", "bridleway"})


@dataclass(frozen=True, slots=True)
class MoveStep:
    dx: int
    dy: int
    direction: str


@dataclass(frozen=True, slots=True)
class CompiledMoveSet:
    steps: tuple[MoveStep, ...]

    @classmethod
    def cardinal(cls) -> CompiledMoveSet:
        return cls.from_directions(CARDINAL_DIRECTIONS)

    @classmethod
    def from_directions(cls, directions: Iterable[str]) -> CompiledMoveSet:
        normalized = frozenset(direction.upper() for direction in directions).intersection(_DIRECTION_STEPS)
        if not normalized:
            normalized = CARDINAL_DIRECTIONS
        return cls(
            tuple(
                MoveStep(dx=dx, dy=dy, direction=direction)
                for direction in _MOVE_ORDER
                if direction in normalized
                for dx, dy in (_DIRECTION_STEPS[direction],)
            )
        )

    @classmethod
    def from_tile_catalog(cls, catalog: Iterable[CMType | Mapping[str, Any]]) -> CompiledMoveSet:
        directions = set()
        for entry in catalog:
            value = entry.direction if isinstance(entry, CMType) else entry.get("direction")
            if value is None:
                continue
            directions.update(_normalize_direction_token(value))
        if not directions:
            return cls.cardinal()
        return cls.from_directions(directions)

    @classmethod
    def for_process_catalog(cls, process: ProcessKind, catalog: Any | None) -> CompiledMoveSet:
        supported = _catalog_supported_step_dirs(catalog)
        if not supported:
            return cls.cardinal()
        if process is ProcessKind.ROAD and not supports_diagonal_directions(supported):
            return cls.from_directions(supported.intersection(CARDINAL_DIRECTIONS))
        return cls.from_directions(supported)


@dataclass(frozen=True, slots=True)
class _RouteAttempt:
    nodes: tuple[GridNode, ...]
    blocked_cells_considered: int
    soft_crossings: int = 0
    hard_blocked_occupied_cells: int = 0
    soft_avoid_cells: int = 0
    tile_feasible_rejections: int = 0
    tile_feasible_failures: tuple[Mapping[str, Any], ...] = ()
    a_star_expansions: int = 0

    @property
    def success(self) -> bool:
        return bool(self.nodes)


class NetworkRouter:
    def __init__(
        self,
        *,
        grid_index: GridIndex,
        occupancy: OccupancyModel | None = None,
        move_set: CompiledMoveSet | None = None,
        corridor_deviation_m: float = 32.0,
        minor_relaxation_m: float = 48.0,
        allow_soft_crossing: bool = False,
        split_long_edge_m: float = 256.0,
        catalogs: Mapping[ProcessKind, Any] | None = None,
        linear_state: LinearNetworkState | None = None,
        interaction_policy: LinearInteractionPolicy | None = None,
    ) -> None:
        if corridor_deviation_m < 0:
            raise ValueError("corridor_deviation_m must be non-negative")
        if minor_relaxation_m < corridor_deviation_m:
            raise ValueError("minor_relaxation_m must be greater than or equal to corridor_deviation_m")
        self.grid_index = grid_index
        self.occupancy = occupancy or OccupancyModel.from_grid_index(grid_index)
        self.move_set = move_set
        self.corridor_deviation_m = corridor_deviation_m
        self.minor_relaxation_m = minor_relaxation_m
        self.allow_soft_crossing = allow_soft_crossing
        self.split_long_edge_m = split_long_edge_m
        self.catalogs = dict(catalogs or {})
        self.process_move_sets = {
            process: CompiledMoveSet.for_process_catalog(process, catalog)
            for process, catalog in self.catalogs.items()
        }
        self.linear_state = linear_state
        self.interaction_policy = interaction_policy or default_linear_interaction_policy()

    def route(self, topology: TopologyGraph) -> NetworkRoutingResult:
        processing_plan = LinearProcessingPlan.from_topology(
            topology,
            interaction_policy=self.interaction_policy,
        )
        linear_state = self.linear_state or LinearNetworkState(
            width=self.grid_index.width,
            height=self.grid_index.height,
            catalogs=self.catalogs,
            interaction_policy=processing_plan.interaction_policy,
        )
        self.linear_state = linear_state
        anchors = {}
        anchor_plans = {}
        anchor_diagnostics: list[Mapping[str, Any]] = []
        routes = []
        for group in processing_plan.groups:
            group_topology = group.topology(topology.nodes)
            anchor_selection = AnchorSelector(
                grid_index=self.grid_index,
                occupancy=self.occupancy,
                catalogs=self.catalogs,
            ).select(group_topology)
            group_anchors = self._node_anchors(group_topology, anchor_selection.plans)
            group_degrees = {node.node_id: group_topology.degree(node.node_id) for node in group_topology.nodes}
            anchors.update(group_anchors)
            anchor_plans.update(anchor_selection.plans)
            anchor_diagnostics.append(anchor_selection.diagnostics)
            ordered_edges = self._fallback_route_order(self._route_order(group.edges, group_degrees), anchor_selection.plans)
            for edge in ordered_edges:
                dropped_route = self._anchor_fallback_dropped_route(edge, anchor_selection.plans)
                if dropped_route is not None:
                    routes.append(dropped_route)
                    continue
                route = self._route_edge(edge, group_anchors, group_degrees, anchor_selection.plans)
                route = self._with_processing_diagnostics(route, group)
                if route.success:
                    reservation = linear_state.reserve_path(route)
                    if not reservation.success:
                        route = self._reservation_failed_route(route, reservation)
                routes.append(route)

        diagnostics = self._diagnostics(routes)
        diagnostics.update(_combined_anchor_diagnostics(anchor_diagnostics))
        diagnostics.update(linear_state.diagnostics())
        diagnostics.update(processing_plan.diagnostics)
        return NetworkRoutingResult(
            routes=tuple(routes),
            node_anchors=anchors,
            anchor_plans=anchor_plans,
            diagnostics=diagnostics,
            linear_state=linear_state,
        )

    def _route_edge(
        self,
        edge: TopologyEdge,
        anchors: Mapping[int, GridNode],
        degrees: Mapping[int, int],
        anchor_plans: Mapping[int, Any],
    ) -> RouteRecord:
        route_started_at = time.perf_counter()
        start = self._anchor_for_edge_node(edge, edge.start_node_id, anchors, anchor_plans)
        goal = self._anchor_for_edge_node(edge, edge.end_node_id, anchors, anchor_plans)
        raster_spine = build_raster_spine(
            topology_edge_id=edge.edge_id,
            line=edge.geometry,
            grid_index=self.grid_index,
        )
        attempts: list[tuple[str | None, float, bool]] = [(None, self.corridor_deviation_m, False)]
        if self._is_minor(edge) and self.minor_relaxation_m > self.corridor_deviation_m:
            attempts.append(("minor_corridor", self.minor_relaxation_m, False))
        if self.allow_soft_crossing:
            attempts.append(("soft_crossing", attempts[-1][1], True))

        total_blocked = 0
        total_hard_blocked = 0
        total_soft_avoid = 0
        total_tile_rejections = 0
        total_expansions = 0
        attempt_count = 0
        tile_failures: list[Mapping[str, Any]] = []
        retry_modes: list[str] = []
        for relaxation, corridor_m, soft_crossing in attempts:
            attempt_count += 1
            retry_mode = _retry_mode(relaxation)
            if retry_mode is not None:
                retry_modes.append(retry_mode)
            attempt = self._a_star(
                edge.geometry,
                start,
                goal,
                corridor_m,
                self._layer_for_edge(edge),
                soft_crossing,
                raster_spine,
                edge.process,
                _edge_top_level_name(edge),
                edge.priority,
            )
            total_blocked += attempt.blocked_cells_considered
            total_hard_blocked += attempt.hard_blocked_occupied_cells
            total_soft_avoid += attempt.soft_avoid_cells
            total_tile_rejections += attempt.tile_feasible_rejections
            total_expansions += attempt.a_star_expansions
            tile_failures.extend(dict(failure) for failure in attempt.tile_feasible_failures)
            if attempt.success:
                route = self._record_success(
                    edge,
                    attempt.nodes,
                    relaxation,
                    total_blocked,
                    attempt.soft_crossings,
                    degrees,
                    raster_spine,
                    hard_blocked_occupied_cells=total_hard_blocked,
                    soft_avoid_cells=total_soft_avoid,
                    retry_modes=tuple(retry_modes),
                    attempt_count=attempt_count,
                    retry_count=len(retry_modes),
                    a_star_expansions=total_expansions,
                    tile_feasible_rejections=total_tile_rejections,
                    tile_feasible_failures=tuple(tile_failures),
                )
                return _route_with_elapsed(route, route_started_at)

        if edge.geometry.length > self.split_long_edge_m:
            split_record = self._try_split_route(edge, start, goal, degrees, raster_spine)
            if split_record is not None:
                diagnostics = {
                    **dict(split_record.diagnostics),
                    "attempt_count": attempt_count + int(split_record.diagnostics.get("attempt_count", 0)),
                    "retry_count": len(retry_modes) + int(split_record.diagnostics.get("retry_count", 0)),
                    "a_star_expansions": total_expansions
                    + int(split_record.diagnostics.get("a_star_expansions", 0)),
                    "tile_feasible_rejections": total_tile_rejections
                    + int(split_record.diagnostics.get("tile_feasible_rejections", 0)),
                }
                return _route_with_elapsed(_route_with_diagnostics(split_record, diagnostics), route_started_at)

        failure_reason = "no_tile_feasible_path" if total_tile_rejections else "no_path"
        route = RouteRecord(
            edge_id=edge.edge_id,
            start_node_id=edge.start_node_id,
            end_node_id=edge.end_node_id,
            process=edge.process,
            config_name=edge.config_name,
            priority=edge.priority,
            raster_spine=raster_spine,
            success=False,
            cm_type=edge.cm_type,
            linear_authority=edge.linear_authority,
            diagnostics={
                **_authority_diagnostics(edge),
                "failure_reason": failure_reason,
                "conflict_family": _conflict_family(tile_failures),
                "source_length_m": edge.geometry.length,
                "raster_spine_cell_count": len(raster_spine.cells),
                "blocked_cells_considered": total_blocked,
                "hard_blocked_occupied_cells": total_hard_blocked,
                "soft_avoid_cells": total_soft_avoid,
                "false_intersection_avoided": bool(edge.diagnostics.get("false_intersection_avoided", False)),
                "corridor_deviation_m": self.corridor_deviation_m,
                "tile_feasible_rejections": total_tile_rejections,
                "tile_feasible_failures": tuple(tile_failures),
                "attempt_count": attempt_count,
                "retry_count": len(retry_modes),
                "a_star_expansions": total_expansions,
                "retry_modes": tuple(retry_modes),
                "source_feature_ids": edge.feature_ids,
                "source_indices": edge.source_indices,
            },
        )
        return _route_with_elapsed(route, route_started_at)

    def _a_star(  # noqa: PLR0915
        self,
        line: LineString,
        start: GridNode,
        goal: GridNode,
        corridor_m: float,
        layer: LayerKind,
        allow_soft_crossing: bool,
        raster_spine: RasterSpine,
        process: ProcessKind,
        top_level_name: str,
        priority: int,
    ) -> _RouteAttempt:
        if start == goal:
            return _RouteAttempt(nodes=(start,), blocked_cells_considered=0)

        window = self._node_window(line, corridor_m)
        move_set = self._move_set_for_process(process)
        start_state = (start.xidx, start.yidx, "")
        open_heap = [(self._heuristic(start, goal, move_set), 0.0, start_state)]
        best_cost = {start_state: 0.0}
        came_from: dict[tuple[int, int, str], tuple[int, int, str] | None] = {start_state: None}
        distance_cache: dict[GridNode, float] = {}
        blocked_cache: dict[GridCell, bool] = {}
        blocked_cells_considered = 0
        soft_crossings = 0
        hard_blocked_occupied_cells = 0
        soft_avoid_cells = 0
        tile_feasible_rejections = 0
        tile_feasible_failures: list[Mapping[str, Any]] = []
        a_star_expansions = 0

        while open_heap:
            _estimated, cost_so_far, state = heapq.heappop(open_heap)
            current = GridNode(state[0], state[1])
            incoming_direction = state[2]
            if current == goal:
                return _RouteAttempt(
                    nodes=self._reconstruct_path(came_from, state),
                    blocked_cells_considered=blocked_cells_considered,
                    soft_crossings=soft_crossings,
                    hard_blocked_occupied_cells=hard_blocked_occupied_cells,
                    soft_avoid_cells=soft_avoid_cells,
                    tile_feasible_rejections=tile_feasible_rejections,
                    tile_feasible_failures=tuple(tile_feasible_failures),
                    a_star_expansions=a_star_expansions,
                )
            if cost_so_far > best_cost[state]:
                continue
            a_star_expansions += 1

            for step in move_set.steps:
                neighbor = GridNode(current.xidx + step.dx, current.yidx + step.dy)
                if not self._node_in_bounds(neighbor):
                    continue
                if neighbor != goal and neighbor != start and not self._cached_node_in_corridor(
                    neighbor,
                    line,
                    corridor_m,
                    window,
                    distance_cache,
                ):
                    continue

                step_decision = self._route_step_decision(
                    came_from=came_from,
                    state=state,
                    current=current,
                    neighbor=neighbor,
                    start=start,
                    goal=goal,
                    incoming_direction=incoming_direction,
                    step_direction=step.direction,
                    process=process,
                    top_level_name=top_level_name,
                    priority=priority,
                )
                if not step_decision.allowed:
                    if step_decision.failures:
                        tile_feasible_rejections += 1
                        hard_blocked_occupied_cells += _hard_blocked_failure_count(step_decision.failures)
                        soft_avoid_cells += _soft_avoid_failure_count(step_decision.failures)
                        _extend_failures(tile_feasible_failures, step_decision.failures)
                    continue
                traversed_cell = self._cell_for_step(current, neighbor)
                blocked = self._cached_cell_is_blocked(traversed_cell, layer, blocked_cache)
                if blocked:
                    blocked_cells_considered += 1
                    if not allow_soft_crossing:
                        hard_blocked_occupied_cells += 1
                        continue
                    soft_crossings += 1
                    soft_avoid_cells += 1

                turn_cost = 0.15 if incoming_direction and incoming_direction != step.direction else 0.0
                distance_cost = (
                    self._cached_node_source_distance(neighbor, line, distance_cache)
                    / max(self.grid_index.cell_size_m, 1.0)
                    * 0.1
                )
                spine_cost = self._spine_alignment_cost(traversed_cell, raster_spine)
                soft_cost = 25.0 if blocked else 0.0
                next_cost = cost_so_far + 1.0 + turn_cost + distance_cost + spine_cost + soft_cost
                next_state = (neighbor.xidx, neighbor.yidx, step.direction)
                if next_cost >= best_cost.get(next_state, math.inf):
                    continue
                best_cost[next_state] = next_cost
                came_from[next_state] = state
                heapq.heappush(open_heap, (next_cost + self._heuristic(neighbor, goal, move_set), next_cost, next_state))

        return _RouteAttempt(
            nodes=(),
            blocked_cells_considered=blocked_cells_considered,
            soft_crossings=soft_crossings,
            hard_blocked_occupied_cells=hard_blocked_occupied_cells,
            soft_avoid_cells=soft_avoid_cells,
            tile_feasible_rejections=tile_feasible_rejections,
            tile_feasible_failures=tuple(tile_feasible_failures),
            a_star_expansions=a_star_expansions,
        )

    def _try_split_route(
        self,
        edge: TopologyEdge,
        start: GridNode,
        goal: GridNode,
        degrees: Mapping[int, int],
        raster_spine: RasterSpine,
    ) -> RouteRecord | None:
        midpoint = edge.geometry.interpolate(0.5, normalized=True)
        midpoint_node = self._clamp_node(self.grid_index.projected_to_cell(midpoint.x, midpoint.y))
        if midpoint_node in {start, goal}:
            return None
        first = self._a_star(
            edge.geometry,
            start,
            midpoint_node,
            self.minor_relaxation_m,
            self._layer_for_edge(edge),
            True,
            raster_spine,
            edge.process,
            _edge_top_level_name(edge),
            edge.priority,
        )
        second = self._a_star(
            edge.geometry,
            midpoint_node,
            goal,
            self.minor_relaxation_m,
            self._layer_for_edge(edge),
            True,
            raster_spine,
            edge.process,
            _edge_top_level_name(edge),
            edge.priority,
        )
        if not first.success or not second.success:
            return None
        nodes = first.nodes + second.nodes[1:]
        record = self._record_success(
            edge,
            nodes,
            "split_long_edge",
            first.blocked_cells_considered + second.blocked_cells_considered,
            first.soft_crossings + second.soft_crossings,
            degrees,
            raster_spine,
            hard_blocked_occupied_cells=first.hard_blocked_occupied_cells + second.hard_blocked_occupied_cells,
            soft_avoid_cells=first.soft_avoid_cells + second.soft_avoid_cells,
            retry_modes=("midpoint_split",),
            attempt_count=2,
            retry_count=1,
            a_star_expansions=first.a_star_expansions + second.a_star_expansions,
            tile_feasible_rejections=first.tile_feasible_rejections + second.tile_feasible_rejections,
            tile_feasible_failures=first.tile_feasible_failures + second.tile_feasible_failures,
        )
        diagnostics = {**dict(record.diagnostics), "split_intersections": 1}
        return RouteRecord(
            edge_id=record.edge_id,
            start_node_id=record.start_node_id,
            end_node_id=record.end_node_id,
            process=record.process,
            config_name=record.config_name,
            priority=record.priority,
            nodes=record.nodes,
            tile_cells=record.tile_cells,
            raster_spine=record.raster_spine,
            success=True,
            diagnostics=diagnostics,
            cm_type=record.cm_type,
            linear_authority=record.linear_authority,
        )

    def _record_success(
        self,
        edge: TopologyEdge,
        nodes: tuple[GridNode, ...],
        relaxation: str | None,
        blocked_cells_considered: int,
        soft_crossings: int,
        degrees: Mapping[int, int],
        raster_spine: RasterSpine,
        *,
        hard_blocked_occupied_cells: int = 0,
        soft_avoid_cells: int = 0,
        retry_modes: tuple[str, ...] = (),
        attempt_count: int = 1,
        retry_count: int = 0,
        a_star_expansions: int = 0,
        tile_feasible_rejections: int = 0,
        tile_feasible_failures: tuple[Mapping[str, Any], ...] = (),
    ) -> RouteRecord:
        tile_cells = tuple(GridCell(node.xidx, node.yidx) for node in nodes)
        route_length = max(0, len(nodes) - 1) * self.grid_index.cell_size_m
        source_length = edge.geometry.length
        spine_diagnostics = self._spine_diagnostics(edge.geometry, nodes, tile_cells, raster_spine)
        diagnostics = {
            **_authority_diagnostics(edge),
            "source_length_m": source_length,
            "route_length_m": route_length,
            "detour_ratio": route_length / source_length if source_length > 0 else 1.0,
            "mean_source_line_distance_m": self._mean_node_source_distance(nodes, edge.geometry),
            "max_source_line_distance_m": max((self._node_source_distance(node, edge.geometry) for node in nodes), default=0.0),
            "blocked_cells_considered": blocked_cells_considered,
            "hard_blocked_occupied_cells": hard_blocked_occupied_cells,
            "soft_avoid_cells": soft_avoid_cells,
            "conflict_family": _conflict_family(tile_feasible_failures),
            "false_intersection_avoided": bool(edge.diagnostics.get("false_intersection_avoided", False)),
            "forced_relaxation": relaxation,
            "soft_crossings": soft_crossings,
            "intersection_importance": max(degrees.get(edge.start_node_id, 0), degrees.get(edge.end_node_id, 0)),
            "retry_modes": retry_modes,
            "attempt_count": attempt_count,
            "retry_count": retry_count,
            "a_star_expansions": a_star_expansions,
            "tile_feasible_rejections": tile_feasible_rejections,
            "tile_feasible_failures": tile_feasible_failures,
            "source_feature_ids": edge.feature_ids,
            "source_indices": edge.source_indices,
            **spine_diagnostics,
        }
        return RouteRecord(
            edge_id=edge.edge_id,
            start_node_id=edge.start_node_id,
            end_node_id=edge.end_node_id,
            process=edge.process,
            config_name=edge.config_name,
            priority=edge.priority,
            nodes=nodes,
            tile_cells=tile_cells,
            raster_spine=raster_spine,
            success=True,
            diagnostics=diagnostics,
            cm_type=edge.cm_type,
            linear_authority=edge.linear_authority,
        )

    def _reservation_failed_route(
        self,
        route: RouteRecord,
        reservation: LinearReservationResult,
    ) -> RouteRecord:
        failure = dict(reservation.failures[0]) if reservation.failures else {"failure_reason": "linear_state_rejected"}
        diagnostics = {
            **dict(route.diagnostics),
            "failure_reason": failure.get("failure_reason", "linear_state_rejected"),
            "conflict_family": _conflict_family(reservation.failures),
            "hard_blocked_occupied_cells": int(route.diagnostics.get("hard_blocked_occupied_cells", 0))
            + _hard_blocked_failure_count(reservation.failures),
            "soft_avoid_cells": int(route.diagnostics.get("soft_avoid_cells", 0))
            + _soft_avoid_failure_count(reservation.failures),
            "linear_state_failures": tuple(dict(item) for item in reservation.failures),
            "tile_feasible_failures": tuple(dict(item) for item in reservation.failures),
            "tile_feasible_rejections": int(route.diagnostics.get("tile_feasible_rejections", 0))
            + len(reservation.failures),
        }
        return RouteRecord(
            edge_id=route.edge_id,
            start_node_id=route.start_node_id,
            end_node_id=route.end_node_id,
            process=route.process,
            config_name=route.config_name,
            priority=route.priority,
            nodes=route.nodes,
            tile_cells=route.tile_cells,
            raster_spine=route.raster_spine,
            success=False,
            diagnostics=diagnostics,
            cm_type=route.cm_type,
            linear_authority=route.linear_authority,
        )

    def _with_processing_diagnostics(
        self,
        route: RouteRecord,
        group: LinearProcessingGroup,
    ) -> RouteRecord:
        diagnostics = {
            **dict(route.diagnostics),
            "processing_group": group.group_index,
            "processing_stage": group.stage,
            "processing_process": group.process.value,
            "processing_config_name": group.config_name,
            "processing_priority": group.priority,
            "processing_rank": group.rank,
        }
        return RouteRecord(
            edge_id=route.edge_id,
            start_node_id=route.start_node_id,
            end_node_id=route.end_node_id,
            process=route.process,
            config_name=route.config_name,
            priority=route.priority,
            nodes=route.nodes,
            tile_cells=route.tile_cells,
            raster_spine=route.raster_spine,
            success=route.success,
            diagnostics=diagnostics,
            cm_type=route.cm_type,
            linear_authority=route.linear_authority,
        )

    def _node_anchors(self, topology: TopologyGraph, anchor_plans: Mapping[int, Any]) -> dict[int, GridNode]:
        anchors = {}
        for node in topology.nodes:
            plan = anchor_plans.get(node.node_id)
            cell = (
                anchor_cell_for_plan(plan)
                if plan is not None
                else self._clamp_node(self.grid_index.projected_to_cell(node.point.x, node.point.y))
            )
            anchors[node.node_id] = GridNode(cell.xidx, cell.yidx)
        return anchors

    def _anchor_for_edge_node(
        self,
        edge: TopologyEdge,
        node_id: int,
        anchors: Mapping[int, GridNode],
        anchor_plans: Mapping[int, Any],
    ) -> GridNode:
        plan = anchor_plans.get(node_id)
        cell = _split_anchor_cell_for_edge(plan, edge, node_id)
        if cell is not None:
            return GridNode(cell.xidx, cell.yidx)
        return anchors[node_id]

    def _route_order(
        self,
        edges: Sequence[TopologyEdge],
        degrees: Mapping[int, int],
    ) -> tuple[TopologyEdge, ...]:
        return tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.priority,
                    _NETWORK_CLASS_RANK.get(edge.config_name, 99),
                    *_authority_sort_key(edge),
                    -edge.geometry.length,
                    -max(degrees.get(edge.start_node_id, 0), degrees.get(edge.end_node_id, 0)),
                    edge.edge_id,
                ),
            )
        )

    def _fallback_route_order(
        self,
        edges: Sequence[TopologyEdge],
        anchor_plans: Mapping[int, Any],
    ) -> tuple[TopologyEdge, ...]:
        edge_positions = {edge.edge_id: index for index, edge in enumerate(edges)}
        return tuple(
            sorted(
                edges,
                key=lambda edge: (
                    _fallback_edge_rank(edge, anchor_plans),
                    edge_positions[edge.edge_id],
                ),
            )
        )

    def _anchor_fallback_dropped_route(
        self,
        edge: TopologyEdge,
        anchor_plans: Mapping[int, Any],
    ) -> RouteRecord | None:
        decision = _anchor_fallback_drop_decision(edge, anchor_plans)
        if decision is None:
            return None
        return RouteRecord(
            edge_id=edge.edge_id,
            start_node_id=edge.start_node_id,
            end_node_id=edge.end_node_id,
            process=edge.process,
            config_name=edge.config_name,
            priority=edge.priority,
            success=False,
            diagnostics={
                **_authority_diagnostics(edge),
                "failure_reason": "anchor_fallback_drop",
                "conflict_family": "same_family",
                "hard_blocked_occupied_cells": 0,
                "soft_avoid_cells": 0,
                "false_intersection_avoided": bool(edge.diagnostics.get("false_intersection_avoided", False)),
                "intersection_fallback_decision": decision,
                "elapsed_ms": 0.0,
                "attempt_count": 0,
                "retry_count": 0,
                "a_star_expansions": 0,
                "tile_feasible_rejections": 0,
                "source_feature_ids": edge.feature_ids,
                "source_indices": edge.source_indices,
            },
            cm_type=edge.cm_type,
            linear_authority=edge.linear_authority,
        )

    def _node_window(self, line: LineString, corridor_m: float) -> tuple[int, int, int, int]:
        min_x, min_y, max_x, max_y = line.bounds
        local_points = [
            self.grid_index.local_from_projected(x, y)
            for x, y in ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))
        ]
        min_x = min(point[0] for point in local_points) - corridor_m
        min_y = min(point[1] for point in local_points) - corridor_m
        max_x = max(point[0] for point in local_points) + corridor_m
        max_y = max(point[1] for point in local_points) + corridor_m
        cell_size = self.grid_index.cell_size_m
        return (
            max(0, math.floor(min_x / cell_size)),
            max(0, math.floor(min_y / cell_size)),
            min(self.grid_index.width - 1, math.ceil(max_x / cell_size)),
            min(self.grid_index.height - 1, math.ceil(max_y / cell_size)),
        )

    def _node_in_corridor(
        self,
        node: GridNode,
        line: LineString,
        corridor_m: float,
        window: tuple[int, int, int, int],
    ) -> bool:
        min_xidx, min_yidx, max_xidx, max_yidx = window
        if not (min_xidx <= node.xidx <= max_xidx and min_yidx <= node.yidx <= max_yidx):
            return False
        return self._node_source_distance(node, line) <= corridor_m + 1e-9

    def _cached_node_in_corridor(
        self,
        node: GridNode,
        line: LineString,
        corridor_m: float,
        window: tuple[int, int, int, int],
        distance_cache: dict[GridNode, float],
    ) -> bool:
        min_xidx, min_yidx, max_xidx, max_yidx = window
        if not (min_xidx <= node.xidx <= max_xidx and min_yidx <= node.yidx <= max_yidx):
            return False
        return self._cached_node_source_distance(node, line, distance_cache) <= corridor_m + 1e-9

    def _node_source_distance(self, node: GridNode, line: LineString) -> float:
        point = self.grid_index.cell_center(GridCell(node.xidx, node.yidx))
        return point.distance(line)

    def _cached_node_source_distance(
        self,
        node: GridNode,
        line: LineString,
        distance_cache: dict[GridNode, float],
    ) -> float:
        distance = distance_cache.get(node)
        if distance is None:
            distance = self._node_source_distance(node, line)
            distance_cache[node] = distance
        return distance

    def _mean_node_source_distance(self, nodes: Sequence[GridNode], line: LineString) -> float:
        if not nodes:
            return 0.0
        return sum(self._node_source_distance(node, line) for node in nodes) / len(nodes)

    def _spine_alignment_cost(self, cell: GridCell, raster_spine: RasterSpine) -> float:
        if not raster_spine.cells:
            return 0.0
        if cell in raster_spine.cells:
            return -0.25
        nearest_cell_steps = min(
            abs(cell.xidx - spine_cell.xidx) + abs(cell.yidx - spine_cell.yidx) for spine_cell in raster_spine.cells
        )
        return nearest_cell_steps * 0.2

    def _route_cell_decision(
        self,
        cell: GridCell,
        *,
        incoming_dir: str | None,
        outgoing_dir: str | None,
        process: ProcessKind,
        top_level_name: str,
        priority: int,
        allow_lower_priority_connection: bool = False,
    ) -> Any:
        if self.linear_state is None:
            return _AllowedCellDecision()
        return self.linear_state.can_enter_cell(
            cell,
            incoming_dir=incoming_dir,
            outgoing_dir=outgoing_dir,
            process=process,
            top_level_name=top_level_name,
            priority=priority,
            allow_lower_priority_connection=allow_lower_priority_connection,
        )

    def _route_step_decision(
        self,
        *,
        came_from: Mapping[tuple[int, int, str], tuple[int, int, str] | None],
        state: tuple[int, int, str],
        current: GridNode,
        neighbor: GridNode,
        start: GridNode,
        goal: GridNode,
        incoming_direction: str,
        step_direction: str,
        process: ProcessKind,
        top_level_name: str,
        priority: int,
    ) -> Any:
        if _state_path_contains(came_from, state, neighbor):
            return _AllowedCellDecision(allowed=False)
        current_decision = self._route_cell_decision(
            GridCell(current.xidx, current.yidx),
            incoming_dir=_opposite(incoming_direction),
            outgoing_dir=step_direction,
            process=process,
            top_level_name=top_level_name,
            priority=priority,
            allow_lower_priority_connection=current in {start, goal},
        )
        if not current_decision.allowed:
            return current_decision
        return self._route_cell_decision(
            self._cell_for_step(current, neighbor),
            incoming_dir=_opposite(step_direction),
            outgoing_dir=None,
            process=process,
            top_level_name=top_level_name,
            priority=priority,
            allow_lower_priority_connection=neighbor in {start, goal},
        )

    def _spine_diagnostics(
        self,
        line: LineString,
        nodes: Sequence[GridNode],
        tile_cells: Sequence[GridCell],
        raster_spine: RasterSpine,
    ) -> dict[str, Any]:
        route_cell_set = set(tile_cells)
        spine_cell_set = set(raster_spine.cells)
        skipped = tuple(cell for cell in raster_spine.cells if cell not in route_cell_set)
        extra = tuple(cell for cell in tile_cells if cell not in spine_cell_set)
        route_to_spine = [self._cell_spine_distance(cell, raster_spine) for cell in tile_cells]
        return {
            "raster_spine_cell_count": len(raster_spine.cells),
            "mean_spine_distance_m": sum(route_to_spine) / len(route_to_spine) if route_to_spine else 0.0,
            "max_spine_distance_m": max(route_to_spine, default=0.0),
            "skipped_spine_cells": skipped,
            "extra_detour_cells": extra,
            "source_spine_length_m": raster_spine.source_length_m,
            "mean_source_line_distance_m": self._mean_node_source_distance(nodes, line),
        }

    def _cell_spine_distance(self, cell: GridCell, raster_spine: RasterSpine) -> float:
        if not raster_spine.cells:
            return 0.0
        point = self.grid_index.cell_center(cell)
        return min(point.distance(self.grid_index.cell_center(spine_cell)) for spine_cell in raster_spine.cells)

    def _cell_for_step(self, _start: GridNode, end: GridNode) -> GridCell:
        return GridCell(
            min(max(end.xidx, 0), self.grid_index.width - 1),
            min(max(end.yidx, 0), self.grid_index.height - 1),
        )

    def _cell_is_blocked(self, cell: GridCell, layer: LayerKind) -> bool:
        placement = PlacementRecord(
            layer=layer,
            grid_kind=GridKind.NORMAL,
            cells=(cell,),
            config_name="network_route_probe",
            feature_id=None,
            priority=0,
            cm_type=CMType(menu="Route", cat1="Route"),
            score=1.0,
        )
        return not self.occupancy.can_place(placement).allowed

    def _cached_cell_is_blocked(
        self,
        cell: GridCell,
        layer: LayerKind,
        blocked_cache: dict[GridCell, bool],
    ) -> bool:
        blocked = blocked_cache.get(cell)
        if blocked is None:
            blocked = self._cell_is_blocked(cell, layer)
            blocked_cache[cell] = blocked
        return blocked

    def _node_in_bounds(self, node: GridNode) -> bool:
        return 0 <= node.xidx < self.grid_index.width and 0 <= node.yidx < self.grid_index.height

    def _clamp_node(self, node: GridNode) -> GridNode:
        return GridNode(
            min(max(node.xidx, 0), self.grid_index.width - 1),
            min(max(node.yidx, 0), self.grid_index.height - 1),
        )

    def _layer_for_edge(self, edge: TopologyEdge) -> LayerKind:
        if edge.process in {ProcessKind.FENCE, ProcessKind.LINEAR, ProcessKind.RAIL}:
            return LayerKind.LINEAR_OBJECT
        return LayerKind.LINEAR_SURFACE

    def _is_minor(self, edge: TopologyEdge) -> bool:
        return edge.config_name in _MINOR_CLASSES or edge.priority >= 7

    def _heuristic(self, current: GridNode, goal: GridNode, move_set: CompiledMoveSet) -> float:
        dx = abs(current.xidx - goal.xidx)
        dy = abs(current.yidx - goal.yidx)
        if any(abs(step.dx) == 1 and abs(step.dy) == 1 for step in move_set.steps):
            return max(dx, dy)
        return dx + dy

    def _move_set_for_process(self, process: ProcessKind) -> CompiledMoveSet:
        if self.move_set is not None:
            return self.move_set
        return self.process_move_sets.get(process, CompiledMoveSet.cardinal())

    def _reconstruct_path(
        self,
        came_from: Mapping[tuple[int, int, str], tuple[int, int, str] | None],
        state: tuple[int, int, str],
    ) -> tuple[GridNode, ...]:
        states = [state]
        while came_from[states[-1]] is not None:
            states.append(came_from[states[-1]])
        states.reverse()
        return tuple(GridNode(xidx, yidx) for xidx, yidx, _direction in states)

    def _diagnostics(self, routes: Sequence[RouteRecord]) -> dict[str, Any]:
        successful = [route for route in routes if route.success]
        failed = [route for route in routes if not route.success]
        detours = [float(route.diagnostics["detour_ratio"]) for route in successful]
        distances = [float(route.diagnostics["max_source_line_distance_m"]) for route in successful]
        spine_distances = [float(route.diagnostics["max_spine_distance_m"]) for route in successful]
        return {
            "successful_routes": len(successful),
            "failed_routes": len(failed),
            "route_count": len(routes),
            "route_attempts": sum(int(route.diagnostics.get("attempt_count", 0)) for route in routes),
            "route_retries": sum(int(route.diagnostics.get("retry_count", 0)) for route in routes),
            "total_a_star_expansions": sum(int(route.diagnostics.get("a_star_expansions", 0)) for route in routes),
        "total_tile_feasible_rejections": sum(
                int(route.diagnostics.get("tile_feasible_rejections", 0))
                for route in routes
            ),
            "hard_blocked_occupied_cells": sum(
                int(route.diagnostics.get("hard_blocked_occupied_cells", 0)) for route in routes
            ),
            "soft_avoid_cells": sum(int(route.diagnostics.get("soft_avoid_cells", 0)) for route in routes),
            "false_intersections_avoided": sum(1 for route in routes if route.diagnostics.get("false_intersection_avoided")),
            "mean_detour_ratio": sum(detours) / len(detours) if detours else None,
            "max_source_line_distance_m": max(distances) if distances else None,
            "max_spine_distance_m": max(spine_distances) if spine_distances else None,
            "raster_spines": sum(1 for route in routes if route.raster_spine is not None),
            "raster_spine_cells": sum(len(route.raster_spine.cells) for route in routes if route.raster_spine is not None),
            "forced_relaxations": sum(1 for route in successful if route.diagnostics.get("forced_relaxation")),
            "soft_crossings": sum(int(route.diagnostics.get("soft_crossings", 0)) for route in successful),
        }


@dataclass(frozen=True, slots=True)
class _AllowedCellDecision:
    allowed: bool = True
    failures: tuple[Mapping[str, Any], ...] = ()


def _route_with_elapsed(route: RouteRecord, started_at: float) -> RouteRecord:
    diagnostics = {
        **dict(route.diagnostics),
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
    }
    return _route_with_diagnostics(route, diagnostics)


def _route_with_diagnostics(route: RouteRecord, diagnostics: Mapping[str, Any]) -> RouteRecord:
    return RouteRecord(
        edge_id=route.edge_id,
        start_node_id=route.start_node_id,
        end_node_id=route.end_node_id,
        process=route.process,
        config_name=route.config_name,
        priority=route.priority,
        nodes=route.nodes,
        tile_cells=route.tile_cells,
        raster_spine=route.raster_spine,
        success=route.success,
        diagnostics=dict(diagnostics),
        cm_type=route.cm_type,
        linear_authority=route.linear_authority,
    )


def _opposite(direction: str | None) -> str | None:
    return None if not direction else _OPPOSITE_DIRECTIONS[direction]


def _catalog_supported_step_dirs(catalog: Any | None) -> frozenset[str]:
    if catalog is None:
        return frozenset()
    if hasattr(catalog, "allowed_step_dirs"):
        return frozenset(str(direction).upper() for direction in catalog.allowed_step_dirs()).intersection(_DIRECTION_STEPS)
    try:
        move_set = CompiledMoveSet.from_tile_catalog(catalog)
    except TypeError:
        return frozenset()
    return frozenset(step.direction for step in move_set.steps)


def _extend_failures(
    collected: list[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    *,
    limit: int = 20,
) -> None:
    if len(collected) >= limit:
        return
    remaining = limit - len(collected)
    collected.extend(dict(failure) for failure in tuple(failures)[:remaining])


def _authority_diagnostics(edge: TopologyEdge) -> dict[str, Any]:
    if edge.linear_authority is not None:
        return dict(edge.linear_authority.as_diagnostics())
    return {
        "top_level_name": edge.process.value,
        "process": edge.process.value,
        "config_priority": edge.priority,
        "cm_type_index": None,
        "tag_rank": None,
        "source_length_m": edge.geometry.length,
        "logical_chain_length_m": edge.geometry.length,
        "stable_source_order": min(edge.source_indices, default=edge.edge_id),
        "source_feature_id": edge.feature_ids[0] if edge.feature_ids else None,
    }


def _edge_top_level_name(edge: TopologyEdge) -> str:
    if edge.linear_authority is not None:
        return edge.linear_authority.top_level_name
    return edge.process.value


def _authority_sort_key(edge: TopologyEdge) -> tuple[int, int, float, int, str]:
    authority = edge.linear_authority
    if authority is None:
        return (99, 99, -edge.geometry.length, min(edge.source_indices, default=edge.edge_id), str(edge.edge_id))
    return (
        99 if authority.cm_type_index is None else authority.cm_type_index,
        99 if authority.first_matching_tag_index is None else authority.first_matching_tag_index,
        -float(authority.logical_chain_length_m or authority.source_feature_length_m),
        authority.stable_source_order,
        str(authority.source_feature_id),
    )


def _conflict_family(failures: Iterable[Mapping[str, Any]]) -> str | None:
    reasons = {str(failure.get("failure_reason")) for failure in failures}
    if reasons.intersection({"process_avoidance", "process_conflict"}):
        return "cross_family"
    if reasons:
        return "same_family"
    return None


def _hard_blocked_failure_count(failures: Iterable[Mapping[str, Any]]) -> int:
    return sum(
        1
        for failure in failures
        if failure.get("failure_reason") not in {"process_avoidance"}
    )


def _soft_avoid_failure_count(failures: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for failure in failures if failure.get("failure_reason") == "process_avoidance")


def _combined_anchor_diagnostics(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        "anchor_nodes": 0,
        "single_anchor_plans": 0,
        "split_anchor_plans": 0,
        "failed_anchor_plans": 0,
        "intersection_fallbacks": 0,
        "intersection_fallback_attached_arms": 0,
        "intersection_fallback_dropped_arms": 0,
        "anchor_candidates": 0,
        "anchor_retry_nodes": 0,
    }
    for diagnostics in items:
        for key in totals:
            totals[key] += int(diagnostics.get(key, 0))
    return totals


def _retry_mode(relaxation: str | None) -> str | None:
    if relaxation == "minor_corridor":
        return "corridor_widening"
    if relaxation == "soft_crossing":
        return "soft_crossing"
    return None


def _state_path_contains(
    came_from: Mapping[tuple[int, int, str], tuple[int, int, str] | None],
    state: tuple[int, int, str],
    node: GridNode,
) -> bool:
    current: tuple[int, int, str] | None = state
    while current is not None:
        if current[0] == node.xidx and current[1] == node.yidx:
            return True
        current = came_from[current]
    return False


def _split_anchor_cell_for_edge(plan: Any, edge: TopologyEdge, node_id: int) -> GridCell | None:
    if getattr(plan, "plan_kind", None) != "split":
        return None
    edge_anchor_cells = getattr(plan, "edge_anchor_cells", None) or {}
    if edge.edge_id in edge_anchor_cells:
        return edge_anchor_cells[edge.edge_id]
    direction = _edge_direction_from_node(edge, node_id)
    if direction is None:
        return getattr(plan, "primary_cell", None)
    split_cells = tuple(getattr(plan, "split_anchor_cells", ()) or ())
    for index, direction_set in enumerate(getattr(plan, "split_direction_sets", ()) or ()):
        if direction in direction_set and index < len(split_cells):
            return split_cells[index]
    return getattr(plan, "primary_cell", None)


def _fallback_edge_rank(edge: TopologyEdge, anchor_plans: Mapping[int, Any]) -> int:
    actions = tuple(_fallback_actions_for_edge(edge, anchor_plans))
    if "preserve" in actions:
        return 0
    if "attach" in actions:
        return 1
    if "drop" in actions:
        return 3
    return 2


def _fallback_actions_for_edge(edge: TopologyEdge, anchor_plans: Mapping[int, Any]) -> tuple[str, ...]:
    actions = []
    for node_id in (edge.start_node_id, edge.end_node_id):
        plan = anchor_plans.get(node_id)
        for decision in getattr(plan, "fallback_decisions", ()) or ():
            if decision.get("edge_id") == edge.edge_id:
                actions.append(str(decision.get("action")))
    return tuple(actions)


def _anchor_fallback_drop_decision(edge: TopologyEdge, anchor_plans: Mapping[int, Any]) -> Mapping[str, Any] | None:
    for node_id in (edge.start_node_id, edge.end_node_id):
        plan = anchor_plans.get(node_id)
        for decision in getattr(plan, "fallback_decisions", ()) or ():
            if decision.get("edge_id") == edge.edge_id and decision.get("action") == "drop":
                return {
                    "node_id": node_id,
                    "edge_id": edge.edge_id,
                    **dict(decision),
                }
    return None


def _edge_direction_from_node(edge: TopologyEdge, node_id: int) -> str | None:
    coords = tuple(edge.geometry.coords)
    if len(coords) < 2:
        return None
    if edge.start_node_id == node_id:
        start, end = coords[0], coords[1]
    elif edge.end_node_id == node_id:
        start, end = coords[-1], coords[-2]
    else:
        return None
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.isclose(dx, 0.0) and math.isclose(dy, 0.0):
        return None
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def _normalize_direction_token(value: Any) -> set[str]:
    if isinstance(value, int):
        return set(CARDINAL_DIRECTIONS)
    text = str(value).upper()
    aliases = {
        "NORTH": "N",
        "EAST": "E",
        "SOUTH": "S",
        "WEST": "W",
    }
    directions = set()
    for token in text.replace("-", "_").replace("+", "_").split("_"):
        direction = aliases.get(token, token)
        if direction in _DIRECTION_STEPS:
            directions.add(direction)
    return directions
