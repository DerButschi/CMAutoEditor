from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from shapely.geometry import LineString
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.linear_network_state import (
    LinearNetworkState,
    LinearReservationResult,
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

_DIRECTION_STEPS = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
}
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
        return cls(
            tuple(
                MoveStep(dx=dx, dy=dy, direction=direction)
                for direction, (dx, dy) in (("N", (0, 1)), ("E", (1, 0)), ("S", (0, -1)), ("W", (-1, 0)))
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
        return cls(tuple(MoveStep(*_DIRECTION_STEPS[direction], direction) for direction in "NESW" if direction in directions))


@dataclass(frozen=True, slots=True)
class _RouteAttempt:
    nodes: tuple[GridNode, ...]
    blocked_cells_considered: int
    soft_crossings: int = 0

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
    ) -> None:
        if corridor_deviation_m < 0:
            raise ValueError("corridor_deviation_m must be non-negative")
        if minor_relaxation_m < corridor_deviation_m:
            raise ValueError("minor_relaxation_m must be greater than or equal to corridor_deviation_m")
        self.grid_index = grid_index
        self.occupancy = occupancy or OccupancyModel.from_grid_index(grid_index)
        self.move_set = move_set or CompiledMoveSet.cardinal()
        self.corridor_deviation_m = corridor_deviation_m
        self.minor_relaxation_m = minor_relaxation_m
        self.allow_soft_crossing = allow_soft_crossing
        self.split_long_edge_m = split_long_edge_m
        self.catalogs = dict(catalogs or {})
        self.linear_state = linear_state

    def route(self, topology: TopologyGraph) -> NetworkRoutingResult:
        anchors = self._node_anchors(topology)
        degrees = {node.node_id: topology.degree(node.node_id) for node in topology.nodes}
        linear_state = self.linear_state or LinearNetworkState(
            width=self.grid_index.width,
            height=self.grid_index.height,
            catalogs=self.catalogs,
        )
        self.linear_state = linear_state
        routes = []
        for edge in self._route_order(topology.edges, degrees):
            route = self._route_edge(edge, anchors, degrees)
            if route.success:
                reservation = linear_state.reserve_path(route)
                if not reservation.success:
                    route = self._reservation_failed_route(route, reservation)
            routes.append(route)

        diagnostics = self._diagnostics(routes)
        diagnostics.update(linear_state.diagnostics())
        return NetworkRoutingResult(
            routes=tuple(routes),
            node_anchors=anchors,
            diagnostics=diagnostics,
            linear_state=linear_state,
        )

    def _route_edge(
        self,
        edge: TopologyEdge,
        anchors: Mapping[int, GridNode],
        degrees: Mapping[int, int],
    ) -> RouteRecord:
        start = anchors[edge.start_node_id]
        goal = anchors[edge.end_node_id]
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
        for relaxation, corridor_m, soft_crossing in attempts:
            attempt = self._a_star(
                edge.geometry,
                start,
                goal,
                corridor_m,
                self._layer_for_edge(edge),
                soft_crossing,
                raster_spine,
            )
            total_blocked += attempt.blocked_cells_considered
            if attempt.success:
                return self._record_success(
                    edge,
                    attempt.nodes,
                    relaxation,
                    total_blocked,
                    attempt.soft_crossings,
                    degrees,
                    raster_spine,
                )

        if edge.geometry.length > self.split_long_edge_m:
            split_record = self._try_split_route(edge, start, goal, degrees, raster_spine)
            if split_record is not None:
                return split_record

        return RouteRecord(
            edge_id=edge.edge_id,
            start_node_id=edge.start_node_id,
            end_node_id=edge.end_node_id,
            process=edge.process,
            config_name=edge.config_name,
            priority=edge.priority,
            raster_spine=raster_spine,
            success=False,
            cm_type=edge.cm_type,
            diagnostics={
                "failure_reason": "no_path",
                "source_length_m": edge.geometry.length,
                "raster_spine_cell_count": len(raster_spine.cells),
                "blocked_cells_considered": total_blocked,
                "corridor_deviation_m": self.corridor_deviation_m,
            },
        )

    def _a_star(
        self,
        line: LineString,
        start: GridNode,
        goal: GridNode,
        corridor_m: float,
        layer: LayerKind,
        allow_soft_crossing: bool,
        raster_spine: RasterSpine,
    ) -> _RouteAttempt:
        if start == goal:
            return _RouteAttempt(nodes=(start,), blocked_cells_considered=0)

        window = self._node_window(line, corridor_m)
        start_state = (start.xidx, start.yidx, "")
        open_heap = [(self._heuristic(start, goal), 0.0, start_state)]
        best_cost = {start_state: 0.0}
        came_from: dict[tuple[int, int, str], tuple[int, int, str] | None] = {start_state: None}
        distance_cache: dict[GridNode, float] = {}
        blocked_cache: dict[GridCell, bool] = {}
        blocked_cells_considered = 0
        soft_crossings = 0

        while open_heap:
            _estimated, cost_so_far, state = heapq.heappop(open_heap)
            current = GridNode(state[0], state[1])
            incoming_direction = state[2]
            if current == goal:
                return _RouteAttempt(
                    nodes=self._reconstruct_path(came_from, state),
                    blocked_cells_considered=blocked_cells_considered,
                    soft_crossings=soft_crossings,
                )
            if cost_so_far > best_cost[state]:
                continue

            for step in self.move_set.steps:
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

                traversed_cell = self._cell_for_step(current, neighbor)
                blocked = self._cached_cell_is_blocked(traversed_cell, layer, blocked_cache)
                if blocked:
                    blocked_cells_considered += 1
                    if not allow_soft_crossing:
                        continue
                    soft_crossings += 1

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
                heapq.heappush(open_heap, (next_cost + self._heuristic(neighbor, goal), next_cost, next_state))

        return _RouteAttempt(nodes=(), blocked_cells_considered=blocked_cells_considered, soft_crossings=soft_crossings)

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
        )
        second = self._a_star(
            edge.geometry,
            midpoint_node,
            goal,
            self.minor_relaxation_m,
            self._layer_for_edge(edge),
            True,
            raster_spine,
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
    ) -> RouteRecord:
        tile_cells = tuple(GridCell(node.xidx, node.yidx) for node in nodes)
        route_length = max(0, len(nodes) - 1) * self.grid_index.cell_size_m
        source_length = edge.geometry.length
        spine_diagnostics = self._spine_diagnostics(edge.geometry, nodes, tile_cells, raster_spine)
        diagnostics = {
            "source_length_m": source_length,
            "route_length_m": route_length,
            "detour_ratio": route_length / source_length if source_length > 0 else 1.0,
            "mean_source_line_distance_m": self._mean_node_source_distance(nodes, edge.geometry),
            "max_source_line_distance_m": max((self._node_source_distance(node, edge.geometry) for node in nodes), default=0.0),
            "blocked_cells_considered": blocked_cells_considered,
            "forced_relaxation": relaxation,
            "soft_crossings": soft_crossings,
            "intersection_importance": max(degrees.get(edge.start_node_id, 0), degrees.get(edge.end_node_id, 0)),
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
            "linear_state_failures": tuple(dict(item) for item in reservation.failures),
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
        )

    def _node_anchors(self, topology: TopologyGraph) -> dict[int, GridNode]:
        return {
            node.node_id: self._clamp_node(self.grid_index.projected_to_cell(node.point.x, node.point.y))
            for node in topology.nodes
        }

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
                    -edge.geometry.length,
                    -max(degrees.get(edge.start_node_id, 0), degrees.get(edge.end_node_id, 0)),
                    edge.edge_id,
                ),
            )
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

    def _heuristic(self, current: GridNode, goal: GridNode) -> float:
        return abs(current.xidx - goal.xidx) + abs(current.yidx - goal.yidx)

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
            "mean_detour_ratio": sum(detours) / len(detours) if detours else None,
            "max_source_line_distance_m": max(distances) if distances else None,
            "max_spine_distance_m": max(spine_distances) if spine_distances else None,
            "raster_spines": sum(1 for route in routes if route.raster_spine is not None),
            "raster_spine_cells": sum(len(route.raster_spine.cells) for route in routes if route.raster_spine is not None),
            "forced_relaxations": sum(1 for route in successful if route.diagnostics.get("forced_relaxation")),
            "soft_crossings": sum(int(route.diagnostics.get("soft_crossings", 0)) for route in successful),
        }


def _normalize_direction_token(value: Any) -> set[str]:
    if isinstance(value, int):
        return set(_DIRECTION_STEPS)
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
