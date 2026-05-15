from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import (
    CMType,
    GridCell,
    GridKind,
    LayerKind,
    PlacementRecord,
    ProcessKind,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
)
from terrain_extraction.osm_extraction.occupancy import OccupancyModel

_DIRECTION_ORDER = {"E": 0, "N": 1, "S": 2, "W": 3}


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    topology_node_id: int
    cell: GridCell
    score: float
    required_dirs_estimate: frozenset[str]
    tile_feasible: bool
    occupancy_feasible: bool
    reasons: tuple[str, ...] = ()
    search_radius: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_dirs_estimate", frozenset(self.required_dirs_estimate))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class SingleAnchorPlan:
    topology_node_id: int
    anchor_cell: GridCell
    selected_candidate: AnchorCandidate
    candidates: tuple[AnchorCandidate, ...]
    plan_kind: str = "single"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True, slots=True)
class SplitAnchorPlan:
    topology_node_id: int
    primary_cell: GridCell
    split_anchor_cells: tuple[GridCell, ...]
    required_dirs_estimate: frozenset[str]
    split_direction_sets: tuple[tuple[str, ...], ...]
    candidates: tuple[AnchorCandidate, ...]
    reason: str
    plan_kind: str = "split"

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_anchor_cells", tuple(self.split_anchor_cells))
        object.__setattr__(self, "required_dirs_estimate", frozenset(self.required_dirs_estimate))
        object.__setattr__(self, "split_direction_sets", tuple(tuple(item) for item in self.split_direction_sets))
        object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True, slots=True)
class FailedAnchorPlan:
    topology_node_id: int
    fallback_cell: GridCell
    required_dirs_estimate: frozenset[str]
    candidates: tuple[AnchorCandidate, ...]
    reason: str
    plan_kind: str = "failed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_dirs_estimate", frozenset(self.required_dirs_estimate))
        object.__setattr__(self, "candidates", tuple(self.candidates))


AnchorPlan = SingleAnchorPlan | SplitAnchorPlan | FailedAnchorPlan


@dataclass(frozen=True, slots=True)
class AnchorSelectionResult:
    plans: Mapping[int, AnchorPlan] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plans", MappingProxyType(dict(self.plans)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def candidates(self) -> tuple[AnchorCandidate, ...]:
        return tuple(candidate for plan in self.plans.values() for candidate in plan.candidates)


class AnchorSelector:
    def __init__(
        self,
        *,
        grid_index: GridIndex,
        occupancy: OccupancyModel | None = None,
        catalogs: Mapping[ProcessKind, Any] | None = None,
        initial_radius: int = 1,
        retry_radii: Iterable[int] = (2, 3),
    ) -> None:
        if initial_radius < 0:
            raise ValueError("initial_radius must be non-negative")
        self.grid_index = grid_index
        self.occupancy = occupancy
        self.catalogs = MappingProxyType(dict(catalogs or {}))
        self.radii = (initial_radius, *tuple(radius for radius in retry_radii if radius > initial_radius))

    def select(self, topology: TopologyGraph) -> AnchorSelectionResult:
        plans = {node.node_id: self._plan_node(node, topology.incident_edges(node.node_id)) for node in topology.nodes}
        diagnostics = {
            "anchor_nodes": len(plans),
            "single_anchor_plans": sum(isinstance(plan, SingleAnchorPlan) for plan in plans.values()),
            "split_anchor_plans": sum(isinstance(plan, SplitAnchorPlan) for plan in plans.values()),
            "failed_anchor_plans": sum(isinstance(plan, FailedAnchorPlan) for plan in plans.values()),
            "anchor_candidates": sum(len(plan.candidates) for plan in plans.values()),
            "anchor_retry_nodes": sum(_selected_radius(plan) > self.radii[0] for plan in plans.values()),
        }
        return AnchorSelectionResult(plans=plans, diagnostics=diagnostics)

    def _plan_node(self, node: TopologyNode, incident_edges: tuple[TopologyEdge, ...]) -> AnchorPlan:
        base_cell = self._clamp_cell(self.grid_index.projected_to_cell(node.point.x, node.point.y))
        required_dirs = frozenset(_ordered_directions(self._required_dirs_estimate(node, incident_edges)))
        process = _primary_process(incident_edges)
        collected: list[AnchorCandidate] = []

        for radius in self.radii:
            candidates = tuple(
                self._candidate(
                    node=node,
                    cell=cell,
                    base_cell=base_cell,
                    required_dirs=required_dirs,
                    process=process,
                    incident_edges=incident_edges,
                    radius=radius,
                )
                for cell in self._cells_at_radius(base_cell, radius)
            )
            collected.extend(candidates)
            feasible = [candidate for candidate in candidates if candidate.tile_feasible and candidate.occupancy_feasible]
            if feasible:
                selected = min(feasible, key=lambda candidate: (candidate.score, candidate.cell.yidx, candidate.cell.xidx))
                return SingleAnchorPlan(
                    topology_node_id=node.node_id,
                    anchor_cell=selected.cell,
                    selected_candidate=selected,
                    candidates=tuple(collected),
                )

            if candidates and not any(candidate.tile_feasible for candidate in candidates):
                split = self._split_plan(node.node_id, base_cell, required_dirs, process, tuple(collected))
                if split is not None:
                    return split

        if collected:
            fallback = min(collected, key=lambda candidate: (candidate.score, candidate.cell.yidx, candidate.cell.xidx))
            return FailedAnchorPlan(
                topology_node_id=node.node_id,
                fallback_cell=fallback.cell,
                required_dirs_estimate=required_dirs,
                candidates=tuple(collected),
                reason="no_tile_and_occupancy_feasible_candidate",
            )
        return FailedAnchorPlan(
            topology_node_id=node.node_id,
            fallback_cell=base_cell,
            required_dirs_estimate=required_dirs,
            candidates=(),
            reason="no_anchor_candidates",
        )

    def _candidate(
        self,
        *,
        node: TopologyNode,
        cell: GridCell,
        base_cell: GridCell,
        required_dirs: frozenset[str],
        process: ProcessKind,
        incident_edges: tuple[TopologyEdge, ...],
        radius: int,
    ) -> AnchorCandidate:
        tile_feasible = self._tile_feasible(process, required_dirs)
        occupancy_feasible = self._occupancy_feasible(cell, process, _min_priority(incident_edges))
        reasons = []
        if not tile_feasible:
            reasons.append("catalog_gap")
        if not occupancy_feasible:
            reasons.append("occupancy_conflict")
        return AnchorCandidate(
            topology_node_id=node.node_id,
            cell=cell,
            score=self._score(node, cell, base_cell),
            required_dirs_estimate=required_dirs,
            tile_feasible=tile_feasible,
            occupancy_feasible=occupancy_feasible,
            reasons=tuple(reasons),
            search_radius=radius,
        )

    def _split_plan(
        self,
        node_id: int,
        base_cell: GridCell,
        required_dirs: frozenset[str],
        process: ProcessKind,
        candidates: tuple[AnchorCandidate, ...],
    ) -> SplitAnchorPlan | None:
        direction_sets = self._split_direction_sets(required_dirs, process)
        if not direction_sets:
            return None
        anchor_cells = tuple(candidate.cell for candidate in sorted(candidates, key=lambda item: item.score)[: len(direction_sets)])
        if not anchor_cells:
            anchor_cells = (base_cell,)
        return SplitAnchorPlan(
            topology_node_id=node_id,
            primary_cell=anchor_cells[0],
            split_anchor_cells=anchor_cells,
            required_dirs_estimate=required_dirs,
            split_direction_sets=direction_sets,
            candidates=candidates,
            reason="single_anchor_catalog_gap",
        )

    def _split_direction_sets(self, required_dirs: frozenset[str], process: ProcessKind) -> tuple[tuple[str, ...], ...]:
        catalog = self.catalogs.get(process)
        if catalog is None or len(required_dirs) < 3:
            return ()
        preferred = []
        for pair in (("E", "W"), ("N", "S")):
            direction_set = frozenset(pair)
            if direction_set.issubset(required_dirs) and catalog.has_tile(direction_set):
                preferred.append(pair)
        covered = frozenset(direction for pair in preferred for direction in pair)
        if covered == required_dirs:
            return tuple(preferred)
        return ()

    def _tile_feasible(self, process: ProcessKind, required_dirs: frozenset[str]) -> bool:
        if len(required_dirs) < 2:
            return True
        catalog = self.catalogs.get(process)
        return catalog is None or bool(catalog.has_tile(required_dirs))

    def _occupancy_feasible(self, cell: GridCell, process: ProcessKind, priority: int) -> bool:
        if self.occupancy is None:
            return True
        placement = PlacementRecord(
            layer=_layer_for_process(process),
            grid_kind=GridKind.NORMAL,
            cells=(cell,),
            config_name="anchor_probe",
            feature_id=None,
            priority=priority,
            cm_type=CMType(menu="Anchor", cat1="Anchor"),
            score=1.0,
        )
        return self.occupancy.can_place(placement).allowed

    def _score(self, node: TopologyNode, cell: GridCell, base_cell: GridCell) -> float:
        center = self.grid_index.cell_center(cell)
        source_distance = center.distance(node.point) / max(self.grid_index.cell_size_m, 1.0)
        movement = max(abs(cell.xidx - base_cell.xidx), abs(cell.yidx - base_cell.yidx))
        base_bonus = -0.001 if cell == base_cell else 0.0
        return source_distance + movement * 0.2 + base_bonus

    def _required_dirs_estimate(self, node: TopologyNode, incident_edges: tuple[TopologyEdge, ...]) -> tuple[str, ...]:
        directions = []
        for edge in incident_edges:
            direction = _edge_direction_from_node(node.node_id, edge)
            if direction:
                directions.append(direction)
        return tuple(directions)

    def _cells_at_radius(self, base_cell: GridCell, radius: int) -> tuple[GridCell, ...]:
        cells = []
        for yidx in range(base_cell.yidx - radius, base_cell.yidx + radius + 1):
            for xidx in range(base_cell.xidx - radius, base_cell.xidx + radius + 1):
                if max(abs(xidx - base_cell.xidx), abs(yidx - base_cell.yidx)) > radius:
                    continue
                cell = GridCell(xidx, yidx)
                if self._cell_in_bounds(cell):
                    cells.append(cell)
        return tuple(cells)

    def _cell_in_bounds(self, cell: GridCell) -> bool:
        return 0 <= cell.xidx < self.grid_index.width and 0 <= cell.yidx < self.grid_index.height

    def _clamp_cell(self, cell: GridCell) -> GridCell:
        return GridCell(
            min(max(cell.xidx, 0), self.grid_index.width - 1),
            min(max(cell.yidx, 0), self.grid_index.height - 1),
        )


def anchor_cell_for_plan(plan: AnchorPlan) -> GridCell:
    if isinstance(plan, SingleAnchorPlan):
        return plan.anchor_cell
    if isinstance(plan, SplitAnchorPlan):
        return plan.primary_cell
    return plan.fallback_cell


def _edge_direction_from_node(node_id: int, edge: TopologyEdge) -> str | None:
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


def _primary_process(edges: tuple[TopologyEdge, ...]) -> ProcessKind:
    if not edges:
        return ProcessKind.ROAD
    return min(edges, key=lambda edge: (edge.priority, edge.edge_id)).process


def _min_priority(edges: tuple[TopologyEdge, ...]) -> int:
    return min((edge.priority for edge in edges), default=0)


def _layer_for_process(process: ProcessKind) -> LayerKind:
    if process in {ProcessKind.FENCE, ProcessKind.LINEAR, ProcessKind.RAIL}:
        return LayerKind.LINEAR_OBJECT
    return LayerKind.LINEAR_SURFACE


def _selected_radius(plan: AnchorPlan) -> int:
    if isinstance(plan, SingleAnchorPlan):
        return plan.selected_candidate.search_radius
    return min((candidate.search_radius for candidate in plan.candidates), default=0)


def _ordered_directions(directions: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(frozenset(directions), key=lambda direction: _DIRECTION_ORDER.get(direction, 99)))
