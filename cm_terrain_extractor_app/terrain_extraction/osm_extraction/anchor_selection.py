from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from terrain_extraction.osm_extraction.direction_resolution import (
    CARDINAL_DIRECTIONS,
    DIAGONAL_DIRECTIONS,
    DIRECTION_ORDER,
    OPPOSITE_DIRECTIONS,
    directions_are_opposite,
    ordered_directions,
    supports_diagonal_directions,
)
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
from terrain_extraction.osm_extraction.raster_spine import build_raster_spine

_DIRECTION_ORDER = DIRECTION_ORDER
_OPPOSITE_DIRECTIONS = OPPOSITE_DIRECTIONS
_MAJOR_ROAD_CLASSES = frozenset({"motorway", "trunk", "primary", "secondary"})
_ROAD_CLASS_RANK = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3}


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    topology_node_id: int
    cell: GridCell
    score: float
    required_dirs_estimate: frozenset[str]
    required_dirs: frozenset[str]
    tile_feasible: bool
    occupancy_feasible: bool
    impossible_arm_count: int = 0
    impossible_arm_severity: float = 0.0
    reasons: tuple[str, ...] = ()
    search_radius: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "required_dirs_estimate", frozenset(self.required_dirs_estimate))
        object.__setattr__(self, "required_dirs", frozenset(self.required_dirs))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class _IncidentArm:
    edge_id: int
    support_cells: tuple[GridCell, ...]
    fallback_direction: str | None
    process: ProcessKind
    config_name: str
    priority: int


@dataclass(frozen=True, slots=True)
class _CandidateArmDirections:
    required_dirs: frozenset[str]
    edge_dirs: Mapping[int, str]
    reasons: tuple[str, ...]
    impossible_arm_count: int
    impossible_arm_severity: float


@dataclass(frozen=True, slots=True)
class _PreservedArmPair:
    arms: tuple[_IncidentArm, _IncidentArm]
    directions: frozenset[str]


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
    edge_anchor_cells: Mapping[int, GridCell]
    preserved_direction_set: tuple[str, ...]
    preserved_edge_ids: tuple[int, ...]
    attached_edge_ids: tuple[int, ...]
    dropped_edge_ids: tuple[int, ...]
    fallback_decisions: tuple[Mapping[str, Any], ...]
    candidates: tuple[AnchorCandidate, ...]
    reason: str
    plan_kind: str = "split"

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_anchor_cells", tuple(self.split_anchor_cells))
        object.__setattr__(self, "required_dirs_estimate", frozenset(self.required_dirs_estimate))
        object.__setattr__(self, "split_direction_sets", tuple(tuple(item) for item in self.split_direction_sets))
        object.__setattr__(self, "edge_anchor_cells", MappingProxyType(dict(self.edge_anchor_cells)))
        object.__setattr__(self, "preserved_direction_set", tuple(self.preserved_direction_set))
        object.__setattr__(self, "preserved_edge_ids", tuple(self.preserved_edge_ids))
        object.__setattr__(self, "attached_edge_ids", tuple(self.attached_edge_ids))
        object.__setattr__(self, "dropped_edge_ids", tuple(self.dropped_edge_ids))
        object.__setattr__(self, "fallback_decisions", tuple(MappingProxyType(dict(item)) for item in self.fallback_decisions))
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
            "intersection_fallbacks": sum(
                isinstance(plan, SplitAnchorPlan) and bool(plan.fallback_decisions) for plan in plans.values()
            ),
            "intersection_fallback_attached_arms": sum(
                len(plan.attached_edge_ids) for plan in plans.values() if isinstance(plan, SplitAnchorPlan)
            ),
            "intersection_fallback_dropped_arms": sum(
                len(plan.dropped_edge_ids) for plan in plans.values() if isinstance(plan, SplitAnchorPlan)
            ),
            "anchor_candidates": sum(len(plan.candidates) for plan in plans.values()),
            "anchor_retry_nodes": sum(_selected_radius(plan) > self.radii[0] for plan in plans.values()),
        }
        return AnchorSelectionResult(plans=plans, diagnostics=diagnostics)

    def _plan_node(self, node: TopologyNode, incident_edges: tuple[TopologyEdge, ...]) -> AnchorPlan:
        base_cell = self._clamp_cell(self.grid_index.projected_to_cell(node.point.x, node.point.y))
        required_dirs = frozenset(_ordered_directions(self._required_dirs_estimate(node, incident_edges)))
        incident_arms = tuple(self._incident_arm(node, edge) for edge in incident_edges)
        continuity_pairs = _major_continuity_pairs(incident_arms)
        process = _primary_process(incident_edges)
        estimate_tile_feasible = self._tile_feasible(process, required_dirs)
        collected: list[AnchorCandidate] = []

        for radius in self.radii:
            candidates = tuple(
                self._candidate(
                    node=node,
                    cell=cell,
                    base_cell=base_cell,
                    required_dirs_estimate=required_dirs,
                    process=process,
                    incident_edges=incident_edges,
                    incident_arms=incident_arms,
                    continuity_pairs=continuity_pairs,
                    radius=radius,
                )
                for cell in self._cells_at_radius(base_cell, radius)
            )
            collected.extend(candidates)
            feasible = [candidate for candidate in candidates if _candidate_selectable(candidate)]
            if not feasible and estimate_tile_feasible:
                feasible = [
                    candidate
                    for candidate in candidates
                    if candidate.tile_feasible and candidate.occupancy_feasible
                ]
            if feasible:
                selected = min(feasible, key=lambda candidate: (candidate.score, candidate.cell.yidx, candidate.cell.xidx))
                return SingleAnchorPlan(
                    topology_node_id=node.node_id,
                    anchor_cell=selected.cell,
                    selected_candidate=selected,
                    candidates=tuple(collected),
                )

            if (
                candidates
                and not estimate_tile_feasible
                and not any(_candidate_has_supported_arms(candidate) for candidate in candidates)
            ):
                split = self._split_plan(
                    node.node_id,
                    base_cell,
                    required_dirs,
                    process,
                    incident_arms,
                    tuple(collected),
                )
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
        required_dirs_estimate: frozenset[str],
        process: ProcessKind,
        incident_edges: tuple[TopologyEdge, ...],
        incident_arms: tuple[_IncidentArm, ...],
        continuity_pairs: tuple[tuple[int, int], ...],
        radius: int,
    ) -> AnchorCandidate:
        arm_dirs = self._candidate_arm_directions(
            cell=cell,
            incident_arms=incident_arms,
            fallback_dirs=required_dirs_estimate,
        )
        tile_feasible = self._tile_feasible(process, arm_dirs.required_dirs)
        occupancy_feasible = self._occupancy_feasible(cell, process, _min_priority(incident_edges))
        reasons = list(arm_dirs.reasons)
        if not tile_feasible:
            reasons.append("catalog_gap")
        if not occupancy_feasible:
            reasons.append("occupancy_conflict")
        continuity_score = _major_continuity_score(arm_dirs.edge_dirs, continuity_pairs)
        return AnchorCandidate(
            topology_node_id=node.node_id,
            cell=cell,
            score=self._score(
                node=node,
                cell=cell,
                base_cell=base_cell,
                tile_feasible=tile_feasible,
                impossible_arm_count=arm_dirs.impossible_arm_count,
                impossible_arm_severity=arm_dirs.impossible_arm_severity,
                continuity_score=continuity_score,
            ),
            required_dirs_estimate=required_dirs_estimate,
            required_dirs=arm_dirs.required_dirs,
            tile_feasible=tile_feasible,
            occupancy_feasible=occupancy_feasible,
            impossible_arm_count=arm_dirs.impossible_arm_count,
            impossible_arm_severity=arm_dirs.impossible_arm_severity,
            reasons=tuple(reasons),
            search_radius=radius,
        )

    def _split_plan(
        self,
        node_id: int,
        base_cell: GridCell,
        required_dirs: frozenset[str],
        process: ProcessKind,
        incident_arms: tuple[_IncidentArm, ...],
        candidates: tuple[AnchorCandidate, ...],
    ) -> SplitAnchorPlan | None:
        catalog = self.catalogs.get(process)
        if catalog is None or len(required_dirs) < 3:
            return None
        preserved_pair = _preserved_arm_pair(incident_arms, catalog)
        if preserved_pair is None:
            direction_sets = self._split_direction_sets(required_dirs, process)
            if not direction_sets:
                return None
            anchor_cells = tuple(candidate.cell for candidate in sorted(candidates, key=lambda item: item.score)[: len(direction_sets)])
            if not anchor_cells:
                anchor_cells = (base_cell,)
            edge_anchor_cells = _legacy_edge_anchor_cells(incident_arms, anchor_cells, direction_sets)
            return SplitAnchorPlan(
                topology_node_id=node_id,
                primary_cell=anchor_cells[0],
                split_anchor_cells=anchor_cells,
                required_dirs_estimate=required_dirs,
                split_direction_sets=direction_sets,
                edge_anchor_cells=edge_anchor_cells,
                preserved_direction_set=direction_sets[0],
                preserved_edge_ids=tuple(sorted(edge_anchor_cells)),
                attached_edge_ids=(),
                dropped_edge_ids=(),
                fallback_decisions=(
                    {
                        "action": "split",
                        "cell": _cell_tuple(anchor_cells[0]),
                        "required_directions": direction_sets[0],
                        "reason": "straight_pair_split",
                    },
                ),
                candidates=candidates,
                reason="single_anchor_catalog_gap",
            )

        primary_cell = _split_primary_cell(base_cell, candidates, preserved_pair.directions)
        edge_anchor_cells: dict[int, GridCell] = {}
        cell_direction_sets: dict[GridCell, set[str]] = {primary_cell: set(preserved_pair.directions)}
        fallback_decisions: list[Mapping[str, Any]] = []
        for arm in preserved_pair.arms:
            edge_anchor_cells[arm.edge_id] = primary_cell
            fallback_decisions.append(
                {
                    "action": "preserve",
                    "edge_id": arm.edge_id,
                    "direction": arm.fallback_direction,
                    "cell": _cell_tuple(primary_cell),
                    "required_directions": _ordered_directions(preserved_pair.directions),
                    "reason": "preserved_priority_pair",
                }
            )

        attached_edge_ids = []
        dropped_edge_ids = []
        remaining_arms = tuple(
            sorted(
                (arm for arm in incident_arms if arm.edge_id not in edge_anchor_cells),
                key=_arm_sort_key,
            )
        )
        for arm in remaining_arms:
            attach_cell = _attachment_cell_for_arm(
                arm,
                primary_cell=primary_cell,
                preserved_dirs=preserved_pair.directions,
                cell_direction_sets=cell_direction_sets,
                candidate_cells=tuple(candidate.cell for candidate in candidates),
                catalog=catalog,
            )
            if attach_cell is None or arm.fallback_direction is None:
                dropped_edge_ids.append(arm.edge_id)
                fallback_decisions.append(
                    {
                        "action": "drop",
                        "edge_id": arm.edge_id,
                        "direction": arm.fallback_direction,
                        "reason": "no_legal_t_junction_attachment",
                    }
                )
                continue

            edge_anchor_cells[arm.edge_id] = attach_cell
            attached_edge_ids.append(arm.edge_id)
            cell_direction_sets.setdefault(attach_cell, set(preserved_pair.directions)).add(arm.fallback_direction)
            fallback_decisions.append(
                {
                    "action": "attach",
                    "edge_id": arm.edge_id,
                    "direction": arm.fallback_direction,
                    "cell": _cell_tuple(attach_cell),
                    "required_directions": _ordered_directions(cell_direction_sets[attach_cell]),
                    "reason": "nearby_t_junction",
                }
            )

        ordered_cells = tuple(sorted(cell_direction_sets, key=lambda cell: (0 if cell == primary_cell else 1, cell.yidx, cell.xidx)))
        direction_sets = tuple(_ordered_directions(cell_direction_sets[cell]) for cell in ordered_cells)
        return SplitAnchorPlan(
            topology_node_id=node_id,
            primary_cell=primary_cell,
            split_anchor_cells=ordered_cells,
            required_dirs_estimate=required_dirs,
            split_direction_sets=direction_sets,
            edge_anchor_cells=edge_anchor_cells,
            preserved_direction_set=_ordered_directions(preserved_pair.directions),
            preserved_edge_ids=tuple(sorted(arm.edge_id for arm in preserved_pair.arms)),
            attached_edge_ids=tuple(attached_edge_ids),
            dropped_edge_ids=tuple(dropped_edge_ids),
            fallback_decisions=tuple(fallback_decisions),
            candidates=candidates,
            reason="intersection_degraded",
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
        if not required_dirs:
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

    def _score(
        self,
        *,
        node: TopologyNode,
        cell: GridCell,
        base_cell: GridCell,
        tile_feasible: bool,
        impossible_arm_count: int,
        impossible_arm_severity: float,
        continuity_score: float,
    ) -> float:
        center = self.grid_index.cell_center(cell)
        source_distance = center.distance(node.point) / max(self.grid_index.cell_size_m, 1.0)
        movement = max(abs(cell.xidx - base_cell.xidx), abs(cell.yidx - base_cell.yidx))
        base_bonus = -0.001 if cell == base_cell else 0.0
        tile_gap_penalty = 0.0 if tile_feasible else 4.0
        impossible_arm_penalty = impossible_arm_count + impossible_arm_severity * 0.25
        return source_distance + movement * 0.2 + tile_gap_penalty + impossible_arm_penalty + continuity_score + base_bonus

    def _incident_arm(self, node: TopologyNode, edge: TopologyEdge) -> _IncidentArm:
        spine = build_raster_spine(topology_edge_id=edge.edge_id, line=edge.geometry, grid_index=self.grid_index)
        support_cells = tuple(dict.fromkeys(spine.cells))
        if edge.end_node_id == node.node_id:
            support_cells = tuple(reversed(support_cells))
        fallback_direction = self._edge_direction_from_node(node.node_id, edge)
        support_cells = _representative_support_cells(
            self._clamp_cell(self.grid_index.projected_to_cell(node.point.x, node.point.y)),
            support_cells,
            fallback_direction,
        )
        return _IncidentArm(
            edge_id=edge.edge_id,
            support_cells=support_cells,
            fallback_direction=fallback_direction,
            process=edge.process,
            config_name=edge.config_name,
            priority=edge.priority,
        )

    def _candidate_arm_directions(
        self,
        *,
        cell: GridCell,
        incident_arms: tuple[_IncidentArm, ...],
        fallback_dirs: frozenset[str],
    ) -> _CandidateArmDirections:
        edge_dirs: dict[int, str] = {}
        reasons = []
        impossible_arm_count = 0
        impossible_arm_severity = 0.0

        for arm in incident_arms:
            allow_diagonal = self._process_allows_diagonal(arm.process)
            direction = _stub_direction_from_cell(
                cell,
                arm.support_cells,
                allow_diagonal=allow_diagonal,
            )
            if allow_diagonal and arm.fallback_direction in DIAGONAL_DIRECTIONS and direction in CARDINAL_DIRECTIONS:
                direction = arm.fallback_direction
                reasons.append(f"geometry_direction_fallback:{arm.edge_id}")
            if direction is None and arm.fallback_direction is not None:
                direction = arm.fallback_direction
                reasons.append(f"geometry_direction_fallback:{arm.edge_id}")
            if direction is None:
                impossible_arm_count += 1
                impossible_arm_severity += 2.0
                reasons.append(f"impossible_arm:{arm.edge_id}:no_stub_direction")
                continue
            edge_dirs[arm.edge_id] = direction

        direction_counts: dict[str, int] = {}
        for direction in edge_dirs.values():
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        for direction, count in sorted(direction_counts.items(), key=lambda item: _DIRECTION_ORDER.get(item[0], 99)):
            if count <= 1:
                continue
            impossible_arm_count += count - 1
            impossible_arm_severity += float(count - 1)
            reasons.append(f"duplicate_arm_direction:{direction}")

        required_dirs = frozenset(edge_dirs.values())
        if not required_dirs and fallback_dirs:
            required_dirs = fallback_dirs
        return _CandidateArmDirections(
            required_dirs=required_dirs,
            edge_dirs=MappingProxyType(edge_dirs),
            reasons=tuple(reasons),
            impossible_arm_count=impossible_arm_count,
            impossible_arm_severity=impossible_arm_severity,
        )

    def _required_dirs_estimate(self, node: TopologyNode, incident_edges: tuple[TopologyEdge, ...]) -> tuple[str, ...]:
        directions = []
        for edge in incident_edges:
            direction = self._edge_direction_from_node(node.node_id, edge)
            if direction:
                directions.append(direction)
        return tuple(directions)

    def _edge_direction_from_node(self, node_id: int, edge: TopologyEdge) -> str | None:
        return _edge_direction_from_node(
            node_id,
            edge,
            allow_diagonal=self._process_allows_diagonal(edge.process),
        )

    def _process_allows_diagonal(self, process: ProcessKind) -> bool:
        catalog = self.catalogs.get(process)
        if catalog is None or not hasattr(catalog, "allowed_step_dirs"):
            return False
        supported = frozenset(catalog.allowed_step_dirs())
        if not supported:
            return False
        if process is ProcessKind.ROAD:
            return supports_diagonal_directions(supported)
        return bool(supported.difference(CARDINAL_DIRECTIONS))

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


def _preserved_arm_pair(incident_arms: tuple[_IncidentArm, ...], catalog: Any) -> _PreservedArmPair | None:
    pairs = []
    for left_index, left in enumerate(incident_arms):
        if left.fallback_direction is None:
            continue
        for right in incident_arms[left_index + 1 :]:
            if right.fallback_direction is None:
                continue
            directions = frozenset((left.fallback_direction, right.fallback_direction))
            if len(directions) != 2 or not catalog.has_tile(directions):
                continue
            pairs.append(_PreservedArmPair(arms=(left, right), directions=directions))
    if not pairs:
        return None
    return min(pairs, key=_preserved_pair_sort_key)


def _preserved_pair_sort_key(pair: _PreservedArmPair) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...]]:
    left, right = pair.arms
    is_major_continuity = int(
        not (
            _major_road_arm(left)
            and _major_road_arm(right)
            and left.config_name == right.config_name
            and _OPPOSITE_DIRECTIONS.get(left.fallback_direction or "") == right.fallback_direction
        )
    )
    is_opposite = int(not _directions_are_opposite(pair.directions))
    priority = min(left.priority, right.priority)
    road_rank = min(_ROAD_CLASS_RANK.get(left.config_name, 99), _ROAD_CLASS_RANK.get(right.config_name, 99))
    direction_rank = tuple(_DIRECTION_ORDER.get(direction, 99) for direction in _ordered_directions(pair.directions))
    edge_ids = tuple(sorted((left.edge_id, right.edge_id)))
    return is_major_continuity, is_opposite, priority, (road_rank, *direction_rank), edge_ids


def _directions_are_opposite(directions: frozenset[str]) -> bool:
    return directions_are_opposite(directions)


def _split_primary_cell(
    base_cell: GridCell,
    candidates: tuple[AnchorCandidate, ...],
    preserved_dirs: frozenset[str],
) -> GridCell:
    for candidate in candidates:
        if candidate.cell == base_cell and candidate.occupancy_feasible:
            return base_cell
    matching = [
        candidate
        for candidate in candidates
        if candidate.occupancy_feasible and preserved_dirs.issubset(candidate.required_dirs)
    ]
    if matching:
        return min(matching, key=lambda candidate: (candidate.score, candidate.cell.yidx, candidate.cell.xidx)).cell
    return base_cell


def _attachment_cell_for_arm(
    arm: _IncidentArm,
    *,
    primary_cell: GridCell,
    preserved_dirs: frozenset[str],
    cell_direction_sets: Mapping[GridCell, set[str]],
    candidate_cells: tuple[GridCell, ...],
    catalog: Any,
) -> GridCell | None:
    if arm.fallback_direction is None:
        return None
    for cell in _attachment_candidate_cells(primary_cell, preserved_dirs, candidate_cells):
        existing_dirs = cell_direction_sets.get(cell, set(preserved_dirs))
        proposed = frozenset((*existing_dirs, arm.fallback_direction))
        if arm.fallback_direction in existing_dirs and cell != primary_cell:
            proposed = frozenset(existing_dirs)
        if catalog.has_tile(proposed):
            return cell
    return None


def _attachment_candidate_cells(
    primary_cell: GridCell,
    preserved_dirs: frozenset[str],
    candidate_cells: tuple[GridCell, ...],
) -> tuple[GridCell, ...]:
    axis_cells = [
        cell
        for cell in dict.fromkeys((primary_cell, *candidate_cells))
        if cell == primary_cell or _cell_is_on_preserved_axis(primary_cell, cell, preserved_dirs)
    ]
    return tuple(
        sorted(
            axis_cells,
            key=lambda cell: (
                0 if cell == primary_cell else 1,
                abs(cell.xidx - primary_cell.xidx) + abs(cell.yidx - primary_cell.yidx),
                _direction_order_from_delta(primary_cell, cell),
                cell.yidx,
                cell.xidx,
            ),
        )
    )


def _cell_is_on_preserved_axis(primary_cell: GridCell, cell: GridCell, preserved_dirs: frozenset[str]) -> bool:
    if preserved_dirs == frozenset({"E", "W"}):
        return cell.yidx == primary_cell.yidx
    if preserved_dirs == frozenset({"N", "S"}):
        return cell.xidx == primary_cell.xidx
    if preserved_dirs == frozenset({"NE", "SW"}):
        return cell.xidx - primary_cell.xidx == cell.yidx - primary_cell.yidx
    if preserved_dirs == frozenset({"NW", "SE"}):
        return cell.xidx - primary_cell.xidx == primary_cell.yidx - cell.yidx
    return max(abs(cell.xidx - primary_cell.xidx), abs(cell.yidx - primary_cell.yidx)) <= 1


def _direction_order_from_delta(primary_cell: GridCell, cell: GridCell) -> int:
    direction = _direction_between_cells(primary_cell, cell)
    return _DIRECTION_ORDER.get(direction or "", 99)


def _legacy_edge_anchor_cells(
    incident_arms: tuple[_IncidentArm, ...],
    anchor_cells: tuple[GridCell, ...],
    direction_sets: tuple[tuple[str, ...], ...],
) -> Mapping[int, GridCell]:
    edge_anchor_cells = {}
    for arm in incident_arms:
        for index, direction_set in enumerate(direction_sets):
            if arm.fallback_direction in direction_set and index < len(anchor_cells):
                edge_anchor_cells[arm.edge_id] = anchor_cells[index]
                break
    return edge_anchor_cells


def _cell_tuple(cell: GridCell) -> tuple[int, int]:
    return cell.xidx, cell.yidx


def _arm_sort_key(arm: _IncidentArm) -> tuple[int, int, int, int]:
    return (
        arm.priority,
        _ROAD_CLASS_RANK.get(arm.config_name, 99),
        _DIRECTION_ORDER.get(arm.fallback_direction or "", 99),
        arm.edge_id,
    )


def _candidate_selectable(candidate: AnchorCandidate) -> bool:
    return candidate.tile_feasible and candidate.occupancy_feasible and candidate.impossible_arm_count == 0


def _candidate_has_supported_arms(candidate: AnchorCandidate) -> bool:
    return candidate.tile_feasible and candidate.impossible_arm_count == 0


def _stub_direction_from_cell(
    cell: GridCell,
    support_cells: tuple[GridCell, ...],
    *,
    allow_diagonal: bool = False,
) -> str | None:
    if not support_cells:
        return None
    if cell in support_cells:
        cell_index = support_cells.index(cell)
        targets = support_cells[cell_index + 1 :]
    else:
        targets = support_cells[:1]
    for target in targets:
        direction = _direction_between_cells(cell, target, allow_diagonal=allow_diagonal)
        if direction is not None:
            return direction
    return None


def _representative_support_cells(
    node_cell: GridCell,
    support_cells: tuple[GridCell, ...],
    direction: str | None,
) -> tuple[GridCell, ...]:
    if direction is None or not support_cells:
        return support_cells
    aligned = tuple(cell for cell in support_cells if _cell_is_on_forward_axis(cell, node_cell, direction))
    return aligned or support_cells


def _cell_is_on_forward_axis(cell: GridCell, node_cell: GridCell, direction: str) -> bool:
    dx = cell.xidx - node_cell.xidx
    dy = cell.yidx - node_cell.yidx
    if direction == "N":
        return dx == 0 and dy >= 0
    if direction == "S":
        return dx == 0 and dy <= 0
    if direction == "E":
        return dy == 0 and dx >= 0
    if direction == "W":
        return dy == 0 and dx <= 0
    if direction == "NE":
        return dx >= 0 and dy >= 0
    if direction == "NW":
        return dx <= 0 and dy >= 0
    if direction == "SE":
        return dx >= 0 and dy <= 0
    if direction == "SW":
        return dx <= 0 and dy <= 0
    return False


def _direction_between_cells(start: GridCell, end: GridCell, *, allow_diagonal: bool = False) -> str | None:
    dx = end.xidx - start.xidx
    dy = end.yidx - start.yidx
    if dx == 0 and dy == 0:
        return None
    if allow_diagonal and dx != 0 and dy != 0:
        return ("N" if dy > 0 else "S") + ("E" if dx > 0 else "W")
    if abs(dx) >= abs(dy):
        return "E" if dx > 0 else "W"
    return "N" if dy > 0 else "S"


def _edge_direction_from_node(node_id: int, edge: TopologyEdge, *, allow_diagonal: bool = False) -> str | None:
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
    if allow_diagonal and not math.isclose(dx, 0.0) and not math.isclose(dy, 0.0):
        return ("N" if dy > 0 else "S") + ("E" if dx > 0 else "W")
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
    return ordered_directions(frozenset(directions))


def _major_continuity_pairs(incident_arms: tuple[_IncidentArm, ...]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for left_index, left in enumerate(incident_arms):
        if not _major_road_arm(left):
            continue
        for right in incident_arms[left_index + 1 :]:
            if not _major_road_arm(right):
                continue
            if left.config_name != right.config_name:
                continue
            if left.fallback_direction and _OPPOSITE_DIRECTIONS.get(left.fallback_direction) == right.fallback_direction:
                pairs.append((left.edge_id, right.edge_id))
    return tuple(pairs)


def _major_road_arm(arm: _IncidentArm) -> bool:
    return arm.process is ProcessKind.ROAD and arm.config_name in _MAJOR_ROAD_CLASSES


def _major_continuity_score(edge_dirs: Mapping[int, str], continuity_pairs: tuple[tuple[int, int], ...]) -> float:
    score = 0.0
    for left_edge_id, right_edge_id in continuity_pairs:
        left_direction = edge_dirs.get(left_edge_id)
        right_direction = edge_dirs.get(right_edge_id)
        if left_direction is None or right_direction is None:
            score += 0.5
        elif _OPPOSITE_DIRECTIONS.get(left_direction) == right_direction:
            score -= 0.05
        else:
            score += 0.5
    return score
