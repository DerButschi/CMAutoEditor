from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from terrain_extraction.osm_extraction.models import (
    ProcessKind,
    TopologyEdge,
    TopologyGraph,
    TopologyNode,
)

_PROCESS_STAGE = {
    ProcessKind.ROAD: 0,
    ProcessKind.RAIL: 1,
    ProcessKind.STREAM: 2,
    ProcessKind.FENCE: 3,
    ProcessKind.LINEAR: 4,
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
    "cycleway": 7,
    "bridleway": 7,
}
_DEFAULT_INTERACTIONS = {
    (ProcessKind.ROAD, ProcessKind.ROAD): "connect",
    (ProcessKind.RAIL, ProcessKind.RAIL): "connect",
    (ProcessKind.STREAM, ProcessKind.STREAM): "connect",
    (ProcessKind.FENCE, ProcessKind.FENCE): "connect",
}
_POLICY_DIAGNOSTIC_PROCESSES = (ProcessKind.ROAD, ProcessKind.RAIL, ProcessKind.STREAM, ProcessKind.FENCE)


@dataclass(frozen=True, slots=True)
class ProcessInteractionDecision:
    existing_process: ProcessKind
    incoming_process: ProcessKind
    interaction: str
    reason: str


@dataclass(frozen=True, slots=True)
class LinearInteractionPolicy:
    interactions: Mapping[tuple[ProcessKind, ProcessKind], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "interactions", MappingProxyType(dict(self.interactions)))

    def decision(self, existing_process: ProcessKind, incoming_process: ProcessKind) -> ProcessInteractionDecision:
        interaction = self.interactions.get((existing_process, incoming_process))
        if interaction is None:
            interaction = self.interactions.get((incoming_process, existing_process), "avoid")
        return ProcessInteractionDecision(
            existing_process=existing_process,
            incoming_process=incoming_process,
            interaction=interaction,
            reason=f"{existing_process.value}_{incoming_process.value}_{interaction}",
        )

    def as_diagnostics(self) -> Mapping[tuple[str, str], str]:
        return MappingProxyType(
            {
                (existing.value, incoming.value): self.decision(existing, incoming).interaction
                for existing in _POLICY_DIAGNOSTIC_PROCESSES
                for incoming in _POLICY_DIAGNOSTIC_PROCESSES
            }
        )


@dataclass(frozen=True, slots=True)
class LinearProcessingGroup:
    group_index: int
    stage: int
    process: ProcessKind
    config_name: str
    priority: int
    rank: int
    edges: tuple[TopologyEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(self.edges))

    @property
    def edge_ids(self) -> tuple[int, ...]:
        return tuple(edge.edge_id for edge in self.edges)

    def topology(self, nodes: Iterable[TopologyNode]) -> TopologyGraph:
        node_ids = {edge.start_node_id for edge in self.edges} | {edge.end_node_id for edge in self.edges}
        return TopologyGraph(nodes=tuple(node for node in nodes if node.node_id in node_ids), edges=self.edges)

    def diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "group_index": self.group_index,
                "stage": self.stage,
                "process": self.process.value,
                "config_name": self.config_name,
                "priority": self.priority,
                "rank": self.rank,
                "edge_ids": self.edge_ids,
            }
        )


@dataclass(frozen=True, slots=True)
class LinearProcessingPlan:
    groups: tuple[LinearProcessingGroup, ...]
    interaction_policy: LinearInteractionPolicy = field(default_factory=lambda: default_linear_interaction_policy())
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @classmethod
    def from_topology(
        cls,
        topology: TopologyGraph,
        *,
        interaction_policy: LinearInteractionPolicy | None = None,
    ) -> LinearProcessingPlan:
        groups_by_key: dict[tuple[int, ProcessKind, int, int, str], list[TopologyEdge]] = {}
        for edge in topology.edges:
            key = (_process_stage(edge.process), edge.process, edge.priority, _config_rank(edge.config_name), edge.config_name)
            groups_by_key.setdefault(key, []).append(edge)

        groups = []
        for group_index, (key, edges) in enumerate(sorted(groups_by_key.items(), key=_group_sort_key)):
            stage, process, priority, rank, config_name = key
            groups.append(
                LinearProcessingGroup(
                    group_index=group_index,
                    stage=stage,
                    process=process,
                    config_name=config_name,
                    priority=priority,
                    rank=rank,
                    edges=tuple(sorted(edges, key=lambda edge: edge.edge_id)),
                )
            )

        policy = interaction_policy or default_linear_interaction_policy()
        return cls(
            groups=tuple(groups),
            interaction_policy=policy,
            diagnostics=_diagnostics(groups, policy),
        )


def default_linear_interaction_policy() -> LinearInteractionPolicy:
    return LinearInteractionPolicy(_DEFAULT_INTERACTIONS)


def _group_sort_key(item: tuple[tuple[int, ProcessKind, int, int, str], list[TopologyEdge]]) -> tuple[int, int, int, str, int]:
    stage, process, priority, rank, config_name = item[0]
    return stage, priority, rank, config_name, min((edge.edge_id for edge in item[1]), default=0)


def _diagnostics(
    groups: tuple[LinearProcessingGroup, ...] | list[LinearProcessingGroup],
    policy: LinearInteractionPolicy,
) -> Mapping[str, Any]:
    stage_order = tuple(dict.fromkeys(group.process.value for group in groups))
    return MappingProxyType(
        {
            "linear_processing_groups": len(groups),
            "linear_processing_stage_order": stage_order,
            "linear_processing_group_order": tuple(group.diagnostics() for group in groups),
            "process_pair_policy": policy.as_diagnostics(),
        }
    )


def _process_stage(process: ProcessKind) -> int:
    return _PROCESS_STAGE.get(process, 99)


def _config_rank(config_name: str) -> int:
    return _NETWORK_CLASS_RANK.get(config_name, 99)
