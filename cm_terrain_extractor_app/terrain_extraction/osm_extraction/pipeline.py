from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import ExtractionResult
from terrain_extraction.osm_extraction.occupancy import OccupancyModel
from terrain_extraction.osm_extraction.stats import ExtractionStats

ProgressCallback = Callable[[str, float, str | None], None]


class LegacyProcessor(Protocol):
    def preprocess_osm_data(self, osm_data: object) -> None: ...

    def run_processors(self) -> None: ...

    def post_process(self) -> None: ...

    def get_output(self) -> Any: ...


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

    def run(self) -> ExtractionResult:
        return ExtractionResult(stats=ExtractionStats(diagnostics={"mode": "stub"}))

    def run_area_rasterizer(
        self,
        *,
        features: tuple[Any, ...],
        config: ExtractionConfig,
        grid_index: GridIndex,
        occupancy: OccupancyModel | None = None,
    ) -> ExtractionResult:
        if not self.context.feature_flags.get("use_new_area_rasterizer", False):
            return ExtractionResult(diagnostics={"area_rasterizer": "disabled"})

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
        if not self.context.feature_flags.get("use_new_network_topology", False):
            return ExtractionResult(diagnostics={"network_topology": "disabled"})

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
