from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _feature(feature_id, config_name, process, geometry, *, priority=5):
    from terrain_extraction.osm_extraction.models import FeatureRecord

    return FeatureRecord(
        feature_id=feature_id,
        source_index=int(str(feature_id).split("-")[-1]) if "-" in str(feature_id) else 0,
        config_name=config_name,
        process=process,
        priority=priority,
        geometry=geometry,
        source_tags={"highway": config_name},
        source_properties={"id": feature_id},
    )


def _clip() -> Polygon:
    return Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])


def test_crossroads_build_one_shared_topology_node() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    graph = NetworkTopologyBuilder(clip_geometry=_clip()).build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 10), (20, 10)]), priority=1),
            _feature("road-1", "primary", ProcessKind.ROAD, LineString([(10, 0), (10, 20)]), priority=1),
        )
    )

    shared_nodes = [node for node in graph.nodes if node.point.distance(Point(10, 10)) < 0.001]
    assert len(shared_nodes) == 1
    assert graph.degree(shared_nodes[0].node_id) == 4
    assert len(graph.edges) == 4
    assert graph.diagnostics["intersection_points"] == 1


def test_t_junction_splits_line_and_marks_degree_three_node() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    graph = NetworkTopologyBuilder(clip_geometry=_clip()).build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 10), (20, 10)]), priority=1),
            _feature("road-1", "primary", ProcessKind.ROAD, LineString([(10, 10), (10, 18)]), priority=1),
        )
    )

    junction = graph.nearest_node(Point(10, 10))
    assert junction is not None
    assert graph.degree(junction.node_id) == 3
    assert len(graph.edges) == 3


def test_near_miss_endpoint_snaps_only_inside_tolerance() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    inside = NetworkTopologyBuilder(snap_tolerance_m=1.0).build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 0), (10, 0)]), priority=1),
            _feature("road-1", "primary", ProcessKind.ROAD, LineString([(10.75, 0), (18, 0)]), priority=1),
        )
    )
    outside = NetworkTopologyBuilder(snap_tolerance_m=0.5).build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 0), (10, 0)]), priority=1),
            _feature("road-1", "primary", ProcessKind.ROAD, LineString([(10.75, 0), (18, 0)]), priority=1),
        )
    )

    assert len(inside.nodes) == 3
    assert inside.diagnostics["snapped_points"] == 1
    assert len(outside.nodes) == 4
    assert outside.diagnostics["snapped_points"] == 0


def test_lines_are_clipped_to_effective_bbox() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    graph = NetworkTopologyBuilder(clip_geometry=_clip()).build(
        (_feature("road-0", "primary", ProcessKind.ROAD, LineString([(-5, 10), (25, 10)]), priority=1),)
    )

    assert len(graph.edges) == 1
    assert graph.edges[0].geometry.equals_exact(LineString([(0, 10), (20, 10)]), tolerance=0.001)
    assert graph.diagnostics["clipped_lines"] == 1


def test_multiline_and_geometry_collection_are_normalized_to_lines() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    geometry = GeometryCollection(
        [
            MultiLineString([[(0, 0), (4, 0)], [(4, 0), (8, 0)]]),
            Polygon([(12, 0), (16, 0), (16, 4), (12, 4)]),
        ]
    )

    graph = NetworkTopologyBuilder().build(
        (_feature("mixed-0", "track", ProcessKind.ROAD, geometry, priority=2),)
    )

    assert graph.diagnostics["source_lines"] == 6
    assert graph.diagnostics["normalized_lines"] == 6
    assert {edge.config_name for edge in graph.edges} == {"track"}


def test_degree_two_chains_with_identical_metadata_are_collapsed() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.network_topology import NetworkTopologyBuilder

    graph = NetworkTopologyBuilder().build(
        (
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 0), (8, 0)]), priority=1),
            _feature("road-1", "primary", ProcessKind.ROAD, LineString([(8, 0), (16, 0)]), priority=1),
            _feature("road-2", "primary", ProcessKind.ROAD, LineString([(16, 0), (24, 0)]), priority=1),
        )
    )

    assert len(graph.edges) == 1
    assert list(graph.edges[0].geometry.coords) == [(0.0, 0.0), (8.0, 0.0), (16.0, 0.0), (24.0, 0.0)]
    assert graph.diagnostics["collapsed_degree_two_nodes"] == 2


def test_pipeline_runs_network_topology_only_when_feature_flag_is_enabled() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    context = ExtractionContext(
        profile="cold_war",
        bbox=None,
        config_path="default_osm_config.json",
        seed=123,
        rng=np.random.default_rng(123),
        feature_flags={"use_new_network_topology": True},
    )
    result = ExtractionPipeline(context).run_network_topology(
        features=(
            _feature("road-0", "primary", ProcessKind.ROAD, LineString([(0, 0), (10, 0)]), priority=1),
        ),
    )

    assert result.stats.counts["topology_edges"] == 1
    assert result.diagnostics["network_topology"].edges[0].feature_ids == ("road-0",)

    disabled = ExtractionPipeline(
        ExtractionContext(
            profile="cold_war",
            bbox=None,
            config_path="default_osm_config.json",
            seed=123,
            rng=np.random.default_rng(123),
        )
    ).run_network_topology(features=())
    assert disabled.diagnostics == {"network_topology": "disabled"}
