from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from shapely.geometry import LineString, Point, Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid():
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    return GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=3,
        height=3,
        cell_size_m=8.0,
        crs_epsg=25832,
    )


def _placement(config_name: str, process, layer, cell):
    from terrain_extraction.osm_extraction.models import CMType, GridKind, PlacementRecord

    return PlacementRecord(
        layer=layer,
        grid_kind=GridKind.NORMAL,
        cells=(cell,),
        config_name=config_name,
        feature_id=config_name,
        priority=1,
        cm_type=CMType(menu="Ground", cat1=config_name),
        score=1.0,
    )


def test_osm_processor_run_processors_dispatches_to_typed_pipeline(monkeypatch) -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.models import (
        ExtractionResult,
        GridCell,
        LayerKind,
        NetworkRoutingResult,
        ProcessKind,
        TopologyGraph,
    )
    from terrain_extraction.osm_processor import OSMProcessor
    from terrain_extraction.osm_utils import processing

    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy osm_utils processing should not run")

    for name in (
        "assign_type_from_tag",
        "assign_type_in_random_clusters",
        "single_object_random",
        "collect_network_data",
        "create_line_graph",
        "create_square_graph_path_search",
        "assign_road_tiles_to_network",
        "collect_building_outlines",
        "process_residential_building_outlines",
        "assign_type_at_linear_feature",
    ):
        monkeypatch.setattr(processing, name, fail_legacy, raising=False)

    raw_config = {
        "meadow": {
            "tags": [["landuse", "meadow"]],
            "cm_types": [{"menu": "Ground", "cat1": "Grass"}],
            "process": ["type_from_tag"],
            "priority": 3,
        },
        "forest": {
            "tags": [["landuse", "forest"]],
            "cm_types": [{"menu": "Foliage", "cat1": "Tree"}],
            "process": ["type_random_clusters"],
            "priority": 4,
        },
        "bench": {
            "tags": [["amenity", "bench"]],
            "cm_types": [{"menu": "Flavor Objects", "cat1": "Bench"}],
            "process": ["single_object_random"],
            "priority": 5,
        },
        "road": {
            "tags": [["highway", "residential"]],
            "cm_types": [{"menu": "Roads", "cat1": "Paved"}],
            "process": ["road_tiles"],
            "priority": 1,
        },
        "road_brush": {
            "tags": [["natural", "scrub"]],
            "cm_types": [{"menu": "Foliage", "cat1": "Brush"}],
            "process": ["type_from_linear"],
            "priority": 6,
            "modifiers": {"linear_name": "road"},
        },
        "houses": {
            "tags": [["building", "house"]],
            "cm_types": [{"menu": "Buildings", "cat1": "House"}],
            "process": ["type_from_residential_building_outline"],
            "priority": 2,
        },
    }
    processor = OSMProcessor.__new__(OSMProcessor)
    processor.config = raw_config
    processor.extraction_config = ExtractionConfig.from_mapping(raw_config)
    processor.profile = "cold_war"
    processor.grid_index = _grid()
    processor.idx_bbox = [0, 0, 2, 2]
    processor.effective_bbox_polygon = Polygon([(0, 0), (24, 0), (24, 24), (0, 24)])
    processor.matched_elements = [
        {"element": SimpleNamespace(properties={"id": "area-1", "landuse": "meadow"}), "geometry": Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]), "name": "meadow", "idx": 0},
        {"element": SimpleNamespace(properties={"id": "forest-1", "landuse": "forest"}), "geometry": Polygon([(8, 0), (16, 0), (16, 8), (8, 8)]), "name": "forest", "idx": 1},
        {"element": SimpleNamespace(properties={"id": "bench-1", "amenity": "bench"}), "geometry": Point(4, 4), "name": "bench", "idx": 2},
        {"element": SimpleNamespace(properties={"id": "road-1", "highway": "residential"}), "geometry": LineString([(0, 0), (16, 0)]), "name": "road", "idx": 3},
        {"element": SimpleNamespace(properties={"id": "house-1", "building": "house"}), "geometry": Polygon([(0, 8), (8, 8), (8, 16), (0, 16)]), "name": "houses", "idx": 4},
    ]
    calls: list[tuple[str, object]] = []

    class FakePipeline:
        context = SimpleNamespace(rng=np.random.default_rng(123))

        def run_area_rasterizer(self, *, features, config, grid_index, occupancy):
            calls.append(("area", tuple(feature.process for feature in features)))
            return ExtractionResult(
                placements=(
                    _placement("meadow", ProcessKind.AREA, LayerKind.GROUND, GridCell(0, 0)),
                    _placement("bench", ProcessKind.POINT, LayerKind.POINT_OBJECT, GridCell(0, 0)),
                )
            )

        def run_network_topology(self, *, features, clip_geometry=None, snap_tolerance_m=1.0):
            calls.append(("topology", tuple(feature.process for feature in features)))
            return ExtractionResult(diagnostics={"network_topology": TopologyGraph()})

        def run_network_router(
            self,
            *,
            topology,
            grid_index,
            occupancy,
            catalogs=None,
            corridor_deviation_m=32.0,
            minor_relaxation_m=48.0,
        ):
            calls.append(("routing", topology))
            return ExtractionResult(diagnostics={"network_routes": NetworkRoutingResult()})

        def run_tile_assignment(self, *, routes, catalogs):
            calls.append(("tiles", tuple(catalogs)))
            return ExtractionResult(
                placements=(_placement("road", ProcessKind.ROAD, LayerKind.LINEAR_SURFACE, GridCell(1, 0)),)
            )

        def run_building_fitter(self, *, features, catalogs, grid_index, occupancy):
            calls.append(("buildings", tuple(feature.process for feature in features)))
            return ExtractionResult(
                placements=(_placement("houses", ProcessKind.BUILDING_OUTLINE, LayerKind.BUILDING, GridCell(0, 1)),)
            )

        def run_output_rows(self, *, placements, bounds):
            calls.append(("output", tuple(placement.config_name for placement in placements)))
            return ExtractionResult(output_rows=({"x": 0, "y": 0, "name": "meadow"},))

    processor.pipeline = FakePipeline()

    processor.run_processors()

    assert [call[0] for call in calls] == ["area", "topology", "routing", "tiles", "buildings", "output"]
    assert calls[0][1] == (ProcessKind.AREA, ProcessKind.RANDOM, ProcessKind.POINT)
    assert calls[1][1] == (ProcessKind.ROAD,)
    assert calls[4][1] == (ProcessKind.BUILDING_OUTLINE,)
    assert processor.output_rows == ({"x": 0, "y": 0, "name": "meadow"},)


def test_legacy_osm_helpers_are_marked_as_quarantined() -> None:
    from terrain_extraction.osm_utils import path_search, processing

    assert "compatibility" in processing.LEGACY_QUARANTINE_REASON
    assert "historical" in path_search.LEGACY_QUARANTINE_REASON
    assert "assign_type_in_random_clusters" in processing.LEGACY_COMPATIBILITY_HELPERS
    assert "search_path2" in path_search.LEGACY_COMPATIBILITY_HELPERS
