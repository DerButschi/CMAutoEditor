from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_typed_processor_commits_roads_and_buildings_before_area_placement() -> None:
    from shapely.geometry import LineString, Point, Polygon
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import (
        CMType,
        FeatureRecord,
        GridCell,
        LayerKind,
        ProcessKind,
    )
    from terrain_extraction.osm_processor import OSMProcessor

    road_cell = GridCell(1, 1)
    building_cell = GridCell(2, 1)
    road_placement = _placement(
        layer=LayerKind.LINEAR_SURFACE,
        cell=road_cell,
        config_name="road",
        priority=1,
        cm_type=CMType(menu="Roads", cat1="Dirt", cat2="Road Tile 1", direction="Direction 2"),
    )
    building_placement = _placement(
        layer=LayerKind.BUILDING,
        cell=building_cell,
        config_name="houses",
        priority=4,
        cm_type=CMType(menu="Buildings", cat1="House", cat2="Small House"),
    )
    area_placement = _placement(
        layer=LayerKind.FOLIAGE,
        cell=GridCell(0, 0),
        config_name="trees",
        priority=5,
        cm_type=CMType(menu="Foliage", cat1="Tree"),
    )
    pipeline = _CrossFeaturePipeline(road_placement, building_placement, area_placement)
    processor = OSMProcessor.__new__(OSMProcessor)
    processor.pipeline = pipeline
    processor.grid_index = GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=4,
        height=3,
    )
    processor.idx_bbox = (0, 0, 3, 2)
    processor.effective_bbox_polygon = None
    processor.extraction_config = ExtractionConfig.from_mapping({})
    processor._typed_features_from_matched_elements = lambda: (
        FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 8), (24, 8)])),
        FeatureRecord("building-1", 1, "houses", ProcessKind.BUILDING_OUTLINE, 4, Polygon([(16, 8), (24, 8), (24, 16), (16, 16)])),
        FeatureRecord("area-1", 2, "trees", ProcessKind.AREA, 5, Point(8, 8).buffer(4)),
    )
    processor._tile_catalogs_for = lambda features: {}
    processor._building_catalogs_for = lambda features: {}
    processor._typed_linear_feature_placements = lambda placements: ()
    processor._set_compatibility_df_from_placements = lambda: None

    processor._run_typed_processors()

    assert pipeline.calls == ["topology", "routing", "tile_assignment", "building", "area", "output"]
    assert road_placement in processor.placements
    assert building_placement in processor.placements
    assert area_placement in processor.placements


def test_output_conflict_resolution_preserves_road_when_area_and_building_exist() -> None:
    from terrain_extraction.osm_extraction.models import CMType, GridCell, LayerKind
    from terrain_extraction.osm_processor import OSMProcessor

    road_cell = GridCell(1, 1)
    placements = (
        _placement(
            layer=LayerKind.GROUND,
            cell=road_cell,
            config_name="meadow",
            priority=3,
            cm_type=CMType(menu="Ground 1", cat1="Grass"),
        ),
        _placement(
            layer=LayerKind.LINEAR_SURFACE,
            cell=road_cell,
            config_name="road",
            priority=1,
            cm_type=CMType(menu="Roads", cat1="Dirt", cat2="Road Tile 1", direction="Direction 2"),
        ),
        _placement(
            layer=LayerKind.BUILDING,
            cell=GridCell(2, 1),
            config_name="houses",
            priority=4,
            cm_type=CMType(menu="Buildings", cat1="House"),
        ),
    )

    resolved = OSMProcessor._resolve_output_layer_conflicts(placements)

    assert placements[1] in resolved
    assert placements[2] in resolved


class _CrossFeaturePipeline:
    def __init__(
        self,
        road_placement,
        building_placement,
        area_placement,
    ) -> None:
        self.calls: list[str] = []
        self.road_placement = road_placement
        self.building_placement = building_placement
        self.area_placement = area_placement
        self.context = SimpleNamespace(rng=None)

    def run_network_topology(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("topology")
        return ExtractionResult(diagnostics={"network_topology": object()})

    def run_network_router(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("routing")
        return ExtractionResult(
            diagnostics={
                "network_routes": SimpleNamespace(
                    routes=(object(),),
                    linear_state=object(),
                )
            }
        )

    def run_tile_assignment(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("tile_assignment")
        return ExtractionResult(
            placements=(self.road_placement,),
            diagnostics={"tile_assignment": object()},
        )

    def run_building_fitter(self, *, occupancy, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult, LayerKind

        self.calls.append("building")
        assert occupancy.object_id_at(LayerKind.LINEAR_SURFACE, self.road_placement.cells[0]) is not None
        occupancy.place(self.building_placement, object_id=self.building_placement.feature_id)
        return ExtractionResult(
            placements=(self.building_placement,),
            diagnostics={"building_fitting": object()},
        )

    def run_area_rasterizer(self, *, occupancy, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult, LayerKind

        self.calls.append("area")
        assert occupancy.object_id_at(LayerKind.LINEAR_SURFACE, self.road_placement.cells[0]) is not None
        assert occupancy.object_id_at(LayerKind.BUILDING, self.building_placement.cells[0]) is not None
        occupancy.place(self.area_placement, object_id=self.area_placement.feature_id)
        return ExtractionResult(placements=(self.area_placement,))

    def run_output_rows(self, *, placements, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("output")
        return ExtractionResult(
            placements=placements,
            output_rows=({"name": "extent_marker"},),
        )


def _placement(*, layer, cell, config_name, priority, cm_type):
    from terrain_extraction.osm_extraction.models import GridKind, PlacementRecord

    return PlacementRecord(
        layer=layer,
        grid_kind=GridKind.NORMAL,
        cells=(cell,),
        config_name=config_name,
        feature_id=f"{config_name}-1",
        priority=priority,
        cm_type=cm_type,
        score=1.0,
    )
