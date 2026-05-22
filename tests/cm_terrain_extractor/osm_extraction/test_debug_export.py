from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
ROOT_DIR = Path(__file__).parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid():
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    return GridIndex(
        origin_x=0.0,
        origin_y=0.0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=4,
        height=4,
        cell_size_m=8.0,
        crs_epsg=3857,
    )


def _feature(feature_id, config_name, process, geometry, *, priority=3):
    from terrain_extraction.osm_extraction.models import FeatureRecord

    return FeatureRecord(
        feature_id=feature_id,
        source_index=0,
        config_name=config_name,
        process=process,
        priority=priority,
        geometry=geometry,
        source_tags={"railway": "rail"} if config_name == "rail" else {"building": "barn"} if config_name == "barns" else {},
        source_properties={"name": config_name},
    )


def _placement(*, config_name, feature_id, layer, cell, menu, cat1, process=None, diagnostics=None):
    from terrain_extraction.osm_extraction.models import CMType, GridKind, PlacementRecord

    return PlacementRecord(
        layer=layer,
        grid_kind=GridKind.NORMAL,
        cells=(cell,),
        config_name=config_name,
        feature_id=feature_id,
        priority=2,
        cm_type=CMType(menu=menu, cat1=cat1),
        score=1.0,
        diagnostics={} if diagnostics is None else {"process": process.value if process else None, **diagnostics},
    )


def test_debug_layers_include_sources_topology_routes_anchors_occupancy_buildings_and_final_rows() -> None:
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
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
        TopologyNode,
    )
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    occupancy = OccupancyModel.from_grid_index(grid)
    road_cell = GridCell(1, 1)
    road_placement = PlacementRecord(
        layer=LayerKind.LINEAR_SURFACE,
        grid_kind=GridKind.NORMAL,
        cells=(road_cell,),
        config_name="road",
        feature_id="road-1",
        priority=2,
        cm_type=CMType(menu="Roads", cat1="Dirt"),
        score=1.0,
        diagnostics={"process": ProcessKind.ROAD.value},
    )
    occupancy.place(road_placement, object_id="road-1")

    building_shape = Polygon([(16, 8), (24, 8), (24, 16), (16, 16)])
    building_placement = _placement(
        config_name="barns",
        feature_id="barn-1",
        layer=LayerKind.BUILDING,
        cell=GridCell(2, 1),
        menu="Buildings",
        cat1="Barn",
        process=ProcessKind.BUILDING_OUTLINE,
        diagnostics={
            "selected_footprint_polygon": building_shape,
            "output_xidx": 1.5,
            "output_yidx": 0.5,
            "iou": 0.91,
        },
    )
    topology = TopologyGraph(
        nodes=(
            TopologyNode(0, Point(0, 8), diagnostics={"kind": "start"}),
            TopologyNode(1, Point(24, 8), diagnostics={"kind": "end"}),
        ),
        edges=(
            TopologyEdge(
                edge_id=10,
                start_node_id=0,
                end_node_id=1,
                geometry=LineString([(0, 8), (24, 8)]),
                feature_ids=("road-1",),
                source_indices=(0,),
                config_name="road",
                process=ProcessKind.ROAD,
                priority=2,
                diagnostics={"source": "fixture"},
            ),
        ),
        diagnostics={"topology_note": "ok"},
    )
    routes = NetworkRoutingResult(
        routes=(
            RouteRecord(
                edge_id=10,
                start_node_id=0,
                end_node_id=1,
                process=ProcessKind.ROAD,
                config_name="road",
                priority=2,
                nodes=(GridNode(0, 1), GridNode(1, 1), GridNode(2, 1)),
                tile_cells=(GridCell(0, 1), road_cell, GridCell(2, 1)),
                raster_spine=RasterSpine(
                    topology_edge_id=10,
                    cells=(GridCell(0, 1), road_cell, GridCell(2, 1)),
                    progress=(0.0, 0.5, 1.0),
                    distance_m=(0.0, 0.0, 0.0),
                    source_length_m=24.0,
                ),
                diagnostics={"detour_ratio": 1.2},
            ),
        ),
        node_anchors={0: GridNode(0, 1), 1: GridNode(2, 1)},
        diagnostics={"failed_routes": 0},
    )
    rows = placements_to_output_rows((road_placement, building_placement), include_internal=True)

    result = build_debug_layers(
        features=(
            _feature("road-1", "road", ProcessKind.ROAD, LineString([(0, 8), (24, 8)])),
            _feature("barn-1", "barns", ProcessKind.BUILDING_OUTLINE, building_shape),
        ),
        topology=topology,
        routing=routes,
        occupancy=occupancy,
        placements=(road_placement, building_placement),
        output_rows=rows,
        grid_index=grid,
    )

    assert set(result.layers) >= {
        "source_features",
        "topology_nodes",
        "topology_edges",
        "routed_paths",
        "raster_spines",
        "route_anchors",
        "occupancy_linear_surface",
        "building_footprints",
        "final_rows",
    }
    assert result.layers["source_features"].loc[0, "feature_id"] == "road-1"
    assert result.layers["routed_paths"].loc[0, "detour_ratio"] == 1.2
    assert result.layers["raster_spines"].edge_id.tolist() == [10, 10, 10]
    assert result.layers["raster_spines"].progress.tolist() == [0.0, 0.5, 1.0]
    assert result.layers["route_anchors"].node_id.tolist() == [0, 1]
    assert result.layers["occupancy_linear_surface"].loc[0, "object_id"] == "road-1"
    assert result.layers["building_footprints"].loc[0, "config_name"] == "barns"
    assert result.layers["final_rows"].name.tolist() == ["road", "barns"]
    assert result.diagnostics["topology"]["topology_note"] == "ok"
    assert result.diagnostics["routing"]["failed_routes"] == 0


def test_rail_and_barn_debug_export_are_revalidated_as_first_class_layers() -> None:
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind, ProcessKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    rail = _placement(
        config_name="rail",
        feature_id="rail-1",
        layer=LayerKind.LINEAR_OBJECT,
        cell=GridCell(0, 0),
        menu="Rail",
        cat1="Track",
        process=ProcessKind.RAIL,
    )
    barn = _placement(
        config_name="barns",
        feature_id="barn-1",
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 0),
        menu="Buildings",
        cat1="Barn",
        process=ProcessKind.BUILDING_OUTLINE,
        diagnostics={
            "selected_footprint_polygon": grid.cell_polygon(GridCell(1, 0)),
            "output_xidx": 0.5,
            "output_yidx": -0.5,
        },
    )

    result = build_debug_layers(
        features=(
            _feature("rail-1", "rail", ProcessKind.RAIL, LineString([(0, 0), (16, 0)])),
            _feature("barn-1", "barns", ProcessKind.BUILDING_OUTLINE, grid.cell_polygon(GridCell(1, 0))),
        ),
        placements=(rail, barn),
        output_rows=placements_to_output_rows((rail, barn), include_internal=True),
        grid_index=grid,
    )

    assert result.layers["source_features"].process.tolist() == ["rail", "building_outline"]
    assert set(result.layers["final_rows"].name.tolist()) == {"rail", "barns"}
    assert result.layers["building_footprints"].feature_id.tolist() == ["barn-1"]


def test_building_footprints_layer_reports_missing_selected_polygon() -> None:
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
    from terrain_extraction.osm_extraction.models import (
        CMType,
        GridCell,
        GridKind,
        LayerKind,
        PlacementRecord,
        ProcessKind,
    )
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    barn = PlacementRecord(
        layer=LayerKind.BUILDING,
        grid_kind=GridKind.SUB_SQUARE,
        cells=(GridCell(1, 0),),
        config_name="barns",
        feature_id="barn-1",
        priority=2,
        cm_type=CMType(menu="Buildings", cat1="Barn"),
        score=1.0,
        diagnostics={
            "process": ProcessKind.BUILDING_OUTLINE.value,
            "output_xidx": 0.5,
            "output_yidx": -0.5,
        },
    )

    result = build_debug_layers(
        placements=(barn,),
        output_rows=placements_to_output_rows((barn,), include_internal=True),
        grid_index=grid,
    )

    assert "building_footprints" not in result.layers
    assert result.diagnostics["layer_errors"] == (
        {
            "layer": "building_footprints",
            "index": 0,
            "error": "ValueError: building placement is missing selected_footprint_polygon",
        },
    )


def test_debug_validation_reports_actual_building_footprint_road_overlap() -> None:
    from terrain_extraction.osm_extraction.debug_export import build_debug_layers
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

    grid = _grid()
    road = _placement(
        config_name="road",
        feature_id="road-1",
        layer=LayerKind.LINEAR_SURFACE,
        cell=GridCell(0, 0),
        menu="Roads",
        cat1="Dirt",
    )
    building = _placement(
        config_name="barns",
        feature_id="barn-1",
        layer=LayerKind.BUILDING,
        cell=GridCell(1, 0),
        menu="Buildings",
        cat1="Barn",
        diagnostics={
            "selected_footprint_polygon": Polygon([(4, 0), (12, 0), (12, 8), (4, 8)]),
            "output_xidx": 0,
            "output_yidx": -0.5,
        },
    )

    result = build_debug_layers(
        placements=(road, building),
        output_rows=placements_to_output_rows((road, building), include_internal=True),
        grid_index=grid,
    )

    assert {
        "layer": "building_road_validation",
        "index": 1,
        "error": "ValueError: building footprint overlaps linear placement",
        "linear_index": 0,
        "cell": (0, 0),
    } in result.diagnostics["layer_errors"]


def test_debug_export_writes_optional_geojson_and_records_layer_failures() -> None:
    from terrain_extraction.osm_extraction.debug_export import (
        build_debug_layers,
        write_debug_geojson,
    )

    result = build_debug_layers(
        features=(_feature("bad", "broken", "bad-process", Point(0, 0)),),
        output_rows=({"xidx": 0, "yidx": 0, "name": "broken", "_layer": "ground"},),
        grid_index=_grid(),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        written = write_debug_geojson(result, output_dir)
        assert (output_dir / "source_features.geojson").exists()
        assert written["source_features"].name == "source_features.geojson"

    assert result.diagnostics["layer_errors"]
    assert "source_features" in result.diagnostics["layer_errors"][0]["layer"]


def test_pipeline_and_osm_processor_use_new_debug_export_by_default() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline
    from terrain_extraction.osm_processor import OSMProcessor

    grid = _grid()
    placement = _placement(
        config_name="rail",
        feature_id="rail-1",
        layer=LayerKind.LINEAR_OBJECT,
        cell=GridCell(0, 0),
        menu="Rail",
        cat1="Track",
        process=ProcessKind.RAIL,
    )
    events: list[str] = []
    pipeline = ExtractionPipeline(
        ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=123,
            progress=lambda stage, value, message=None: events.append(stage),
        )
    )

    result = pipeline.run_debug_export(
        features=(_feature("rail-1", "rail", ProcessKind.RAIL, LineString([(0, 0), (8, 0)])),),
        placements=(placement,),
        output_rows=({"xidx": 0, "yidx": 0, "name": "rail", "_layer": "linear_object"},),
        grid_index=grid,
    )

    assert result.diagnostics["debug_export"].layers["final_rows"].name.tolist() == ["rail"]
    assert result.stats.counts["debug_layers"] >= 2
    assert events == ["debug_export"]

    processor = OSMProcessor.__new__(OSMProcessor)
    processor.pipeline = pipeline
    processor.features = ()
    processor.placements = (placement,)
    processor.output_rows = result.output_rows
    processor.grid_index = grid
    processor.idx_bbox = [0, 0, 3, 3]

    geometries = processor.get_geometries(crs=CRS.from_epsg(3857))

    assert len(geometries["rail"]) == 1
    assert geometries["rail"][0].equals(grid.cell_polygon(GridCell(0, 0)))
