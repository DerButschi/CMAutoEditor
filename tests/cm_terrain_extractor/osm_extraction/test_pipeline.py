from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_pipeline_records_progress_and_wraps_legacy_processor() -> None:
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    events: list[tuple[str, float, str | None]] = []
    context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=123,
        progress=lambda stage, value, message=None: events.append((stage, value, message)),
    )
    processor = _LegacyProcessor()

    result = ExtractionPipeline(context).run_legacy(processor, osm_data={"features": []})

    assert processor.calls == ["preprocess_osm_data", "run_processors", "post_process", "get_output"]
    assert result.output_rows == ({"x": 1, "y": 2, "name": "forest"},)
    assert result.stats.counts["output_rows"] == 1
    assert [event[0] for event in events] == [
        "preprocess_osm_data",
        "run_processors",
        "post_process",
        "output_assembly",
    ]
    assert events[-1] == ("output_assembly", 1.0, "Legacy output rows assembled")


def test_context_initializes_deterministic_rng_and_noop_progress() -> None:
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    first = ExtractionContext.create(profile="cold_war", bbox=object(), config_path="cfg.json", seed=99)
    second = ExtractionContext.create(profile="cold_war", bbox=object(), config_path="cfg.json", seed=99)

    assert first.rng.integers(0, 10_000, size=5).tolist() == second.rng.integers(
        0,
        10_000,
        size=5,
    ).tolist()
    assert first.progress("stage", 0.5, None) is None


def test_building_fitter_occupancy_uses_resolved_linear_rows() -> None:
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import (
        CMType,
        GridCell,
        GridKind,
        LayerKind,
        PlacementRecord,
    )
    from terrain_extraction.osm_extraction.pipeline import _resolved_occupancy_for_building_fitter

    grid = GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=4,
        height=2,
    )
    weak = PlacementRecord(
        layer=LayerKind.LINEAR_SURFACE,
        grid_kind=GridKind.NORMAL,
        cells=(GridCell(0, 0), GridCell(1, 0)),
        config_name="road",
        feature_id="weak-road",
        priority=5,
        cm_type=CMType(menu="Roads", cat1="Dirt"),
        score=1.0,
    )
    strong = PlacementRecord(
        layer=LayerKind.LINEAR_SURFACE,
        grid_kind=GridKind.NORMAL,
        cells=(GridCell(1, 0), GridCell(2, 0)),
        config_name="road",
        feature_id="strong-road",
        priority=1,
        cm_type=CMType(menu="Roads", cat1="Dirt"),
        score=1.0,
    )

    occupancy = _resolved_occupancy_for_building_fitter(grid, (weak, strong))

    assert occupancy.object_id_at(LayerKind.LINEAR_SURFACE, GridCell(0, 0)) == "weak-road"
    assert occupancy.object_id_at(LayerKind.LINEAR_SURFACE, GridCell(1, 0)) == "strong-road"
    assert occupancy.object_id_at(LayerKind.LINEAR_SURFACE, GridCell(2, 0)) == "strong-road"


def test_osm_processor_imports_without_streamlit_and_holds_pipeline(monkeypatch) -> None:
    config_path = "default_osm_config.json"

    for module_name in ["terrain_extraction.osm_processor", "streamlit"]:
        sys.modules.pop(module_name, None)

    original_import = builtins.__import__

    def import_without_streamlit(name: str, *args: object, **kwargs: object) -> object:
        if name == "streamlit":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_streamlit)

    module = importlib.import_module("terrain_extraction.osm_processor")
    processor = module.OSMProcessor("cold_war", SimpleNamespace(), config_path)

    assert processor.pipeline.context.profile == "cold_war"
    assert processor.pipeline.context.config_path == config_path
    assert hasattr(module.st, "progress")


def test_pipeline_run_owns_typed_orchestration_and_catalog_gap_diagnostics() -> None:
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
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

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
    pipeline = _TypedPipelineHarness(road_placement, building_placement, area_placement)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )
    grid_index = GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=4,
        height=3,
    )
    features = (
        FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 8), (24, 8)])),
        FeatureRecord(
            "building-1",
            1,
            "houses",
            ProcessKind.BUILDING_OUTLINE,
            4,
            Polygon([(16, 8), (24, 8), (24, 16), (16, 16)]),
        ),
        FeatureRecord("area-1", 2, "trees", ProcessKind.AREA, 5, Point(8, 8).buffer(4)),
    )

    config = ExtractionConfig.from_mapping({"linear_route_faithfulness": {"high": {"max_distance_m": 18.0}}})
    result = pipeline.run(
        features=features,
        config=config,
        grid_index=grid_index,
        bounds=(0, 0, 3, 2),
        linear_catalog_provider=lambda _features: {
            ProcessKind.ROAD: SimpleNamespace(
                catalog_gap_diagnostics=lambda _required_sets: (
                    {"process": "road", "required_directions": ("E", "N", "S", "W"), "failure_reason": "catalog_gap"},
                )
            )
        },
        building_catalog_provider=lambda _features: {},
    )

    assert pipeline.calls == ["topology", "routing", "tile_assignment", "building", "area", "output"]
    assert result.placements == (road_placement, building_placement, area_placement)
    assert result.output_rows == ({"name": "extent_marker"},)
    assert result.diagnostics["catalog_gaps"] == (
        {"process": "road", "required_directions": ("E", "N", "S", "W"), "failure_reason": "catalog_gap"},
    )
    for stage in (
        "topology",
        "routing",
        "tile_assignment",
        "building_fitting",
        "area_rasterization",
        "output_row_assembly",
        "road_validation",
    ):
        assert result.diagnostics["timings"][stage]["elapsed_ms"] >= 0.0
    assert result.diagnostics["routing_diagnostics"]["route_count"] == 1
    assert pipeline.route_kwargs["route_faithfulness_config"] is config.linear_route_faithfulness
    assert result.diagnostics["routing_diagnostics"]["placed_source_length_fraction_by_cm_type"]["0"] == 1.0
    assert result.diagnostics["tile_assignment_diagnostics"]["top_slowest_components"] == ()
    assert result.diagnostics["building_fitting_diagnostics"]["building_feature_count"] == 0


def test_pipeline_run_defaults_to_warn_for_invalid_road_output() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import (
        CMType,
        FeatureRecord,
        GridCell,
        LayerKind,
        ProcessKind,
    )
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    road_placement = _placement(
        layer=LayerKind.LINEAR_SURFACE,
        cell=GridCell(0, 0),
        config_name="road",
        priority=4,
        cm_type=CMType(menu="Roads", cat1="Paved 2", cat2="Road Tile 1", direction="Direction 2"),
    )
    pipeline = _RoadValidationPipelineHarness(road_placement)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    result = pipeline.run(
        features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 4, LineString([(0, 0), (8, 0)])),),
        config=ExtractionConfig.from_mapping({}),
        grid_index=GridIndex(
            origin_x=0,
            origin_y=0,
            x_axis_unit=(1.0, 0.0),
            y_axis_unit=(0.0, 1.0),
            width=3,
            height=3,
        ),
        bounds=(0, 0, 2, 2),
        linear_catalog_provider=lambda _features: {},
    )

    assert result.output_rows[0]["name"] == "road"
    assert result.diagnostics["road_validation_status"]["mode"] == "warn"
    assert result.diagnostics["road_validation_status"]["is_valid"] is False
    assert not result.diagnostics["road_validation"].is_valid


@pytest.mark.parametrize(
    ("config", "explicit_mode"),
    [
        ({"road_validation_mode": "strict"}, None),
        ({}, "strict"),
    ],
)
def test_pipeline_run_strict_mode_rejects_invalid_road_output(config, explicit_mode) -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import (
        CMType,
        FeatureRecord,
        GridCell,
        LayerKind,
        ProcessKind,
    )
    from terrain_extraction.osm_extraction.output_rows import OutputRowValidationError
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    road_placement = _placement(
        layer=LayerKind.LINEAR_SURFACE,
        cell=GridCell(0, 0),
        config_name="road",
        priority=4,
        cm_type=CMType(menu="Roads", cat1="Paved 2", cat2="Road Tile 1", direction="Direction 2"),
    )
    pipeline = _RoadValidationPipelineHarness(road_placement)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    with pytest.raises(OutputRowValidationError, match="road output issues"):
        pipeline.run(
            features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 4, LineString([(0, 0), (8, 0)])),),
            config=ExtractionConfig.from_mapping(config),
            grid_index=GridIndex(
                origin_x=0,
                origin_y=0,
                x_axis_unit=(1.0, 0.0),
                y_axis_unit=(0.0, 1.0),
                width=3,
                height=3,
            ),
            bounds=(0, 0, 2, 2),
            linear_catalog_provider=lambda _features: {},
            road_validation_mode=explicit_mode,
        )


def test_pipeline_warn_mode_contains_tile_assignment_failures_locally() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    routes, linear_state = _tile_failure_routes_and_state()
    pipeline = _TileFailureContainmentPipelineHarness(routes, linear_state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    result = pipeline.run(
        features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)])),),
        config=ExtractionConfig.from_mapping({}),
        grid_index=GridIndex(
            origin_x=0,
            origin_y=0,
            x_axis_unit=(1.0, 0.0),
            y_axis_unit=(0.0, 1.0),
            width=6,
            height=6,
        ),
        bounds=(0, 0, 5, 5),
        linear_catalog_provider=lambda _features: {ProcessKind.ROAD: _tile_failure_catalog(include_four_way=False)},
        road_validation_mode="warn",
    )

    assert result.diagnostics["tile_assignment_failures"] == (
        {
            "process": "road",
            "cell": (1, 1),
            "required_directions": ("E", "N", "S", "W"),
            "failure_reason": "catalog_gap",
            "hard_failure": True,
            "route_ids": (1, 2),
        },
    )
    road_rows = {
        (row["x"], row["y"])
        for row in result.output_rows
        if isinstance(row.get("cat2"), str) and row["cat2"].startswith("Road Tile ")
    }
    assert road_rows == {(0, 4), (1, 4), (2, 4)}
    assert {placement.feature_id for placement in result.placements} == {3}
    assert all(
        not {1, 2}.intersection(placement.diagnostics.get("contributing_route_ids", ()))
        for placement in result.placements
    )


def test_pipeline_strict_mode_fails_on_tile_assignment_failures() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, TileAssignmentError

    routes, linear_state = _tile_failure_routes_and_state()
    pipeline = _TileFailureContainmentPipelineHarness(routes, linear_state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    with pytest.raises(TileAssignmentError, match="tile assignment failures.*catalog_gap"):
        pipeline.run(
            features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)])),),
            config=ExtractionConfig.from_mapping({}),
            grid_index=GridIndex(
                origin_x=0,
                origin_y=0,
                x_axis_unit=(1.0, 0.0),
                y_axis_unit=(0.0, 1.0),
                width=6,
                height=6,
            ),
            bounds=(0, 0, 5, 5),
            linear_catalog_provider=lambda _features: {ProcessKind.ROAD: _tile_failure_catalog(include_four_way=False)},
            road_validation_mode="strict",
        )


def test_pipeline_warn_mode_reports_side_signature_component_failures() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    routes, linear_state = _side_signature_failure_routes_and_state()
    pipeline = _TileFailureContainmentPipelineHarness(routes, linear_state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    result = pipeline.run(
        features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)])),),
        config=ExtractionConfig.from_mapping({}),
        grid_index=GridIndex(
            origin_x=0,
            origin_y=0,
            x_axis_unit=(1.0, 0.0),
            y_axis_unit=(0.0, 1.0),
            width=6,
            height=6,
        ),
        bounds=(0, 0, 5, 5),
        linear_catalog_provider=lambda _features: {ProcessKind.ROAD: _side_signature_failure_catalog()},
        road_validation_mode="warn",
    )

    failures = result.diagnostics["tile_assignment_failures"]
    assert len(failures) == 1
    assert failures[0]["failure_reason"] == "no_compatible_tile_component"
    assert failures[0]["component_cells"] == ((0, 1), (1, 1))
    assert failures[0]["incompatible_edges"] == (
        {"cell_a": (0, 1), "direction": "E", "cell_b": (1, 1)},
    )
    tile_diagnostics = result.diagnostics["tile_assignment_diagnostics"]
    assert tile_diagnostics["state_component_diagnostics"][0]["solver_used"] == "path_dp"
    assert tile_diagnostics["top_slowest_components"][0]["elapsed_ms"] >= 0.0


def test_pipeline_strict_mode_fails_on_side_signature_component_failures() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, TileAssignmentError

    routes, linear_state = _side_signature_failure_routes_and_state()
    pipeline = _TileFailureContainmentPipelineHarness(routes, linear_state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    with pytest.raises(TileAssignmentError, match="no_compatible_tile_component"):
        pipeline.run(
            features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)])),),
            config=ExtractionConfig.from_mapping({}),
            grid_index=GridIndex(
                origin_x=0,
                origin_y=0,
                x_axis_unit=(1.0, 0.0),
                y_axis_unit=(0.0, 1.0),
                width=6,
                height=6,
            ),
            bounds=(0, 0, 5, 5),
            linear_catalog_provider=lambda _features: {ProcessKind.ROAD: _side_signature_failure_catalog()},
            road_validation_mode="strict",
        )


def test_pipeline_warn_mode_reports_intersection_fallback_drops() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    routes, linear_state = _intersection_fallback_routes_and_state()
    pipeline = _TileFailureContainmentPipelineHarness(routes, linear_state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    result = pipeline.run(
        features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)])),),
        config=ExtractionConfig.from_mapping({}),
        grid_index=GridIndex(
            origin_x=0,
            origin_y=0,
            x_axis_unit=(1.0, 0.0),
            y_axis_unit=(0.0, 1.0),
            width=6,
            height=6,
        ),
        bounds=(0, 0, 5, 5),
        linear_catalog_provider=lambda _features: {ProcessKind.ROAD: _tile_failure_catalog(include_four_way=False)},
        road_validation_mode="warn",
    )

    assert result.diagnostics["intersection_fallback_failures"] == (
        {
            "process": "road",
            "route_id": 2,
            "config_name": "road",
            "failure_reason": "anchor_fallback_drop",
            "decision": {
                "node_id": 99,
                "edge_id": 2,
                "action": "drop",
                "reason": "no_legal_t_junction_attachment",
            },
        },
    )
    assert result.output_rows


def test_pipeline_strict_mode_fails_on_intersection_fallback_drops() -> None:
    from shapely.geometry import LineString
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import FeatureRecord, ProcessKind
    from terrain_extraction.osm_extraction.pipeline import (
        ExtractionContext,
        IntersectionFallbackError,
    )

    routes, linear_state = _intersection_fallback_routes_and_state()
    pipeline = _TileFailureContainmentPipelineHarness(routes, linear_state)
    pipeline.context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=0,
    )

    with pytest.raises(IntersectionFallbackError, match="intersection fallback failures.*anchor_fallback_drop"):
        pipeline.run(
            features=(FeatureRecord("road-1", 0, "road", ProcessKind.ROAD, 1, LineString([(0, 0), (16, 0)])),),
            config=ExtractionConfig.from_mapping({}),
            grid_index=GridIndex(
                origin_x=0,
                origin_y=0,
                x_axis_unit=(1.0, 0.0),
                y_axis_unit=(0.0, 1.0),
                width=6,
                height=6,
            ),
            bounds=(0, 0, 5, 5),
            linear_catalog_provider=lambda _features: {ProcessKind.ROAD: _tile_failure_catalog(include_four_way=False)},
            road_validation_mode="strict",
        )


def test_osm_processor_run_processors_delegates_to_pipeline_run() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_processor import OSMProcessor

    pipeline = _ProcessorDelegationPipeline()
    processor = OSMProcessor.__new__(OSMProcessor)
    processor.pipeline = pipeline
    processor.grid_index = GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=2,
        height=2,
    )
    processor.idx_bbox = (0, 0, 1, 1)
    processor.effective_bbox_polygon = None
    processor.extraction_config = ExtractionConfig.from_mapping({})
    processor._typed_features_from_matched_elements = lambda: ()
    processor._tile_catalogs_for = lambda features: {}
    processor._building_catalogs_for = lambda features: {}
    processor._set_compatibility_df_from_placements = lambda: None

    processor.run_processors()

    assert pipeline.calls == ["run"]
    assert processor.features == ()
    assert processor.placements == ()
    assert processor.output_rows == ({"name": "extent_marker"},)
    assert processor.stats == "stats"


def test_source_aware_road_validation_diagnostics_classify_topology_endpoints() -> None:
    from shapely.geometry import LineString, Point
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        ProcessKind,
        TopologyEdge,
        TopologyGraph,
        TopologyNode,
    )
    from terrain_extraction.osm_extraction.pipeline import _source_aware_road_validation_diagnostics

    grid_index = GridIndex(
        origin_x=0,
        origin_y=0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=4,
        height=1,
    )
    topology = TopologyGraph(
        nodes=(
            TopologyNode(0, Point(0.5, 0.5)),
            TopologyNode(1, Point(16.5, 0.5)),
        ),
        edges=(TopologyEdge(0, 0, 1, LineString([(0.5, 0.5), (16.5, 0.5)]), ("road-1",), (0,), "road", ProcessKind.ROAD, 1),),
    )
    road_validation = SimpleNamespace(
        dangling_arms=(
            SimpleNamespace(cell=GridCell(0, 0)),
            SimpleNamespace(cell=GridCell(2, 0)),
            SimpleNamespace(cell=GridCell(3, 0)),
        )
    )

    diagnostics = _source_aware_road_validation_diagnostics(
        road_validation=road_validation,
        topology=topology,
        grid_index=grid_index,
    )

    assert diagnostics["topology_endpoint_cells"] == ((0, 0), (2, 0))
    assert diagnostics["unexplained_dangling_arm_cells"] == ((3, 0),)


class _LegacyProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def preprocess_osm_data(self, osm_data: object) -> None:
        self.calls.append("preprocess_osm_data")

    def run_processors(self) -> None:
        self.calls.append("run_processors")

    def post_process(self) -> None:
        self.calls.append("post_process")

    def get_output(self) -> pd.DataFrame:
        self.calls.append("get_output")
        return pd.DataFrame([{"x": 1, "y": 2, "name": "forest"}])


class _RoadValidationPipelineHarness:
    def __init__(self, road_placement) -> None:
        from terrain_extraction.osm_extraction.pipeline import ExtractionContext

        self.context = ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=0,
        )
        self.road_placement = road_placement

    from terrain_extraction.osm_extraction.pipeline import ExtractionPipeline

    run = ExtractionPipeline.run
    run_output_rows = ExtractionPipeline.run_output_rows

    def run_network_topology(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        return ExtractionResult(diagnostics={"network_topology": object()})

    def run_network_router(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

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

        return ExtractionResult(
            placements=(self.road_placement,),
            diagnostics={"tile_assignment": object()},
        )

    def run_area_rasterizer(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        return ExtractionResult()


class _TypedPipelineHarness:
    def __init__(self, road_placement, building_placement, area_placement) -> None:
        from terrain_extraction.osm_extraction.pipeline import ExtractionContext

        self.context = ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=0,
        )
        self.calls: list[str] = []
        self.road_placement = road_placement
        self.building_placement = building_placement
        self.area_placement = area_placement
        self.route_kwargs = {}

    from terrain_extraction.osm_extraction.pipeline import ExtractionPipeline

    run = ExtractionPipeline.run

    def run_network_topology(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("topology")
        return ExtractionResult(diagnostics={"network_topology": object()})

    def run_network_router(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("routing")
        self.route_kwargs = dict(_kwargs)
        return ExtractionResult(
            diagnostics={
                "network_routes": SimpleNamespace(
                    routes=(object(),),
                    linear_state=object(),
                    diagnostics={
                        "route_count": 1,
                        "placed_source_length_fraction_by_cm_type": {"0": 1.0},
                    },
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
            stats="stats",
            diagnostics={"road_validation": object()},
        )


class _TileFailureContainmentPipelineHarness:
    def __init__(self, routes, linear_state) -> None:
        from terrain_extraction.osm_extraction.pipeline import ExtractionContext

        self.context = ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=0,
        )
        self.routes = routes
        self.linear_state = linear_state

    from terrain_extraction.osm_extraction.pipeline import ExtractionPipeline

    run = ExtractionPipeline.run
    run_tile_assignment = ExtractionPipeline.run_tile_assignment
    run_output_rows = ExtractionPipeline.run_output_rows

    def run_network_topology(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        return ExtractionResult(diagnostics={"network_topology": object()})

    def run_network_router(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult, NetworkRoutingResult

        return ExtractionResult(
            diagnostics={
                "network_routes": NetworkRoutingResult(
                    routes=self.routes,
                    linear_state=self.linear_state,
                )
            }
        )

    def run_area_rasterizer(self, **_kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        return ExtractionResult()


class _ProcessorDelegationPipeline:
    def __init__(self) -> None:
        from terrain_extraction.osm_extraction.pipeline import ExtractionContext

        self.context = ExtractionContext.create(
            profile="cold_war",
            bbox=object(),
            config_path="default_osm_config.json",
            seed=0,
        )
        self.calls: list[str] = []

    def run(self, **kwargs):
        from terrain_extraction.osm_extraction.models import ExtractionResult

        self.calls.append("run")
        assert kwargs["features"] == ()
        assert kwargs["bounds"] == (0, 0, 1, 1)
        assert callable(kwargs["linear_catalog_provider"])
        assert callable(kwargs["building_catalog_provider"])
        return ExtractionResult(
            output_rows=({"name": "extent_marker"},),
            stats="stats",
            diagnostics={
                "occupancy": object(),
                "network_topology": object(),
                "network_routes": object(),
                "tile_assignment": object(),
            },
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


def _tile_failure_catalog(*, include_four_way: bool):
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    rows = [
        {"direction": 6, "row": 0, "col": 2, "r": (2, 3), "cost": 1.0},
        {"direction": 7, "row": 0, "col": 2, "l": (2, 3), "cost": 1.0},
        {"direction": 8, "row": 0, "col": 2, "u": (2, 3), "cost": 1.0},
        {"direction": 9, "row": 0, "col": 2, "d": (2, 3), "cost": 1.0},
        {"direction": 1, "row": 0, "col": 0, "l": (2, 3), "r": (2, 3), "cost": 1.0},
        {"direction": 0, "row": 0, "col": 1, "u": (2, 3), "d": (2, 3), "cost": 1.0},
    ]
    if include_four_way:
        rows.append(
            {
                "direction": 5,
                "row": 2,
                "col": 2,
                "l": (2, 3),
                "r": (2, 3),
                "u": (2, 3),
                "d": (2, 3),
                "cost": 2.0,
            }
        )
    return CompiledTileCatalog.from_records(rows, process=ProcessKind.ROAD)


def _tile_failure_routes_and_state():
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    def route(edge_id: int, cells: tuple[tuple[int, int], ...]) -> RouteRecord:
        tile_cells = tuple(GridCell(xidx, yidx) for xidx, yidx in cells)
        return RouteRecord(
            edge_id=edge_id,
            start_node_id=edge_id * 2,
            end_node_id=edge_id * 2 + 1,
            process=ProcessKind.ROAD,
            config_name="road",
            priority=1,
            nodes=tuple(GridNode(cell.xidx, cell.yidx) for cell in tile_cells),
            tile_cells=tile_cells,
            diagnostics={"source_feature_ids": (f"osm-{edge_id}",)},
        )

    routes = (
        route(1, ((0, 1), (1, 1), (2, 1))),
        route(2, ((1, 0), (1, 1), (1, 2))),
        route(3, ((0, 4), (1, 4), (2, 4))),
    )
    full_catalog = _tile_failure_catalog(include_four_way=True)
    state = LinearNetworkState(width=6, height=6, catalogs={ProcessKind.ROAD: full_catalog})
    for candidate in routes:
        assert state.reserve_path(candidate).success
    return routes, state


def _side_signature_failure_catalog():
    from terrain_extraction.osm_extraction.models import ProcessKind
    from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

    return CompiledTileCatalog.from_records(
        ({"direction": 1, "row": 0, "col": 0, "l": ("left-only",), "r": ("right-only",), "cost": 1.0},),
        process=ProcessKind.ROAD,
    )


def _side_signature_failure_routes_and_state():
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    tile_cells = (GridCell(0, 1), GridCell(1, 1))
    route = RouteRecord(
        edge_id=1,
        start_node_id=0,
        end_node_id=1,
        process=ProcessKind.ROAD,
        config_name="road",
        priority=1,
        nodes=tuple(GridNode(cell.xidx, cell.yidx) for cell in tile_cells),
        tile_cells=tile_cells,
        diagnostics={"source_feature_ids": ("osm-1",)},
    )
    catalog = _side_signature_failure_catalog()
    state = LinearNetworkState(width=6, height=6, catalogs={ProcessKind.ROAD: catalog})
    assert state.reserve_path(route).success
    return (route,), state


def _intersection_fallback_routes_and_state():
    from terrain_extraction.osm_extraction.linear_network_state import LinearNetworkState
    from terrain_extraction.osm_extraction.models import (
        GridCell,
        GridNode,
        ProcessKind,
        RouteRecord,
    )

    def route(edge_id: int, cells: tuple[tuple[int, int], ...], *, success: bool = True, diagnostics=None) -> RouteRecord:
        tile_cells = tuple(GridCell(xidx, yidx) for xidx, yidx in cells)
        return RouteRecord(
            edge_id=edge_id,
            start_node_id=99,
            end_node_id=edge_id,
            process=ProcessKind.ROAD,
            config_name="road",
            priority=1,
            nodes=tuple(GridNode(cell.xidx, cell.yidx) for cell in tile_cells),
            tile_cells=tile_cells,
            success=success,
            diagnostics={} if diagnostics is None else diagnostics,
        )

    kept = route(1, ((0, 1), (1, 1), (2, 1)))
    dropped = route(
        2,
        (),
        success=False,
        diagnostics={
            "failure_reason": "anchor_fallback_drop",
            "intersection_fallback_decision": {
                "node_id": 99,
                "edge_id": 2,
                "action": "drop",
                "reason": "no_legal_t_junction_attachment",
            },
        },
    )
    state = LinearNetworkState(width=6, height=6, catalogs={ProcessKind.ROAD: _tile_failure_catalog(include_four_way=False)})
    assert state.reserve_path(kept).success
    return (kept, dropped), state
