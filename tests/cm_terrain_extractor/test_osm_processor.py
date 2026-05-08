from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from pyproj.crs import CRS
from shapely import GeometryCollection, LineString, Point, Polygon

APP_DIR = Path(__file__).parents[2] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

def _processor_with_df(df: pd.DataFrame):
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor.config = {
        "default_ground": {
            "cm_types": [{"menu": "Ground", "cat1": "Grass"}],
        },
        "road": {
            "cm_types": [
                {"menu": "Roads", "cat1": "Paved"},
                {"menu": "Roads", "cat1": "Dirt"},
            ],
        },
        "forest": {
            "cm_types": [{"menu": "Ground", "cat1": "Forest"}],
        },
        "negative_overlay": {
            "cm_types": [{"menu": "Overlay", "cat1": "Mud"}],
        },
    }
    processor.df = df

    return processor


def test_post_process_resolves_duplicate_priorities_and_cm_rank() -> None:
    df = pd.DataFrame(
        [
            {
                "xidx": 0,
                "yidx": 0,
                "z": -1,
                "menu": "Ground",
                "cat1": "Grass",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "default_ground",
                "priority": -999,
            },
            {
                "xidx": 0,
                "yidx": 0,
                "z": -1,
                "menu": "Roads",
                "cat1": "Paved",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "road",
                "priority": 4,
            },
            {
                "xidx": 0,
                "yidx": 0,
                "z": -1,
                "menu": "Ground",
                "cat1": "Forest",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "forest",
                "priority": 5,
            },
            {
                "xidx": 0,
                "yidx": 0,
                "z": -1,
                "menu": "Overlay",
                "cat1": "Mud",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "negative_overlay",
                "priority": -1,
            },
            {
                "xidx": 1,
                "yidx": 0,
                "z": -1,
                "menu": "Roads",
                "cat1": "Dirt",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "road",
                "priority": 4,
            },
            {
                "xidx": 1,
                "yidx": 0,
                "z": -1,
                "menu": "Roads",
                "cat1": "Paved",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "road",
                "priority": 4,
            },
            {
                "xidx": 2,
                "yidx": 0,
                "z": -1,
                "menu": "Ground",
                "cat1": "Grass",
                "cat2": -1,
                "direction": -1,
                "id": -1,
                "name": "default_ground",
                "priority": -999,
            },
        ]
    )
    processor = _processor_with_df(df)

    processor.post_process()

    assert processor.df.loc[:, ["xidx", "yidx", "name", "cat1", "priority"]].to_dict("records") == [
        {"xidx": 0, "yidx": 0, "name": "road", "cat1": "Paved", "priority": 4},
        {"xidx": 1, "yidx": 0, "name": "road", "cat1": "Paved", "priority": 4},
        {"xidx": 2, "yidx": 0, "name": "default_ground", "cat1": "Grass", "priority": -999},
    ]


def test_layered_output_conflicts_keep_positive_priority_winner() -> None:
    from terrain_extraction.osm_extraction.models import (
        CMType,
        GridCell,
        GridKind,
        LayerKind,
        PlacementRecord,
    )
    from terrain_extraction.osm_processor import OSMProcessor

    cell = GridCell(209, 196)
    stream = PlacementRecord(
        layer=LayerKind.LINEAR_SURFACE,
        grid_kind=GridKind.NORMAL,
        cells=(cell,),
        config_name="stream",
        feature_id="stream-1",
        priority=-1,
        cm_type=CMType(menu="Roads", cat1="Stream"),
        score=1.0,
    )
    road = PlacementRecord(
        layer=LayerKind.LINEAR_SURFACE,
        grid_kind=GridKind.NORMAL,
        cells=(cell,),
        config_name="road",
        feature_id="road-1",
        priority=4,
        cm_type=CMType(menu="Roads", cat1="Paved"),
        score=1.0,
    )

    assert OSMProcessor._resolve_output_layer_conflicts((stream, road)) == (road,)


class _FeatureCollection:
    def __init__(self, features: list[SimpleNamespace]) -> None:
        self.features = features

    def __getitem__(self, key: str) -> list[SimpleNamespace]:
        if key != "features":
            raise KeyError(key)
        return self.features


def test_preprocess_matches_tags_before_projecting_irrelevant_geometry(monkeypatch) -> None:
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor.config = {
        "forest": {
            "tags": [["landuse", "forest"]],
            "cm_types": [{"menu": "Ground", "cat1": "Forest"}],
            "process": ["type_from_tag"],
            "priority": 2,
        }
    }
    processor._config_order = {"forest": 0}
    processor._tag_to_config_names = processor._build_tag_to_config_names()
    processor.bbox = SimpleNamespace(crs_projected=CRS.from_epsg(25832))
    processor.matched_elements = []
    processor._init_grid = lambda bbox: None
    processor._get_projected_geometry = lambda geometry: (_ for _ in ()).throw(
        AssertionError("irrelevant feature geometry should not be projected")
    )
    monkeypatch.setattr(
        "terrain_extraction.osm_processor.st.progress",
        lambda *args, **kwargs: SimpleNamespace(progress=lambda *args, **kwargs: None),
    )
    osm_data = _FeatureCollection(
        [
            SimpleNamespace(
                properties={"amenity": "bench"},
                geometry={"type": "Point", "coordinates": [8.0, 50.0]},
            )
        ]
    )

    processor.preprocess_osm_data(osm_data)

    assert processor.matched_elements == []


def test_preprocess_throttles_progress_updates(monkeypatch) -> None:
    from terrain_extraction.osm_processor import OSMProcessor

    progress_values = []
    processor = OSMProcessor.__new__(OSMProcessor)
    processor.config = {
        "forest": {
            "tags": [["landuse", "forest"]],
            "cm_types": [{"menu": "Ground", "cat1": "Forest"}],
            "process": ["type_from_tag"],
            "priority": 2,
        }
    }
    processor._config_order = {"forest": 0}
    processor._tag_to_config_names = processor._build_tag_to_config_names()
    processor.bbox = SimpleNamespace(crs_projected=CRS.from_epsg(25832))
    processor.matched_elements = []
    processor._init_grid = lambda bbox: None
    processor._get_projected_geometry = lambda geometry: geometry
    monkeypatch.setattr(
        "terrain_extraction.osm_processor.st.progress",
        lambda *args, **kwargs: SimpleNamespace(
            progress=lambda value, *args, **kwargs: progress_values.append(value)
        ),
    )
    osm_data = _FeatureCollection(
        [
            SimpleNamespace(
                properties={"landuse": "forest"},
                geometry=object(),
            )
            for _ in range(250)
        ]
    )

    processor.preprocess_osm_data(osm_data)

    assert len(progress_values) < 20
    assert progress_values[-1] == 1.0


def test_typed_features_are_clipped_to_effective_bbox() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor.extraction_config = ExtractionConfig.from_mapping(
        {
            "field": {
                "tags": [["landuse", "field"]],
                "cm_types": [{"menu": "Ground 3", "cat1": "Crop 1"}],
                "process": ["type_from_tag"],
                "priority": 1,
            },
            "bench": {
                "tags": [["amenity", "bench"]],
                "cm_types": [{"menu": "Flavor Objects 1", "cat1": "Bench"}],
                "process": ["single_object_random"],
                "priority": 1,
            },
        }
    )
    processor.effective_bbox_polygon = Polygon([(0, 0), (16, 0), (16, 16), (0, 16)])
    processor.matched_elements = [
        {
            "element": SimpleNamespace(properties={"id": "field-1", "landuse": "field"}),
            "geometry": Polygon([(8, 0), (24, 0), (24, 8), (8, 8)]),
            "name": "field",
            "idx": 0,
        },
        {
            "element": SimpleNamespace(properties={"id": "bench-1", "amenity": "bench"}),
            "geometry": Point(24, 24),
            "name": "bench",
            "idx": 1,
        },
    ]

    features = processor._typed_features_from_matched_elements()

    assert len(features) == 1
    assert features[0].feature_id == "field-1"
    assert features[0].geometry.bounds == (8.0, 0.0, 16.0, 8.0)


def test_grid_cell_indices_match_exact_pairs() -> None:
    from terrain_extraction.osm_utils.processing import _get_grid_indices_for_cells

    gdf = pd.DataFrame(
        {
            "xidx": [1, 1, 2, 2, 3],
            "yidx": [1, 2, 1, 2, 3],
            "value": [11, 12, 21, 22, 33],
        },
        index=[10, 11, 12, 13, 14],
    )

    assert _get_grid_indices_for_cells(gdf, [(1, 1), (2, 2)]).tolist() == [10, 13]


def test_cm_type_rank_distinguishes_cat2_values() -> None:
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor.config = {
        "foliage": {
            "cm_types": [
                {"menu": "Foliage", "cat1": "Tree A", "cat2": "density 1"},
                {"menu": "Foliage", "cat1": "Tree A", "cat2": "density 2"},
            ],
        }
    }

    assert processor._get_cm_type_rank("foliage", "Foliage", "Tree A", "density 2") == 1


def test_get_geometries_returns_building_geometry_from_diagonal_grid(monkeypatch) -> None:
    from terrain_extraction import osm_processor as osm_processor_module
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor._df_parts = []
    processor.df = pd.DataFrame(
        [
            {
                "xidx": 0.5,
                "yidx": 0.0,
                "menu": "Buildings",
                "cat1": "House",
                "cat2": "Building 1",
                "direction": "Direction 1",
                "name": "houses",
            }
        ]
    )
    processor.config = {"houses": {"process": ["type_from_residential_building_outline"]}}
    processor.bbox = SimpleNamespace(
        crs_projected=CRS.from_epsg(4326),
        get_rotation_angle=lambda: 0,
    )
    processor.gdf = gpd.GeoDataFrame(
        {"xidx": [0], "yidx": [0]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    processor.sub_square_grid_gdf = gpd.GeoDataFrame(
        {"xidx": [99], "yidx": [99]},
        geometry=[Point(99, 99).buffer(1)],
        crs="EPSG:4326",
    )
    processor.sub_square_grid_diagonal_gdf = gpd.GeoDataFrame(
        {"xidx": [0.5], "yidx": [0.0]},
        geometry=[Point(10, 10).buffer(1)],
        crs="EPSG:4326",
    )
    monkeypatch.setitem(
        osm_processor_module.process_to_building_type,
        "type_from_residential_building_outline",
        "residential_buildings",
    )
    monkeypatch.setattr(
        osm_processor_module,
        "get_building_outline_by_df_entry",
        lambda *args: (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), True),
    )
    monkeypatch.setattr(
        osm_processor_module.affinity,
        "rotate",
        lambda geometry, *args, **kwargs: geometry,
    )
    monkeypatch.setattr(
        osm_processor_module.affinity,
        "translate",
        lambda geometry, *args, **kwargs: geometry,
    )

    geometries = processor.get_geometries(crs=CRS.from_epsg(4326))

    assert len(geometries["houses"]) == 1


def test_get_geometries_returns_linear_feature_cells() -> None:
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor.__new__(OSMProcessor)
    processor._df_parts = []
    processor.df = pd.DataFrame([{"xidx": 0, "yidx": 0, "name": "road"}])
    processor.config = {"road": {"process": ["road_tiles"]}}
    processor.bbox = SimpleNamespace(crs_projected=CRS.from_epsg(4326))
    processor.gdf = gpd.GeoDataFrame(
        {"xidx": [0], "yidx": [0]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    processor.sub_square_grid_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    processor.sub_square_grid_diagonal_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    geometries = processor.get_geometries(crs=CRS.from_epsg(4326))

    assert len(geometries["road"]) == 1


def test_collect_network_data_keeps_linestrings_from_geometry_collection() -> None:
    from terrain_extraction.osm_utils.processing import collect_network_data

    processor = SimpleNamespace(
        effective_bbox_polygon=Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)]),
        network_graphs={},
    )
    element_entry = {
        "name": "road",
        "idx": 3,
        "geometry": GeometryCollection(
            [
                LineString([(0, 0), (1, 1)]),
                Point(0, 1),
            ]
        ),
    }

    collect_network_data(processor, {}, element_entry)

    lines = processor.network_graphs["road"]["lines"]
    assert lines.element_idx.tolist() == [3]
    assert lines.geometry.iloc[0].equals(LineString([(0, 0), (1, 1)]))


def test_create_line_graph_does_not_emit_query_bulk_future_warning() -> None:
    from terrain_extraction.osm_utils.processing import create_line_graph

    lines = gpd.GeoDataFrame(
        {"element_idx": [0, 1]},
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 1), (1, 0)]),
        ],
    )
    processor = SimpleNamespace(network_graphs={"road": {"lines": lines}})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        create_line_graph(
            processor,
            {"road": {"priority": 1}},
            "road",
            tqdm_string="test",
        )

    assert processor.network_graphs["road"]["line_graph"].number_of_edges() > 0


def test_get_matched_squares_does_not_emit_query_bulk_future_warning() -> None:
    from terrain_extraction.osm_utils.processing import _get_matched_squares

    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    processor = SimpleNamespace(
        sub_square_grid_gdf=gpd.GeoDataFrame(
            {"xidx": [0], "yidx": [0]},
            geometry=[square],
        ),
        sub_square_grid_diagonal_gdf=gpd.GeoDataFrame(
            {"xidx": [0.5], "yidx": [0.0]},
            geometry=[Point(0.5, 0.5).buffer(0.5)],
        ),
        occupancy_gdf=gpd.GeoDataFrame(
            {"priority": [10], "name": ["other"]},
            geometry=[Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])],
        ),
        _flush_occupancy_gdf_parts=lambda: None,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        squares = _get_matched_squares(processor, 1, square, diagonal=False)

    assert squares[["xidx", "yidx"]].to_dict("records") == [{"xidx": 0, "yidx": 0}]


def test_process_building_outlines_does_not_touch_matplotlib_when_debug_disabled(monkeypatch) -> None:
    from terrain_extraction.osm_utils import processing

    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    processor = SimpleNamespace(
        building_outlines={"houses": {}},
        _flush_occupancy_gdf_parts=lambda: None,
        gdf=gpd.GeoDataFrame(geometry=[square]),
        occupancy_gdf=gpd.GeoDataFrame(
            {"priority": [], "name": []},
            geometry=[],
        ),
        sub_square_grid_diagonal_gdf=gpd.GeoDataFrame(
            {"xidx": [0.5], "yidx": [0.0]},
            geometry=[Point(0.5, 0.5).buffer(0.5)],
        ),
        sub_square_grid_gdf=gpd.GeoDataFrame(
            {"xidx": [0], "yidx": [0]},
            geometry=[square],
        ),
        profile="cold_war",
    )
    monkeypatch.setattr(
        processing,
        "get_building_tiles",
        lambda building_type, profile: pd.DataFrame(),
    )
    monkeypatch.setattr(
        processing.plt,
        "gca",
        lambda: (_ for _ in ()).throw(AssertionError("debug plotting should be disabled")),
    )

    processing.process_building_outlines(
        processor,
        {"houses": {"priority": 1}},
        "houses",
        "residential_buildings",
        "test",
    )
