from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pyproj.crs import CRS

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
