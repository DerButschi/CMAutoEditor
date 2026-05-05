from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from shapely.geometry import LineString, Point, mapping

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_feature_matcher_extracts_nested_tags_and_preserves_source_order() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.feature_matcher import FeatureMatcher
    from terrain_extraction.osm_extraction.models import ProcessKind

    config = ExtractionConfig.from_mapping(
        {
            "roads": {
                "tags": [["highway", "primary"]],
                "cm_types": [{"menu": "Roads", "cat1": "Paved 1", "tags": [["highway", "primary"]]}],
                "process": ["road_tiles"],
                "priority": 1,
            },
            "pavement": {
                "tags": [["surface", "asphalt"]],
                "cm_types": [{"menu": "Ground 2", "cat1": "Pavement", "tags": [["surface", "asphalt"]]}],
                "process": ["type_from_tag"],
                "priority": 4,
            },
        }
    )
    feature = SimpleNamespace(
        properties={"id": "way/7", "tags": {"highway": "primary", "surface": "asphalt"}},
        geometry=mapping(LineString([(0, 0), (1, 1)])),
    )

    records = FeatureMatcher(config).match_features([feature])

    assert [(record.config_name, record.process) for record in records] == [
        ("roads", ProcessKind.ROAD),
        ("pavement", ProcessKind.AREA),
    ]
    assert records[0].feature_id == "way/7"
    assert records[0].source_index == 0
    assert records[0].source_tags["highway"] == "primary"
    assert records[0].source_properties["id"] == "way/7"


def test_feature_matcher_supports_direct_properties_and_id_filters() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.feature_matcher import FeatureMatcher

    config = ExtractionConfig.from_mapping(
        {
            "allowed_tree": {
                "tags": [["natural", "tree"]],
                "allowed_ids": [11],
                "cm_types": [
                    {"menu": "Flavor Objects 2", "cat1": "Tree", "tags": [["natural", "tree"]]}
                ],
                "process": ["single_object_random"],
                "priority": 0,
            }
        }
    )
    features = [
        {"id": 11, "properties": {"natural": "tree"}, "geometry": mapping(Point(0, 0))},
        {"id": 12, "properties": {"natural": "tree"}, "geometry": mapping(Point(1, 1))},
    ]

    records = FeatureMatcher(config).match_features(features)

    assert [record.feature_id for record in records] == [11]
    assert records[0].source_tags == {"natural": "tree"}


def test_feature_matcher_is_deterministic_under_same_config_seed() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.feature_matcher import FeatureMatcher

    config = ExtractionConfig.from_mapping(
        {
            "streams": {
                "tags": [["waterway", "stream"]],
                "cm_types": [{"menu": "Roads", "cat1": "Stream", "tags": [["waterway", "stream"]]}],
                "process": ["stream_tiles"],
                "priority": 2,
            }
        },
        seed=123,
    )
    features = [
        {
            "id": f"way/{idx}",
            "properties": {"tags": {"waterway": "stream"}},
            "geometry": mapping(LineString([(idx, 0), (idx, 1)])),
        }
        for idx in range(3)
    ]

    first = FeatureMatcher(config).match_features(features)
    second = FeatureMatcher(config).match_features(features)

    assert first == second
    assert config.seed == 123
