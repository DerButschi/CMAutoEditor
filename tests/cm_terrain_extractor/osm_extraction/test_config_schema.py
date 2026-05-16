from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyproj.crs import CRS
from shapely.geometry import Polygon

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_config_schema_validates_matching_filters_and_allowed_ids() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.models import ProcessKind

    config = ExtractionConfig.from_mapping(
        {
            "forest": {
                "tags": [["landuse", "forest"]],
                "required_tags": [["leaf_type", "broadleaved"]],
                "exclude_tags": [["access", "private"]],
                "allowed_ids": ["way/1"],
                "cm_types": [{"menu": "Ground 2", "cat1": "Woods", "tags": [["landuse", "forest"]]}],
                "process": ["type_from_tag"],
                "priority": 3,
            }
        }
    )

    entry = config.entry_by_name("forest")

    assert entry.processes == (ProcessKind.AREA,)
    assert entry.matches_tags({"landuse": "forest", "leaf_type": "broadleaved"}, "way/1")
    assert not entry.matches_tags({"landuse": "forest"}, "way/1")
    assert not entry.matches_tags(
        {"landuse": "forest", "leaf_type": "broadleaved", "access": "private"},
        "way/1",
    )
    assert not entry.matches_tags({"landuse": "forest", "leaf_type": "broadleaved"}, "way/2")


def test_config_schema_defaults_road_validation_mode_to_warn() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig

    config = ExtractionConfig.from_mapping({})

    assert config.road_validation_mode == "warn"


@pytest.mark.parametrize("mode", ["strict", "warn"])
def test_config_schema_accepts_road_validation_modes(mode: str) -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig

    config = ExtractionConfig.from_mapping({"road_validation_mode": mode})

    assert config.road_validation_mode == mode


def test_config_schema_rejects_unknown_road_validation_mode() -> None:
    from terrain_extraction.osm_extraction.config_schema import (
        ConfigValidationError,
        ExtractionConfig,
    )

    with pytest.raises(ConfigValidationError, match="road_validation_mode"):
        ExtractionConfig.from_mapping({"road_validation_mode": "repair"})


def test_config_schema_fails_unknown_process_early() -> None:
    from terrain_extraction.osm_extraction.config_schema import (
        ConfigValidationError,
        ExtractionConfig,
    )

    with pytest.raises(ConfigValidationError, match="bogus_process"):
        ExtractionConfig.from_mapping(
            {
                "bad": {
                    "tags": [["amenity", "bench"]],
                    "cm_types": [{"menu": "Flavor Objects 1", "cat1": "Bench"}],
                    "process": ["bogus_process"],
                    "priority": 1,
                }
            }
        )


def test_matched_or_first_cm_type_uses_feature_tags() -> None:
    from terrain_extraction.osm_extraction.config_schema import (
        ExtractionConfig,
        matched_or_first_cm_type,
    )

    config = ExtractionConfig.from_mapping(
        {
            "road": {
                "tags": [["highway", "primary"], ["highway", "unclassified"]],
                "cm_types": [
                    {"menu": "Roads", "cat1": "Paved 1", "tags": [["highway", "primary"]]},
                    {"menu": "Roads", "cat1": "Paved 2", "tags": [["highway", "unclassified"]]},
                ],
                "process": ["road_tiles"],
                "priority": 4,
            }
        }
    )

    entry = config.entry_by_name("road")

    assert matched_or_first_cm_type(entry, {"highway": "unclassified"}).cat1 == "Paved 2"
    assert matched_or_first_cm_type(entry, {"highway": "primary"}).cat1 == "Paved 1"


def test_barn_process_spelling_is_canonical_and_profile_compatible() -> None:
    from terrain_extraction.osm_extraction.config_schema import ExtractionConfig
    from terrain_extraction.osm_extraction.models import ProcessKind

    from profiles import process_to_building_type

    config = ExtractionConfig.from_mapping(
        {
            "barns": {
                "tags": [["building", "barn"]],
                "cm_types": [
                    {
                        "menu": "Independent Buildings",
                        "cat1": "Church",
                        "tags": [["building", "barn"]],
                    }
                ],
                "process": ["type_from_barn_outline"],
                "priority": 5,
            }
        }
    )

    assert config.entry_by_name("barns").legacy_processes == ("type_from_barn_outline",)
    assert config.entry_by_name("barns").processes == (ProcessKind.BUILDING_OUTLINE,)
    assert process_to_building_type["type_from_barn_outline"] == "barns"
    assert process_to_building_type["type_from_barn_outlines"] == "barns"


def test_osm_processor_exposes_path_to_config_with_read_only_legacy_alias() -> None:
    from terrain_extraction.osm_processor import OSMProcessor

    processor = OSMProcessor("cold_war", SimpleNamespace(), "default_osm_config.json")

    assert processor.path_to_config == "default_osm_config.json"
    assert processor.path_to_congih == "default_osm_config.json"
    with pytest.raises(AttributeError):
        processor.path_to_congih = "other.json"


def test_non_wgs84_bounding_box_uses_epsg_code_value() -> None:
    from terrain_extraction.bbox_utils import BoundingBox

    polygon = Polygon(
        [
            (400_000, 5_700_000),
            (400_080, 5_700_000),
            (400_080, 5_700_080),
            (400_000, 5_700_080),
        ]
    )

    bbox = BoundingBox(polygon, CRS.from_epsg(25832))

    assert bbox.crs_orig.to_epsg() == 25832
    assert bbox.polygon_wgs84.bounds[0] < 20
