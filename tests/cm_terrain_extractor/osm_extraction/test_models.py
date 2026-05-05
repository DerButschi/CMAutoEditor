from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import MappingProxyType

import pytest
from shapely.geometry import Point

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_models_import_and_records_are_immutable() -> None:
    from terrain_extraction.osm_extraction.models import (
        CMType,
        ExtractionResult,
        FeatureRecord,
        GridCell,
        GridKind,
        GridNode,
        LayerKind,
        PlacementRecord,
        ProcessKind,
    )
    from terrain_extraction.osm_extraction.stats import ExtractionStats

    cm_type = CMType(menu="Ground", cat1="Grass", cat2=None, direction=None, tile_id=None)
    feature = FeatureRecord(
        feature_id="way/1",
        source_index=0,
        config_name="forest",
        process=ProcessKind.AREA,
        priority=3,
        geometry=Point(0, 0),
        source_tags=MappingProxyType({"landuse": "forest"}),
        source_properties=MappingProxyType({"id": "way/1"}),
    )
    placement = PlacementRecord(
        layer=LayerKind.GROUND,
        grid_kind=GridKind.NORMAL,
        cells=(GridCell(1, 2),),
        config_name="forest",
        feature_id=feature.feature_id,
        priority=feature.priority,
        cm_type=cm_type,
        score=1.0,
        diagnostics=MappingProxyType({"source": "test"}),
    )
    result = ExtractionResult(
        features=(feature,),
        placements=(placement,),
        output_rows=(MappingProxyType({"x": 1, "y": 2}),),
        stats=ExtractionStats(),
    )

    assert feature.process is ProcessKind.AREA
    assert GridNode(1, 2).xidx == 1
    assert result.placements == (placement,)
    assert dataclasses.is_dataclass(feature)

    with pytest.raises(dataclasses.FrozenInstanceError):
        feature.priority = 4  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        placement.cells = ()  # type: ignore[misc]


def test_process_kind_knows_legacy_process_names() -> None:
    from terrain_extraction.osm_extraction.models import ProcessKind

    assert ProcessKind.from_legacy("road_tiles") is ProcessKind.ROAD
    assert ProcessKind.from_legacy("rail_tiles") is ProcessKind.RAIL
    assert ProcessKind.from_legacy("stream_tiles") is ProcessKind.STREAM
    assert ProcessKind.from_legacy("fence_tiles") is ProcessKind.FENCE
    assert ProcessKind.from_legacy("type_from_barn_outline") is ProcessKind.BUILDING_OUTLINE
