from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _placement(layer, cells, *, priority=10, feature_id="feature-1", config_name="test"):
    from terrain_extraction.osm_extraction.models import CMType, GridKind, PlacementRecord

    return PlacementRecord(
        layer=layer,
        grid_kind=GridKind.NORMAL,
        cells=tuple(cells),
        config_name=config_name,
        feature_id=feature_id,
        priority=priority,
        cm_type=CMType(menu="Ground", cat1="Grass"),
        score=1.0,
    )


def test_occupancy_places_releases_and_reports_layer_conflicts() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    occupancy = OccupancyModel(width=3, height=3)
    road = _placement(LayerKind.LINEAR_SURFACE, [GridCell(1, 1)], priority=5, feature_id="road-1")

    decision = occupancy.place(road, object_id="road-1")

    assert decision.allowed is True
    assert occupancy.is_blocked(GridCell(1, 1), LayerKind.BUILDING)
    assert occupancy.object_id_at(LayerKind.LINEAR_SURFACE, GridCell(1, 1)) == "road-1"

    building = _placement(LayerKind.BUILDING, [GridCell(1, 1)], priority=2, feature_id="building-1")
    blocked = occupancy.can_place(building)

    assert blocked.allowed is False
    assert any("linear_surface" in reason for reason in blocked.reasons)

    occupancy.release("road-1")

    assert not occupancy.is_blocked(GridCell(1, 1), LayerKind.BUILDING)
    assert occupancy.place(building, object_id="building-1").allowed is True


def test_reserved_cells_block_all_normal_placement_with_readable_reason() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    occupancy = OccupancyModel(width=2, height=2)
    cell = GridCell(0, 1)

    assert occupancy.reserve([cell], object_id="corridor", metadata={"source": "route"}).allowed is True

    foliage = _placement(LayerKind.FOLIAGE, [cell], priority=9, feature_id="trees")
    decision = occupancy.can_place(foliage)

    assert decision.allowed is False
    assert decision.conflicts[0].blocking_layer is LayerKind.RESERVED
    assert decision.conflicts[0].blocking_object_id == "corridor"
    assert "reserved" in decision.reasons[0]
    assert occupancy.metadata["corridor"]["source"] == "route"


def test_occupancy_tracks_rank_and_allows_explicit_stronger_replacement() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    occupancy = OccupancyModel(width=2, height=2)
    cell = GridCell(1, 1)
    weaker = _placement(LayerKind.FOLIAGE, [cell], priority=20, feature_id="weak")
    stronger = _placement(LayerKind.FOLIAGE, [cell], priority=5, feature_id="strong")

    assert occupancy.place(weaker, object_id="weak").allowed is True

    blocked = occupancy.can_place(stronger)
    assert blocked.allowed is False
    assert "occupied" in blocked.reasons[0]

    replaced = occupancy.place(stronger, object_id="strong", allow_replace=True)
    assert replaced.allowed is True
    assert occupancy.object_id_at(LayerKind.FOLIAGE, cell) == "strong"
    assert occupancy.rank_at(LayerKind.FOLIAGE, cell) == 5
    assert "weak" not in occupancy.metadata


def test_occupancy_rejects_out_of_bounds_cells_before_array_access() -> None:
    from terrain_extraction.osm_extraction.models import GridCell, LayerKind
    from terrain_extraction.osm_extraction.occupancy import OccupancyModel

    occupancy = OccupancyModel(width=1, height=1)
    placement = _placement(LayerKind.GROUND, [GridCell(1, 0)])
    decision = occupancy.can_place(placement)

    assert decision.allowed is False
    assert decision.conflicts == ()
    assert decision.reasons == ("cell (1, 0) is outside the grid",)

    with pytest.raises(ValueError, match="outside the grid"):
        occupancy.place(placement, object_id="outside")
