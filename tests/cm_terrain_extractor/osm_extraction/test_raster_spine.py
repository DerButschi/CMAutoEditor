from __future__ import annotations

import sys
from pathlib import Path

import pytest
from shapely.geometry import LineString

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def _grid(*, rotated: bool = False):
    from terrain_extraction.osm_extraction.grid_index import GridIndex

    if rotated:
        return GridIndex(
            origin_x=100.0,
            origin_y=50.0,
            x_axis_unit=(0.0, 1.0),
            y_axis_unit=(-1.0, 0.0),
            width=8,
            height=6,
            cell_size_m=8.0,
        )
    return GridIndex(
        origin_x=0.0,
        origin_y=0.0,
        x_axis_unit=(1.0, 0.0),
        y_axis_unit=(0.0, 1.0),
        width=8,
        height=6,
        cell_size_m=8.0,
    )


def _assert_ordered_progress(spine) -> None:
    assert len(spine.cells) == len(spine.progress) == len(spine.distance_m)
    assert spine.progress == tuple(sorted(spine.progress))
    assert spine.progress[0] == pytest.approx(0.0)
    assert spine.progress[-1] == pytest.approx(1.0)
    assert all(distance >= 0.0 for distance in spine.distance_m)


def test_horizontal_line_spine_uses_intersected_tile_cells() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.raster_spine import build_raster_spine

    spine = build_raster_spine(
        topology_edge_id=7,
        line=LineString([(4, 4), (28, 4)]),
        grid_index=_grid(),
    )

    assert spine.topology_edge_id == 7
    assert spine.cells == (GridCell(0, 0), GridCell(1, 0), GridCell(2, 0), GridCell(3, 0))
    assert spine.source_length_m == pytest.approx(24.0)
    _assert_ordered_progress(spine)


def test_vertical_line_spine_uses_intersected_tile_cells() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.raster_spine import build_raster_spine

    spine = build_raster_spine(
        topology_edge_id=8,
        line=LineString([(12, 4), (12, 28)]),
        grid_index=_grid(),
    )

    assert spine.cells == (GridCell(1, 0), GridCell(1, 1), GridCell(1, 2), GridCell(1, 3))
    _assert_ordered_progress(spine)


def test_diagonalish_line_spine_contains_ordered_line_support_cells() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.raster_spine import build_raster_spine

    spine = build_raster_spine(
        topology_edge_id=9,
        line=LineString([(4, 4), (28, 20)]),
        grid_index=_grid(),
    )

    assert spine.cells[0] == GridCell(0, 0)
    assert spine.cells[-1] == GridCell(3, 2)
    assert len(spine.cells) > 2
    assert len(set(spine.cells)) == len(spine.cells)
    _assert_ordered_progress(spine)


def test_spine_generation_respects_rotated_grid_axes() -> None:
    from terrain_extraction.osm_extraction.models import GridCell
    from terrain_extraction.osm_extraction.raster_spine import build_raster_spine

    grid = _grid(rotated=True)
    start = grid.projected_from_local(4, 4)
    end = grid.projected_from_local(28, 4)

    spine = build_raster_spine(
        topology_edge_id=10,
        line=LineString([(start.x, start.y), (end.x, end.y)]),
        grid_index=grid,
    )

    assert spine.cells == (GridCell(0, 0), GridCell(1, 0), GridCell(2, 0), GridCell(3, 0))
    _assert_ordered_progress(spine)
