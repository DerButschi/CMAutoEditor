from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from shapely.geometry import Point

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


class _FakeCRS:
    def to_epsg(self) -> int:
        return 25832


class _FakeBBox:
    crs_projected = _FakeCRS()

    def __init__(self, origin: Point, x_axis_end: Point, y_axis_end: Point) -> None:
        self._origin = origin
        self._x_axis_end = x_axis_end
        self._y_axis_end = y_axis_end

    def get_reference_points(self, crs: _FakeCRS | None = None) -> tuple[Point, Point, Point]:
        return self._origin, self._x_axis_end, self._y_axis_end

    def get_rotation_angle(self) -> float:
        dx = self._x_axis_end.x - self._origin.x
        dy = self._x_axis_end.y - self._origin.y
        return math.degrees(math.atan2(dy, dx))


def test_grid_index_matches_legacy_representative_grid_views() -> None:
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import GridCell, GridKind
    from terrain_extraction.osm_utils.grid import get_all_grids

    grid = GridIndex.from_bbox(_FakeBBox(Point(100, 200), Point(124, 200), Point(100, 216)))
    legacy_normal, legacy_diagonal, legacy_sub_square = get_all_grids(100, 200, 124, 216, 3, 2)

    assert grid.width == 3
    assert grid.height == 2
    assert grid.cell_center(GridCell(1, 0)).equals_exact(Point(112, 204), tolerance=0.001)
    assert grid.cell_polygon(GridCell(1, 0)).area == pytest.approx(64.0)

    normal_view = grid.to_geodataframe(GridKind.NORMAL)
    sub_square_view = grid.to_geodataframe(GridKind.SUB_SQUARE)
    diagonal_view = grid.to_geodataframe(GridKind.DIAGONAL)

    assert normal_view.loc[:, ["xidx", "yidx", "x", "y"]].values.tolist() == legacy_normal.loc[
        :, ["xidx", "yidx", "x", "y"]
    ].values.tolist()
    assert sub_square_view.loc[:, ["xidx", "yidx", "x", "y"]].values.tolist() == legacy_sub_square.loc[
        :, ["xidx", "yidx", "x", "y"]
    ].values.tolist()
    assert diagonal_view.loc[:, ["xidx", "yidx", "x", "y"]].values.tolist() == legacy_diagonal.loc[
        :, ["xidx", "yidx", "x", "y"]
    ].values.tolist()


def test_grid_index_projects_rotated_coordinates_with_affine_math() -> None:
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import GridCell, GridNode

    angle = math.radians(30)
    origin = Point(100, 200)
    x_axis = Point(origin.x + 24 * math.cos(angle), origin.y + 24 * math.sin(angle))
    y_axis = Point(origin.x - 16 * math.sin(angle), origin.y + 16 * math.cos(angle))
    grid = GridIndex.from_bbox(_FakeBBox(origin, x_axis, y_axis))

    cell = GridCell(1, 1)
    center = grid.cell_center(cell)
    assert grid.projected_to_cell(center.x, center.y) == cell

    near_node = grid.projected_from_local(15.7, 0.2)
    assert grid.projected_to_nearest_node(near_node.x, near_node.y) == GridNode(2, 0)

    assert grid.clipped_cell_window(-5, -2, 99, 1) == (0, 0, 2, 1)
    assert grid.clipped_cell_window(50, 50, 60, 60) is None


def test_sub_square_and_diagonal_helpers_document_fractional_conventions() -> None:
    from terrain_extraction.osm_extraction.grid_index import GridIndex
    from terrain_extraction.osm_extraction.models import GridCell

    grid = GridIndex.from_bbox(_FakeBBox(Point(0, 0), Point(16, 0), Point(0, 16)))

    sub_centers = grid.sub_square_centers(GridCell(1, 0))
    assert sorted(sub_centers) == [(0.75, -0.25), (0.75, 0.25), (1.25, -0.25), (1.25, 0.25)]
    assert sub_centers[(0.75, -0.25)].equals_exact(Point(10, 2), tolerance=0.001)
    assert grid.sub_square_polygon(1.25, 0.25).area == pytest.approx(16.0)

    diagonal_centers = grid.diagonal_centers(GridCell(1, 0))
    assert sorted(diagonal_centers) == [(0.5, 0), (1, -0.5), (1, 0.5), (1.5, 0)]
    assert diagonal_centers[(1.5, 0)].equals_exact(Point(16, 4), tolerance=0.001)
    assert grid.diagonal_polygon(1.5, 0).area == pytest.approx(32.0)
