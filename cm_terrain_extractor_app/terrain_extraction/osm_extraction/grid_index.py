from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Point, Polygon
from terrain_extraction.osm_extraction.models import GridCell, GridKind, GridNode

_DEFAULT_ROW = {
    "z": -1,
    "menu": -1,
    "cat1": -1,
    "cat2": -1,
    "direction": -1,
    "id": -1,
    "name": -1,
    "priority": -1,
}


@dataclass(slots=True)
class GridIndex:
    origin_x: float
    origin_y: float
    x_axis_unit: tuple[float, float]
    y_axis_unit: tuple[float, float]
    width: int
    height: int
    cell_size_m: float = 8.0
    crs_epsg: int | None = None
    _gdf_cache: dict[GridKind, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_bbox(cls, bbox: Any, cell_size_m: float = 8.0) -> GridIndex:
        p0, p1, p2 = bbox.get_reference_points(bbox.crs_projected)
        x_dx = p1.x - p0.x
        x_dy = p1.y - p0.y
        y_dx = p2.x - p0.x
        y_dy = p2.y - p0.y
        x_len = math.hypot(x_dx, x_dy)
        y_len = math.hypot(y_dx, y_dy)
        if x_len <= 0 or y_len <= 0:
            raise ValueError("Bounding box reference axes must have positive length")

        crs_epsg = None
        if getattr(bbox, "crs_projected", None) is not None:
            crs_epsg = bbox.crs_projected.to_epsg()

        return cls(
            origin_x=p0.x,
            origin_y=p0.y,
            x_axis_unit=(x_dx / x_len, x_dy / x_len),
            y_axis_unit=(y_dx / y_len, y_dy / y_len),
            width=math.floor(x_len / cell_size_m),
            height=math.floor(y_len / cell_size_m),
            cell_size_m=cell_size_m,
            crs_epsg=crs_epsg,
        )

    def projected_from_local(self, local_x: float, local_y: float) -> Point:
        x = self.origin_x + self.x_axis_unit[0] * local_x + self.y_axis_unit[0] * local_y
        y = self.origin_y + self.x_axis_unit[1] * local_x + self.y_axis_unit[1] * local_y
        return Point(x, y)

    def local_from_projected(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.origin_x
        dy = y - self.origin_y
        return (
            dx * self.x_axis_unit[0] + dy * self.x_axis_unit[1],
            dx * self.y_axis_unit[0] + dy * self.y_axis_unit[1],
        )

    def projected_to_cell(self, x: float, y: float) -> GridCell:
        local_x, local_y = self.local_from_projected(x, y)
        return GridCell(
            math.floor(local_x / self.cell_size_m + 1e-9),
            math.floor(local_y / self.cell_size_m + 1e-9),
        )

    def projected_to_nearest_node(self, x: float, y: float) -> GridNode:
        local_x, local_y = self.local_from_projected(x, y)
        return GridNode(
            math.floor(local_x / self.cell_size_m + 0.5),
            math.floor(local_y / self.cell_size_m + 0.5),
        )

    def cell_center(self, cell: GridCell) -> Point:
        return self.projected_from_local(
            (cell.xidx + 0.5) * self.cell_size_m,
            (cell.yidx + 0.5) * self.cell_size_m,
        )

    def cell_polygon(self, cell: GridCell) -> Polygon:
        min_x = cell.xidx * self.cell_size_m
        min_y = cell.yidx * self.cell_size_m
        return self._polygon_from_local_offsets(
            [
                (min_x, min_y),
                (min_x + self.cell_size_m, min_y),
                (min_x + self.cell_size_m, min_y + self.cell_size_m),
                (min_x, min_y + self.cell_size_m),
            ]
        )

    def sub_square_centers(self, cell: GridCell) -> dict[tuple[float, float], Point]:
        quarter = 0.25
        offset = self.cell_size_m / 4
        center_x = (cell.xidx + 0.5) * self.cell_size_m
        center_y = (cell.yidx + 0.5) * self.cell_size_m
        return {
            (cell.xidx - quarter, cell.yidx - quarter): self.projected_from_local(center_x - offset, center_y - offset),
            (cell.xidx - quarter, cell.yidx + quarter): self.projected_from_local(center_x - offset, center_y + offset),
            (cell.xidx + quarter, cell.yidx - quarter): self.projected_from_local(center_x + offset, center_y - offset),
            (cell.xidx + quarter, cell.yidx + quarter): self.projected_from_local(center_x + offset, center_y + offset),
        }

    def sub_square_polygon(self, xidx: float, yidx: float) -> Polygon:
        center_x = (xidx + 0.5) * self.cell_size_m
        center_y = (yidx + 0.5) * self.cell_size_m
        half_size = self.cell_size_m / 4
        return self._polygon_from_center_offsets(
            center_x,
            center_y,
            [(-half_size, -half_size), (half_size, -half_size), (half_size, half_size), (-half_size, half_size)],
        )

    def diagonal_centers(self, cell: GridCell) -> dict[tuple[float, float], Point]:
        half = 0.5
        center_x = (cell.xidx + 0.5) * self.cell_size_m
        center_y = (cell.yidx + 0.5) * self.cell_size_m
        offset = self.cell_size_m / 2
        return {
            (cell.xidx - half, cell.yidx): self.projected_from_local(center_x - offset, center_y),
            (cell.xidx + half, cell.yidx): self.projected_from_local(center_x + offset, center_y),
            (cell.xidx, cell.yidx - half): self.projected_from_local(center_x, center_y - offset),
            (cell.xidx, cell.yidx + half): self.projected_from_local(center_x, center_y + offset),
        }

    def diagonal_polygon(self, xidx: float, yidx: float) -> Polygon:
        center_x = (xidx + 0.5) * self.cell_size_m
        center_y = (yidx + 0.5) * self.cell_size_m
        radius = self.cell_size_m / 2
        return self._polygon_from_center_offsets(
            center_x,
            center_y,
            [(radius, 0), (0, -radius), (-radius, 0), (0, radius)],
        )

    def clipped_cell_window(
        self,
        min_xidx: int,
        min_yidx: int,
        max_xidx: int,
        max_yidx: int,
    ) -> tuple[int, int, int, int] | None:
        clipped_min_x = max(0, min_xidx)
        clipped_min_y = max(0, min_yidx)
        clipped_max_x = min(self.width - 1, max_xidx)
        clipped_max_y = min(self.height - 1, max_yidx)
        if clipped_min_x > clipped_max_x or clipped_min_y > clipped_max_y:
            return None
        return clipped_min_x, clipped_min_y, clipped_max_x, clipped_max_y

    def to_geodataframe(self, grid_kind: GridKind = GridKind.NORMAL):
        if grid_kind in self._gdf_cache:
            return self._gdf_cache[grid_kind]

        import geopandas

        rows = []
        geometry = []
        for xidx in range(self.width):
            for yidx in range(self.height):
                cell = GridCell(xidx, yidx)
                if grid_kind is GridKind.NORMAL:
                    center = self.cell_center(cell)
                    rows.append({"x": center.x, "y": center.y, "xidx": xidx, "yidx": yidx, **_DEFAULT_ROW})
                    geometry.append(self.cell_polygon(cell))
                elif grid_kind is GridKind.SUB_SQUARE:
                    for sub_xidx, sub_yidx, center in self._ordered_sub_square_entries(cell):
                        rows.append({"x": center.x, "y": center.y, "xidx": sub_xidx, "yidx": sub_yidx, **_DEFAULT_ROW})
                        geometry.append(self.sub_square_polygon(sub_xidx, sub_yidx))
                elif grid_kind is GridKind.DIAGONAL:
                    for diag_xidx, diag_yidx, center in self._ordered_diagonal_entries(cell):
                        rows.append({"x": center.x, "y": center.y, "xidx": diag_xidx, "yidx": diag_yidx, **_DEFAULT_ROW})
                        geometry.append(self.diagonal_polygon(diag_xidx, diag_yidx))
                else:
                    raise ValueError(f"Unknown grid kind: {grid_kind}")

        gdf = geopandas.GeoDataFrame(rows, geometry=geometry)
        if self.crs_epsg is not None:
            gdf = gdf.set_crs(epsg=self.crs_epsg)
        self._gdf_cache[grid_kind] = gdf
        return gdf

    def _ordered_sub_square_entries(self, cell: GridCell) -> list[tuple[float, float, Point]]:
        centers = self.sub_square_centers(cell)
        keys = [
            (cell.xidx - 0.25, cell.yidx - 0.25),
            (cell.xidx - 0.25, cell.yidx + 0.25),
            (cell.xidx + 0.25, cell.yidx - 0.25),
            (cell.xidx + 0.25, cell.yidx + 0.25),
        ]
        return [(xidx, yidx, centers[(xidx, yidx)]) for xidx, yidx in keys]

    def _ordered_diagonal_entries(self, cell: GridCell) -> list[tuple[float, float, Point]]:
        centers = self.diagonal_centers(cell)
        keys = [
            (cell.xidx - 0.5, cell.yidx),
            (cell.xidx + 0.5, cell.yidx),
            (cell.xidx, cell.yidx - 0.5),
            (cell.xidx, cell.yidx + 0.5),
        ]
        return [(xidx, yidx, centers[(xidx, yidx)]) for xidx, yidx in keys]

    def _polygon_from_center_offsets(
        self,
        center_x: float,
        center_y: float,
        offsets: list[tuple[float, float]],
    ) -> Polygon:
        return self._polygon_from_local_offsets([(center_x + x_offset, center_y + y_offset) for x_offset, y_offset in offsets])

    def _polygon_from_local_offsets(self, local_points: list[tuple[float, float]]) -> Polygon:
        return Polygon([(point.x, point.y) for point in (self.projected_from_local(x, y) for x, y in local_points)])
