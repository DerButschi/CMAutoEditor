# Copyright (C) 2022  Nicolas Möser

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import geopandas
import pandas
from shapely import Polygon, Point

def _create_geodataframe(xarr, yarr, xiarr, yiarr, geometry):
    gdf = geopandas.GeoDataFrame({
        'x': xarr, 
        'y': yarr, 
        'xidx': xiarr, 
        'yidx': yiarr, 
        'z': [-1] * len(xarr),
        'menu': [-1] * len(xarr),
        'cat1': [-1] * len(xarr),
        'cat2': [-1] * len(xarr),
        'direction': [-1] * len(xarr),
        'id': [-1] * len(xarr),
        'name': [-1] * len(xarr),
        'priority': [-1] * len(xarr),
    }, 
    geometry=geometry
    )

    return gdf

def _rotate_grid(gdf, rotation_angle, rotation_center=None):
    if rotation_center is None:
        rotation_center = 'center'
    gdf.geometry = gdf.rotate(rotation_angle, origin=rotation_center)
    gdf.x = gdf.geometry.centroid.x
    gdf.y = gdf.geometry.centroid.y
    
    return gdf


def get_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    xarr = []
    yarr = []
    xiarr = []
    yiarr = []
    cell_size_x = (xmax - xmin) / n_squares_x
    cell_size_y = (ymax - ymin) / n_squares_y
    for xidx, x in enumerate(np.linspace(xmin + cell_size_x / 2, xmax - cell_size_x / 2, n_squares_x)):
        for yidx, y in enumerate(np.linspace(ymin + cell_size_y / 2, ymax - cell_size_y / 2, n_squares_y)):
            xarr.append(x)
            yarr.append(y)
            xiarr.append(xidx)
            yiarr.append(yidx)

    geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, cap_style=3)

    gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, geometry)

    if rotation_angle is not None:
        gdf = _rotate_grid(gdf, rotation_angle, rotation_center)
    
    return gdf

def get_diagonal_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    xarr = []
    yarr = []
    xiarr = []
    yiarr = []
    cell_size_x = (xmax - xmin) / n_squares_x
    cell_size_y = (ymax - ymin) / n_squares_y
    for xidx, x in enumerate(np.linspace(xmin + cell_size_x / 2, xmax - cell_size_x / 2, n_squares_x)):
        for yidx, y in enumerate(np.linspace(ymin + cell_size_y / 2, ymax - cell_size_y / 2, n_squares_y)):
            xarr.append(x - cell_size_x / 2)
            yarr.append(y)
            xiarr.append(xidx - 0.5)
            yiarr.append(yidx)
            xarr.append(x + cell_size_x / 2)
            yarr.append(y)
            xiarr.append(xidx + 0.5)
            yiarr.append(yidx)
            xarr.append(x)
            yarr.append(y - cell_size_y / 2)
            xiarr.append(xidx)
            yiarr.append(yidx - 0.5)
            xarr.append(x)
            yarr.append(y + cell_size_y / 2)
            xiarr.append(xidx)
            yiarr.append(yidx + 0.5)
        
    grid_geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, resolution=1)
    diagonal_gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, grid_geometry)

    if rotation_angle is not None:
        diagonal_gdf = _rotate_grid(diagonal_gdf, rotation_angle, rotation_center)

    return diagonal_gdf

def get_sub_square_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    xarr = []
    yarr = []
    xiarr = []
    yiarr = []
    cell_size_x = (xmax - xmin) / n_squares_x
    cell_size_y = (ymax - ymin) / n_squares_y
    for xidx, x in enumerate(np.linspace(xmin + cell_size_x / 2, xmax - cell_size_x / 2, n_squares_x)):
        for yidx, y in enumerate(np.linspace(ymin + cell_size_y / 2, ymax - cell_size_y / 2, n_squares_y)):
            xarr.append(x - 2)
            yarr.append(y - 2)
            xiarr.append(xidx - 0.25)
            yiarr.append(yidx - 0.25)
            xarr.append(x - 2)
            yarr.append(y + 2)
            xiarr.append(xidx - 0.25)
            yiarr.append(yidx + 0.25)
            xarr.append(x + 2)
            yarr.append(y - 2)
            xiarr.append(xidx + 0.25)
            yiarr.append(yidx - 0.25)
            xarr.append(x + 2)
            yarr.append(y + 2)
            xiarr.append(xidx + 0.25)
            yiarr.append(yidx + 0.25)

    geometry = geopandas.points_from_xy(xarr, yarr).buffer(2, cap_style=3)

    sub_square_grid_gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, geometry)

    if rotation_angle is not None:
        sub_square_grid_gdf = _rotate_grid(sub_square_grid_gdf, rotation_angle, rotation_center)

    return sub_square_grid_gdf

def get_all_grids(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    grid_gdf = get_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle, rotation_center)
    diagonal_grid_gdf = get_diagonal_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle, rotation_center)
    sub_square_grid_gdf = get_sub_square_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle, rotation_center)

    return grid_gdf, diagonal_grid_gdf, sub_square_grid_gdf

def get_reference_rectanlge_points(polygon, ref_rectangle):
    polygon_points = [Point(coord[0], coord[1]) for coord in polygon.exterior.coords]

    if not ref_rectangle.exterior.is_ccw:
        ref_rectangle = Polygon(ref_rectangle.exterior.coords[::-1])
    # get closest point in minimum rotated rectangle to first point of bounding box
    rectangle_points = [Point(*coord) for coord in ref_rectangle.exterior.coords]
    dist = [polygon_points[0].distance(pt) for pt in rectangle_points]
    min_idx = np.argmin(dist)

    # get rotation angle of x-axis, assumed to be defined by (x0, y0) -> (x1, y1)
    # since the last point in a polygon is always identical to the first point and np.argmin returns the first match,
    # there should always be min_idx + 1 within the array
    p0 = rectangle_points[min_idx]
    p1 = rectangle_points[min_idx + 1]
    if min_idx + 2 == len(rectangle_points):
        p2 = rectangle_points[1]
    else:
        p2 = rectangle_points[min_idx + 2]

    return p0, p1, p2
