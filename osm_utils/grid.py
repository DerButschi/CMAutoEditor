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
            xarr.append(x)
            yarr.append(y)
            xiarr.append(xidx)
            yiarr.append(yidx)
        
    grid_geometry1 = geopandas.points_from_xy(xarr, yarr)
    grid_geometry1_gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, grid_geometry1)

    xarr = []
    yarr = []
    xiarr = []
    yiarr = []
    for xidx, x in enumerate(np.linspace(xmin, xmax, n_squares_x + 1)):
        for yidx, y in enumerate(np.linspace(ymin, ymax, n_squares_y + 1)):
            xarr.append(x)
            yarr.append(y)
            xiarr.append(xidx-0.5)
            yiarr.append(yidx-0.5)

    grid_geometry2 = geopandas.points_from_xy(xarr, yarr)
    grid_geometry2_gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, grid_geometry2)

    diagonal_gdf = pandas.concat((grid_geometry1_gdf, grid_geometry2_gdf), ignore_index=True)
    diagonal_gdf.geometry = diagonal_gdf.geometry.buffer(4, resolution=1)

    if rotation_angle is not None:
        diagonal_gdf = _rotate_grid(diagonal_gdf, rotation_angle, rotation_center)

    return diagonal_gdf

def get_sub_square_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    xarr = []
    yarr = []
    xiarr = []
    yiarr = []
    for xidx, x in enumerate(np.linspace(xmin, xmax, n_squares_x * 2 + 1)):
        for yidx, y in enumerate(np.linspace(ymin, ymax, n_squares_y * 2 + 1)):
            xarr.append(x)
            yarr.append(y)
            xiarr.append(xidx / 2 - 0.5)
            yiarr.append(yidx / 2 - 0.5)

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