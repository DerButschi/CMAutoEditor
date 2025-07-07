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
from shapely import Polygon, Point, box
from shapely.affinity import rotate

def _create_geodataframe(xarr, yarr, xiarr, yiarr, geometry):
    neg1 = np.full(len(xarr), -1, dtype=int)
    gdf = geopandas.GeoDataFrame({
        'x': xarr, 
        'y': yarr, 
        'xidx': xiarr, 
        'yidx': yiarr, 
        'z': neg1,
        'menu': neg1,
        'cat1': neg1,
        'cat2': neg1,
        'direction': neg1,
        'id': neg1,
        'name': neg1,
        'priority': neg1,
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
    print('calculating grid...')    
    # xarr = []
    # yarr = []
    # xiarr = []
    # yiarr = []
    cell_size_x = (xmax - xmin) / n_squares_x
    cell_size_y = (ymax - ymin) / n_squares_y
    # for xidx, x in enumerate(np.linspace(xmin + cell_size_x / 2, xmax - cell_size_x / 2, n_squares_x)):
    #     for yidx, y in enumerate(np.linspace(ymin + cell_size_y / 2, ymax - cell_size_y / 2, n_squares_y)):
    #         xarr.append(x)
    #         yarr.append(y)
    #         xiarr.append(xidx)
    #         yiarr.append(yidx)

    xs = np.linspace(xmin + cell_size_x/2, xmax - cell_size_x/2, n_squares_x)
    ys = np.linspace(ymin + cell_size_y/2, ymax - cell_size_y/2, n_squares_y)

    # 2D grids of coords and indices
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    XI, YI = np.meshgrid(np.arange(n_squares_x), np.arange(n_squares_y), indexing='ij')

    # flatten to 1D
    xarr, yarr = X.ravel(), Y.ravel()
    xiarr, yiarr = XI.ravel(), YI.ravel()

    x0, x1 = xarr - cell_size_x / 2, xarr + cell_size_x / 2
    y0, y1 = yarr - cell_size_y / 2, yarr + cell_size_y / 2
    geometry = [box(x0i, y0i, x1i, y1i) for x0i,y0i,x1i,y1i in zip(x0, y0, x1, y1)]    

    # geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, cap_style=3)

    gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, geometry)

    if rotation_angle is not None:
        gdf = _rotate_grid(gdf, rotation_angle, rotation_center)
    
    return gdf

def get_diagonal_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    print('calculating diagonal grid...')    
    cell_size_x = (xmax - xmin) / n_squares_x
    cell_size_y = (ymax - ymin) / n_squares_y

    # 1D centers
    xs = np.linspace(xmin + cell_size_x/2, xmax - cell_size_x/2, n_squares_x)
    ys = np.linspace(ymin + cell_size_y/2, ymax - cell_size_y/2, n_squares_y)

    # 2D grids of centers and indices (matching your x‐outer, y‐inner order)
    Xc, Yc = np.meshgrid(xs, ys, indexing='ij')   # shapes (n_x, n_y)
    XI, YI = np.meshgrid(np.arange(n_squares_x), np.arange(n_squares_y), indexing='ij')

    # flatten to 1D
    Xc = Xc.ravel()   # len = n_x * n_y
    Yc = Yc.ravel()
    XI = XI.ravel()
    YI = YI.ravel()    
    # offsets
    offs = np.array([
        (-cell_size_x/2,  0.0,  -0.5,  0.0),   # x_off, y_off, xi_off, yi_off
        (+cell_size_x/2,  0.0,  +0.5,  0.0),
        ( 0.0,  -cell_size_y/2,  0.0,  -0.5),
        ( 0.0,  +cell_size_y/2,  0.0,  +0.5),
    ])   # shape (4,4)

    # repeat center arrays 4×
    X_rep = np.repeat(Xc, 4)
    Y_rep = np.repeat(Yc, 4)
    XI_rep = np.repeat(XI, 4)
    YI_rep = np.repeat(YI, 4)

    # tile offsets to match length
    off_tile = np.tile(offs, (Xc.size,1))  # shape (N*4,4)

    # apply
    xarr = X_rep + off_tile[:,0]
    yarr = Y_rep + off_tile[:,1]
    xiarr = XI_rep + off_tile[:,2]
    yiarr = YI_rep + off_tile[:,3]

    grid_geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, resolution=1)
    diagonal_gdf = _create_geodataframe(xarr, yarr, xiarr, yiarr, grid_geometry)

    if rotation_angle is not None:
        diagonal_gdf = _rotate_grid(diagonal_gdf, rotation_angle, rotation_center)

    return diagonal_gdf

def get_sub_square_grid(xmin, ymin, xmax, ymax, n_squares_x, n_squares_y, rotation_angle=None, rotation_center=None):
    print('calculating sub square grid...')    
    cell_size_x = (xmax - xmin) / n_squares_x
    cell_size_y = (ymax - ymin) / n_squares_y

    xs = np.linspace(xmin + cell_size_x/2, xmax - cell_size_x/2, n_squares_x)
    ys = np.linspace(ymin + cell_size_y/2, ymax - cell_size_y/2, n_squares_y)
    Xc, Yc = np.meshgrid(xs, ys, indexing='ij')
    XI, YI = np.meshgrid(np.arange(n_squares_x), np.arange(n_squares_y), indexing='ij')
    Xc, Yc, XI, YI = (arr.ravel() for arr in (Xc, Yc, XI, YI))

    # 2) quarter-cell offsets
    offsets = np.array([[-cell_size_x/4, -cell_size_y/4, -0.25, -0.25],
                        [-cell_size_x/4, +cell_size_y/4, -0.25, +0.25],
                        [+cell_size_x/4, -cell_size_y/4, +0.25, -0.25],
                        [+cell_size_x/4, +cell_size_y/4, +0.25, +0.25]])
    N = Xc.size
    centers = np.repeat(np.stack([Xc,Yc,XI,YI],axis=1), 4, axis=0)
    offs = np.tile(offsets, (N,1))
    xarr, yarr, xiarr, yiarr = (centers[:,i] + offs[:,i] for i in range(4))

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
