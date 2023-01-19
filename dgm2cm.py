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

import pandas
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
import argparse
from shapely import Point, Polygon, MultiPoint
from shapely.ops import nearest_points
from pyproj.transformer import Transformer
import geopandas
from osm_utils.grid import get_all_grids

def z_on_plain(p1, p2, x, y):
    f1 = p2[0] - p1[0]
    f2 = p2[1] - p1[1]
    f3 = p2[2] - p1[2]

    z = f1 * f3 / (np.square(f1) + np.square(f2)) * (x - p1[0]) + f2 * f3 / (np.square(f1) + np.square(f2)) * (y - p1[1]) + p1[2]

    return z

def get_projected_bbox(input_crs, bbox_crs, points):
    transformer = Transformer.from_crs('epsg:{}'.format(input_crs), 'epsg:{}'.format(bbox_crs), always_xy=True)
    projected_points = []
    for point in points:
        projected_points.append(Point(transformer.transform(point.x, point.y)))
    
    return projected_points

def get_lon_lat_bounds(input_crs, polygon):
    transformer = Transformer.from_crs('epsg:{}'.format(input_crs), 'epsg:{}'.format(4326), always_xy=True)
    bounds = polygon.bounds
    return [*transformer.transform(bounds[0], bounds[1]), *transformer.transform(bounds[2], bounds[3])]

argparser = argparse.ArgumentParser()
argparser.add_argument('--dgm-dir', '-d', required=True, help='directory in which the dgm files are located')
argparser.add_argument('--bounding-box', '-b', required=True, help='Coordinates of box in which to extract data. 2 or 4 points. If an additional number is provided, the first number is interpreted as epsg-code.'
    'Otherwise 4326 (longitude/latitude) is assumend.', type=float, nargs='+'
)
argparser.add_argument('--input-crs', required=False, type=int, help='epsg-code of input data. [default: 4326]', default=4326)
argparser.add_argument('--contour', '-c', required=False, type=float, help='contour level distance (default: 5 m)', default=5.0)
argparser.add_argument('--output-name', '-o', required=False, type=str, help='output name (without file extension) (default: output)', default='output')
argparser.add_argument('--water-level-correction', '-w', required=False, type=float, nargs=4, help='correct elevation for the fact that in CM water does not flow downhill expects x,y coordinates of lowest and highest water level of one river.')
argparser.add_argument('--stride', '-s', required=False, type=int, help='ouput will contain only every stride-th point')

args = argparser.parse_args()

bbox_epsg = 4326
bbox_points = []
if not len(args.bounding_box) in [4, 5, 8, 9]:
    raise argparse.ArgumentError('--bounding-box requires either 2 or 4 points with an optional epsg-code prepended.')
elif len(args.bounding_box) in [5, 9]:
    bbox_epsg = int(args.bounding_box[0])
    bbox_points = get_projected_bbox(bbox_epsg, args.input_crs, 
        [Point(args.bounding_box[i], args.bounding_box[i+1]) for i in range(1, len(args.bounding_box), 2)])
else:
    bbox_points = get_projected_bbox(bbox_epsg, args.input_crs, 
        [Point(args.bounding_box[i], args.bounding_box[i+1]) for i in range(0, len(args.bounding_box), 2)])

if len(bbox_points) == 2:
    bbox_polygon = MultiPoint(bbox_points).envelope
    bbox_polygon_orig = bbox_polygon
else:
    bbox_polygon = MultiPoint(bbox_points).minimum_rotated_rectangle
    bbox_polygon_orig = Polygon([(p.x, p.y) for p in bbox_points])
    print('Rotated Rectangle: {} {} {} {} {} {} {} {}'.format(*np.array([bbox_polygon.exterior.coords[i] for i in range(len(bbox_polygon.exterior.coords)-1)]).flatten()))

print('Bounds: {}'.format(get_lon_lat_bounds(args.input_crs, bbox_polygon)))
xmin_request, ymin_request, xmax_request, ymax_request = bbox_polygon.bounds

df = None
print('Processing input files...')
for filename in os.listdir(args.dgm_dir):
    # if filename.endswith('.xyz.gz'):
    if os.path.isfile(os.path.join(args.dgm_dir, filename)):
        df_file = pandas.read_csv(os.path.join(args.dgm_dir, filename), delimiter=r"\s+", names=['x','y','z'])
        # df_file = df_file[df_file.x.between(xmin_request, xmax_request) & df_file.y.between(ymin_request, ymax_request)]
        if df is not None:
            df = pandas.concat((df, df_file))
        else:
            df = df_file

grid_cell_x = np.sort(df.x.unique())[1] - np.sort(df.x.unique())[0]
grid_cell_y = np.sort(df.y.unique())[1] - np.sort(df.y.unique())[0]
grid_size_x = df.x.max() + grid_cell_x / 2 - (df.x.min() - grid_cell_x / 2) 
grid_size_y = df.y.max() + grid_cell_y / 2 - (df.y.min() - grid_cell_y / 2)

print('Extracted data grid has size is {}m x {}m with a cell size of {}m x {}m.'.format(grid_size_x, grid_size_y, grid_cell_x, grid_cell_y))
print('This corresponds to {} x {} squares in CM.'.format(np.floor(grid_size_x / 8).astype(int), np.floor(grid_size_y / 8).astype(int)))

grid_size_x_cm = np.floor(grid_size_x / 8).astype(int)
grid_size_y_cm = np.floor(grid_size_y / 8).astype(int)

if args.water_level_correction is not None:
    p1_x_condition = df.x.between(args.water_level_correction[0] - grid_cell_x / 2, args.water_level_correction[0] + grid_cell_x / 2)
    p1_y_condition = df.y.between(args.water_level_correction[1] - grid_cell_y / 2, args.water_level_correction[1] + grid_cell_y / 2)
    p2_x_condition = df.x.between(args.water_level_correction[2] - grid_cell_x / 2, args.water_level_correction[2] + grid_cell_x / 2)
    p2_y_condition = df.y.between(args.water_level_correction[3] - grid_cell_y / 2, args.water_level_correction[3] + grid_cell_y / 2)

    z1 = df.loc[p1_x_condition & p1_y_condition, 'z']
    z2 = df.loc[p2_x_condition & p2_y_condition, 'z']

    if len(z1) == 0 or len(z2) == 0:
        print('One of the specified points is outside of the selected area. Water level correction cannot be applied.')
    else:
        z1 = z1.values[0]
        z2 = z2.values[0]
        print('Water level correction requested. The difference in height between both points is {} m.'.format(z2 - z1))

        zp = z_on_plain((args.water_level_correction[0], args.water_level_correction[1], z1), (args.water_level_correction[2], args.water_level_correction[3], z2), df.x, df.y)

        df.z = df.z - (zp - z1)

# df = df[df.x.between(xmin_request, xmax_request, inclusive='left') & df.y.between(ymin_request, ymax_request, inclusive='left')]

x_offset = df.x.min()
y_offset = df.y.min()
df_orig = df.copy()
df_orig.x = df_orig.x - x_offset
df_orig.y = df_orig.y - y_offset

if df.z.min() < 20:
    df.z = df.z - np.floor(df.z.min()) + 20

# df = df[df.x.between(0, grid_size_x_cm * 8, inclusive='left') & df.y.between(0, grid_size_y_cm * 8, inclusive='left')]

height_map_orig = np.zeros((int(df_orig.x.max()) + 1, int(df_orig.y.max()) + 1))

x = np.array(df_orig.x.values, dtype=int)
y = np.array(df_orig.y.values, dtype=int)
z = df_orig.z.values

height_map_orig[x, y] = z
plt.figure()
plt.imshow(height_map_orig.transpose(1,0), origin='lower')
plt.plot(bbox_polygon_orig.exterior.xy[0] - x_offset, bbox_polygon_orig.exterior.xy[1] - y_offset, ':y')
plt.plot(bbox_polygon.exterior.xy[0] - x_offset, bbox_polygon.exterior.xy[1] - y_offset, '-r')
plt.tight_layout(pad=1.00)
plt.savefig("{}_orig.png".format(args.output_name), bbox_inches='tight', dpi=1200, pad_inches=0)

df = df[df.x.between(xmin_request, xmax_request, inclusive='left') & df.y.between(ymin_request, ymax_request, inclusive='left')]

x_offset = df.x.min()
y_offset = df.y.min()
df.x = df.x - x_offset
df.y = df.y - y_offset

height_map = np.zeros((int(df.x.max()) + 1, int(df.y.max()) + 1))

x = np.array(df.x.values, dtype=int)
y = np.array(df.y.values, dtype=int)
z = df.z.values

height_map[x, y] = z

if len(args.bounding_box) in [8,9]:
    # make sure, bounding box polygon is counter-clockwise
    if not bbox_polygon.exterior.is_ccw:
        bbox_polygon = Polygon(bbox_polygon.exterior.coords[::-1])
    # get closest point in minimum rotated rectangle to first point of bounding box
    polygon_points = [Point(*coord) for coord in bbox_polygon.exterior.coords]
    dist = [bbox_points[0].distance(pt) for pt in polygon_points]
    min_idx = np.argmin(dist)

    # get rotation angle of x-axis, assumed to be defined by (x0, y0) -> (x1, y1)
    # since the last point in a polygon is always identical to the first point and np.argmin returns the first match,
    # there should always be min_idx + 1 within the array
    p0 = polygon_points[min_idx]
    p1 = polygon_points[min_idx + 1]
    if min_idx + 2 == len(polygon_points):
        p2 = polygon_points[1]
    else:
        p2 = polygon_points[min_idx + 2]

    center = (np.round((df.x.max()) / 2).astype(int), np.round(df.y.max() / 2).astype(int))

    plt.figure()
    plt.imshow(height_map.transpose(1,0), origin='lower')
    plt.plot(center[0], center[1], 'ro')
    plt.plot([p0.x - x_offset, p1.x - x_offset, p2.x - x_offset], [p0.y - y_offset, p1.y - y_offset, p2.y - y_offset], ':rD')
    plt.tight_layout(pad=1.00)
    plt.savefig("{}_bounds.png".format(args.output_name), bbox_inches='tight', dpi=1200, pad_inches=0)

    rotation_angle = np.arctan2(p1.y - p0.y, p1.x - p0.x) * 180.0 / np.pi
    size_x = p0.distance(p1)
    size_y = p1.distance(p2)

    height_map = skimage.transform.rotate(height_map, -rotation_angle, resize=True, cval=-1, preserve_range=True, clip=True, center=center)
    # center must be recalculated because img is resized!
    center_rotated = (height_map.shape[0] / 2 - 0.5, height_map.shape[1] / 2 - 0.5)

    # centre according to skimage rotate default
    lower_left = (max(0, center_rotated[0] - size_x / 2 / grid_cell_x), max(center_rotated[1] - size_y / 2 / grid_cell_y, 0))
    lower_left = (np.round(lower_left[0]).astype(int), np.round(lower_left[1]).astype(int))

    upper_right = (
        lower_left[0] + min(int((height_map.shape[0]-1 - lower_left[0]) / 8) * 8, int(size_x / 8) * 8),
        lower_left[1] + min(int((height_map.shape[1]-1 - lower_left[1]) / 8) * 8, int(size_y / 8) * 8)
    )

    plt.figure()
    plt.imshow(height_map.transpose(1,0), origin='lower')
    plt.plot(center_rotated[0], center_rotated[1], 'ro')
    plt.plot([lower_left[0], upper_right[0], upper_right[0], lower_left[0], lower_left[0]], [lower_left[1], lower_left[1], upper_right[1], upper_right[1], lower_left[1]], ':rD')
    plt.tight_layout(pad=1.00)
    plt.savefig("{}_rotated.png".format(args.output_name), bbox_inches='tight', dpi=1200, pad_inches=0)


    height_map = height_map[lower_left[0]:upper_right[0], lower_left[1]:upper_right[1]]

    trf_lower_left = (
        center[0] - (np.cos(rotation_angle / 180.0 * np.pi) * (center_rotated[0] - lower_left[0] - grid_cell_x / 2) - np.sin(rotation_angle / 180.0 * np.pi) * (center_rotated[1] - lower_left[1] - grid_cell_y / 2)) + x_offset,
        center[1] - (np.sin(rotation_angle / 180.0 * np.pi) * (center_rotated[0] - lower_left[0] - grid_cell_x / 2) + np.cos(rotation_angle / 180.0 * np.pi) * (center_rotated[1] - lower_left[1] - grid_cell_y / 2)) + y_offset
    )

    grid_gdf, diagonal_grid_gdf, sub_square_grid_gdf = get_all_grids(
        trf_lower_left[0], 
        trf_lower_left[1], 
        trf_lower_left[0] + height_map.shape[0] + grid_cell_x, 
        trf_lower_left[1] + height_map.shape[1] + grid_cell_y,
        int(height_map.shape[0] / 8),
        int(height_map.shape[1] / 8),
        rotation_angle=rotation_angle,
        rotation_center=trf_lower_left
    )

    grid_gdf.to_file(args.output_name + '_grid.json', driver='GeoJSON', crs=f'epsg:{args.input_crs}')
    diagonal_grid_gdf.to_file(args.output_name + '_diagonal_grid.json', driver='GeoJSON', crs=f'epsg:{args.input_crs}')
    sub_square_grid_gdf.to_file(args.output_name + '_sub_square_grid.json', driver='GeoJSON', crs=f'epsg:{args.input_crs}')

height_map_reduced = skimage.transform.rescale(height_map, (grid_cell_x / 8, grid_cell_y / 8), cval=1, preserve_range=True, clip=True, anti_aliasing=True)

x_arr = []
y_arr = []
z_arr = []
for xx in range(height_map_reduced.shape[0]):
    for yy in range(height_map_reduced.shape[1]):
        x_arr.append(xx)
        y_arr.append(yy)
        z_arr.append(height_map_reduced[xx, yy])

height_map_reduced_df = pandas.DataFrame({'x': x_arr, 'y': y_arr, 'z': z_arr})
if args.stride is not None:
    df_out = height_map_reduced_df.iloc[::args.stride]
    df_out = pandas.concat((df_out, pandas.DataFrame(
        {
            'x': [height_map_reduced_df.x.max()], 
            'y': [height_map_reduced_df.y.max()], 
            'z': [-1]
        }
    )))
    df_out.to_csv('{}.csv'.format(args.output_name))
else:
    height_map_reduced_df.to_csv('{}.csv'.format(args.output_name))

hm = height_map_reduced.T
zmin = np.floor(height_map_reduced_df.z.min()).astype(int)
zmax = np.ceil(height_map_reduced_df.z.max()).astype(int)
plt.figure(); plt.axis('equal'); cs = plt.contour(hm, levels=np.arange(0, zmax+args.contour, args.contour), colors='k', linewidths=0.05); plt.clabel(cs, inline=0, fontsize=1, fmt='%d'); plt.axis('off')
plt.tight_layout(pad=1.00)
plt.savefig("{}_contour.png".format(args.output_name), bbox_inches='tight', dpi=1200, pad_inches=0)

df_contour = None
for level_idx in range(len(cs.levels)):
    if len(cs.allsegs[level_idx]) == 0:
        continue

    
    for seg_idx in range(len(cs.allsegs[level_idx])):
        seg = cs.allsegs[level_idx][seg_idx]
        x_contour = seg[:,0].round()
        y_contour = seg[:,1].round()
        z_contour = [cs.levels[level_idx]] * len(x_contour)

        if df_contour is None:
            df_contour = pandas.DataFrame({'x': x_contour, 'y': y_contour, 'z': z_contour})
        else:
            df_contour = pandas.concat((df_contour, pandas.DataFrame({'x': x_contour, 'y': y_contour, 'z': z_contour})))

if df_contour is not None:
    df_contour = df_contour.drop_duplicates()
    if args.stride is not None:
        df_contour = df_contour.iloc[::args.stride]
        df_contour = pandas.concat((df_contour, pandas.DataFrame(
            {
                'x': [height_map_reduced_df.x.max()], 
                'y': [height_map_reduced_df.y.max()], 
                'z': [-1]
            }
        )))

    df_contour.to_csv("{}_contour.csv".format(args.output_name))

plt.figure()
plt.imshow(height_map_reduced.T, vmin=zmin, vmax=zmax, origin='lower')
plt.savefig('{}.png'.format(args.output_name))


