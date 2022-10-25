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

argparser = argparse.ArgumentParser()
argparser.add_argument('--dgm-dir', '-d', required=True, help='directory in which the dgm files are located')
argparser.add_argument('--bounding-box', '-b', required=True, help='lower left and upper right corner of the box in which data is to be extracted (min x, min y, max x, max y)',
                        type=float, nargs=4
)
argparser.add_argument('--contour', '-c', required=False, type=float, help='contour level distance (default: 5 m)', default=5.0)
argparser.add_argument('--output-name', '-o', required=False, type=str, help='output name (without file extension) (default: output)', default='output')

args = argparser.parse_args()

xmin_request, ymin_request, xmax_request, ymax_request = args.bounding_box

df = None
print('Processing input files...')
for filename in os.listdir(args.dgm_dir):
    if filename.endswith('.xyz.gz'):
        df_file = pandas.read_csv(os.path.join(args.dgm_dir, filename), delimiter=r"\s+", names=['x','y','z'])
        df_file = df_file[df_file.x.between(xmin_request, xmax_request) & df_file.y.between(ymin_request, ymax_request)]
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

df.x = df.x - df.x.min()
df.y = df.y - df.y.min()

if df.z.min() < 20:
    df.z = df.z - np.floor(df.z.min()) + 20

df = df[df.x.between(0, grid_size_x_cm * 8, inclusive='left') & df.y.between(0, grid_size_y_cm * 8, inclusive='left')]

height_map = np.zeros((int(grid_size_x_cm * 8), int(grid_size_y_cm * 8)))

x = np.array(df.x.values, dtype=int)
y = np.array(df.y.values, dtype=int)
z = df.z.values

height_map[x, y] = z

height_map_reduced = skimage.measure.block_reduce(height_map, (int(8 / grid_cell_x), int(8 / grid_cell_y)), np.mean)

x_arr = []
y_arr = []
z_arr = []
for xx in range(height_map_reduced.shape[0]):
    for yy in range(height_map_reduced.shape[1]):
        x_arr.append(xx)
        y_arr.append(yy)
        z_arr.append(height_map_reduced[xx, yy])

height_map_reduced_df = pandas.DataFrame({'x': x_arr, 'y': y_arr, 'z': z_arr})
height_map_reduced_df.to_csv('{}.csv'.format(args.output_name))

hm = height_map_reduced.T
zmin = np.floor(height_map_reduced_df.z.min()).astype(int)
zmax = np.ceil(height_map_reduced_df.z.max()).astype(int)
plt.figure(); plt.axis('equal'); cs = plt.contour(hm, levels=np.arange(zmin-20, zmax+args.contour, args.contour), colors='k', linewidths=0.05); plt.clabel(cs, inline=0, fontsize=1, fmt='%d'); plt.axis('off')
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

df_contour = df_contour.drop_duplicates()

if df_contour is not None:
    df_contour.to_csv("{}_contour.csv".format(args.output_name))

# a = 1
plt.figure()
plt.imshow(height_map_reduced.T[::-1,:], vmin=zmin, vmax=zmax)
# plt.savefig('heightmap.png')
# pickle.dump(height_map_reduced, open('heightmap.pkl', 'wb'))
plt.savefig('{}.png'.format(args.output_name))
# pickle.dump(height_map_reduced, open('heightmap.pkl', 'wb'))


