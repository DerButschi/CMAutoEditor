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

import matplotlib.pyplot as plt
import numpy as np
import pandas
import rasterio
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj.transformer import Transformer
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import griddata
from scipy.signal import argrelextrema
from tqdm import tqdm

# bbox_utm = [404574.5693375174, 5321911.884501877, 409440.47559354035, 5326019.8956525605]
bbox_utm = [404906.392,5323566.913, 407839.804,5326576.343]
# https://gis.stackexchange.com/questions/428728/get-lanlon-and-values-from-geotiff-using-python

# data = rasterio.open("alos/N048E037/ALPSMLC30_N048E037_DSM.tif")

# aoi = AreaOfInterest(*data.bounds)

# crs_list = query_utm_crs_info(area_of_interest=aoi)

# crs = None
# for crs_candidate in crs_list:
#     if 'WGS 84 / UTM' in crs_candidate.name:
#         crs = crs_candidate
        
# if crs is None:
#     crs = crs_list[-1]


# dst_crs = '{}:{}'.format(crs.auth_name, crs.code)

# transform, width, height = calculate_default_transform(
#     data.crs, dst_crs, data.width, data.height, *data.bounds)

# transformer = Transformer.from_crs('epsg:{}'.format(crs.code), 'epsg:{}'.format(data.crs.to_epsg()), always_xy=True)

# dst = np.zeros((width, height))
with rasterio.open('alos/N048E037/ALPSMLC30_N048E037_DSM.tif') as src:
    aoi = AreaOfInterest(*src.bounds)

    crs_list = query_utm_crs_info(area_of_interest=aoi)

    crs = None
    for crs_candidate in crs_list:
        if 'WGS 84 / UTM' in crs_candidate.name:
            crs = crs_candidate
            
    if crs is None:
        crs = crs_list[-1]

    dst_crs = '{}:{}'.format(crs.auth_name, crs.code)

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # with rasterio.open('alos/N048E037/ALPSMLC30_N048E037_DSM_utm.tif', 'w', **kwargs) as dst:
    with MemoryFile() as memfile:
        with memfile.open(**kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    data_transform=src.transform,
                    data_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

            n_bins_x = int(np.floor((bbox_utm[2] - bbox_utm[0]) / 8) + 1)
            n_bins_y = int(np.floor((bbox_utm[3] - bbox_utm[1]) / 8) + 1)
            data = np.zeros((n_bins_x, n_bins_y))

            pic_xmin_idx, pic_ymin_idx = dst.index(bbox_utm[0], bbox_utm[1])
            pic_xmax_idx, pic_ymax_idx = dst.index(bbox_utm[2], bbox_utm[3])

            pic_data = dst.read(1)
            pic_data_bbox = pic_data[pic_xmax_idx:pic_xmin_idx+1, pic_ymin_idx:pic_ymax_idx+1]

            pic_grid_x, pic_grid_y = np.mgrid[bbox_utm[0]:bbox_utm[2]:complex(pic_xmin_idx-pic_xmax_idx+1), bbox_utm[1]:bbox_utm[3]:complex(pic_ymax_idx-pic_ymin_idx+1)]
            points = np.zeros((pic_grid_x.shape[0] * pic_grid_x.shape[1], 2))
            values = pic_data_bbox.flatten()
            points[:,0] = pic_grid_x.flatten()
            points[:,1] = pic_grid_y.flatten()
            grid_x, grid_y = np.mgrid[bbox_utm[0]:bbox_utm[2]:complex(n_bins_y), bbox_utm[1]:bbox_utm[3]:complex(n_bins_x)]
            interp_data = griddata(points, values, (grid_x, grid_y))

            # maxima_yindices, maxima_xindices = argrelextrema(interp_data, np.greater)
            # minima_yindices, minima_xindices = argrelextrema(interp_data, np.less)
            x_arr = []
            y_arr = []
            z_arr = []
            for xidx in range(n_bins_x):
                for yidx in range(n_bins_y):
            # for idx in range(len(maxima_xindices)):
            #     xidx = maxima_xindices[idx]
            #     yidx = maxima_yindices[idx]
            #     x_arr.append(xidx)
            #     y_arr.append(yidx)
            #     z_arr.append(interp_data[n_bins_y-yidx-1, xidx])
            # for idx in range(len(minima_yindices)):
            #     xidx = minima_xindices[idx]
            #     yidx = minima_yindices[idx]
                    x_arr.append(xidx)
                    y_arr.append(yidx)
                    z_arr.append(interp_data[n_bins_y-yidx-1, xidx])

            df = pandas.DataFrame({'x': x_arr, 'y': y_arr, 'z': z_arr})

            df.to_csv('alos_test.csv')





