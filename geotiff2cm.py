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
from contextlib import nullcontext
import argparse
from tqdm import tqdm

def process_geotiff(args):
    with rasterio.open(args.input_file) as src:
        
        aoi = AreaOfInterest(*src.bounds)

        crs_list = query_utm_crs_info(area_of_interest=aoi)

        if not src.crs.is_projected:
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

        with MemoryFile() if not src.crs.is_projected else nullcontext() as memfile:
            with memfile.open(**kwargs) if not src.crs.is_projected else src as dst:
                if not src.crs.is_projected:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            data_transform=src.transform,
                            data_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest)

                if args.bounding_box is not None:
                    epsg = int(args.bounding_box[0])
                    bbox_transformer = Transformer.from_crs('epsg:{}'.format(epsg), 'epsg:{}'.format(dst.crs.to_epsg()), always_xy=True)
                    bbox = [
                        *bbox_transformer.transform(float(args.bounding_box[1]), float(args.bounding_box[2])),
                        *bbox_transformer.transform(float(args.bounding_box[3]), float(args.bounding_box[4])), 
                    ]
                else:
                    bbox = [*dst.bounds]
                
                n_bins_x = int(np.floor((bbox[2] - bbox[0]) / 8) + 1)
                n_bins_y = int(np.floor((bbox[3] - bbox[1]) / 8) + 1)

                data = dst.read(
                    out_shape=(
                        dst.count,
                        int(dst.height * (dst.res[1] / 8.0)),
                        int(dst.width * (dst.res[0] / 8.0))
                    ),
                    resampling=Resampling.bilinear
                )

                transform = dst.transform * dst.transform.scale(
                    (dst.width / data.shape[-1]),
                    (dst.height / data.shape[-2])
                )
                transformer = rasterio.transform.AffineTransformer(transform)

                pic_bottom, pic_left = transformer.rowcol(bbox[0], bbox[1])
                pic_top, pic_right = transformer.rowcol(bbox[2], bbox[3])

                pic_data = data[0]
                pic_data[pic_data == dst.nodatavals] = -1
                x_arr = []
                y_arr = []
                z_arr = []
                pbar = tqdm(total=(pic_bottom - pic_top + 2) * (pic_right - pic_left + 2))
                for row in range(pic_top-1, pic_bottom+2):
                    for col in range(pic_left-1, pic_right+2):
                        x, y = transformer.xy(row, col)
                        z = pic_data[row, col]
                        x_arr.append(x)
                        y_arr.append(y)
                        z_arr.append(z)
                        pbar.update(1)
                pbar.close()

                points = np.zeros((len(x_arr), 2))
                points[:,0] = x_arr
                points[:,1] = y_arr

                # picture is (row, col) with (0, 0) at top left!
                # pic_data_bbox = pic_data_bbox[::-1,:].transpose(0,1)


                # pic_grid_x, pic_grid_y = np.mgrid[bbox_utm[0]:bbox_utm[2]:complex(pic_xmin_idx-pic_xmax_idx+1), bbox_utm[1]:bbox_utm[3]:complex(pic_ymax_idx-pic_ymin_idx+1)]
                # points = np.zeros((pic_grid_x.shape[0] * pic_grid_x.shape[1], 2))
                # values = pic_data_bbox.flatten('F')
                # points[:,0] = pic_grid_x.flatten()
                # points[:,1] = pic_grid_y.flatten()
                # grid_x, grid_y = np.mgrid[bbox_utm[0]:bbox_utm[2]:complex(n_bins_x), bbox_utm[1]:bbox_utm[3]:complex(n_bins_y)]
                # interp_data = griddata(points, values, (grid_x, grid_y))

                # maxima_yindices, maxima_xindices = argrelextrema(interp_data, np.greater)
                # minima_yindices, minima_xindices = argrelextrema(interp_data, np.less)
                x_interp_arr = []
                y_interp_arr = []
                xidx_arr = []
                yidx_arr = []
                for xidx, x in enumerate(np.linspace(bbox[0] + 4, bbox[0] + 4 + (n_bins_x - 1) * 8, n_bins_x)):
                    for yidx, y in enumerate(np.linspace(bbox[1] + 4, bbox[1] + 4 + (n_bins_y - 1) * 8, n_bins_y)):
                        x_interp_arr.append(x)
                        y_interp_arr.append(y)
                        xidx_arr.append(xidx)
                        yidx_arr.append(yidx)

                interp_data = griddata(points, z_arr, (x_interp_arr, y_interp_arr))

                df = pandas.DataFrame({'x': xidx_arr, 'y': yidx_arr, 'z': interp_data})

                df.to_csv(args.output_file)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input-file', type=str, help='input geotiff file')
    argparser.add_argument('-d', '--input-directory', type=str, help='directory with input geotiff files')
    argparser.add_argument('-o', '--output-file', type=str, help='name of output file', default='output.csv')
    argparser.add_argument('-e', '--file-extension', type=str, help='extension of files to be read in case of input directory [default: tif]', default='tif')
    argparser.add_argument('-b', '--bounding-box', nargs=5, help='Box in which to extract data. EPSG no, left, bottom, right, top')

    args = argparser.parse_args()
    if args.input_file is None and args.input_directory is None:
        raise argparse.ArgumentError(args.input_file, 'Either input file or input directory must be provided.')
    if args.input_file is not None and args.input_directory is not None:
        raise argparse.ArgumentError(args.input_file, '--input-file and --input-directory are mutually exclusive.')



    process_geotiff(args)




