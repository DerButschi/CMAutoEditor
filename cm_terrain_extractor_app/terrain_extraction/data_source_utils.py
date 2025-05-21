from abc import ABC, abstractmethod
from typing import List, Tuple
from shapely import Polygon, Point
from shapely.affinity import rotate
import pandas
from affine import Affine
from rasterio.enums import Resampling
from rasterio import open as rasterio_open
from rasterio.warp import calculate_default_transform, reproject
from rasterio.io import MemoryFile
from rasterio.fill import fillnodata
from rasterio import band
import numpy as np
from rasterio.merge import merge
import os
from pyproj.crs import CRS
import skimage
from PIL import Image
import matplotlib as mpl
import geopandas
import zipfile
import gzip
import math

from terrain_extraction.projection_utils import reproject_array, transform_point, reproject_geotiff
from terrain_extraction.bbox_utils import get_rectangle_rotation_angle, get_polygon_node_points, BoundingBox
from terrain_extraction.data_container import ElevationDataContainer

def clip_dataframe_to_bounding_box(df: pandas.DataFrame, bounds: Tuple[float]) -> pandas.DataFrame:
    xmin_request, ymin_request, xmax_request, ymax_request = bounds
    df_clip = df[df.x.between(xmin_request, xmax_request, inclusive='left') & df.y.between(ymin_request, ymax_request, inclusive='left')]    
    return df_clip

def dataframe2ndarray(df: pandas.DataFrame, origin_bottom: bool = True, resolution: Tuple = (1.0, 1.0)) -> np.ndarray:
    x_offset = df.x.min()
    y_offset = df.y.min()

    arr_shape0 = int((df.x.max() - x_offset) / resolution[0]) + 1
    arr_shape1 = int((df.y.max() - y_offset) / resolution[1]) + 1

    arr = np.zeros((arr_shape0, arr_shape1))

    x = np.array((df.x.values - x_offset) / resolution[0], dtype=int)
    y = np.array((df.y.values - y_offset) / resolution[1], dtype=int)
    z = df.z.values

    arr[x, y] = z
    if not origin_bottom:
        arr = arr.transpose(1,0)
        arr = arr[::-1,:]

    return arr

def get_map_center(df: pandas.DataFrame) -> Tuple[float]:
    return (np.round((df.x.max()) / 2).astype(int), np.round(df.y.max() / 2).astype(int))

def rescale_height_map(height_map: np.ndarray, calculation_resolution: Tuple = (1.0, 1.0), output_resolution: Tuple = (8.0, 8.0)) -> np.ndarray:
    # NOTE: assuming height map has 1m resolution!
    return skimage.transform.rescale(
        height_map, 
        (calculation_resolution[0] / output_resolution[0], calculation_resolution[1] / output_resolution[1]), 
        cval=1, 
        preserve_range=True, 
        clip=True, 
        anti_aliasing=True
    )

def rotate_height_map(height_map: np.ndarray, 
                      rotation_angle: float, 
                      center: Tuple[float],
                      size_x: float,
                      size_y: float,
                      res_x: float,
                      res_y: float,
                      out_res_x: float = 8.0,
                      out_res_y: float = 8.0
                     ) -> np.ndarray:

    height_map = skimage.transform.rotate(height_map, -rotation_angle, resize=True, mode='edge', preserve_range=True, clip=True, center=center)

    # center must be recalculated because img is resized!
    center_rotated = (height_map.shape[0] / 2 - 0.5, height_map.shape[1] / 2 - 0.5)

    # centre according to skimage rotate default
    lower_left = (max(0, center_rotated[0] - size_x / 2 / res_x), max(center_rotated[1] - size_y / 2 / res_y, 0))
    lower_left = (np.round(lower_left[0]).astype(int), np.round(lower_left[1]).astype(int))

    upper_right = (
        int(lower_left[0] + min(int((height_map.shape[0]-1 - lower_left[0]) / out_res_x) * out_res_x, int(size_x / res_x / out_res_x) * out_res_x)),
        int(lower_left[1] + min(int((height_map.shape[1]-1 - lower_left[1]) / out_res_y) * out_res_y, int(size_y / res_y / out_res_y) * out_res_y))
    )

    height_map = height_map[lower_left[0]:upper_right[0], lower_left[1]:upper_right[1]]

    return height_map

def ndarray2dataframe(arr: np.ndarray, x_offset: float = 0, y_offset: float = 0, origin_bottom: bool = True, resolution: Tuple[float] = (1.0, 1.0)) -> pandas.DataFrame:
    x_arr = []
    y_arr = []
    z_arr = []
    if not origin_bottom:
        arr_tmp = arr.transpose(1,0)[:,::-1]
    else:
        arr_tmp = arr
    for xx in range(arr_tmp.shape[0]):
        for yy in range(arr_tmp.shape[1]):
            x_arr.append(xx * resolution[0])
            y_arr.append(yy * resolution[1])
            z_arr.append(arr_tmp[xx, yy])

    df = pandas.DataFrame({'x': x_arr, 'y': y_arr, 'z': z_arr})
    df.loc[:, 'x'] += x_offset
    df.loc[:, 'y'] += y_offset
                          
    return df

def height_map_to_png(height_map: np.ndarray, file_path: str):
    # height_map = height_map.T
    # height_map = height_map[:,::-1]
    norm = mpl.colors.Normalize(vmin=np.min(height_map[height_map > 0]), vmax=np.max(height_map))
    is_zero_value = height_map <= 0
    height_map_rgba = mpl.colormaps['PuBu_r'](norm(height_map))
    height_map_rgba[is_zero_value] = [1, 1, 1, 0]
    height_map_rgba = (height_map_rgba * 255).astype(np.uint8)
    img = Image.fromarray(height_map_rgba)
    img.save(file_path)


def dataframe_in_bbox_to_png(df: pandas.DataFrame, bounding_box: BoundingBox, file_path: str):
    gdf = geopandas.GeoDataFrame(
        df, 
        geometry=geopandas.points_from_xy(df.x, df.y)
    )
    gdf.loc[~gdf.index.isin(gdf.sindex.query(bounding_box.get_box(bounding_box.crs_projected), predicate='contains')), 'z'] = -9999
    gdf = clip_dataframe_to_bounding_box(gdf, bounding_box.get_bounds(bounding_box.crs_projected))

    height_map = dataframe2ndarray(gdf)
    height_map = height_map.transpose(1,0)[::-1,:]

    reprojected_height_map = reproject_array(
        height_map, 
        (gdf.x.min()-0.5,gdf.y.min()-0.5,gdf.x.max()+0.5,gdf.y.max()+0.5),
        bounding_box.crs_projected,
        CRS.from_epsg(4326),
        (1.0 / 3600.0 / 4.0, 1.0 / 3600.0 / 4.0),
    )

    height_map_to_png(reprojected_height_map, file_path)

def reproject_dataframe(df: pandas.DataFrame, source_crs: CRS, target_crs: CRS, target_resolution: Tuple[float]):
    offset_point = Point(df.x.min(), df.y.min())
    offset_point = transform_point(offset_point, from_epsg=source_crs.to_epsg(), to_epsg=target_crs.to_epsg())

    df_arr = dataframe2ndarray(df, origin_bottom=False)
    reprojected_df_arr = reproject_array(
        df_arr,
        (df.x.min()-0.5, df.y.min()-0.5, df.x.max()+0.5, df.y.max()+0.5),
        source_crs,
        target_crs,
        target_resolution,
    )
    reprojected_df = ndarray2dataframe(reprojected_df_arr, x_offset=offset_point.x, y_offset=offset_point.y, origin_bottom=False)
    return reprojected_df

def ndarray2dataframe_transform(
        arr: np.ndarray,
        transform: Affine,
        nodata: float = None
    ) -> pandas.DataFrame:
    """
    Convert a 2D array into a DataFrame with x, y, z columns.
    
    Parameters
    ----------
    arr : np.ndarray
        2D array of shape (rows, cols), e.g. your extracted window.
    transform : Affine
        The Affine mapping pixel (col, row) → map (x, y).
    nodata : float, optional
        Value to ignore/drop (e.g. np.nan or src.nodata). If None, all cells are included.
    
    Returns
    -------
    df : pd.DataFrame
        Columns: x (Easting), y (Northing), z (pixel value).
    """
    # 1. Build the grid of column and row indices
    n_rows, n_cols = arr.shape
    cols = np.arange(n_cols)
    rows = np.arange(n_rows)
    col_idxs, row_idxs = np.meshgrid(cols, rows)
    
    # 2. Convert pixel‐indices to map coords
    #    x = A * col + B * row + C
    #    y = D * col + E * row + F
    xs, ys = transform * (col_idxs, row_idxs)  # vectorized Affine
    
    # 3. Flatten everything
    xs_flat = xs.ravel()
    ys_flat = ys.ravel()
    zs_flat = arr.ravel()
    
    # 4. Build DataFrame
    df = pandas.DataFrame({
        'x': xs_flat,
        'y': ys_flat,
        'z': zs_flat
    })
    
    # 5. Optionally drop nodata
    if nodata is not None:
        df = df[df['z'] != nodata].reset_index(drop=True)
    
    return df


def check_zip_file(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            bad_file = zip_ref.testzip()
            if bad_file is not None:
                return False
    except zipfile.BadZipFile:
        return False
    
    return True

def check_gzip_file(file_path):
    try: 
        with gzip.open(file_path, 'rb') as f:
            while f.read(10000000) != b'':
                pass
    
    except EOFError:
        return False
        
    except gzip.BadGzipFile:
        return False
    
    return True


class DataSource(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_data(self, bounding_box: BoundingBox, cache_dir: str, output_resolution: Tuple = (1.0, 1.0)) -> pandas.DataFrame:
        pass

    @abstractmethod
    def intersects_bounding_box(self, bounding_box: BoundingBox) -> bool:
        pass

    def get_gdf(self) -> geopandas.GeoDataFrame:
        if self.gdf is None:
            self.gdf = geopandas.GeoDataFrame.from_file(self.gdf_geojson_path)

        return self.gdf


    def intersects_bounding_box(self, bounding_box: BoundingBox) -> bool:
        if not bounding_box.get_box(self.crs).intersects(self.envelope):
            return False
        gdf = self.get_gdf()
        return gdf.sindex.query(bounding_box.get_box(self.crs), predicate='intersects').any()

    def cut_out_bounding_box(
            self, 
            df: pandas.DataFrame, 
            bounding_box: BoundingBox, 
            calculation_resoultion: Tuple = (1.0, 1.0), 
            output_resolution: Tuple = (8.0, 8.0)
        ):
        box = bounding_box.get_box(bounding_box.crs_projected)
        p0, p1, p2, _ = get_polygon_node_points(box)
        df = clip_dataframe_to_bounding_box(df, box.bounds)
        rotation_angle = get_rectangle_rotation_angle(box, p0)
        height_map = dataframe2ndarray(df, resolution=calculation_resoultion)

        center = get_map_center(df)

        size_x = p0.distance(p1)
        size_y = p1.distance(p2)

        height_map = rotate_height_map(height_map, rotation_angle, None, size_x, size_y, calculation_resoultion[0], calculation_resoultion[1], output_resolution[0], output_resolution[1])

        if calculation_resoultion != output_resolution:
            height_map = rescale_height_map(height_map)

        height_map_df = ndarray2dataframe(height_map)

        return height_map_df
    
    def get_png(self, bounding_box: BoundingBox, cache_dir: str):
        if self.cached_data is None or not self.cached_data_bounding_box.equals(bounding_box):
            self.get_data(bounding_box, cache_dir)

        elevation_data = self.cached_data
        file_path = os.path.join(cache_dir, 'current_height_map.png')
        elevation_data.to_png(file_path, bounding_box.crs_orig)

        return file_path


class GeoTiffDataSource(DataSource):
    def __init__(self):
        self.data_type: str = 'geotiff'

    def merge_image_files(self, image_files, out_dir, bounding_box: BoundingBox):
        self.current_merged_image_path = os.path.join(out_dir, self.data_folder, 'merged_images_in_bbox.tif')
        # merge(image_files, dst_path=self.current_merged_image_path, res=(1.0,1.0), resampling=Resampling.bilinear)
        bounds = bounding_box.get_buffer(self.crs).bounds
        merge(image_files, dst_path=self.current_merged_image_path, bounds=bounds, resampling=Resampling.bilinear)

    def get_merged_elevation_data_in_bounding_box(
        self,
        bounding_box: BoundingBox,                       # Shapely Polygon in EPSG:32632
        target_res: Tuple[float] = (10.0,10.0) # (Δx, Δy) in metres
    ) -> ElevationDataContainer:
        
        return ElevationDataContainer.from_geotiff(self.current_merged_image_path, bounding_box, target_res)

    def reproject_rotate_and_crop_merged_image(self, bounding_box: BoundingBox, output_resolution: Tuple = (1.0, 1.0), resampling=Resampling.bilinear):
        with rasterio_open(self.current_merged_image_path) as src:
            dst_crs = bounding_box.crs_projected
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            with MemoryFile() as memfile:
                with memfile.open(**kwargs) as utm_raster:
                    reproject(
                        source=band(src, 1),
                        destination=band(utm_raster, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

        box = bounding_box.get_box(bounding_box.crs_projected)
        p0, p1, p2, _ = get_polygon_node_points(box)
        rotation_angle = get_rectangle_rotation_angle(box, p0)

        # Rotate the rectangle so it becomes axis-aligned
        rotated_rect = rotate(bounding_box.box_utm, rotation_angle, origin='centroid', use_radians=False)
        minx, miny, maxx, maxy = rotated_rect.bounds

        # Desired resolution
        res = (1.0, 1.0)  # or any other (xres, yres)

        width = int((maxx - minx) / output_resolution[0])
        height = int((maxy - miny) / output_resolution[1])

        # Build transform: rotate, then translate to minx/miny
        rotation = Affine.rotation(rotation_angle)
        translation = Affine.translation(minx, maxy)  # note: y origin is top

        # Scaling to match resolution
        scaling = Affine.scale(output_resolution[0], -output_resolution[1])  # negative y for top-down

        dst_transform = translation * rotation * scaling

        profile = kwargs.copy()

        profile.update({
            'height': height,
            'width': width,
            'transform': dst_transform,
            'crs': dst_crs
        })

        with memfile.open() as utm_raster:
            with rasterio_open("output_rotated_crop.tif", 'w', **profile) as dst:
                reproject(
                    source=band(utm_raster, 1),
                    destination=band(dst, 1),
                    src_transform=utm_raster.transform,
                    src_crs=utm_raster.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )


        """
        Script to:
        1. Reproject a GeoTIFF from WGS84 to UTM Zone 32N (EPSG:32632).
        2. Rotate the reprojected image so a user-defined non-axis-aligned rectangle becomes axis-aligned.
        3. Crop the rotated image exactly to that rectangle extents.

        Dependencies:
        - rasterio
        - shapely
        - affine

        Usage:
        Set your file paths, the target CRS, desired output resolution, and the rectangle corners in UTM.
        """
        # import math
        # import rasterio
        # from rasterio.warp import calculate_default_transform, reproject, Resampling
        # from rasterio.io import MemoryFile
        # from affine import Affine
        # from shapely.geometry import Polygon, mapping

        # ----- User parameters -----
        # Paths
        # src_path = "input_wgs84.tif"              # Input GeoTIFF in WGS84
        # dst_path = "output_rotated_cropped.tif"   # Final output

        # # Target CRS and resolution
        # dst_crs = 'EPSG:32632'  # UTM Zone 32N
        # # Output pixel size (in CRS units), e.g. (30, 30) for 30m resolution
        # dst_resolution = None  # None to inherit source resolution

        # # Define four corners of the non-axis-aligned rectangle in EPSG:32632
        # # Order: (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        # utm_rect = [
        #     (500000, 5200000),
        #     (500500, 5200100),
        #     (501000, 5200050),
        #     (500500, 5200000)
        # ]
        # # ----------------------------

        # 1) Reproject input to UTM32N into memory
        # with rasterio_open(self.current_merged_image_path) as src:
        #     transform, width, height = calculate_default_transform(
        #         src.crs, bounding_box.crs_projected, src.width, src.height, *src.bounds, resolution=output_resolution
        #     )
        #     meta = src.meta.copy()
        #     meta.update({ 'crs': bounding_box.crs_projected, 'transform': transform, 'width': width, 'height': height })

        #     memfile = MemoryFile()
        #     with memfile.open(**meta) as tmp:
        #         for b in range(1, src.count+1):
        #             reproject(
        #                 source=band(src, b),
        #                 destination=band(tmp, b),
        #                 src_transform=src.transform, src_crs=src.crs,
        #                 dst_transform=transform, dst_crs=bounding_box.crs_projected,
        #                 resampling=resampling
        #             )

        # # 2) Compute rotation angle so that edge (pt1->pt2) aligns with the x-axis
        # #    and build an Affine transform: translate -> rotate -> crop
        # # box = bounding_box.get_box(bounding_box.crs_projected)
        # # p0, p1, p2, _ = get_polygon_node_points(box)
        # # rotation_angle = get_rectangle_rotation_angle(box, p0)

        # rect = bounding_box.box_utm
        # utm_rect = [coord for coord in rect.exterior.coords]
        # (x1, y1), (x2, y2) = utm_rect[0], utm_rect[1]
        # dx = x2 - x1
        # dy = y2 - y1
        # angle = math.atan2(dy, dx)  # negative to rotate cylinder to horizontal

        # rotated_rect = rotate(bounding_box.box_utm, angle * 180 / math.pi, origin='centroid', use_radians=False)
        # minx, miny, maxx, maxy = rotated_rect.bounds

        # out_width = int((maxx - minx) / output_resolution[0])
        # out_height = int((maxy - miny) / output_resolution[1])

        # # Build transform: rotate, then translate to minx/miny
        # rotation = Affine.rotation(angle * 180 / math.pi)
        # translation = Affine.translation(minx, maxy)  # note: y origin is top

        # # Scaling to match resolution
        # scaling = Affine.scale(output_resolution[0], -output_resolution[1])  # negative y for top-down

        # dst_transform = translation * rotation * scaling

        # # Update metadata for final output
        # out_meta = meta.copy()
        # out_meta.update({
        #     'crs': bounding_box.crs_projected,
        #     'transform': dst_transform,
        #     'width': out_width,
        #     'height': out_height
        # })

        # # 5) Reproject from in-memory UTM to the rotated & cropped output
        # with memfile.open() as src:
        #     with rasterio_open(os.path.join(os.path.dirname(self.current_merged_image_path), 'proj_crop_' + os.path.basename(self.current_merged_image_path)), 'w', **out_meta) as dst:
        #         for b in range(1, src.count+1):
        #             reproject(
        #                 source=band(src, b),
        #                 destination=band(dst, b),
        #                 src_transform=src.transform, src_crs=src.crs,
        #                 dst_transform=dst_transform, dst_crs=bounding_box.crs_projected,
        #                 dst_width=out_width, dst_height=out_height,
        #                 resampling=resampling
        #             )
        
    def get_merged_dataframe(self, bounding_box: BoundingBox, calculation_resolution: Tuple = (1.0, 1.0)) -> pandas.DataFrame:
        with rasterio_open(self.current_merged_image_path) as src:
            data = src.read(1)

            while len(data[data == src.nodata]) > 0:
                mask = np.full(data.shape, 255)
                mask[data == src.nodata] = 0
                data = fillnodata(data, mask)

            reprojected_data = reproject_array(
                data,
                src.bounds,
                self.crs,
                bounding_box.crs_projected,
                calculation_resolution,
            )
            upper_left = transform_point(Point(*src.xy(0,0)), self.crs.to_epsg(), bounding_box.crs_projected.to_epsg())
            lower_right = transform_point(Point(*src.xy(data.shape[0] - 1, data.shape[1] - 1)), self.crs.to_epsg(), bounding_box.crs_projected.to_epsg())
            # x_axis = np.tile(np.linspace(src.xy(0,0)[0], src.xy(*data.shape)[0], data.shape[1]), (data.shape[0],1))
            # y_axis = np.tile(np.linspace(src.xy(0,0)[1], src.xy(*data.shape)[1], data.shape[0]), (data.shape[1],1)).T

            x_axis = np.tile(np.linspace(upper_left.x, lower_right.x, reprojected_data.shape[1]), (reprojected_data.shape[0],1))
            y_axis = np.tile(np.linspace(upper_left.y, lower_right.y, reprojected_data.shape[0]), (reprojected_data.shape[1],1)).T


            # Create a Pandas DataFrame
            df = pandas.DataFrame({
                'x': x_axis.flatten(),
                'y': y_axis.flatten(),
                'z': reprojected_data.flatten().astype(np.float32)
            })

        return df


class XYZDataSource(DataSource):
    def __init__(self) -> None:
        self.data_type: str = 'xyz'
        super().__init__()

    def get_merged_dataframe(self, bounding_box: BoundingBox, data_files: List[str]):
        df = None
        for filename in data_files:
            df_file = pandas.read_csv(filename, delimiter=self.data_delimiter, names=['x','y','z'])
            df_file = clip_dataframe_to_bounding_box(df_file, bounding_box.get_buffer(self.crs).bounds)
            if df is not None:
                df = pandas.concat((df, df_file))
            else:
                df = df_file

        reprojected_df = reproject_dataframe(
            df,
            self.crs,
            bounding_box.crs_projected,
            (1.0, 1.0)
        )

        return reprojected_df


class ASCDataSource(DataSource):
    def __init__(self) -> None:
        self.data_type: str = 'asc'
        super().__init__()


