from abc import ABC, abstractmethod
from typing import List, Tuple
from shapely import Polygon, Point
import pandas
from rasterio.enums import Resampling
from rasterio import open as rasterio_open
from rasterio.fill import fillnodata
import numpy as np
from rasterio.merge import merge
import os
from pyproj.crs import CRS
import skimage
from PIL import Image
import matplotlib as mpl
import geopandas

from terrain_extraction.projection_utils import reproject_array, transform_point
from terrain_extraction.bbox_utils import get_rectangle_rotation_angle, get_polygon_node_points, BoundingBox

def clip_dataframe_to_bounding_box(df: pandas.DataFrame, bounds: Tuple[float]) -> pandas.DataFrame:
    xmin_request, ymin_request, xmax_request, ymax_request = bounds
    df_clip = df[df.x.between(xmin_request, xmax_request, inclusive='left') & df.y.between(ymin_request, ymax_request, inclusive='left')]    
    return df_clip

def dataframe2ndarray(df: pandas.DataFrame, origin_bottom: bool = True) -> np.ndarray:
    x_offset = df.x.min()
    y_offset = df.y.min()

    arr = np.zeros((int(df.x.max() - x_offset) + 1, int(df.y.max() - y_offset) + 1))

    x = np.array(df.x.values - x_offset, dtype=int)
    y = np.array(df.y.values - y_offset, dtype=int)
    z = df.z.values

    arr[x, y] = z
    if not origin_bottom:
        arr = arr.transpose(1,0)
        arr = arr[::-1,:]

    return arr

def get_map_center(df: pandas.DataFrame) -> Tuple[float]:
    return (np.round((df.x.max()) / 2).astype(int), np.round(df.y.max() / 2).astype(int))

def rescale_height_map(height_map: np.ndarray) -> np.ndarray:
    # NOTE: assuming height map has 1m resolution!
    return skimage.transform.rescale(height_map, (1.0 / 8, 1.0 / 8), cval=1, preserve_range=True, clip=True, anti_aliasing=True)

def rotate_height_map(height_map: np.ndarray, 
                      rotation_angle: float, 
                      center: Tuple[float],
                      size_x: float,
                      size_y: float,
                      res_x: float,
                      res_y: float) -> np.ndarray:
    height_map = skimage.transform.rotate(height_map, -rotation_angle, resize=True, cval=-1, preserve_range=True, clip=True, center=center)

    # center must be recalculated because img is resized!
    center_rotated = (height_map.shape[0] / 2 - 0.5, height_map.shape[1] / 2 - 0.5)

    # centre according to skimage rotate default
    lower_left = (max(0, center_rotated[0] - size_x / 2 / res_x), max(center_rotated[1] - size_y / 2 / res_y, 0))
    lower_left = (np.round(lower_left[0]).astype(int), np.round(lower_left[1]).astype(int))

    upper_right = (
        lower_left[0] + min(int((height_map.shape[0]-1 - lower_left[0]) / 8) * 8, int(size_x / 8) * 8),
        lower_left[1] + min(int((height_map.shape[1]-1 - lower_left[1]) / 8) * 8, int(size_y / 8) * 8)
    )

    height_map = height_map[lower_left[0]:upper_right[0], lower_left[1]:upper_right[1]]

    return height_map

def ndarray2dataframe(arr: np.ndarray, x_offset: float = 0, y_offset: float = 0, origin_bottom: bool = True) -> pandas.DataFrame:
    x_arr = []
    y_arr = []
    z_arr = []
    if not origin_bottom:
        arr_tmp = arr.transpose(1,0)[:,::-1]
    else:
        arr_tmp = arr
    for xx in range(arr_tmp.shape[0]):
        for yy in range(arr_tmp.shape[1]):
            x_arr.append(xx)
            y_arr.append(yy)
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



class DataSource(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_data(self, bounding_box: BoundingBox, cache_dir: str) -> pandas.DataFrame:
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

    def cut_out_bounding_box(self, df: pandas.DataFrame, bounding_box: BoundingBox):
        box = bounding_box.get_box(bounding_box.crs_projected)
        p0, p1, p2, _ = get_polygon_node_points(box)
        df = clip_dataframe_to_bounding_box(df, box.bounds)
        rotation_angle = get_rectangle_rotation_angle(box, p0)
        height_map = dataframe2ndarray(df)

        center = get_map_center(df)

        size_x = p0.distance(p1)
        size_y = p1.distance(p2)

        height_map = rotate_height_map(height_map, rotation_angle, center, size_x, size_y, 1.0, 1.0)
        height_map_reduced = rescale_height_map(height_map)

        height_map_df = ndarray2dataframe(height_map_reduced)

        return height_map_df
    
    def get_png(self, bounding_box: BoundingBox, cache_dir: str):
        if self.cached_data is None or not self.cached_data_bounding_box.equals(bounding_box):
            self.get_data(bounding_box, cache_dir)

        df = self.cached_data
        file_path = os.path.join(cache_dir, 'current_height_map.png')
        dataframe_in_bbox_to_png(df, self.cached_data_bounding_box, file_path)

        return file_path


class GeoTiffDataSource(DataSource):
    def __init__(self):
        self.data_type: str = 'geotiff'

    def merge_image_files(self, image_files, out_dir, bounding_box: BoundingBox):
        self.current_merged_image_path = os.path.join(out_dir, self.data_folder, 'merged_images_in_bbox.tif')
        # merge(image_files, dst_path=self.current_merged_image_path, res=(1.0,1.0), resampling=Resampling.bilinear)
        bounds = bounding_box.get_buffer(self.crs).bounds
        merge(image_files, dst_path=self.current_merged_image_path, bounds=bounds, resampling=Resampling.bilinear)

            
        
    def get_merged_dataframe(self, bounding_box: BoundingBox) -> pandas.DataFrame:
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
                (1.0, 1.0),
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


