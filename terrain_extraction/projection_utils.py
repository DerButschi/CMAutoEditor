from typing import Tuple
from shapely import Polygon, Point
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj.crs import CRS
import numpy as np
import rasterio


def get_projection_epsg_code_from_bbox(bbox):
    aoi = AreaOfInterest(*bbox.bounds)

    crs_list = query_utm_crs_info(area_of_interest=aoi)

    crs = None
    for crs_candidate in crs_list:
        if 'WGS 84 / UTM' in crs_candidate.name:
            crs = crs_candidate
        
    if crs is None:
        crs = crs_list[-1]

    return crs.code


def transform_polygon(polygon: Polygon, from_epsg, to_epsg) -> Polygon:
    transformer = Transformer.from_crs('epsg:{}'.format(from_epsg), 'epsg:{}'.format(to_epsg), always_xy=True)

    polygon_xy = transformer.transform([coord[0] for coord in polygon.exterior.coords], [coord[1] for coord in polygon.exterior.coords])
    projected_polygon = Polygon([(polygon_xy[0][i], polygon_xy[1][i]) for i in range(len(polygon_xy[0]))])

    return projected_polygon

def transform_point(point: Point, from_epsg, to_epsg) -> Point:
    transformer = Transformer.from_crs('epsg:{}'.format(from_epsg), 'epsg:{}'.format(to_epsg), always_xy=True)
    return Point(transformer.transform(point.x, point.y))

def reproject_array(arr: np.ndarray, source_bounds: Tuple[float], source_crs: CRS, destination_crs: CRS, destination_resolution: Tuple[float], resampling=rasterio.enums.Resampling.nearest) -> np.ndarray:
    affine_transform = rasterio.transform.from_bounds(*source_bounds, arr.shape[1], arr.shape[0])
    trf=rasterio.warp.calculate_default_transform(
        source_crs, 
        destination_crs, 
        arr.shape[1], 
        arr.shape[0], 
        left=source_bounds[0],
        bottom=source_bounds[1],
        right=source_bounds[2],
        top=source_bounds[3], 
        resolution=destination_resolution
    )

    reprojected_arr, _ = rasterio.warp.reproject(
        source=arr, 
        destination=np.zeros((trf[2], trf[1])), 
        src_transform=affine_transform, 
        dst_transform=trf[0], 
        src_crs=source_crs,
        dst_crs=destination_crs, 
        resampling=rasterio.enums.Resampling.nearest
    )

    return reprojected_arr