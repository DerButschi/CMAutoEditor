from typing import Tuple
from shapely import Polygon, Point, LineString
from pyproj import Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj.crs import CRS
import numpy as np
from rasterio.warp import calculate_default_transform, reproject
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio import open as rasterio_open
from rasterio import band


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

def transform_linestring(ls: LineString, from_epsg, to_epsg) -> LineString:
    return LineString([transform_point(Point(coord[0], coord[1]), from_epsg, to_epsg) for coord in ls.coords])

def reproject_array(arr: np.ndarray, source_bounds: Tuple[float], source_crs: CRS, destination_crs: CRS, destination_resolution: Tuple[float], resampling=Resampling.bilinear) -> np.ndarray:
    affine_transform = from_bounds(*source_bounds, arr.shape[1], arr.shape[0])
    trf=calculate_default_transform(
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

    reprojected_arr, _ = reproject(
        source=arr, 
        destination=np.zeros((trf[2], trf[1])), 
        src_transform=affine_transform, 
        dst_transform=trf[0], 
        src_crs=source_crs,
        dst_crs=destination_crs, 
        resampling=resampling
    )

    return reprojected_arr

def reproject_geotiff(source_path: str, destination_path: str, destination_crs: CRS, destination_resolution: Tuple[float] = (1.0, 1.0), resampling=Resampling.bilinear):
    """
    Script to reproject a GeoTIFF from WGS84 to UTM Zone 32N (EPSG:32632).

    Features:
    - Reproject from WGS84 to UTM32N
    - Optional custom output resolution (in CRS units)
    """

    with rasterio_open(source_path) as src:
        # Compute transform, width, height for dest
        transform, width, height = calculate_default_transform(
            src.crs,
            destination_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=destination_resolution
        )

        # Update metadata for destination
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': destination_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Write reprojected raster
        with rasterio_open(destination_path, 'w', **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=band(src, band_idx),
                    destination=band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=destination_crs,
                    resampling=resampling
                )
