from typing import Dict
import geojson
from terrain_extraction.bbox_utils import BoundingBox
from pyproj.crs import CRS
from shapely import Polygon, union_all
from geopandas import GeoDataFrame
from streamlit import cache_data

@cache_data
def read_file(path_to_file: str):
    with open(path_to_file, encoding='utf8') as f:
        return geojson.load(f)
    
def read_file_object(file_object):
    return geojson.load(file_object)
    
@cache_data
def get_bounding_box_from_file_object(file_object):
    return get_bounding_box(osm_data=read_file_object(file_object))

def get_bounding_box(osm_data: Dict):
    if 'crs' in osm_data and 'properties' in osm_data['crs'] and 'name' in osm_data['crs']['properties']:
        crs = CRS.from_string(osm_data['crs']['properties']['name'])
    else:
        crs = CRS.from_epsg(4326)

    if 'bbox' in osm_data:
        xmin, ymin, xmax, ymax = osm_data['bbox']
        bbox_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    else:
        gdf = GeoDataFrame.from_features(osm_data['features'])
        bbox_polygon = union_all(gdf.geometry.buffer(0)).envelope
    return BoundingBox(bbox_polygon, crs)

