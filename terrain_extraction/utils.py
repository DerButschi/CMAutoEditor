from shapely import Polygon
from pyproj.crs import CRS
from terrain_extraction.projection_utils import get_projection_epsg_code_from_bbox, transform_polygon

class BoundingBox:
    def __init__(self, polygon: Polygon, crs: CRS) -> None:
        if not CRS.to_epsg == 4326:
            self.polygon_wgs84 = transform_polygon(polygon, from_epsg=crs.to_epsg, to_epsg=4326)
        else:
            self.polygon_wgs84 = polygon

        epsg_code_utm = get_projection_epsg_code_from_bbox(self.polygon_wgs84)
        self.polygon_utm = transform_polygon(self.polygon_wgs84, from_epsg=4326, to_epsg=epsg_code_utm)

        self.crs_orig = crs
        self.crs_projected = epsg_code_utm

        box = self.polygon_utm.minimum_rotated_rectangle
        



