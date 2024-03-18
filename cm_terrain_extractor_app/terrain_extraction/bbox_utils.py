from typing import List, Self
from shapely import Polygon, Point
import numpy as np
from pyproj.crs import CRS
from terrain_extraction.projection_utils import get_projection_epsg_code_from_bbox, transform_polygon, transform_point
import pandas

def make_polygon_counter_clockwise(polygon: Polygon) -> Polygon:
    if not polygon.exterior.is_ccw:
        polygon = Polygon(polygon.exterior.coords[::-1])

    return polygon

def find_idx_closest_point_in_list(points: List[Point], ref_point: Point) -> int:
    dist = [ref_point.distance(pt) for pt in points]
    min_idx = np.argmin(dist)
    return min_idx

def get_polygon_node_points(polygon: Polygon) -> List[Point]:
    return [Point(*coord) for coord in polygon.exterior.coords][:-1]


def find_idx_closest_polygon_node(polygon: Polygon, ref_point: Point) -> int:
    polygon_points = get_polygon_node_points(polygon)
    return find_idx_closest_point_in_list(polygon_points, ref_point)

def permute_list_to_idx(lst: List, start_index: int) -> List:
    if start_index < 0 or start_index >= len(lst):
        raise ValueError("Invalid start index")

    # Use slicing to create two parts: from start_index to the end and from the beginning to start_index
    part1 = lst[start_index:]
    part2 = lst[:start_index]

    # Concatenate the two parts to get the permuted list
    permuted_list = part1 + part2

    return permuted_list

def permute_polygon_to_idx(polygon: Polygon, node_idx: int) -> Polygon:
    polygon_points = get_polygon_node_points(polygon)
    permutated_polygon_points = permute_list_to_idx(polygon_points, node_idx)
    return Polygon(permutated_polygon_points)

def get_rectangle_rotation_angle(rectangle: Polygon, origin_point: Point, degrees=True):
    # get rotation angle of x-axis, assumed to be defined by (x0, y0) -> (x1, y1)
    # since the last point in a polygon is always identical to the first point and np.argmin returns the first match,
    # there should always be min_idx + 1 within the array
    origin_idx = find_idx_closest_polygon_node(rectangle, origin_point)
    rectangle = permute_polygon_to_idx(rectangle, origin_idx)
    rectangle_points = get_polygon_node_points(rectangle)
    p0 = rectangle_points[0]
    p1 = rectangle_points[1]

    angle = np.arctan2(p1.y - p0.y, p1.x - p0.x)
    if degrees:
        angle = angle * 180.0 / np.pi   
    return angle       



class BoundingBox:
    def __init__(self, polygon: Polygon, crs: CRS = CRS.from_epsg(4326)) -> None:
        orig_polygon_points = get_polygon_node_points(polygon)
        if not crs.to_epsg() == 4326:
            self.polygon_wgs84 = transform_polygon(polygon, from_epsg=crs.to_epsg, to_epsg=4326)
        else:
            self.polygon_wgs84 = polygon

        epsg_code_utm = get_projection_epsg_code_from_bbox(self.polygon_wgs84)
        self.polygon_utm = transform_polygon(self.polygon_wgs84, from_epsg=4326, to_epsg=epsg_code_utm)

        self.crs_orig: CRS = crs
        self.crs_projected: CRS = CRS.from_epsg(epsg_code_utm)

        orig_p0_utm = transform_point(orig_polygon_points[0], self.crs_orig.to_epsg(), self.crs_projected.to_epsg())

        box = make_polygon_counter_clockwise(self.polygon_utm.minimum_rotated_rectangle)
        origin_idx = find_idx_closest_polygon_node(box, orig_p0_utm)
        self.box_utm = permute_polygon_to_idx(box, origin_idx)

        self.box_wgs84 = transform_polygon(self.box_utm, self.crs_projected.to_epsg(), 4326)

    def get_box(self, crs: CRS = CRS.from_epsg(4326)) -> Polygon:
        if crs.to_epsg() == 4326:
            return self.box_wgs84
        elif crs.to_epsg() == self.crs_projected.to_epsg():
            return self.box_utm
        else:
            return transform_polygon(self.box_wgs84, 4326, crs.to_epsg())
        
    def get_dataframe(self, crs: CRS = CRS.from_epsg(4326)) -> pandas.DataFrame:
        box = self.get_box(crs)
        df = pandas.DataFrame({
            'x': [coord[0] for coord in box.exterior.coords][:-1],
            'y': [coord[1] for coord in box.exterior.coords][:-1]
        })
        return df
    
    def get_length_xaxis(self) -> float:
        return self.get_length_axis(axis=0)
    
    def get_length_yaxis(self) -> float:
        return self.get_length_axis(axis=1)

    def get_length_axis(self, axis=0):
        box_points = get_polygon_node_points(self.box_utm)
        if axis == 0:
            return box_points[0].distance(box_points[1])
        else:
            return box_points[0].distance(box_points[3])
        
    def get_area(self):
        return self.box_utm.area
    
    def get_bounds(self, crs: CRS = CRS.from_epsg(4326)):
        return self.get_box(crs).bounds
    
    def get_coordinates(self, crs: CRS = CRS.from_epsg(4326), xy=True):
        box = self.get_box(crs)
        if xy:
            return [(coord[0], coord[1]) for coord in box.exterior.coords]
        else:
            return [(coord[1], coord[0]) for coord in box.exterior.coords]

    def equals(self, other: Self):
        this_box = self.get_box(self.crs_projected)
        other_box = other.get_box(self.crs_projected)
        return this_box.equals_exact(other_box, tolerance=0.1)
    
    def get_buffer(self, crs: CRS = CRS.from_epsg(4326), buffer_size: float = 100.0):
        return transform_polygon(self.get_box(self.crs_projected).buffer(buffer_size), self.crs_projected.to_epsg(), crs.to_epsg())

    def cycle_origin(self):
        self.box_utm = permute_polygon_to_idx(self.box_utm, 1)
        self.box_wgs84 = permute_polygon_to_idx(self.box_wgs84, 1)

    def get_rotation_angle(self):
        box_points = get_polygon_node_points(self.box_utm)
        return get_rectangle_rotation_angle(self.box_utm, box_points[0])
    
    def get_origin_point(self, crs: CRS = CRS.from_epsg(4326)):
        box_points = get_polygon_node_points(self.get_box(crs))
        return box_points[0]
    
    def get_reference_points(self, crs: CRS = CRS.from_epsg(4326)):
        box_points = get_polygon_node_points(self.get_box(crs))
        return box_points[0], box_points[1], box_points[3]
        



    
