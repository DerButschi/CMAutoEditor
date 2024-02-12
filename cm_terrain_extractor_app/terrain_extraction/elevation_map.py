import pandas
from shapely import Polygon

from terrain_extraction.bbox_utils import get_rectangle_rotation_angle, get_polygon_node_points






def cut_out_bounding_box(df: pandas.DataFrame, bounding_box: Polygon, data_epsg_code: int, cache_dir: str):
    p0, p1, p2, _ = get_polygon_node_points(bounding_box)
    dataframe_in_bbox_to_png(df, bounding_box, data_epsg_code, cache_dir)
    df = clip_dataframe_to_bounding_box(df, bounding_box)
    rotation_angle = get_rectangle_rotation_angle(bounding_box, p0)
    height_map_orig = dataframe2ndarray(df)


    # img = Image.fromarray(height_map_orig).convert('RGB')
    # img.save('height_map.png')

    center = get_map_center(df)

    size_x = p0.distance(p1)
    size_y = p1.distance(p2)

    height_map = rotate_height_map(height_map_orig, rotation_angle, center, size_x, size_y, 1.0, 1.0)
    height_map_reduced = rescale_height_map(height_map)

    # img = Image.fromarray(height_map_reduced).convert('RGB')
    # img.save('height_map_reduced.png')

    height_map_df = ndarray2dataframe(height_map_reduced)

    return height_map_df
