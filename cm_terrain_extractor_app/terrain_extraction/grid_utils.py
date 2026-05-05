from affine import Affine
import numpy as np
from terrain_extraction.bbox_utils import BoundingBox
def make_rotated_affine(bbox: BoundingBox, pixel_size):
    p0 = bbox.get_origin_point(bbox.crs_projected)
    theta  = -bbox.get_rotation_angle() / 180.0 * np.pi
    return Affine(
        np.cos(theta)*pixel_size,  np.sin(theta)*pixel_size,  p0.x,
       -np.sin(theta)*pixel_size,  np.cos(theta)*pixel_size,  p0.y
    )
