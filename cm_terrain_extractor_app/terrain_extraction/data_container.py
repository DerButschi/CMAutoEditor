import numpy as np
from affine import Affine
from pyproj.crs import CRS
from terrain_extraction.bbox_utils import BoundingBox
from rasterio.enums import Resampling
from rasterio import open as rasterio_open
from rasterio.warp import calculate_default_transform, reproject
from rasterio import band
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from PIL import Image
import pandas

from typing import Tuple

class ElevationDataContainer:
    def __init__(self):
        self.projected_array = None
        self.transform = None
        self.projected_crs = None
        self.n_cells_x = None
        self.n_cells_y = None
        self.z_scale = None

    @classmethod
    def from_geotiff(
        cls,
        path_to_file: str,
        bounding_box: BoundingBox,
        target_res: Tuple[float] = (1.0, 1.0)
    ):
        with rasterio_open(path_to_file) as src:
            # 1. Get the first three corners of the UTM‐polygon
            x0,y0 = bounding_box.box_utm.exterior.coords[0]
            x1,y1 = bounding_box.box_utm.exterior.coords[1]
            x2,y2 = bounding_box.box_utm.exterior.coords[2]

            # 2. Compute rotation, size, array dims
            theta      = np.arctan2(y1-y0, x1-x0)
            wm     = np.hypot(x1-x0, y1-y0)
            hm     = np.hypot(x2-x1, y2-y1)
            delta_x, delta_y  = target_res
            nx      = int(np.ceil(wm/delta_x))
            ny      = int(np.ceil(hm/delta_y))

            # 3. UTM Affine (scale → rotate → translate)
            T_UTM = (
                Affine.translation(x0, y0)
                * Affine.rotation(np.degrees(theta))
                * Affine.scale(delta_x, delta_y)
            )

            # 4. Reproject into UTM grid
            bands = src.count
            dst = np.full((bands, ny, nx),
                        src.nodata if src.nodata is not None else np.nan,
                        dtype=src.dtypes[0])

            reproject(
                source         = band(src, list(range(1, bands+1))),
                destination    = dst,
                src_transform  = src.transform,
                src_crs        = src.crs,
                dst_transform  = T_UTM,
                dst_crs        = bounding_box.crs_projected,
                resampling     = Resampling.bilinear,
                dst_nodata     = src.nodata
            )

        data_container = ElevationDataContainer()
        data_container.projected_array = dst[0, ...]
        data_container.transform = T_UTM
        data_container.projected_crs = bounding_box.crs_projected
        data_container.n_cells_x = dst.shape[2]
        data_container.n_cells_y = dst.shape[1]
        data_container.z_scale = dst[0, ...].max()

        return data_container

    def to_png(
        self,
        output_path: str,
        out_crs: CRS,
        resampling: Resampling = Resampling.bilinear,
        vmin: float = None,
        vmax: float = None
    ) -> None:
        """
        Warp a UTM‐projected array back to WGS84 and save as an 8-bit PNG.

        Parameters
        ----------
        arr : np.ndarray
            2D (H×W) or 3D (bands×H×W) array in UTM.
        src_transform : Affine
            Geotransform of `arr` in UTM units (metres).
        src_crs : str
            CRS of `arr` (default 'EPSG:32632').
        output_path : str
            File path for the output PNG.
        resampling : Resampling
            Resampling method for reprojection.
        vmin, vmax : float, optional
            Value range for stretching to [0,255]. If None, uses array min/max.

        Writes
        ------
        output_path : a PNG file in WGS84.
        """

        arr = self.projected_array
        src_crs = self.projected_crs
        src_transform = self.transform

        # 1. Determine source height/width
        if arr.ndim == 2:
            bands = 1
            height, width = arr.shape
            src_data = arr[np.newaxis, ...]
        else:
            bands, height, width = arr.shape
            src_data = arr

        # 2. Compute WGS84 extents & target transform/shape
        left, bottom, right, top = array_bounds(height, width, src_transform)
        dst_crs = out_crs
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs,
            width, height,
            left, bottom, right, top
        )

        # 3. Allocate destination array (bands × dst_height × dst_width)
        dst = np.zeros((bands, dst_height, dst_width), dtype=src_data.dtype)

        # 4. Reproject each band
        for b in range(bands):
            reproject(
                source      = src_data[b],
                destination = dst[b],
                src_transform = src_transform,
                src_crs       = src_crs,
                dst_transform = dst_transform,
                dst_crs       = dst_crs,
                resampling    = resampling
            )

        # 5. Stack or squeeze to H×W×bands for PIL
        if bands == 1:
            img = dst[0]
        else:
            # move to H×W×bands
            img = np.moveaxis(dst, 0, -1)

        # 6. Scale to uint8
        if vmin is None: vmin = np.nanmin(img)
        if vmax is None: vmax = np.nanmax(img)
        img = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255)
        img8 = img.astype(np.uint8)

        # 7. Save with PIL
        im = Image.fromarray(img8)
        im.save(output_path)

    def to_dataframe(
        self,
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
        n_rows, n_cols = self.projected_array.shape
        cols = np.arange(n_cols)
        rows = np.arange(n_rows)
        col_idxs, row_idxs = np.meshgrid(cols, rows)
        
        # 2. Convert pixel‐indices to map coords
        #    x = A * col + B * row + C
        #    y = D * col + E * row + F
        xs, ys = self.transform * (col_idxs, row_idxs)  # vectorized Affine
        
        # 3. Flatten everything
        xs_flat = xs.ravel()
        ys_flat = ys.ravel()
        zs_flat = self.projected_array.ravel()
        
        # 4. Build DataFrame
        df = pandas.DataFrame({
            'x': xs_flat,
            'y': ys_flat,
            'z': zs_flat
        })
        
        return df
    
    def to_bytes(self):
        arr = self.projected_array
        arr = arr.T
        return ((arr - arr.min()) / arr.max() * 65536).astype('uint16').tobytes(order='C')

    
        
