import json
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Polygon, union_all
import streamlit as st
from typing import List
import os
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from pyproj.crs import CRS
import pandas
import gzip
import shutil

from rasterio.transform import xy as transform_xy
from terrain_extraction.data_source_utils import XYZDataSource
from terrain_extraction.bbox_utils import BoundingBox


class NRWDataSource(XYZDataSource):
    def __init__(self):
        self.name = 'NRW DGM1'
        self.data_type = 'xyz'
        self.model_type = 'DTM'
        self.resolution = '1 m'
        self.crs = CRS.from_epsg(25832)
        self.gdf = None
        self.outline: MultiPolygon = None
        self.data_folder = 'nrw_dgm1'
        self.current_merged_image_path = None,
        self.cached_data: pandas.DataFrame = None
        self.cached_data_bounding_box: BoundingBox = None
        self.envelope = Polygon([(279999.5, 5575999.5), (531999.5, 5575999.5), (531999.5, 5821999.5), (279999.5, 5821999.5), (279999.5, 5575999.5)])
        self.gdf_geojson_path = 'terrain_extraction\\data_sources\\nrw_dgm1\\nrw_dgm1.geojson'
        self.data_delimiter = r"\s+"

    def get_outline(self):
        if self.outline is None:
            gdf = self.get_gdf()
            self.outline = union_all(gdf.geometry)

        return self.outline
    
    
    def get_missing_files(self, bounding_box: BoundingBox, data_storage_folder: str) -> List[str]:
        gdf = self.get_gdf()
        missing_files = []
        for _, entry in gdf[gdf.intersects(bounding_box.get_box(self.crs))].iterrows():
            relative_location = os.path.join(self.data_folder, entry['url'].split('/')[-1])
            if not os.path.isfile(os.path.join(data_storage_folder, relative_location)):
                missing_files.append(entry['url'])

        return missing_files
    





    def get_images_in_bounding_box(self, bounding_box: BoundingBox, outdir):
        image_files = []
        for dir_path, dir_names, file_names in os.walk(os.path.join(outdir, self.data_folder)):
            for file_name in file_names:
                if file_name.endswith('.gz'):
                    x, y = file_name.split('_')[2:4]
                    x = float(x) * 1000
                    y = float(y) * 1000
                    image_bounds = Polygon([
                        (x - 0.5, y - 0.5), (x + 1000 - 0.5, y - 0.5), (x + 1000 - 0.5, y + 1000 - 0.5), (x - 0.5, y + 1000 - 0.5)
                    ])
                    image_name = '.'.join(file_name.split('.')[:-1])
                    if image_bounds.intersects(bounding_box.get_box(self.crs)):
                        image_files.append(os.path.join(dir_path, file_name.split('.')[0], image_name))
                        if not os.path.isfile(os.path.join(dir_path, file_name.split('.')[0], image_name)):
                            with gzip.open(os.path.join(dir_path, file_name), 'rb') as f_in:
                                os.makedirs(os.path.join(dir_path, file_name.split('.')[0]), exist_ok=True)
                                with open(os.path.join(dir_path, file_name.split('.')[0], image_name), 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)

        return image_files
    
    
    def download_overlapping_data(self, missing_files: List[str], out_dir: str):
        for url in missing_files:
            file_name = url.split('/')[-1]
            r_head = requests.head(url)
            content_length = r_head.headers['content-length']
            r = requests.get(url, stream=True)
            os.makedirs(os.path.join(out_dir, self.data_folder), exist_ok=True)
            size_downloaded = 0
            total_size = float(content_length)
            progressbar = st.progress(0, "Downloading missing file {}...".format(file_name))
            with open(os.path.join(os.path.join(out_dir, self.data_folder, file_name)), 'wb') as fd:
                for chunk in r.iter_content(chunk_size=4096):
                    fd.write(chunk)
                    size_downloaded += 4096
                    progress = min(int(size_downloaded / total_size * 100), 100)
                    progressbar.progress(progress, text='Downloading missing file {}... {} of {} MB'.format(file_name, int(size_downloaded / 1024 / 1024), int(total_size / 1024 / 1024)))

    def get_data(self, bounding_box: BoundingBox, cache_dir: str):
        if self.cached_data is not None and self.cached_data_bounding_box.equals(bounding_box):
            df = self.cached_data
        else:
            st.write("Checking data cache...")
            missing_files = self.get_missing_files(bounding_box, cache_dir)
            self.download_overlapping_data(missing_files, cache_dir)
            st.write("Unpacking xyz-files intersecting with bound box...")
            data_files = self.get_images_in_bounding_box(bounding_box, cache_dir)
            # st.write("Merging geotiffs...")
            # self.merge_image_files(image_files, cache_dir)
            st.write('Reading elevation data from xyz...')
            df = self.get_merged_dataframe(bounding_box, data_files)
            self.cached_data = df
            self.cached_data_bounding_box = bounding_box

        st.write('Cutting out data in selected area...')
        df = self.cut_out_bounding_box(df, bounding_box)

        return df
