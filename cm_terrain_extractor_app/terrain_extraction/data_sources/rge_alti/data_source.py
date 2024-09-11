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
import py7zr

from rasterio.transform import xy as transform_xy
from terrain_extraction.data_source_utils import ASCDataSource, GeoTiffDataSource
from terrain_extraction.bbox_utils import BoundingBox


class FranceDataSource(ASCDataSource, GeoTiffDataSource):
    def __init__(self):
        self.name = 'RGE Alti'
        self.data_type = 'asc'
        self.model_type = 'DTM'
        self.resolution = '5 m'
        self.country = 'France'
        self.crs = CRS.from_epsg(2154)
        self.gdf = None
        self.outline: MultiPolygon = None
        self.data_folder = 'rge_alti'
        self.current_merged_image_path = None,
        self.cached_data: pandas.DataFrame = None
        self.cached_data_bounding_box: BoundingBox = None
        self.envelope = Polygon([(99101.77742169285, 6046555.784040092), (1242435.995219805, 6046555.784040092), (1242435.995219805, 7110479.964018984), (99101.77742169285, 7110479.964018984), (99101.77742169285, 6046555.784040092)])
        self.gdf_geojson_path = 'cm_terrain_extractor_app/terrain_extraction\\data_sources\\rge_alti\\rge_alti.geojson'
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
# with py7zr.SevenZipFile('archive.7z', 'r') as archive:
#     allfiles = archive.getnames()
#     selective_files = [f for f in allfiles if filter_pattern.match(f)]
#     archive.extract(targets=selective_files)    

    def get_images_in_bounding_box(self, bounding_box: BoundingBox, outdir):
        image_files = []
        for dir_path, dir_names, file_names in os.walk(os.path.join(outdir, self.data_folder)):
            for file_name in file_names:
                if file_name.endswith('.7z'):
                    with py7zr.SevenZipFile(os.path.join(dir_path, file_name), 'r') as archive:
                        files_to_extract = []
                        for image_name in archive.getnames():
                            if not image_name.endswith('.asc'):
                                continue
                            x, y = os.path.basename(image_name).split('.')[0].split('_')[2:4]
                            x = float(x) * 1000
                            y = float(y) * 1000
                            image_bounds = Polygon([
                                (x - 2.5, y - 2.5), (x - 2.5 + 5000 - 2.5, y), (x - 2.5 + 5000 - 2.5, y - 5000 - 2.5), (x - 2.5, y - 5000 - 2.5)
                            ])
                            if image_bounds.intersects(bounding_box.get_box(self.crs)):
                                image_files.append(os.path.join(outdir, self.data_folder, image_name))
                                if not os.path.isfile(os.path.join(outdir, self.data_folder, image_name)):
                                    files_to_extract.append(image_name)

                        archive.extract(path=os.path.join(outdir, self.data_folder), targets=files_to_extract)

        return image_files
    
    def download_overlapping_data(self, missing_files: List[str], out_dir: str):
        for url in missing_files:
            file_name = url.split('/')[-1]
            total_size = None
            r = requests.get(url, stream=True)
            if 'content-length' in r.headers.keys():
                total_size = float(r.headers['content-length'])
            os.makedirs(os.path.join(out_dir, self.data_folder), exist_ok=True)
            size_downloaded = 0
            progressbar = st.progress(0, "Downloading missing file {}...".format(file_name))
            with open(os.path.join(os.path.join(out_dir, self.data_folder, file_name)), 'wb') as fd:
                for chunk in r.iter_content(chunk_size=4096):
                    fd.write(chunk)
                    size_downloaded += 4096
                    if total_size is not None:
                        progress = min(int(size_downloaded / total_size * 100), 100)
                        text = 'Downloading missing file {}... {} of {} MB'.format(file_name, int(size_downloaded / 1024 / 1024), int(total_size / 1024 / 1024))
                    else:
                        progress = int(size_downloaded / 1024 / 1024)
                        text = 'Downloading missing file {}... {} MB'.format(file_name, int(size_downloaded / 1024 / 1024))

                    progressbar.progress(progress, text=text)

    def get_data(self, bounding_box: BoundingBox, cache_dir: str):
        if self.cached_data is not None and self.cached_data_bounding_box.equals(bounding_box):
            df = self.cached_data
        else:
            st.write("Checking data cache...")
            missing_files = self.get_missing_files(bounding_box, cache_dir)
            self.download_overlapping_data(missing_files, cache_dir)
            st.write("Unpacking asc-files intersecting with bounding box...")
            data_files = self.get_images_in_bounding_box(bounding_box, cache_dir)
            st.write("Merging geotiffs...")
            self.merge_image_files(data_files, cache_dir, bounding_box)
            st.write('Reading elevation data from geotiff...')
            df = self.get_merged_dataframe(bounding_box)
            self.cached_data = df
            self.cached_data_bounding_box = bounding_box

        st.write('Cutting out data in selected area...')
        df = self.cut_out_bounding_box(df, bounding_box)

        return df
