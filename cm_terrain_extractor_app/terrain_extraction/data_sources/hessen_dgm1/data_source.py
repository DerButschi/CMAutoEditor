import json
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Polygon, union_all
import streamlit as st
from typing import List
import os
import requests
from datetime import datetime
import zipfile
import matplotlib.pyplot as plt
from pyproj.crs import CRS
import pandas

from rasterio.transform import xy as transform_xy
from terrain_extraction.data_source_utils import GeoTiffDataSource
from terrain_extraction.bbox_utils import BoundingBox


class HessenDataSource(GeoTiffDataSource):
    def __init__(self):
        self.name = 'Hessen DGM1'
        self.data_type = 'geotiff'
        self.model_type = 'DTM'
        self.resolution = '1 m'
        self.crs = CRS.from_epsg(25832)
        self.gdf = None
        self.outline: MultiPolygon = None
        self.data_folder = 'hessen_dgm1'
        self.current_merged_image_path = None,
        self.cached_data: pandas.DataFrame = None
        self.cached_data_bounding_box: BoundingBox = None
        self.envelope = Polygon([(412133.984, 5471408.535), (586408.646, 5471408.535), (586408.646, 5723092.335), (412133.984, 5723092.335), (412133.984, 5471408.535)])
        self.gdf_geojson_path = 'cm_terrain_extractor_app/terrain_extraction/data_sources/hessen_dgm1/hessen_dgm1.geojson'

    def get_outline(self):
        if self.outline is None:
            gdf = self.get_gdf()
            self.outline = union_all(gdf.geometry)

        return self.outline
    
    def intersects_bounding_box(self, bounding_box: BoundingBox) -> bool:
        if not bounding_box.get_box(self.crs).intersects(self.envelope):
            return False
        gdf = self.get_gdf()
        return gdf.sindex.query(bounding_box.get_box(self.crs), predicate='intersects').any()
    
    def get_missing_files(self, bounding_box: BoundingBox, data_storage_folder: str) -> List[str]:
        gdf = self.get_gdf()
        missing_files = []
        for _, entry in gdf[gdf.intersects(bounding_box.get_box(self.crs))].iterrows():
            relative_location = os.path.join(self.data_folder, entry['folder'], entry['file_name'] + '.zip')
            if not os.path.isfile(os.path.join(data_storage_folder, relative_location)):
                missing_files.append((entry['folder'], entry['file_name']))

        return missing_files
    
    def get_images_in_bounding_box(self, bounding_box: BoundingBox, outdir):
        image_files = []
        for dir_path, dir_names, file_names in os.walk(os.path.join(outdir, self.data_folder)):
            for file_name in file_names:
                if file_name.endswith('.zip'):
                    with zipfile.ZipFile(os.path.join(dir_path, file_name), 'r') as zip_ref:
                        for image_name in zip_ref.namelist():
                            if image_name.endswith('.tfw'):
                                continue
                            x, y = image_name.split('_')[2:4]
                            x = float(x) * 1000
                            y = float(y) * 1000
                            image_bounds = Polygon([
                                (x - 0.5, y - 0.5), (x + 1000 - 0.5, y - 0.5), (x + 1000 - 0.5, y + 1000 - 0.5), (x - 0.5, y + 1000 - 0.5)
                            ])
                            if image_bounds.intersects(bounding_box.get_box(self.crs)):
                                image_files.append(os.path.join(dir_path, os.path.splitext(file_name)[0], image_name))
                                if not os.path.isfile(os.path.join(dir_path, os.path.splitext(file_name)[0], image_name)):
                                    zip_ref.extract(image_name, os.path.join(dir_path, os.path.splitext(file_name)[0]))

        return image_files
    
    
    def download_overlapping_data(self, missing_files: List[str], out_dir: str):
        for dir_name, file_name in missing_files:
            date_str = '{}{}{}'.format(datetime.now().year, str(datetime.now().month).zfill(2), str(datetime.now().day).zfill(2))
            url = 'https://gds.hessen.de/downloadcenter/{}/3D-Daten/Digitales Geländemodell (DGM1)/{}/{} - DGM1.zip'.format(
                date_str,
                dir_name,
                file_name 
            )
            r_head = requests.head(url)
            content_length = r_head.headers['content-length']
            r = requests.get(url, stream=True)
            os.makedirs(os.path.join(out_dir, self.data_folder, dir_name), exist_ok=True)
            size_downloaded = 0
            total_size = float(content_length)
            progressbar = st.progress(0, "Downloading missing file {}...".format(file_name + '.zip'))
            with open(os.path.join(os.path.join(out_dir, self.data_folder, dir_name, file_name + '.zip')), 'wb') as fd:
                for chunk in r.iter_content(chunk_size=4096):
                    fd.write(chunk)
                    size_downloaded += 4096
                    progress = min(int(size_downloaded / total_size * 100), 100)
                    progressbar.progress(progress, text='Downloading missing file {}... {} of {} MB'.format(file_name + '.zip', int(size_downloaded / 1024 / 1024), int(total_size / 1024 / 1024)))

    def get_data(self, bounding_box: BoundingBox, cache_dir: str):
        if self.cached_data is not None and self.cached_data_bounding_box.equals(bounding_box):
            df = self.cached_data
        else:
            st.write("Checking data cache...")
            missing_files = self.get_missing_files(bounding_box, cache_dir)
            self.download_overlapping_data(missing_files, cache_dir)
            st.write("Unpacking geotiffs intersecting with bounding box...")
            image_files = self.get_images_in_bounding_box(bounding_box, cache_dir)
            st.write("Merging geotiffs...")
            self.merge_image_files(image_files, cache_dir, bounding_box)
            st.write('Reading elevation data from geotiff...')
            df = self.get_merged_dataframe(bounding_box)
            self.cached_data = df
            self.cached_data_bounding_box = bounding_box

        st.write('Cutting out data in selected area...')
        df = self.cut_out_bounding_box(df, bounding_box)

        return df


if __name__ == '__main__':
    ds = DataSource()
    image_files = ds.get_images_in_bounding_box(
        Polygon([
            [546207.0187136664, 5604318.09105179],
            [546274.1791940755, 5602101.795198287],
            [549826.9686077208, 5602041.350765919],
            [549645.635310616, 5604412.115724363]
        ]),
        "C:\\Users\\der_b\\Downloads\\cm_terrain_extraction",
    )

    ds.merge_image_files(image_files, "C:\\Users\\der_b\\Downloads\\cm_terrain_extraction")

