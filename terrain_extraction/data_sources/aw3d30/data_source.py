from geopandas import GeoDataFrame
from shapely import MultiPolygon, union_all, Polygon
import streamlit as st
from typing import List
import os
import zipfile
from pyproj.crs import CRS
import pandas

from terrain_extraction.data_source_utils import GeoTiffDataSource
from terrain_extraction.bbox_utils import BoundingBox
from ftplib import FTP

class FTPProgress:
    def __init__(self, file_name, total_size):
        self.file_name = file_name
        self.total_size = total_size
        self.size_downloaded = 0
        self.progressbar = st.progress(0, "Downloading missing file {}...".format(file_name))
        self.file_handle = None

    def set_file_handle(self, file_handle):
        self.file_handle = file_handle

    def update_progress(self, chunk):
        self.file_handle.write(chunk)
        self.size_downloaded += len(chunk)
        progress = min(int(self.size_downloaded / self.total_size * 100), 100)
        self.progressbar.progress(progress, text='Downloading missing file {}... {} of {} MB'.format(self.file_name, int(self.size_downloaded / 1024 / 1024), int(self.total_size / 1024 / 1024)))


class AW3D30DataSource(GeoTiffDataSource):
    def __init__(self):
        self.name = 'ALOS World 3D'
        self.data_type = 'geotiff'
        self.model_type = 'DSM'
        self.resolution = '1 arcsec (~30 m)'
        self.crs = CRS.from_epsg(4326)
        self.gdf = None
        self.outline: MultiPolygon = None
        self.data_folder = 'aw3d30'
        self.current_merged_image_path = None
        self.cached_data: pandas.DataFrame = None
        self.cached_data_bounding_box: BoundingBox = None
        self.envelope = Polygon([(-180.0, -84.0), (180.0, -84.0), (180.0, 84.0), (-180.0, 84.0), (-180.0, -84.0)])
        self.gdf_geojson_path = 'terrain_extraction/data_sources/aw3d30/aw3d30.geojson'

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
            relative_location = os.path.join(self.data_folder, entry['folder'], entry['file_name'])
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
                            if not image_name.endswith('_DSM.tif'):
                                continue
                            image_files.append(os.path.join(dir_path, os.path.splitext(file_name)[0], image_name))
                            if not os.path.isfile(os.path.join(dir_path, os.path.splitext(file_name)[0], image_name)):
                                zip_ref.extract(image_name, os.path.join(dir_path, os.path.splitext(file_name)[0]))

        return image_files
    
    def download_overlapping_data(self, missing_files: List[str], out_dir: str):
        ftp = FTP('ftp.eorc.jaxa.jp')
        ftp.login()
        ftp.cwd('pub/ALOS/ext1/AW3D30/release_v2303/')

        for dir_name, file_name in missing_files:
            ftp.cwd(dir_name)
            ftp.sendcmd('TYPE i')
            total_size = ftp.size(file_name)
            ftp.sendcmd('TYPE A')

            os.makedirs(os.path.join(out_dir, self.data_folder, dir_name), exist_ok=True)
            with open(os.path.join(out_dir, self.data_folder, dir_name, file_name), 'wb') as alos_zip_handle:
                ftp_progress = FTPProgress(file_name, total_size)
                ftp_progress.set_file_handle(alos_zip_handle)
                ftp.retrbinary('RETR {}'.format(file_name), callback=ftp_progress.update_progress)

            ftp.cwd('..')
        ftp.close()

    def get_data(self, bounding_box: BoundingBox, cache_dir: str):
        if self.cached_data is not None and self.cached_data_bounding_box.equals(bounding_box):
            df = self.cached_data
        else:
            st.write("Checking data cache...")
            missing_files = self.get_missing_files(bounding_box, cache_dir)
            self.download_overlapping_data(missing_files, cache_dir)
            st.write("Unpacking geotiffs intersecting with bound box...")
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
