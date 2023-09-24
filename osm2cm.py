# Copyright (C) 2022  Nicolas Möser

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import geojson
import geopandas
import numpy as np
import pandas
import pyproj
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyproj import CRS

from matplotlib.font_manager import json_load
from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                              MultiPolygon, Point, Polygon, shape)
from shapely.ops import nearest_points, snap, split, transform
from tqdm import tqdm

import osm_utils.processing
from osm_utils.grid import get_all_grids, get_reference_rectanlge_points
from profiles import available_profiles
import PySimpleGUI as sg

import sys

def run_startup_gui():
    sg.theme('Dark')
    sg.theme_button_color('#002366')

    layout = [
        [sg.Titlebar('OSM to CM Converter')],
        [sg.Text('Profile: '), sg.Combo(values=list(available_profiles.keys()), default_value=list(available_profiles.keys())[0], 
                                        key='cm_profile')],
        [sg.Text('OSM Input File: '), sg.Input(key='input_file_path'), sg.FileBrowse(file_types=(('GeoJSON', '*.geojson'),))],
        [sg.Text('Output File Name: '), sg.Input('output.csv', key='output_file_name')],
        [sg.Text('OSM Configuration File: '), sg.Input('default_osm_config.json', key='osm_config_name'), sg.FileBrowse(file_types=(('JSON', '*.json'), ))],
        [sg.Column([[sg.Radio('Read grid/bounding box from grid-file', key='read_grid_from_grid_file', default=True, enable_events=True, group_id='bbox_input'),
                     sg.Radio('Enter bounding box', key='enter_bbox', default=False, enable_events=True, group_id='bbox_input'),
                     sg.Radio('Take bounding box from OSM-File', key='read_bbox_from_osm_file', default=False, enable_events=True, group_id='bbox_input'),
                     ]])],
        [sg.pin(
            sg.Column([
                [sg.Text('Grid File: '), sg.Input(key='grid_file_name'), sg.FileBrowse(file_types=(('ESRI Shape File', '*.shp'), ))]
            ], visible=True, key='grid_file_input')
        )],
        [sg.pin(sg.Column([[sg.Radio('axis parallel', group_id='bbox_type_selection', key='bbox_axis_parallel', enable_events=True, default=True), sg.Radio('freely rotatable', group_id='bbox_type_selection', key='bbox_freely_rotatable', enable_events=True)],
                          [sg.Text('Bounding box coordinates CRS: EPSG:'), sg.InputText('4326', enable_events=True, key='bbox_crs')]],
                          visible=False, key='enter_bbox_type'))],
        [sg.pin(sg.Column([
            [sg.Text('Select data within a rectangle where the sides are parallel to the x-axis.')],
            [sg.Text('The x-axis will be the W <-> E axis in CM.')],
            [sg.Column([
                [sg.Text('')],
                [sg.Text('lower left point')],
                [sg.Text('upper right point')]
            ]),
            sg.Column([
                [sg.Text('x')],
                [sg.InputText(key='bb_xmin', enable_events=True)],
                [sg.InputText(key='bb_xmax', enable_events=True)],
            ]),
            sg.Column([
                [sg.Text('y')],
                [sg.InputText(key='bb_ymin', enable_events=True)],
                [sg.InputText(key='bb_ymax', enable_events=True)],
            ]),
        ]], key='bbox_two_points', visible=False))],
        [sg.pin(sg.Column([
            [sg.Text('Select data within an arbitrary rectangle.')],
            [sg.Text('Point 1 will be the lower left corner in CM. From there enter the other corners of the rectangle in counter-clockwise order.')],
            [sg.Column([
                [sg.Text('')],
                [sg.Text('point 1')],
                [sg.Text('point 2')],
                [sg.Text('point 3')],
                [sg.Text('point 4')],
            ]),
            sg.Column([
                [sg.Text('x')],
                [sg.InputText(key='bb_point1_x', enable_events=True)],
                [sg.InputText(key='bb_point2_x', enable_events=True)],
                [sg.InputText(key='bb_point3_x', enable_events=True)],
                [sg.InputText(key='bb_point4_x', enable_events=True)],
            ]),
            sg.Column([
                [sg.Text('y')],
                [sg.InputText(key='bb_point1_y', enable_events=True)],
                [sg.InputText(key='bb_point2_y', enable_events=True)],
                [sg.InputText(key='bb_point3_y', enable_events=True)],
                [sg.InputText(key='bb_point4_y', enable_events=True)],
            ]),
        ]], key='bbox_four_points', visible=False))],

        [sg.Push(), sg.Submit('Start OSM to CM Converter', key='start'), sg.Exit(), sg.Push()]

    ]

    window = sg.Window('OSM Converter', layout)
    while True:
        event, values = window.read()

        if event == 'read_grid_from_grid_file':
            window['grid_file_input'].update(visible=True)
            window['enter_bbox_type'].update(visible=False)
            window['bbox_two_points'].update(visible=False)
            window['bbox_four_points'].update(visible=False)
        elif event == 'enter_bbox':
            window['grid_file_input'].update(visible=False)
            window['enter_bbox_type'].update(visible=True)
            window['bbox_two_points'].update(visible=True)
            window['bbox_four_points'].update(visible=False)
        elif event == 'read_bbox_from_osm_file':
            window['grid_file_input'].update(visible=False)
            window['enter_bbox_type'].update(visible=False)
            window['bbox_two_points'].update(visible=False)
            window['bbox_four_points'].update(visible=False)
        elif event == 'bbox_axis_parallel':
            window['bbox_two_points'].update(visible=True)
            window['bbox_four_points'].update(visible=False)
        elif event == 'bbox_freely_rotatable':
            window['bbox_two_points'].update(visible=False)
            window['bbox_four_points'].update(visible=True)
        elif event == sg.WIN_CLOSED or event == 'Exit':
            sys.exit(0)
        elif event == 'start':
            break

    outlist = ['-o', values['output_file_name'], '-c', values['osm_config_name'], '-i', values['input_file_path'], 
               '-p', available_profiles[values['cm_profile']]]
    if values['grid_file_name'] != '':
        outlist.extend(['-g', values['grid_file_name']])
    elif values['bb_xmin'] != '':
        outlist.extend(['-b', values['bbox_crs'], values['bb_xmin'], values['bb_ymin'], values['bb_xmax'], values['bb_ymax']])
    elif values['bb_point1_x'] != '':
        outlist.extend(['-b', values['bbox_crs'],
                        values['bb_point1_x'], values['bb_point1_y'], values['bb_point2_x'], values['bb_point2_y'],
                        values['bb_point3_x'], values['bb_point3_y'], values['bb_point4_x'], values['bb_point4_y']])
    
    window.close()
    return outlist


class OSMProcessor:
    def __init__(self, config: Dict, profile: str, bbox: Optional[List[float]] = None, bbox_lon_lat: Optional[List[float]] = None, grid_file: Optional[str] = None):
        self.config = config
        self.profile = profile
        self.bbox = bbox
        self.bbox_lon_lat = bbox_lon_lat
        self.grid_file = grid_file
        self.idx_bbox = None
        self.effective_bbox_polygon = None
        self.transformer = None
        self.gdf = None
        self.df = None
        self.network_graphs = {}
        self.grids = {}
        self.building_outlines = {}
        self.grid_graph = None
        self.occupancy_gdf = geopandas.GeoDataFrame(columns=['geometry', 'priority', 'name'])

        self.matched_elements = []

        self.processing_stages = {
            "type_from_tag": [(0, "assign_type_from_tag", "by_element")],
            "type_random_area": [(0, "assign_type_randomly_in_area", "by_element")],
            "type_random_individual": [(0, "assign_type_randomly_for_each_square", "by_element")],
            "type_random_clusters": [(0, "assign_type_in_random_clusters", "by_element")],
            "type_from_linear": [(6, "assign_type_at_linear_feature", "by_config_name")],
            "single_object_random": [(0, "single_object_random", "by_element")],
            "road_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (2, "create_square_graph_path_search", "by_config_name"), 
                (3, "assign_road_tiles_to_network", "by_config_name")
            ],
            "rail_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (2, "create_square_graph_path_search", "by_config_name"), 
                (3, "assign_rail_tiles_to_network", "by_config_name")
            ],
            "stream_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (2, "create_square_graph_path_search", "by_config_name"), 
                (3, "assign_stream_tiles_to_network", "by_config_name")
            ],
            "fence_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (4, "create_square_graph_path_search", "by_config_name"), 
                (5, "assign_fence_tiles_to_network", "by_config_name")
            ],
            "type_from_residential_building_outline": [
                (0, "collect_building_outlines", "by_element"),
                (1, "process_residential_building_outlines", "by_config_name")
            ],
            "type_from_church_outline": [
                (0, "collect_building_outlines", "by_element"),
                (1, "process_church_outlines", "by_config_name")
            ],
            "type_from_barn_outline": [
                (0, "collect_building_outlines", "by_element"),
                (1, "process_barn_outlines", "by_config_name")
            ],
        }

        self.logger = logging.getLogger('osm2cm')
        self.logger.setLevel(logging.DEBUG)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('[%(name)s] [%(levelname)s]: %(message)s'))
        stream_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('osm2cm.log', encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)
        self.logger.debug('Initialization complete.')

    def _init_grid(self, bbox_polygon: Polygon):
        # grid_gdf, diagonal_grid_gdf, sub_square_grid_gdf = get_all_grids(
        #     trf_lower_left[0], 
        #     trf_lower_left[1], 
        #     trf_lower_left[0] + height_map.shape[0], 
        #     trf_lower_left[1] + height_map.shape[1],
        #     int(height_map.shape[0] / 8),
        #     int(height_map.shape[1] / 8),
        #     rotation_angle=rotation_angle,
        #     rotation_center=trf_lower_left
        # )

        bbox_rectangle = bbox_polygon.minimum_rotated_rectangle
        p0, p1, p2 = get_reference_rectanlge_points(bbox_polygon, bbox_rectangle)

        n_bins_x = np.floor((p0.distance(p1)) / 8).astype(int)
        n_bins_y = np.floor((p1.distance(p2)) / 8).astype(int)

        xmin, ymin = p0.x, p0.y
        xmax = xmin + n_bins_x * 8
        ymax = ymin + n_bins_y * 8
        rotation_angle = np.arctan2(p1.y - p0.y, p1.x - p0.x) * 180.0 / np.pi
        
        grid_gdf, diagonal_grid_gdf, sub_square_grid_gdf = get_all_grids(xmin, ymin, xmax, ymax, n_bins_x, n_bins_y, 
                                                                         rotation_angle=rotation_angle, rotation_center=[p0.x, p0.y])
        self.gdf = grid_gdf
        self.sub_square_grid_diagonal_gdf = diagonal_grid_gdf
        self.sub_square_grid_gdf = sub_square_grid_gdf

        grid_polygons = MultiPolygon(self.gdf.geometry.values)
        self.effective_bbox_polygon = grid_polygons.buffer(0)

        self.idx_bbox = [0, 0, self.gdf.xidx.max(), self.gdf.yidx.max()]

    def _load_grid(self):
        file_name_base = str.join('_', self.grid_file.split('_')[:-1])
        self.gdf = geopandas.GeoDataFrame.from_file(self.grid_file)
        self.sub_square_grid_diagonal_gdf = geopandas.GeoDataFrame.from_file(file_name_base + '_diagonal_grid.shp')
        self.sub_square_grid_gdf = geopandas.GeoDataFrame.from_file(file_name_base + '_sub_square_grid.shp')

        grid_polygons = MultiPolygon(self.gdf.geometry.values)
        self.effective_bbox_polygon = grid_polygons.buffer(0)

        self.idx_bbox = [0, 0, self.gdf.xidx.max(), self.gdf.yidx.max()]

    def _get_epsg_code_from_bbox(self, bbox_crs, bbox):
        aoi = AreaOfInterest(*bbox.bounds)

        crs_list = query_utm_crs_info(area_of_interest=aoi)

        if not bbox_crs.is_projected:
            crs = None
            for crs_candidate in crs_list:
                if 'WGS 84 / UTM' in crs_candidate.name:
                    crs = crs_candidate
                
            if crs is None:
                crs = crs_list[-1]

            return crs.code
        else:
            return bbox_crs.to_epsg()

    def _get_geometry(self, geojson_geometry):
        try:
            return shape(geojson_geometry)
        except:
            return None
        


    def _get_projected_geometry(self, geojson_geometry):
            # geometry = geopandas.GeoSeries(shape(geojson_geometry))
            # geometry = geometry.set_crs(epsg=4326)
            # geometry = geometry.to_crs(epsg=25832)
            # return geometry[0]

            geometry_object = self._get_geometry(geojson_geometry)

            if geometry_object is not None:
                projected_geometry_object = transform(self.transformer.transform, geometry_object)

                return projected_geometry_object
            else:
                return None



    def preprocess_osm_data(self, osm_data: Dict):
        bbox_polygon = None
        if self.grid_file is not None:
            self._load_grid()
            epsg_code = self.gdf.crs.to_epsg()
            bbox_from_data = False
        elif self.bbox is not None:
            bbox_crs = CRS.from_epsg(self.bbox[0])
            bbox_polygon = Polygon([(self.bbox[i-1], self.bbox[i]) for i in range(2, len(self.bbox), 2)])
            epsg_code = self._get_epsg_code_from_bbox(bbox_crs=bbox_crs, bbox=bbox_polygon)
            bbox_from_data = False
        elif 'bbox' in osm_data and osm_data.bbox is not None:
            bbox_crs = CRS.from_epsg(4326)
            bbox_polygon = Polygon([(osm_data.bbox[0], osm_data.bbox[1]), 
                                    (osm_data.bbox[2], osm_data.bbox[1]),
                                    (osm_data.bbox[2], osm_data.bbox[3]),
                                    (osm_data.bbox[0], osm_data.bbox[3])])
            epsg_code = self._get_epsg_code_from_bbox(bbox_crs=bbox_crs, bbox=bbox_polygon)
            bbox_from_data = False
        else:
            epsg_code = None  # By default use WGS84/Transverse Mercator as in JOSM
            bbox_from_data = True
            xmin = np.inf
            xmax = -np.inf
            ymin = np.inf
            ymax = -np.inf

        self.transformer = None
        if epsg_code is not None:
            self.transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:{}'.format(epsg_code), always_xy=True)
            

        element_idx = 0
        unprocessed_tags = {'key': [], 'value': []}
        for element in tqdm(osm_data.features, 'Preprocessing OSM Data'):
            element_matched = False
            if bbox_from_data and epsg_code is None:
                raw_geometry = self._get_geometry(element.geometry)
                if raw_geometry is not None:
                    epsg_code = self._get_epsg_code_from_bbox(bbox_crs=CRS.from_epsg(4326), bbox=raw_geometry)
                    self.transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:{}'.format(epsg_code), always_xy=True)
                else:
                    continue

            geometry = self._get_projected_geometry(element.geometry)

            if geometry is None:
                continue

            if bbox_from_data:
                xmin = min(xmin, geometry.bounds[0])
                xmax = max(xmax, geometry.bounds[2])
                ymin = min(ymin, geometry.bounds[1])
                ymax = max(ymax, geometry.bounds[3])

            element_tags = {}
            if 'tags' in element.properties:
                element_tags = element.properties['tags']
            else:
                element_tags = element.properties
            element_id = None
            if 'id' in element.properties:
                element_id = element.properties["id"]

            for name in config:
                if 'active' in config[name] and not config[name]['active']:
                    continue
                matched = False
                excluded = False
                if 'exclude_tags' in config[name]:
                    for key, value in config[name]['exclude_tags']:
                        if key in element_tags.keys() and element_tags[key] == value:
                            excluded = True

                if 'required_tags' in config[name]:
                    for key, value in config[name]['required_tags']:
                        if not (key in element_tags.keys() and element_tags[key] == value):
                            excluded = True

                if 'exclude_ids' in config[name]:
                    exclude_ids = config[name]['exclude_ids']
                    if element_id in exclude_ids:
                        excluded = True

                if 'allowed_ids' in config[name]:
                    allowed_ids = config[name]['allowed_ids']
                    if not element_id in allowed_ids:
                        excluded = True

                if excluded:
                    continue

                for tag_key, tag_value in config[name]['tags']:
                    if tag_key in element_tags and element_tags[tag_key] == tag_value:
                        matched = True

                if matched:
                    self.matched_elements.append({'element': element, 'geometry': geometry, 'name': name, 'idx': element_idx})
                    element_idx += 1
                    element_matched = True

            if not element_matched:
                for key, value in element_tags.items():
                    unprocessed_tags['key'].append(key)
                    unprocessed_tags['value'].append(value)

        unprocessed_tags_df = pandas.DataFrame(unprocessed_tags)
        unprocessed_tags_df = unprocessed_tags_df.drop_duplicates()
        unprocessed_tags_df.to_csv('unprocessed_tags.csv')

        if self.grid_file is None:
            if bbox_from_data:
                bbox_polygon = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            else:
                bbox_polygon = transform(self.transformer.transform, bbox_polygon)
            
            self._init_grid(bbox_polygon)

        # add default entries
        if "default_ground" in config:
            self.matched_elements.append({'element': None, 'geometry': self.effective_bbox_polygon, 
                                          'name': 'default_ground', 'idx': element_idx + 1})
        if "default_foliage" in config:
            self.matched_elements.append({'element': None, 'geometry': self.effective_bbox_polygon, 
                                          'name': 'default_foliage', 'idx': element_idx + 2})


    def _collect_stages(self):
        stages = {}
        for entry_idx, entry in enumerate(self.matched_elements):
            name = entry['name']
            priority = config[name]['priority']
            if priority not in stages:
                stages[priority] = {}
            for process in config[name]['process']:
                for stage_idx, stage, processing_type in self.processing_stages[process]:
                    if stage_idx not in stages[priority]:
                        stages[priority][stage_idx] = {}
                    if stage not in stages[priority][stage_idx]:
                        stages[priority][stage_idx][stage] = []
                    if processing_type == "by_element":
                        stages[priority][stage_idx][stage].append(entry_idx)
                    elif processing_type == "by_config_name" and name not in stages[priority][stage_idx][stage]:
                        stages[priority][stage_idx][stage].append(name)
                    

        return stages

    def _get_sub_df(self, idx, gdf=None):
        if gdf is None:
            return self.gdf.loc[idx, ['xidx', 'yidx', 'z', 'menu', 'cat1', 'cat2', 'direction', 'id', 'name', 'priority']].copy(deep=True)
        else:
            return gdf.loc[idx, ['xidx', 'yidx', 'z', 'menu', 'cat1', 'cat2', 'direction', 'id', 'name', 'priority']].copy(deep=True)

    def _append_to_df(self, sub_df: pandas.DataFrame):
        if self.df is None:
            self.df = sub_df
        else:
            self.df = pandas.concat((self.df, sub_df), ignore_index=True)

    def run_processors(self):
        stages = self._collect_stages()

        # plt.figure()
        for priority in sorted(stages.keys()):
            for stage_idx in sorted(stages[priority].keys()):
                for stage in stages[priority][stage_idx]:
                    if type(stages[priority][stage_idx][stage][0]) == int:
                        for item in tqdm(stages[priority][stage_idx][stage], 'Processing Priority {}, Stage {}, processor {}'.format(priority, stage_idx, stage)):
                            osm_utils.processing.__getattribute__(stage)(self, config, self.matched_elements[item])
                    else:
                        for item in stages[priority][stage_idx][stage]:
                            osm_utils.processing.__getattribute__(stage)(self, config, item, tqdm_string='Processing Priority {}, Stage {}, processor {}'.format(priority, stage_idx, stage))

                            

        # plt.axis('equal')
        # plt.show()

    def post_process(self):
        # remove invalid entries
        self.df = self.df.drop_duplicates()
        self.df = self.df.drop(self.df[(self.df.menu == -1) & (self.df.z == -1)].index)

        duplicate_indices = self.df[self.df.duplicated(subset=['xidx', 'yidx'])].index
        indices_to_drop = []
        for idx in tqdm(duplicate_indices, 'Postprocessing OSM Data'):
            xidx = self.df.loc[idx].xidx
            yidx = self.df.loc[idx].yidx

            # min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx)].priority.max()
            contains_default = len(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == -999)]) > 0

            if contains_default:
                min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > -999)].priority.min()
            else:
                min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > 0)].priority.min()
            # if min_valid_priority < 1:
            #     continue
            if np.isnan(min_valid_priority):
                continue

            if contains_default:
                indices_to_drop.extend(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == -999)].index)
            else:
                indices_to_drop.extend(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > min_valid_priority)].index)
                indices_to_drop.extend(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority < 0)].index)

            min_priority_names = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == min_valid_priority)].name.unique()

            for name in min_priority_names:
                if len(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == min_valid_priority) & (self.df.name == name)]) > 1:
                    indices = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == min_valid_priority) & (self.df.name == name)].index
                    cm_types = self.config[name]['cm_types']
                    indices_with_cm_rank = []
                    for sub_idx in indices:
                        menu = self.df.loc[sub_idx].menu
                        cat1 = self.df.loc[sub_idx].cat1
                        cat2 = self.df.loc[sub_idx].cat2

                        for cidx, cm_type in enumerate(cm_types):
                            if cm_type['menu'] == menu and cm_type['cat1'] == cat1 and (cat2 not in cm_type or ('cat2' in cm_type and cm_type['cat2'] == cat2)):
                                indices_with_cm_rank.append((sub_idx, cidx))
                                break
                    indices_with_rank = [sub_idx[0] for sub_idx in sorted(indices_with_cm_rank, key=lambda x: x[1])[1:]]
                    indices_to_drop.extend(indices_with_rank)
                else:
                    pass
        
        if len(indices_to_drop) > 0:
            self.df = self.df.drop(indices_to_drop)







        

    def write_to_file(self, output_file_name):
        xmax = self.idx_bbox[2]
        ymax = self.idx_bbox[3]
        sub_df = self._get_sub_df((self.gdf.xidx == xmax) & (self.gdf.yidx == ymax))
        sub_df.x = xmax
        sub_df.y = ymax
        self.df = pandas.concat((self.df, sub_df), ignore_index=True)
        self.df = self.df.rename(columns={"xidx": "x", "yidx": "y"})
        self.df = self.df.loc[
            (self.df.x.between(self.idx_bbox[0], self.idx_bbox[2])) &
            (self.df.y.between(self.idx_bbox[1], self.idx_bbox[3]))
        ]
        self.df.x = self.df.x - self.idx_bbox[0]
        self.df.y = self.df.y - self.idx_bbox[1]
        self.df.to_csv(output_file_name)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--osm-input', required=True)
    argparser.add_argument('-g', '--grid-file', required=False)
    argparser.add_argument('-b', '--bounding-box', required=False, help='Coordinates of box in which to extract data. 2 or 4 points. If an additional number is provided, the first number is interpreted as epsg-code.'
        'Otherwise 4326 (longitude/latitude) is assumend.', type=float, nargs='+')
    argparser.add_argument('-c', '--config-file', required=False, default='default_osm_config.json')
    argparser.add_argument('-o', '--output-file', required=True)
    argparser.add_argument('-p', '--profile', type=str, required=False, default='cold_war')

    if len(sys.argv) == 1:
        argv_list = run_startup_gui()
        args = argparser.parse_args(argv_list)
    else:
        args = argparser.parse_args()

    os.makedirs('debug', exist_ok=True)

    config = json.load(open(args.config_file, 'r'))

    osm_data = geojson.load(open(args.osm_input, encoding='utf8'))
    if args.grid_file is not None:
        osm_processor = OSMProcessor(config=config, grid_file=args.grid_file, profile=args.profile)
    elif args.bounding_box is not None:
        osm_processor = OSMProcessor(config=config, bbox=args.bounding_box, profile=args.profile)
    else:

        osm_processor = OSMProcessor(config=config, profile=args.profile)

    osm_processor.preprocess_osm_data(osm_data=osm_data)
    osm_processor.run_processors()
    osm_processor.post_process()
    osm_processor.write_to_file(args.output_file)

    sg.popup('OSM conversion complete.')

