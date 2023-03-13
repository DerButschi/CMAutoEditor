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
from matplotlib.font_manager import json_load
from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                              MultiPolygon, Point, Polygon, shape)
from shapely.ops import nearest_points, snap, split, transform
from tqdm import tqdm

import osm_utils.processing
from osm_utils.grid import get_all_grids

import PySimpleGUI as sg

import sys

def run_startup_gui():
    sg.theme('Dark')
    sg.theme_button_color('#002366')

    layout = [
        [sg.Titlebar('OSM to CM Converter')],
        [sg.Text('OSM Input File: '), sg.Input(key='input_file_path'), sg.FileBrowse(file_types=(('GeoJSON', '*.geojson'),))],
        [sg.Text('Output File Name: '), sg.Input('output.csv', key='output_file_name')],
        [sg.Text('OSM Configuration File: '), sg.Input('default_osm_config.json', key='osm_config_name'), sg.FileBrowse(file_types=(('JSON', '*.json'), ))],
        [sg.Checkbox('Read grid from file', key='read_grid_from_file', default=False, enable_events=True)],
        [sg.pin(
            sg.Column([
                [sg.Text('Grid File: '), sg.Input(key='grid_file_name'), sg.FileBrowse(file_types=(('ESRI Shape File', '*.shp'), ))]
            ], visible=False, key='grid_file_input')
        )],
        [sg.Push(), sg.Submit('Start OSM to CM Converter', key='start'), sg.Exit(), sg.Push()]

    ]

    window = sg.Window('OSM Converter', layout)
    while True:
        event, values = window.read()

        if event == 'read_grid_from_file':
            window['grid_file_input'].update(visible=values['read_grid_from_file'])
        elif event == 'start':
            break

    if values['grid_file_name'] == '':
        outlist = ['-i', values['input_file_path'], '-o', values['output_file_name'], '-c', values['osm_config_name']]
    else:
        outlist = ['-i', values['input_file_path'], '-o', values['output_file_name'], '-c', values['osm_config_name'], '-g', values['grid_file_name']]
    
    window.close()
    return outlist


class OSMProcessor:
    def __init__(self, config: Dict, bbox: Optional[List[float]] = None, bbox_lon_lat: Optional[List[float]] = None, grid_file: Optional[str] = None):
        self.config = config
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
            "type_from_linear": [(4, "assign_type_at_linear_feature", "by_config_name")],
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

    def _init_grid(self, bounding_box_data):
        if self.bbox is None:
            self.bbox = bounding_box_data

        xmin, ymin, xmax, ymax = self.bbox
        n_bins_x = np.ceil((xmax - xmin) / 8).astype(int)
        n_bins_y = np.ceil((ymax - ymin) / 8).astype(int)
        xmax = xmin + n_bins_x * 8
        ymax = ymin + n_bins_y * 8
        
        grid_gdf, diagonal_grid_gdf, sub_square_grid_gdf = get_all_grids(xmin, ymin, xmax, ymax, n_bins_x, n_bins_y)
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

    def _get_projected_geometry(self, geojson_geometry):
            # geometry = geopandas.GeoSeries(shape(geojson_geometry))
            # geometry = geometry.set_crs(epsg=4326)
            # geometry = geometry.to_crs(epsg=25832)
            # return geometry[0]

            try:
                geometry_object = shape(geojson_geometry)
            except:
                return None

            projected_geometry_object = transform(self.transformer.transform, geometry_object)

            return projected_geometry_object



    def preprocess_osm_data(self, osm_data: Dict):
        self.transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:25832', always_xy=True)

        if self.bbox_lon_lat is not None and self.bbox is None:
            self.bbox = [
                *self.transformer.transform(*self.bbox_lon_lat[0:2]), 
                *self.transformer.transform(*self.bbox_lon_lat[2:4])
            ]

        bbox_from_data = False
        if 'bbox' in osm_data:
            xmin, ymin = self.transformer.transform(osm_data['bbox'][0], osm_data['bbox'][1])
            xmax, ymax = self.transformer.transform(osm_data['bbox'][2], osm_data['bbox'][3])
        elif self.bbox is not None:
            x_dist = self.bbox[2] - self.bbox[0]
            y_dist = self.bbox[3] - self.bbox[1]
            xmin = self.bbox[0] - x_dist * 0.05
            xmax = self.bbox[2] + x_dist * 0.05
            ymin = self.bbox[1] - y_dist * 0.05
            ymax = self.bbox[3] + y_dist * 0.05
        else:
            bbox_from_data = True
            xmin = np.inf
            xmax = -np.inf
            ymin = np.inf
            ymax = -np.inf


        element_idx = 0
        unprocessed_tags = {'key': [], 'value': []}
        for element in tqdm(osm_data.features, 'Preprocessing OSM Data'):
            element_matched = False
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
            self._init_grid([xmin, ymin, xmax, ymax])
        else:
            self._load_grid()

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
        for idx in duplicate_indices:
            xidx = self.df.loc[idx].xidx
            yidx = self.df.loc[idx].yidx

            # min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx)].priority.max()
            min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > 0)].priority.min()
            # if min_valid_priority < 1:
            #     continue
            if np.isnan(min_valid_priority):
                continue
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
    argparser.add_argument('-c', '--config-file', required=False, default='default_osm_config.json')
    argparser.add_argument('-o', '--output-file', required=True)

    if len(sys.argv) == 1:
        argv_list = run_startup_gui()
        args = argparser.parse_args(argv_list)
    else:
        args = argparser.parse_args()

    os.makedirs('debug', exist_ok=True)

    config = json.load(open(args.config_file, 'r'))

    osm_data = geojson.load(open(args.osm_input, encoding='utf8'))
    if args.grid_file is not None:
        osm_processor = OSMProcessor(config=config, grid_file=args.grid_file)
    else:
        osm_processor = OSMProcessor(config=config)

    osm_processor.preprocess_osm_data(osm_data=osm_data)
    osm_processor.run_processors()
    osm_processor.post_process()
    osm_processor.write_to_file(args.output_file)

    sg.popup('OSM conversion complete.')

