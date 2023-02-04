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
from typing import Dict, List, Optional

import geojson
import geopandas
import matplotlib.pyplot as plt
import networkx as nx
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
            "type_from_building_outline": [
                (0, "collect_building_outlines", "by_element"),
                (1, "process_building_outlines", "by_config_name")
                # (1, "assing_buildings_to_outlines", "by_config_name")
            ]
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
        file_name_base = self.grid_file.split('_')[0]
        import datetime
        dt = datetime.datetime.now()
        self.gdf = geopandas.GeoDataFrame.from_file(self.grid_file)
        self.sub_square_grid_diagonal_gdf = geopandas.GeoDataFrame.from_file(file_name_base + '_diagonal_grid.shp')
        self.sub_square_grid_gdf = geopandas.GeoDataFrame.from_file(file_name_base + '_sub_square_grid.shp')
        print((datetime.datetime.now() - dt).total_seconds())

        total_bounds = self.gdf.geometry.total_bounds
        # self.effective_bbox_polygon = Polygon([
        #     (total_bounds[0], total_bounds[1]),
        #     (total_bounds[2], total_bounds[1]),
        #     (total_bounds[2], total_bounds[3]),
        #     (total_bounds[0], total_bounds[3])
        # ])
        grid_polygons = MultiPolygon(self.gdf.geometry.values)
        self.effective_bbox_polygon = grid_polygons.buffer(0)

        p_origin = self.gdf.loc[(self.gdf.xidx == self.gdf.xidx.min()) & (self.gdf.yidx == self.gdf.yidx.min()), ['x', 'y']].values[0]
        p_xaxis_max = self.gdf.loc[(self.gdf.xidx == self.gdf.xidx.max()) & (self.gdf.yidx == self.gdf.yidx.min()), ['x', 'y']].values[0]
        p_yaxis_max = self.gdf.loc[(self.gdf.xidx == self.gdf.xidx.min()) & (self.gdf.yidx == self.gdf.yidx.max()), ['x', 'y']].values[0]

        idx_origin_point = Point(p_origin[0] - 4, p_origin[1] - 4)
        idx_point_x_axis_max = Point(p_xaxis_max[0] + 4, p_xaxis_max[1] - 4)
        idx_point_y_axis_max = Point(p_yaxis_max[0] - 4, p_yaxis_max[1] + 4)

        eff_bbox_points = [Point(*coords) for coords in self.effective_bbox_polygon.exterior.coords]

        origin_point = nearest_points(idx_origin_point, eff_bbox_points)
        point_x_axis_max = nearest_points(idx_point_x_axis_max, eff_bbox_points)
        point_y_axis_max = nearest_points(idx_point_y_axis_max, eff_bbox_points)


        self.idx_bbox = [0, 0, self.gdf.xidx.max(), self.gdf.yidx.max()]

    def _get_projected_geometry(self, geojson_geometry):
            # geometry = geopandas.GeoSeries(shape(geojson_geometry))
            # geometry = geometry.set_crs(epsg=4326)
            # geometry = geometry.to_crs(epsg=25832)
            # return geometry[0]

            geometry_object = shape(geojson_geometry)
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
        for element in tqdm(osm_data.features, 'Preprocessing OSM Data'):
            geometry = self._get_projected_geometry(element.geometry)

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
                    for item in tqdm(stages[priority][stage_idx][stage], 'Processing Priority {}, Stage {}, processor {}'.format(priority, stage_idx, stage)):
                        if type(item) == int:
                            osm_utils.processing.__getattribute__(stage)(self, config, self.matched_elements[item])
                        else:
                            osm_utils.processing.__getattribute__(stage)(self, config, item)

        # plt.axis('equal')
        # plt.show()
        

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

    args = argparser.parse_args()

    config = json.load(open(args.config_file, 'r'))

    # with codecs.open('scenarios/rhine_crossing/rhine_crossing_area1.osm', 'r', encoding='utf-8') as data:
    #     xml = data.read()

    # osm_data2 = osm2geojson.xml2geojson(xml, filter_used_refs=False, log_level='ERROR')
    # with open('scenarios/rhine_crossing/rhine_crossing_area1.geojson', 'w', encoding='utf8') as geojson_file:
    #     geojson_file.write(json.dumps(osm_data2))

    # osm_data = geojson.load(open('scenarios/agger_valley/schlingenbach/osm/schlingenbach.geojson', encoding='utf8'))
    osm_data = geojson.load(open(args.osm_input, encoding='utf8'))
    # osm_data = geojson.load(open('test/industrial_fences.geojson', encoding='utf8'))
    # osm_processor = OSMProcessor(config=config, bbox=[379877.0, 5643109.0, 381461.0, 5645022.0])
    # osm_processor = OSMProcessor(config=config, bbox=[383148.0, 5647828.0, 385543.0, 5649632.0])
    # osm_processor = OSMProcessor(config=config, bbox=[361607.305,5625049.525, 365540.291, 5627329.787])
    if args.grid_file is not None:
        osm_processor = OSMProcessor(config=config, grid_file='schlingenbach_grid.shp')
    else:
        osm_processor = OSMProcessor(config=config)

    # osm_processor = OSMProcessor(config=config, bbox_lon_lat=[7.2961798, 50.9429712, 7.3008123, 50.9447395])
    # osm_processor = OSMProcessor(config=config)
    osm_processor.preprocess_osm_data(osm_data=osm_data)
    osm_processor.run_processors()
    # osm_processor.write_to_file('scenarios/agger_valley/schlingenbach/schlingenbach_osm_civil.csv')
    osm_processor.write_to_file(args.output_file)
    # osm_processor.write_to_file('test/industrial_fences.csv')



