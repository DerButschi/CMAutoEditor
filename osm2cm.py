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

from typing import Dict, List, Optional
from unicodedata import category
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from OSMPythonTools.api import Api
from matplotlib.font_manager import json_load
import pyproj
import pandas
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, Point, MultiPoint, shape
from shapely.ops import split, snap
import matplotlib.pyplot as plt
# from skimage.draw import line, line_aa, line_nd, polygon
import geopandas
import re
import networkx as nx
from sklearn import neighbors
# from profile.general import road_tiles
import json
import geojson
from tqdm import tqdm
import osm_utils.processing
import logging



config = json.load(open('osm_buildings_only_config.json', 'r'))







                


# overpass = Overpass()
# api = Api()
# # query = overpassQueryBuilder(bbox=[7.30153, 50.93133, 7.30745, 50.93588], elementType='way')

# # bbox = [50.93133, 7.30153, 50.93588, 7.30745] # lat_min, lon_min, lat_max, lon_max
# projection = Proj(proj='utm', zone=32, ellps='WGS84')

# # bbox_utm = [379964.0, 5643796.0, 380804.0-8, 5644444.0-8] # overath
# bbox_utm = [379877.0, 5643109.0, 381461.0, 5645022.0] # overath extended
# # bbox_utm = [550894, 5586630, 553442, 5589362] # doellbach
# lon_min, lat_min = projection(bbox_utm[0], bbox_utm[1], inverse=True)
# lon_max, lat_max = projection(bbox_utm[2], bbox_utm[3], inverse=True)

# bbox = [lat_min, lon_min, lat_max, lon_max]


# query = overpassQueryBuilder(bbox=bbox, elementType=['way', 'relation'], includeGeometry=True, out='body')

# result = overpass.query(query)


# n_bins_x = np.floor((bbox_utm[2] - bbox_utm[0]) / 8).astype(int)
# n_bins_y = np.floor((bbox_utm[3] - bbox_utm[1]) / 8).astype(int)

# bins_x = np.linspace(bbox_utm[0], bbox_utm[0] + n_bins_x * 8, n_bins_x + 1)
# bins_y = np.linspace(bbox_utm[1], bbox_utm[1] + n_bins_y * 8, n_bins_y + 1)

# xarr = []
# yarr = []
# xiarr = []
# yiarr = []
# for xidx, x in enumerate(np.linspace(bbox_utm[0] + 4, bbox_utm[0] + 4 + (n_bins_x - 1) * 8, n_bins_x)):
#     for yidx, y in enumerate(np.linspace(bbox_utm[1] + 4, bbox_utm[1] + 4 + (n_bins_y - 1) * 8, n_bins_y)):
#         xarr.append(x)
#         yarr.append(y)
#         xiarr.append(xidx)
#         yiarr.append(yidx)

# geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, cap_style=3)
# # gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr, 'filled': [False] * len(xarr),
# #                               'pattern': [-1] * len(xarr), 'tile_page': [-1] * len(xarr), 'tile_row': [-1] * len(xarr), 'tile_col': [-1] * len(xarr), 
# #                               'z': [-1] * len(xarr), 'category': [-1] * len(xarr), 'type': [-1] * len(xarr), 'sub_type': [-1] * len(xarr)
# #                              }, geometry=geometry)

# gdf = geopandas.GeoDataFrame({
#     'x': xarr, 
#     'y': yarr, 
#     'xidx': xiarr, 
#     'yidx': yiarr, 
#     'z': [-1] * len(xarr),
#     'menu': [-1] * len(xarr),
#     'cat1': [-1] * len(xarr),
#     'cat2': [-1] * len(xarr),
#     'direction': [-1] * len(xarr),
#     'id': [-1] * len(xarr),
#     'name': [-1] * len(xarr),
#     'dist_along_way': [-1] * len(xarr),
#     }, geometry=geometry)



class OSMProcessor:
    def __init__(self, config: Dict, bbox: Optional[List[float]] = None, bbox_lon_lat: Optional[List[float]] = None):
        self.config = config
        self.bbox = bbox
        self.bbox_lon_lat = bbox_lon_lat
        self.idx_bbox = None
        self.effective_bbox_polygon = None
        self.transformer = None
        self.gdf = None
        self.df = None
        self.network_graphs = {}
        self.grids = {}
        self.building_outlines = None
        self.occupancy_gdf = geopandas.GeoDataFrame(columns=['geometry', 'priority', 'name'])

        self.matched_elements = []

        self.processing_stages = {
            "type_from_tag": [(0, "assign_type_from_tag", "by_element")],
            "type_random_area": [(0, "assign_type_randomly_in_area", "by_element")],
            "type_random_individual": [(0, "assign_type_randomly_for_each_square", "by_element")],
            "road_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (2, "create_square_graph", "by_config_name"), 
                (3, "assign_road_tiles_to_network", "by_config_name")
            ],
            "rail_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (2, "create_square_graph", "by_config_name"), 
                (3, "assign_rail_tiles_to_network", "by_config_name")
            ],
            "stream_tiles": [
                (0, "collect_network_data", "by_element"), 
                (1, "create_line_graph", "by_config_name"), 
                (2, "create_square_graph", "by_config_name"), 
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
        n_bins_x = np.floor((self.bbox[2] - self.bbox[0]) / 8).astype(int)
        n_bins_y = np.floor((self.bbox[3] - self.bbox[1]) / 8).astype(int)
        self.logger.info('Requested bounding box has {} x {} squares.'.format(n_bins_x, n_bins_y))

        n_bins_pad_x_min = np.ceil(max(0, self.bbox[0] - bounding_box_data[0]) / 8).astype(int)
        n_bins_pad_y_min = np.ceil(max(0, self.bbox[1] - bounding_box_data[1]) / 8).astype(int)
        n_bins_pad_x_max = np.ceil(max(0, bounding_box_data[2] - (self.bbox[0] + n_bins_x * 8)) / 8).astype(int)
        n_bins_pad_y_max = np.ceil(max(0, bounding_box_data[3] - (self.bbox[1] + n_bins_y * 8)) / 8).astype(int)

        self.logger.debug('Input data adds squares: {} W, {} E, {} S, {} N'.format(n_bins_pad_x_min, n_bins_pad_x_max, n_bins_pad_y_min, n_bins_pad_y_max))

        n_bins_x_data = n_bins_x + n_bins_pad_x_min + n_bins_pad_x_max
        n_bins_y_data = n_bins_y + n_bins_pad_y_min + n_bins_pad_y_max

        self.idx_bbox = [n_bins_pad_x_min, n_bins_pad_y_min, n_bins_pad_x_min + n_bins_x - 1, n_bins_pad_y_min + n_bins_y - 1]
        self.effective_bbox_polygon = Polygon([
            (self.bbox[0] - n_bins_pad_x_min * 8, self.bbox[1] - n_bins_pad_y_min * 8),
            (self.bbox[0] - n_bins_pad_x_min * 8 + n_bins_x_data * 8, self.bbox[1] - n_bins_pad_y_min * 8),
            (self.bbox[0] - n_bins_pad_x_min * 8 + n_bins_x_data * 8, self.bbox[1] - n_bins_pad_y_min * 8 + n_bins_y_data * 8),
            (self.bbox[0] - n_bins_pad_x_min * 8, self.bbox[1] - n_bins_pad_y_min * 8 + n_bins_y_data * 8),
        ])

        xarr = []
        yarr = []
        xiarr = []
        yiarr = []
        for xidx, x in enumerate(np.linspace(self.bbox[0] - n_bins_pad_x_min * 8 + 4, self.bbox[0] - n_bins_pad_x_min * 8 + 4 + (n_bins_x_data - 1) * 8, n_bins_x_data)):
            for yidx, y in enumerate(np.linspace(self.bbox[1] - n_bins_pad_y_min * 8 + 4, self.bbox[1] - n_bins_pad_y_min * 8 + 4 + (n_bins_y_data - 1) * 8, n_bins_y_data)):
                xarr.append(x)
                yarr.append(y)
                xiarr.append(xidx)
                yiarr.append(yidx)

        geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, cap_style=3)


        self.gdf = geopandas.GeoDataFrame({
            'x': xarr, 
            'y': yarr, 
            'xidx': xiarr, 
            'yidx': yiarr, 
            'z': [-1] * len(xarr),
            'menu': [-1] * len(xarr),
            'cat1': [-1] * len(xarr),
            'cat2': [-1] * len(xarr),
            'direction': [-1] * len(xarr),
            'id': [-1] * len(xarr),
            'name': [-1] * len(xarr),
            'priority': [-1] * len(xarr),
            }, geometry=geometry)

        octagon_geometry = geopandas.points_from_xy(xarr, yarr).buffer(4 / np.cos(22.5 / 180 * np.pi), cap_style=1, resolution=2)
        self.octagon_gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr}, geometry=octagon_geometry.rotate(22.5))
        circle_geometry1 = geopandas.points_from_xy(xarr, yarr).buffer(2 * np.sqrt(2), cap_style=1)
        self.circle_geometry1_gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr}, geometry=circle_geometry1)
        sub_square_grid_geometry1 = geopandas.points_from_xy(xarr, yarr)

        sub_square_grid_geometry1_gdf = geopandas.GeoDataFrame({
            'x': xarr, 
            'y': yarr, 
            'xidx': xiarr, 
            'yidx': yiarr, 
            'z': [-1] * len(xarr),
            'menu': [-1] * len(xarr),
            'cat1': [-1] * len(xarr),
            'cat2': [-1] * len(xarr),
            'direction': [-1] * len(xarr),
            'id': [-1] * len(xarr),
            'name': [-1] * len(xarr),
            'priority': [-1] * len(xarr),
            }, geometry=sub_square_grid_geometry1)


        xarr = []
        yarr = []
        xiarr = []
        yiarr = []
        for xidx, x in enumerate(np.linspace(self.bbox[0] - n_bins_pad_x_min * 8, self.bbox[0] - n_bins_pad_x_min * 8 + (n_bins_x_data) * 8, n_bins_x_data + 1)):
            for yidx, y in enumerate(np.linspace(self.bbox[1] - n_bins_pad_y_min * 8, self.bbox[1] - n_bins_pad_y_min * 8 + (n_bins_y_data) * 8, n_bins_y_data + 1)):
                xarr.append(x)
                yarr.append(y)
                xiarr.append(xidx-0.5)
                yiarr.append(yidx-0.5)

        filler_square_geometry = geopandas.points_from_xy(xarr, yarr).buffer(4 * np.tan(22.5 / 180 * np.pi), cap_style=3)
        filler_square_gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr}, geometry=filler_square_geometry.rotate(45))
        circle_geometry2 = geopandas.points_from_xy(xarr, yarr).buffer(2 * np.sqrt(2), cap_style=1)
        self.circle_geometry2_gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr}, geometry=circle_geometry2)
        sub_square_grid_geometry2 = geopandas.points_from_xy(xarr, yarr)
        sub_square_grid_geometry2_gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr}, geometry=sub_square_grid_geometry2)

        sub_square_grid_geometry2_gdf = geopandas.GeoDataFrame({
            'x': xarr, 
            'y': yarr, 
            'xidx': xiarr, 
            'yidx': yiarr, 
            'z': [-1] * len(xarr),
            'menu': [-1] * len(xarr),
            'cat1': [-1] * len(xarr),
            'cat2': [-1] * len(xarr),
            'direction': [-1] * len(xarr),
            'id': [-1] * len(xarr),
            'name': [-1] * len(xarr),
            'priority': [-1] * len(xarr),
            }, geometry=sub_square_grid_geometry2)

        # self.octagon_gdf = pandas.concat((self.octagon_gdf, filler_square_gdf), ignore_index=True)
        self.circle_gdf = pandas.concat((self.circle_geometry1_gdf, self.circle_geometry2_gdf), ignore_index=True)
        self.sub_square_grid_diagonal_gdf = pandas.concat((sub_square_grid_geometry1_gdf, sub_square_grid_geometry2_gdf), ignore_index=True)
        self.sub_square_grid_diagonal_gdf.geometry = self.sub_square_grid_diagonal_gdf.geometry.buffer(4, resolution=1)
        
        for xidx, x in enumerate(np.linspace(self.bbox[0] - n_bins_pad_x_min * 8, self.bbox[0] - n_bins_pad_x_min * 8 + (n_bins_x_data) * 8, n_bins_x_data * 2 + 1)):
            for yidx, y in enumerate(np.linspace(self.bbox[1] - n_bins_pad_y_min * 8, self.bbox[1] - n_bins_pad_y_min * 8 + (n_bins_y_data) * 8, n_bins_y_data * 2 + 1)):
                xarr.append(x)
                yarr.append(y)
                xiarr.append(xidx / 2 - 0.5)
                yiarr.append(yidx / 2 - 0.5)

        geometry = geopandas.points_from_xy(xarr, yarr).buffer(2, cap_style=3)

        self.sub_square_grid_gdf = geopandas.GeoDataFrame({
            'x': xarr, 
            'y': yarr, 
            'xidx': xiarr, 
            'yidx': yiarr, 
            'z': [-1] * len(xarr),
            'menu': [-1] * len(xarr),
            'cat1': [-1] * len(xarr),
            'cat2': [-1] * len(xarr),
            'direction': [-1] * len(xarr),
            'id': [-1] * len(xarr),
            'name': [-1] * len(xarr),
            'priority': [-1] * len(xarr),
            }, geometry=geometry)



    def _get_projected_geometry(self, geojson_geometry):
            geometry = geopandas.GeoSeries(shape(geojson_geometry))
            geometry = geometry.set_crs(epsg=4326)
            geometry = geometry.to_crs(epsg=25832)
            return geometry[0]



    def preprocess_osm_data(self, osm_data: Dict):
        self.transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:25832', always_xy=True)

        if self.bbox_lon_lat is not None and self.bbox is None:
            self.bbox = [
                *self.transformer.transform(*self.bbox_lon_lat[0:2]), 
                *self.transformer.transform(*self.bbox_lon_lat[2:4])
            ]

        xmin, ymin = self.transformer.transform(osm_data['bbox'][0], osm_data['bbox'][1])
        xmax, ymax = self.transformer.transform(osm_data['bbox'][2], osm_data['bbox'][3])

        self._init_grid([xmin, ymin, xmax, ymax])

        element_idx = 0
        for element in tqdm(osm_data.features, 'Preprocessing OSM Data'):
            geometry = self._get_projected_geometry(element.geometry)

            element_tags = element.properties

            for name in config:
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

                if excluded:
                    continue

                for tag_key, tag_value in config[name]['tags']:
                    if tag_key in element_tags and element_tags[tag_key] == tag_value:
                        matched = True

                if matched:
                    self.matched_elements.append({'element': element, 'geometry': geometry, 'name': name, 'idx': element_idx})
                    element_idx += 1

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





# df = pandas.DataFrame(columns=['x', 'y', 'z', 'menu', 'cat1', 'cat2', 'direction', 'id', 'name'])
# df = None
# generic_node_cnt = 0
# road_graphs = {}

# for element in result.elements():
#     if element.type() not in ('relation', 'way'):
#         continue
#     element_tags = element.tags()
#     if element.tags() is None:
#         print(element.tags(), element.id())
#         continue

#     matched = False
#     for name in config:
#         # if config['name']['pass'] != i_pass:
#         #     continue
#         excluded = False
#         if 'exclude_tags' in config[name]:
#             for element_tag_key in element_tags:
#                 if element_tag_key in config[name]['exclude_tags'] and config[name]['exclude_tags'][element_tag_key] == element_tags[element_tag_key]:
#                     excluded = True

#         if 'required_tags' in config[name]:
#             for required_tag_key in config[name]['required_tags']:
#                 if not (required_tag_key in element_tags and config[name]['required_tags'][required_tag_key] == element_tags[required_tag_key]):
#                     excluded = True

#         if excluded:
#             continue

#         for tag_key, tag_value in config[name]['tags']:
#             if tag_key in element_tags and element_tags[tag_key] == tag_value:
#                 to_fill = None
#                 matched = True
#                 element_geometry = element.geometry()
#                 if element_geometry['type'] == 'Polygon':
#                     exterior_coords = [(projection(coord[0], coord[1])) for coord in element_geometry['coordinates'][0]]
#                     interiors = []
#                     for interior_idx in range(1, len(element_geometry['coordinates'])):
#                         interior_coords = [(projection(coord[0], coord[1])) for coord in element_geometry['coordinates'][interior_idx]]
#                         interiors.append(interior_coords)

#                     polygon = Polygon(exterior_coords, holes=interiors)
#                     within = gdf.geometry.within(polygon)
#                     intersecting = gdf.geometry.intersects(polygon)
#                     is_border = np.bitwise_and(intersecting, ~within)
#                     is_largest_square_area = gdf.loc[is_border].geometry.intersection(polygon).area > 32

#                     to_fill = np.bitwise_or(within, is_largest_square_area)
#                 elif element_geometry['type'] == 'LineString':
#                     coords = [(projection(coord[0], coord[1])) for coord in element.geometry()['coordinates']]
#                     ls = LineString(coords)
#                     to_fill = np.bitwise_or(gdf.geometry.crosses(ls), gdf.geometry.contains(ls))
#                 elif element_geometry['type'] == 'MultiPolygon':
#                     polygons = []
#                     for polygon_idx in range(len(element_geometry['coordinates'])):
#                         polygon_coordinates = element_geometry['coordinates'][polygon_idx]
#                         exterior_coords = [(projection(coord[0], coord[1])) for coord in polygon_coordinates[0]]
#                         interiors = []
#                         for interior_idx in range(1, len(polygon_coordinates)):
#                             interior_coords = [(projection(coord[0], coord[1])) for coord in polygon_coordinates[interior_idx]]
#                             interiors.append(interior_coords)
                        
#                         polygons.append(Polygon(exterior_coords, holes=interiors))

#                     multipolygon = MultiPolygon(polygons)
#                     within = gdf.geometry.within(polygon)
#                     intersecting = gdf.geometry.intersects(polygon)
#                     is_border = np.bitwise_and(intersecting, ~within)
#                     is_largest_square_area = gdf.loc[is_border].geometry.intersection(polygon).area > 32

#                     to_fill = np.bitwise_or(within, is_largest_square_area)
#                 else:
#                     raise Exception('geometry {} of element {}/{} not yet covered'.format(element_geometry['type'], element.type(), element.id()))

#                 if to_fill is not None:
#                     element_df = gdf.loc[to_fill, ['xidx', 'yidx', 'z', 'menu', 'cat1', 'cat2', 'direction', 'id', 'name']].copy(deep=True)
#                     element_df['id'] = element.id()
#                     element_df['name'] = name

#                     cm_types = config[name]['cm_types']
#                     if 'process' in cm_types:
#                         for func_name in cm_types['process']:
#                             element_df = globals()[func_name](element_df, gdf, element, cm_types)

#                     if df is None:
#                         df = element_df
#                     else:
#                         df = pandas.concat([df, element_df], ignore_index=True)

#     if not matched:
#         print(element.tags(), element.id())

# # node_pos = {}
# # for node in road_graphs['road'].nodes:
# #     node_pos[node] = road_graphs['road'].nodes[node]['square']

# # plt.figure()
# # plt.axis('equal')
# # nx.draw_networkx(road_graphs['road'], pos=node_pos)
# # plt.show()

# for name in config:
#     cm_types = config[name]['cm_types']
#     if 'post_process' in cm_types:
#         for func_name in cm_types['post_process']:
#             globals()[func_name](df, gdf, name, cm_types)





# # plt.show()
# df = pandas.concat((
#     df, 
#     pandas.DataFrame({
#         'xidx': [gdf.xidx.max()], 
#         'yidx': [gdf.yidx.max()], 
#         'z': [-1], 
#         'menu': [-1], 
#         'cat1': [-1], 
#         'cat2': [-1], 
#         'direction': [-1], 
#         'id': [-1], 
#         'name': [-1]
#     })
# ))

# df_out = df.rename(columns={"xidx": "x", "yidx": "y"})
# df_out.to_csv('overath_extended_osm_roads.csv')


if __name__ == '__main__':
    osm_data = geojson.load(open('scenarios/loope/loope2.geojson', encoding='utf8'))
    # osm_data = geojson.load(open('test/fields.geojson', encoding='utf8'))
    # osm_processor = OSMProcessor(config=config, bbox=[379877.0, 5643109.0, 381461.0, 5645022.0])
    osm_processor = OSMProcessor(config=config, bbox=[383148.0, 5647828.0, 385543.0, 5649632.0])

    # osm_processor = OSMProcessor(config=config, bbox_lon_lat=[7.2961798, 50.9429712, 7.3008123, 50.9447395])
    # osm_processor = OSMProcessor(config=config)
    osm_processor.preprocess_osm_data(osm_data=osm_data)
    osm_processor.run_processors()
    osm_processor.write_to_file('scenarios/loope/loope2.csv')



