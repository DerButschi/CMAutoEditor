from typing import Dict

import pyproj
from terrain_extraction.bbox_utils import BoundingBox
from terrain_extraction.osm_utils.grid import get_all_grids
import terrain_extraction.osm_utils.processing
import geopandas
import numpy as np
from shapely import MultiPolygon, Polygon, transform, union_all, affinity
from shapely.geometry import shape
import streamlit as st
import json
import pandas
from pyproj.crs import CRS
from profiles import get_building_tiles, get_building_cat2, process_to_building_type

class OSMProcessor:
    def __init__(self, profile: str, bbox: BoundingBox, path_to_config: str = "default_osm_config.json"):
        self.path_to_congih = path_to_config
        self.config = json.load(open(path_to_config, 'r'))
        self.profile = profile
        self.bbox = bbox
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

        # self.logger = logging.getLogger('osm2cm')
        # self.logger.setLevel(logging.DEBUG)
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(logging.Formatter('[%(name)s] [%(levelname)s]: %(message)s'))
        # stream_handler.setLevel(logging.DEBUG)
        # file_handler = logging.FileHandler('osm2cm.log', encoding='utf-8', mode='w')
        # file_handler.setLevel(logging.DEBUG)
        # self.logger.addHandler(stream_handler)
        # self.logger.addHandler(file_handler)
        # self.logger.debug('Initialization complete.')

    def _init_grid(self, bbox: BoundingBox):
        p0, p1, p2 = bbox.get_reference_points(bbox.crs_projected)

        n_bins_x = np.floor((p0.distance(p1)) / 8).astype(int)
        n_bins_y = np.floor((p1.distance(p2)) / 8).astype(int)

        xmin, ymin = p0.x, p0.y
        xmax = xmin + n_bins_x * 8
        ymax = ymin + n_bins_y * 8
        rotation_angle = bbox.get_rotation_angle()
        
        grid_gdf, diagonal_grid_gdf, sub_square_grid_gdf = get_all_grids(xmin, ymin, xmax, ymax, n_bins_x, n_bins_y, 
                                                                         rotation_angle=rotation_angle, rotation_center=[p0.x, p0.y])
        self.gdf = grid_gdf
        self.sub_square_grid_diagonal_gdf = diagonal_grid_gdf
        self.sub_square_grid_gdf = sub_square_grid_gdf

        self.gdf = self.gdf.set_crs(epsg=bbox.crs_projected.to_epsg())
        self.sub_square_grid_gdf = self.sub_square_grid_gdf.set_crs(epsg=bbox.crs_projected.to_epsg())
        self.sub_square_grid_diagonal_gdf = self.sub_square_grid_diagonal_gdf.set_crs(epsg=bbox.crs_projected.to_epsg())

        grid_polygons = MultiPolygon(self.gdf.geometry.values)
        self.effective_bbox_polygon = grid_polygons.buffer(0)

        self.idx_bbox = [0, 0, self.gdf.xidx.max(), self.gdf.yidx.max()]

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
            def _trf(coords):
                return np.array([self.transformer.transform(*coord) for coord in coords])

            if geometry_object is not None:
                projected_geometry_object = transform(geometry_object, _trf)

                return projected_geometry_object
            else:
                return None

    def preprocess_osm_data(self, osm_data: Dict):
        bbox_crs = self.bbox.crs_projected
        bbox_polygon = self.bbox.box_utm

        self.transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:{}'.format(bbox_crs.to_epsg()), always_xy=True)

        element_idx = 0
        unprocessed_tags = {'key': [], 'value': []}
        progress_bar = st.progress(0.0, 'Preprocessing OSM Data')
        for eidx, element in enumerate(osm_data.features):
            element_matched = False

            geometry = self._get_projected_geometry(element.geometry)

            if geometry is None:
                continue

            element_tags = {}
            if 'tags' in element.properties:
                element_tags = element.properties['tags']
            else:
                element_tags = element.properties
            element_id = None
            if 'id' in element.properties:
                element_id = element.properties["id"]

            for name in self.config:
                if 'active' in self.config[name] and not self.config[name]['active']:
                    continue
                matched = False
                excluded = False
                if 'exclude_tags' in self.config[name]:
                    for key, value in self.config[name]['exclude_tags']:
                        if key in element_tags.keys() and element_tags[key] == value:
                            excluded = True

                if 'required_tags' in self.config[name]:
                    for key, value in self.config[name]['required_tags']:
                        if not (key in element_tags.keys() and element_tags[key] == value):
                            excluded = True

                if 'exclude_ids' in self.config[name]:
                    exclude_ids = self.config[name]['exclude_ids']
                    if element_id in exclude_ids:
                        excluded = True

                if 'allowed_ids' in self.config[name]:
                    allowed_ids = self.config[name]['allowed_ids']
                    if not element_id in allowed_ids:
                        excluded = True

                if excluded:
                    continue

                for tag_key, tag_value in self.config[name]['tags']:
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

            progress_bar.progress(eidx / len(osm_data['features']), 'Preprocessing OSM Data')

        # unprocessed_tags_df = pandas.DataFrame(unprocessed_tags)
        # unprocessed_tags_df = unprocessed_tags_df.drop_duplicates()
        # unprocessed_tags_df.to_csv('unprocessed_tags.csv')
            
        self._init_grid(self.bbox)

        # add default entries
        if "default_ground" in self.config and (True if not 'active' in self.config['default_ground'] else self.config['default_ground']['active']):
            self.matched_elements.append({'element': None, 'geometry': self.effective_bbox_polygon, 
                                          'name': 'default_ground', 'idx': element_idx + 1})
        if "default_foliage" in self.config and (True if not 'active' in self.config['default_foliage'] else self.config['default_foliage']['active']):
            self.matched_elements.append({'element': None, 'geometry': self.effective_bbox_polygon, 
                                          'name': 'default_foliage', 'idx': element_idx + 2})


    def _collect_stages(self):
        stages = {}
        for entry_idx, entry in enumerate(self.matched_elements):
            name = entry['name']
            priority = self.config[name]['priority']
            if priority not in stages:
                stages[priority] = {}
            for process in self.config[name]['process']:
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
                        for item in stages[priority][stage_idx][stage]:
                            terrain_extraction.osm_utils.processing.__getattribute__(stage)(self, self.config, self.matched_elements[item])
                    else:
                        for item in stages[priority][stage_idx][stage]:
                            terrain_extraction.osm_utils.processing.__getattribute__(stage)(self, self.config, item, tqdm_string='Processing Priority {}, Stage {}, processor {}'.format(priority, stage_idx, stage))

                            

        # plt.axis('equal')
        # plt.show()

    def post_process(self):
        # remove invalid entries
        self.df = self.df.drop_duplicates()
        self.df = self.df.drop(self.df[(self.df.menu == -1) & (self.df.z == -1)].index)

        # duplicate_indices = self.df[self.df.duplicated(subset=['xidx', 'yidx'])].index
        indices_to_drop = []

        for _, group in self.df.groupby(by=['xidx', 'yidx']):
            if len(group) < 2:
                continue
            indices_to_drop_in_group = []
            indices_to_drop_in_group.extend(group[group.priority == -999].index)
            min_priority = group[group.priority > 0].priority.min()
            if min_priority > 0:
                indices_to_drop_in_group.extend(group[group.priority > min_priority].index)
                indices_to_drop_in_group.extend(group[(group.priority < 0) & (group.priority > -999)].index)

            surviving_group = group[~group.index.isin(indices_to_drop_in_group)]
            if len(surviving_group) > 1:
                for _, prio_group in surviving_group.groupby(by=['priority', 'name']):
                    if len(prio_group) == 1:
                        continue
                    priority_names = prio_group.name.unique()

                    for name in priority_names:
                        indices = prio_group[(prio_group.name == name)].index
                        cm_types = self.config[name]['cm_types']
                        indices_with_cm_rank = []
                        for sub_idx in indices:
                            menu = prio_group.loc[sub_idx].menu
                            cat1 = prio_group.loc[sub_idx].cat1
                            cat2 = prio_group.loc[sub_idx].cat2

                            for cidx, cm_type in enumerate(cm_types):
                                if cm_type['menu'] == menu and cm_type['cat1'] == cat1 and (cat2 not in cm_type or ('cat2' in cm_type and cm_type['cat2'] == cat2)):
                                    indices_with_cm_rank.append((sub_idx, cidx))
                                    break
                        indices_with_rank = [sub_idx[0] for sub_idx in sorted(indices_with_cm_rank, key=lambda x: x[1])[1:]]
                        indices_to_drop_in_group.extend(indices_with_rank)

            indices_to_drop.extend(indices_to_drop_in_group)

        # for idx in tqdm(duplicate_indices, 'Postprocessing OSM Data'):
        #     xidx = self.df.loc[idx].xidx
        #     yidx = self.df.loc[idx].yidx

        #     # min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx)].priority.max()
        #     contains_default = len(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == -999)]) > 0

        #     if contains_default:
        #         min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > -999)].priority.min()
        #     else:
        #         min_valid_priority = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > 0)].priority.min()
        #     # if min_valid_priority < 1:
        #     #     continue
        #     if np.isnan(min_valid_priority):
        #         continue

        #     if contains_default:
        #         indices_to_drop.extend(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == -999)].index)
        #     else:
        #         indices_to_drop.extend(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority > min_valid_priority)].index)
        #         indices_to_drop.extend(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority < 0)].index)

        #     min_priority_names = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == min_valid_priority)].name.unique()

        #     for name in min_priority_names:
        #         if len(self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == min_valid_priority) & (self.df.name == name)]) > 1:
        #             indices = self.df[(self.df.xidx == xidx) & (self.df.yidx == yidx) & (self.df.priority == min_valid_priority) & (self.df.name == name)].index
        #             cm_types = self.config[name]['cm_types']
        #             indices_with_cm_rank = []
        #             for sub_idx in indices:
        #                 menu = self.df.loc[sub_idx].menu
        #                 cat1 = self.df.loc[sub_idx].cat1
        #                 cat2 = self.df.loc[sub_idx].cat2

        #                 for cidx, cm_type in enumerate(cm_types):
        #                     if cm_type['menu'] == menu and cm_type['cat1'] == cat1 and (cat2 not in cm_type or ('cat2' in cm_type and cm_type['cat2'] == cat2)):
        #                         indices_with_cm_rank.append((sub_idx, cidx))
        #                         break
        #             indices_with_rank = [sub_idx[0] for sub_idx in sorted(indices_with_cm_rank, key=lambda x: x[1])[1:]]
        #             indices_to_drop.extend(indices_with_rank)
        #         else:
        #             pass
        
        if len(indices_to_drop) > 0:
            self.df = self.df.drop(indices_to_drop)

    def write_to_file(self, output_file_name):
        xmax = self.idx_bbox[2]
        ymax = self.idx_bbox[3]
        sub_df = self._get_sub_df((self.gdf.xidx == xmax) & (self.gdf.yidx == ymax))
        sub_df.x = xmax
        sub_df.y = ymax
        out_df = pandas.concat((self.df, sub_df), ignore_index=True)
        out_df = out_df.rename(columns={"xidx": "x", "yidx": "y"})
        out_df = out_df.loc[
            (out_df.x.between(self.idx_bbox[0], self.idx_bbox[2])) &
            (out_df.y.between(self.idx_bbox[1], self.idx_bbox[3]))
        ]
        out_df.x = out_df.x - self.idx_bbox[0]
        out_df.y = out_df.y - self.idx_bbox[1]
        out_df.to_csv(output_file_name)

    def get_output(self):
        xmax = self.idx_bbox[2]
        ymax = self.idx_bbox[3]
        sub_df = self._get_sub_df((self.gdf.xidx == xmax) & (self.gdf.yidx == ymax))
        sub_df.x = xmax
        sub_df.y = ymax
        out_df = pandas.concat((self.df, sub_df), ignore_index=True)
        out_df = out_df.rename(columns={"xidx": "x", "yidx": "y"})
        out_df = out_df.loc[
            (out_df.x.between(self.idx_bbox[0], self.idx_bbox[2])) &
            (out_df.y.between(self.idx_bbox[1], self.idx_bbox[3]))
        ]
        out_df.x = out_df.x - self.idx_bbox[0]
        out_df.y = out_df.y - self.idx_bbox[1]

        return out_df
    
    def get_geometries(self, crs: CRS = CRS.from_epsg(4326)):
        gdf = self.gdf.to_crs(epsg=crs.to_epsg())
        sgdf = self.sub_square_grid_gdf


        geometry_dict = {}
        for name in self.df.name.unique():
            if name in ['default_ground', 'default_foliage']:
                continue
            geometry_dict[name] = []
            is_linear = False
            is_building = False
            for process in self.config[name]['process']:
                if process in ['stream_tiles', 'road_tiles', 'fence_tiles']:
                    is_linear = True
                    break
                elif process.endswith('outline'):
                    is_building = True
                    break
            
            if not is_building or is_linear:
                df = self.df[self.df.name == name].merge(gdf, on=['xidx', 'yidx'])
                if len(df) == 0:
                    continue
                geometry = union_all(df.geometry)
                if geometry.geom_type == 'MultiPolygon':
                    geometry_dict[name].extend(list(geometry.geoms))
                elif geometry.geom_type == 'Polygon':
                    geometry_dict[name].append(geometry)
            elif is_building:
                process = [p for p in self.config[name]['process'] if p.endswith('outline')][0]
                if process in process_to_building_type:
                    building_type = process_to_building_type[process]
                else:
                    continue
                try:
                    building_tiles = get_building_tiles(building_type, 'cold_war')
                    building_geometries = []
                    for _, row in building_tiles.iterrows():
                        p0 = np.array([0,0])
                        if row['is_diagonal']:
                            p1 = p0 + np.array([0.5, -0.5]) * 8 * row['width']
                            p2 = p1 + np.array([0.5, 0.5]) * 8 * row['height']
                            p3 = p2 + np.array([-0.5, 0.5]) * 8 * row['width']
                        else:
                            p1 = p0 + np.array([0.5, 0]) * 8 * row['width']
                            p2 = p1 + np.array([0, 0.5]) * 8 * row['height']
                            p3 = p2 + np.array([-0.5, 0]) * 8 * row['width']
                        
                        building_geometries.append(affinity.rotate(Polygon([p0, p1, p2, p3]), self.bbox.get_rotation_angle(), origin=(0,0)))
                    building_tiles = building_tiles.assign(building_geometry=building_geometries)
                    building_tiles = building_tiles.assign(cat2_x=[get_building_cat2(building_type, row['row'], row['col'], 'cold_war') for _, row in building_tiles.iterrows()])
                    df = self.df[self.df.name == name].merge(sgdf, on=['xidx', 'yidx'])
                    df = df.merge(building_tiles, on=['cat2_x'])
                    df.geometry = df.geometry.apply(lambda x: x.centroid)
                    df.geometry = df.apply(lambda x: affinity.translate(x.building_geometry, xoff=x.geometry.x, yoff=x.geometry.y), axis=1)
                    df = geopandas.GeoDataFrame(df).set_crs(self.bbox.crs_projected).to_crs(epsg=crs.to_epsg())
                    geometry = union_all(df.geometry)
                    if geometry.geom_type == 'MultiPolygon':
                        geometry_dict[name].extend(list(geometry.geoms))
                    elif geometry.geom_type == 'Polygon':
                        geometry_dict[name].append(geometry)
                except:
                    a = 1

                
        return geometry_dict






