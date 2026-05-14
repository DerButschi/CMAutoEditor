import json

import geopandas
import numpy as np
import pandas
import pyproj
from pyproj.crs import CRS
from shapely import affinity, transform, union_all
from shapely.geometry import shape
from terrain_extraction.bbox_utils import BoundingBox
from terrain_extraction.osm_extraction.config_schema import (
    ExtractionConfig,
    matched_or_first_cm_type,
)
from terrain_extraction.osm_extraction.grid_index import GridIndex
from terrain_extraction.osm_extraction.models import (
    CMType,
    FeatureRecord,
    GridCell,
    GridKind,
    LayerKind,
    PlacementRecord,
    ProcessKind,
)
from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline
from terrain_extraction.osm_extraction.tile_assignment import CompiledTileCatalog

from profiles import get_building_outline_by_df_entry, get_building_tiles, process_to_building_type
from profiles.general import fence_tiles, rail_tiles, road_tiles, stream_tiles

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


class _NullProgress:
    def progress(self, *args, **kwargs):
        return self


class _NullStreamlit:
    def progress(self, *args, **kwargs):
        return _NullProgress()


if st is None:
    st = _NullStreamlit()


class OSMProcessor:
    def __init__(self, profile: str, bbox: BoundingBox, path_to_config: str = "default_osm_config.json"):
        self.path_to_config = path_to_config
        with open(path_to_config) as config_file:
            self.config = json.load(config_file)
        self.extraction_config = ExtractionConfig.from_mapping(self.config)
        self._config_order = {name: idx for idx, name in enumerate(self.config)}
        self._tag_to_config_names = self._build_tag_to_config_names()
        self.profile = profile
        self.bbox = bbox
        self.pipeline = ExtractionPipeline(
            ExtractionContext.create(
                profile=profile,
                bbox=bbox,
                config_path=path_to_config,
                feature_flags=self.extraction_config.feature_flags,
            )
        )
        self.idx_bbox = None
        self.effective_bbox_polygon = None
        self.transformer = None
        self.gdf = None
        self.df = None
        self._df_parts = []
        self.network_graphs = {}
        self._network_line_parts = {}
        self._grid_cell_indices_by_gdf_id = {}
        self.grids = {}
        self.building_outlines = {}
        self.grid_graph = None
        self.occupancy_gdf = geopandas.GeoDataFrame(columns=['geometry', 'priority', 'name'])
        self._occupancy_gdf_parts = []

        self.matched_elements = []
        self.features = ()
        self.placements = ()
        self.output_rows = ()
        self.topology = None
        self.routing = None
        self.stats = None

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

    @property
    def path_to_congih(self):
        return self.path_to_config

    def _build_tag_to_config_names(self):
        tag_to_config_names = {}
        for name, config_entry in self.config.items():
            if 'active' in config_entry and not config_entry['active']:
                continue
            for tag in config_entry['tags']:
                tag_to_config_names.setdefault(tuple(tag), []).append(name)

        return tag_to_config_names

    def _config_entry_allows_element(self, name, element_tags, element_id):
        config_entry = self.config[name]

        if 'exclude_tags' in config_entry:
            for key, value in config_entry['exclude_tags']:
                if element_tags.get(key) == value:
                    return False

        if 'required_tags' in config_entry:
            for key, value in config_entry['required_tags']:
                if element_tags.get(key) != value:
                    return False

        if 'exclude_ids' in config_entry and element_id in config_entry['exclude_ids']:
            return False

        return not ('allowed_ids' in config_entry and element_id not in config_entry['allowed_ids'])

    def _get_matching_config_names(self, element_tags, element_id):
        candidate_names = set()
        for tag_key, tag_value in element_tags.items():
            candidate_names.update(self._tag_to_config_names.get((tag_key, tag_value), ()))

        return [
            name
            for name in sorted(candidate_names, key=self._config_order.get)
            if self._config_entry_allows_element(name, element_tags, element_id)
        ]

    def _update_preprocessing_progress(self, progress_bar, feature_idx, total_features, progress_update_stride):
        if total_features == 0:
            return
        if feature_idx == total_features - 1 or feature_idx % progress_update_stride == 0:
            progress_bar.progress((feature_idx + 1) / total_features, 'Preprocessing OSM Data')

    def _init_grid(self, bbox: BoundingBox):
        self.grid_index = GridIndex.from_bbox(bbox)
        p0, p1, p2 = bbox.get_reference_points(bbox.crs_projected)

        n_bins_x = np.floor((p0.distance(p1)) / 8).astype(int)
        n_bins_y = np.floor((p0.distance(p2)) / 8).astype(int)

        xidx = np.repeat(np.arange(n_bins_x, dtype=np.int32), n_bins_y)
        yidx = np.tile(np.arange(n_bins_y, dtype=np.int32), n_bins_x)
        self.gdf = pandas.DataFrame({"xidx": xidx, "yidx": yidx})
        self.sub_square_grid_diagonal_gdf = geopandas.GeoDataFrame(geometry=[], crs=bbox.crs_projected)
        self.sub_square_grid_gdf = geopandas.GeoDataFrame(geometry=[], crs=bbox.crs_projected)

        p3 = (p1.x + p2.x - p0.x, p1.y + p2.y - p0.y)
        self.effective_bbox_polygon = shape(
            {
                "type": "Polygon",
                "coordinates": [[(p0.x, p0.y), (p1.x, p1.y), p3, (p2.x, p2.y), (p0.x, p0.y)]],
            }
        )

        self.idx_bbox = [0, 0, n_bins_x - 1, n_bins_y - 1]

    def _get_geometry(self, geojson_geometry):
        try:
            return shape(geojson_geometry)
        except Exception:
            return None
        
    def _get_projected_geometry(self, geojson_geometry):
            # geometry = geopandas.GeoSeries(shape(geojson_geometry))
            # geometry = geometry.set_crs(epsg=4326)
            # geometry = geometry.to_crs(epsg=25832)
            # return geometry[0]

            geometry_object = self._get_geometry(geojson_geometry)
            def _trf(coords):
                x, y = self.transformer.transform(coords[:, 0], coords[:, 1])
                return np.column_stack([x, y])

            if geometry_object is not None:
                projected_geometry_object = transform(geometry_object, _trf)

                return projected_geometry_object
            else:
                return None

    def preprocess_osm_data(self, osm_data: dict):
        bbox_crs = self.bbox.crs_projected

        self.transformer = pyproj.Transformer.from_crs('epsg:4326', f'epsg:{bbox_crs.to_epsg()}', always_xy=True)

        element_idx = 0
        unprocessed_tags = {'key': [], 'value': []}
        progress_bar = st.progress(0.0, 'Preprocessing OSM Data')
        total_features = len(osm_data['features'])
        progress_update_stride = max(1, total_features // 10)
        for eidx, element in enumerate(osm_data.features):
            element_properties = dict(element.properties or {})
            if isinstance(element_properties.get('tags'), dict):
                element_tags = element_properties['tags']
            else:
                element_tags = element_properties
            element_id = element_properties.get("id")

            matching_names = self._get_matching_config_names(element_tags, element_id)
            if len(matching_names) == 0:
                for key, value in element_tags.items():
                    unprocessed_tags['key'].append(key)
                    unprocessed_tags['value'].append(value)
                self._update_preprocessing_progress(progress_bar, eidx, total_features, progress_update_stride)
                continue

            geometry = self._get_projected_geometry(element.geometry)
            if geometry is None:
                self._update_preprocessing_progress(progress_bar, eidx, total_features, progress_update_stride)
                continue

            for name in matching_names:
                self.matched_elements.append({'element': element, 'geometry': geometry, 'name': name, 'idx': element_idx})
                element_idx += 1

            self._update_preprocessing_progress(progress_bar, eidx, total_features, progress_update_stride)

        # unprocessed_tags_df = pandas.DataFrame(unprocessed_tags)
        # unprocessed_tags_df = unprocessed_tags_df.drop_duplicates()
        # unprocessed_tags_df.to_csv('unprocessed_tags.csv')
            
        self._init_grid(self.bbox)

        # add default entries
        if "default_ground" in self.config and self.config['default_ground'].get('active', True):
            self.matched_elements.append({'element': None, 'geometry': self.effective_bbox_polygon, 
                                          'name': 'default_ground', 'idx': element_idx + 1})
        if "default_foliage" in self.config and self.config['default_foliage'].get('active', True):
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

    def _get_grid_indices_for_cells(self, gdf, cells):
        if not hasattr(self, '_grid_cell_indices_by_gdf_id'):
            self._grid_cell_indices_by_gdf_id = {}
        cell_index = self._grid_cell_indices_by_gdf_id.get(id(gdf))
        if cell_index is None:
            cell_index = {
                (xidx, yidx): idx
                for idx, xidx, yidx in gdf.loc[:, ['xidx', 'yidx']].itertuples()
            }
            self._grid_cell_indices_by_gdf_id[id(gdf)] = cell_index

        indices = []
        seen_cells = set()
        for cell in cells:
            if cell in seen_cells:
                continue
            if cell in cell_index:
                indices.append(cell_index[cell])
                seen_cells.add(cell)

        return pandas.Index(indices)

    def _append_to_df(self, sub_df: pandas.DataFrame):
        if len(sub_df) > 0:
            if not hasattr(self, '_df_parts'):
                self._df_parts = []
            self._df_parts.append(sub_df)

    def _flush_df_parts(self):
        if len(getattr(self, '_df_parts', [])) == 0:
            return
        if self.df is None:
            self.df = pandas.concat(self._df_parts, ignore_index=True, copy=False)
        else:
            self.df = pandas.concat([self.df, *self._df_parts], ignore_index=True, copy=False)
        self._df_parts = []

    def _append_network_lines(self, name, element_gdf):
        if len(element_gdf) > 0:
            if not hasattr(self, '_network_line_parts'):
                self._network_line_parts = {}
            self._network_line_parts.setdefault(name, []).append(element_gdf)

    def _flush_network_line_parts(self, name=None):
        names = [name] if name is not None else list(getattr(self, '_network_line_parts', {}))
        for line_name in names:
            parts = self._network_line_parts.get(line_name, [])
            if len(parts) == 0:
                continue
            self.network_graphs.setdefault(line_name, {})
            existing_lines = self.network_graphs[line_name].get('lines')
            if existing_lines is None:
                self.network_graphs[line_name]['lines'] = pandas.concat(parts, ignore_index=True, copy=False)
            else:
                self.network_graphs[line_name]['lines'] = pandas.concat([existing_lines, *parts], ignore_index=True, copy=False)
            self._network_line_parts[line_name] = []

    def _append_occupancy_gdf(self, occupancy_entry):
        if len(occupancy_entry) > 0:
            if not hasattr(self, '_occupancy_gdf_parts'):
                self._occupancy_gdf_parts = []
            self._occupancy_gdf_parts.append(occupancy_entry)

    def _flush_occupancy_gdf_parts(self):
        if len(getattr(self, '_occupancy_gdf_parts', [])) == 0:
            return
        self.occupancy_gdf = pandas.concat([self.occupancy_gdf, *self._occupancy_gdf_parts], ignore_index=True, copy=False)
        self._occupancy_gdf_parts = []

    def run_processors(self):
        self._run_typed_processors()

    def _run_typed_processors(self):
        grid_index = getattr(self, "grid_index", None)
        if grid_index is None:
            grid_index = GridIndex.from_bbox(self.bbox)
            self.grid_index = grid_index

        from terrain_extraction.osm_extraction.occupancy import OccupancyModel

        self.occupancy = OccupancyModel.from_grid_index(grid_index)
        features = self._typed_features_from_matched_elements()
        self.features = features

        placements = []
        area_features = tuple(
            feature
            for feature in features
            if feature.process in {ProcessKind.AREA, ProcessKind.RANDOM, ProcessKind.POINT}
        )
        area_result = self.pipeline.run_area_rasterizer(
            features=area_features,
            config=self.extraction_config,
            grid_index=grid_index,
            occupancy=self.occupancy,
        )
        placements.extend(area_result.placements)

        linear_features = tuple(
            feature
            for feature in features
            if feature.process in {ProcessKind.ROAD, ProcessKind.RAIL, ProcessKind.STREAM, ProcessKind.FENCE}
        )
        if linear_features:
            topology_result = self.pipeline.run_network_topology(
                features=linear_features,
                clip_geometry=getattr(self, "effective_bbox_polygon", None),
            )
            self.topology = topology_result.diagnostics["network_topology"]
            routing_result = self.pipeline.run_network_router(
                topology=self.topology,
                grid_index=grid_index,
                occupancy=self.occupancy,
            )
            self.routing = routing_result.diagnostics["network_routes"]
            tile_result = self.pipeline.run_tile_assignment(
                routes=self.routing.routes,
                catalogs=self._tile_catalogs_for(linear_features),
            )
            placements.extend(tile_result.placements)
            self._reserve_output_placements(tile_result.placements)

        linear_dependent_placements = self._typed_linear_feature_placements(tuple(placements))
        placements.extend(linear_dependent_placements)
        self._reserve_output_placements(linear_dependent_placements)

        building_features = tuple(feature for feature in features if feature.process is ProcessKind.BUILDING_OUTLINE)
        if building_features:
            building_result = self.pipeline.run_building_fitter(
                features=building_features,
                catalogs=self._building_catalogs_for(building_features),
                grid_index=grid_index,
                occupancy=self.occupancy,
            )
            placements.extend(building_result.placements)

        self.placements = self._resolve_output_layer_conflicts(tuple(placements))
        output_result = self.pipeline.run_output_rows(
            placements=self.placements,
            bounds=tuple(self.idx_bbox),
        )
        self.output_rows = output_result.output_rows
        self.stats = output_result.stats
        self._set_compatibility_df_from_placements()

    def _set_compatibility_df_from_placements(self):
        from terrain_extraction.osm_extraction.output_rows import (
            OUTPUT_ROW_COLUMNS,
            placements_to_output_rows,
        )

        rows = placements_to_output_rows(self.placements)
        self.df = pandas.DataFrame.from_records(rows, columns=OUTPUT_ROW_COLUMNS)

    def _typed_features_from_matched_elements(self):
        features = []
        clip_geometry = getattr(self, "effective_bbox_polygon", None)
        for fallback_index, element_entry in enumerate(self.matched_elements):
            name = element_entry["name"]
            try:
                config_entry = self.extraction_config.entry_by_name(name)
            except KeyError:
                continue
            properties = self._element_properties(element_entry.get("element"))
            source_tags = properties.get("tags") if isinstance(properties.get("tags"), dict) else properties
            feature_id = properties.get("id", element_entry.get("idx", fallback_index))
            for process in config_entry.processes:
                if process is ProcessKind.DEFAULT:
                    continue
                geometry = self._clip_feature_geometry_to_bbox(
                    element_entry["geometry"],
                    process=process,
                    clip_geometry=clip_geometry,
                )
                if geometry is None:
                    continue
                features.append(
                    FeatureRecord(
                        feature_id=feature_id,
                        source_index=int(element_entry.get("idx", fallback_index)),
                        config_name=name,
                        process=process,
                        priority=config_entry.priority,
                        geometry=geometry,
                        source_tags=source_tags,
                        source_properties=properties,
                        cm_type=matched_or_first_cm_type(config_entry, source_tags),
                    )
                )
        return tuple(features)

    @staticmethod
    def _clip_feature_geometry_to_bbox(geometry, *, process, clip_geometry):
        if clip_geometry is None:
            return geometry
        if process is ProcessKind.POINT:
            return geometry if clip_geometry.covers(geometry) else None
        if not geometry.intersects(clip_geometry):
            return None
        clipped = geometry.intersection(clip_geometry)
        return None if clipped.is_empty else clipped

    @staticmethod
    def _element_properties(element):
        if element is None:
            return {}
        properties = getattr(element, "properties", {})
        return dict(properties or {})

    def _tile_catalogs_for(self, features):
        tile_sources = {
            ProcessKind.ROAD: road_tiles,
            ProcessKind.RAIL: rail_tiles,
            ProcessKind.STREAM: stream_tiles,
            ProcessKind.FENCE: fence_tiles,
        }
        processes = {feature.process for feature in features}
        return {
            process: CompiledTileCatalog.from_records(
                tile_sources[process],
                process=process,
                base_cm_type=self._base_cm_type_for_process(process),
            )
            for process in sorted(processes, key=lambda item: item.value)
        }

    def _base_cm_type_for_process(self, process):
        for entry in self.extraction_config.entries:
            if process in entry.processes and entry.cm_types:
                return entry.cm_types[0]
        return None

    def _building_catalogs_for(self, features):
        catalogs = {}
        for feature in features:
            if feature.config_name in catalogs:
                continue
            entry = self.extraction_config.entry_by_name(feature.config_name)
            building_type = None
            for legacy_process in entry.legacy_processes:
                if legacy_process in process_to_building_type:
                    building_type = process_to_building_type[legacy_process]
                    break
            if building_type is not None:
                catalogs[feature.config_name] = get_building_tiles(building_type, self.profile)
        return catalogs

    def _typed_linear_feature_placements(self, placements):
        linear_placements = []
        cells_by_name = {}
        for placement in placements:
            for cell in placement.cells:
                cells_by_name.setdefault(placement.config_name, set()).add(cell)

        for entry in self.extraction_config.entries:
            if ProcessKind.LINEAR not in entry.processes:
                continue
            source_name = entry.modifiers.get("linear_name")
            if not source_name:
                continue
            for cell in sorted(cells_by_name.get(source_name, ()), key=lambda item: (item.xidx, item.yidx)):
                cm_type = self._choose_cm_type(entry.cm_types)
                if cm_type is None:
                    continue
                linear_placements.append(
                    PlacementRecord(
                        layer=self._layer_for_cm_type(cm_type),
                        grid_kind=GridKind.NORMAL,
                        cells=(GridCell(cell.xidx, cell.yidx),),
                        config_name=entry.name,
                        feature_id=f"{entry.name}:{source_name}:{cell.xidx}:{cell.yidx}",
                        priority=entry.priority,
                        cm_type=cm_type,
                        score=1.0,
                        diagnostics={"derived_from_linear": source_name},
                    )
                )
        return tuple(linear_placements)

    def _choose_cm_type(self, cm_types):
        if not cm_types:
            return None
        weights = np.array([float(cm_type.modifiers.get("weight", 1.0)) for cm_type in cm_types], dtype=float)
        probabilities = weights / weights.sum()
        cm_type = cm_types[int(self.pipeline.context.rng.choice(len(cm_types), p=probabilities))]
        return None if cm_type.modifiers.get("dummy") is True else cm_type

    @staticmethod
    def _layer_for_cm_type(cm_type: CMType):
        menu = cm_type.menu.lower()
        if menu.startswith("foliage") or menu.startswith("brush"):
            return LayerKind.FOLIAGE
        if menu.startswith("flavor objects"):
            return LayerKind.POINT_OBJECT
        if menu.startswith("walls") or menu.startswith("fence"):
            return LayerKind.LINEAR_OBJECT
        if menu.startswith("roads"):
            return LayerKind.LINEAR_SURFACE
        if "building" in menu:
            return LayerKind.BUILDING
        return LayerKind.GROUND

    def _reserve_output_placements(self, placements):
        occupancy = getattr(self, "occupancy", None)
        if occupancy is None:
            return
        for placement in placements:
            object_id = (
                placement.feature_id
                if placement.feature_id is not None
                else f"{placement.config_name}:{placement.cells[0].xidx}:{placement.cells[0].yidx}"
            )
            occupancy.place(placement, object_id=object_id, allow_replace=False)

    @staticmethod
    def _resolve_output_layer_conflicts(placements):
        winners_by_cell = {}
        for placement_idx, placement in enumerate(placements):
            for cell in placement.cells:
                key = (placement.layer, cell)
                winner = winners_by_cell.get(key)
                candidate = (OSMProcessor._output_priority_key(placement), placement_idx)
                if winner is None or candidate < winner[0]:
                    winners_by_cell[key] = (candidate, placement)

        resolved = []
        for placement in placements:
            cells = tuple(
                cell
                for cell in placement.cells
                if winners_by_cell.get((placement.layer, cell), (None, None))[1] is placement
            )
            if not cells:
                continue
            if cells == placement.cells:
                resolved.append(placement)
                continue
            resolved.append(
                PlacementRecord(
                    layer=placement.layer,
                    grid_kind=placement.grid_kind,
                    cells=cells,
                    config_name=placement.config_name,
                    feature_id=placement.feature_id,
                    priority=placement.priority,
                    cm_type=placement.cm_type,
                    score=placement.score,
                    diagnostics=placement.diagnostics,
                )
            )
        return tuple(resolved)

    @staticmethod
    def _output_priority_key(placement):
        if placement.priority > 0:
            return 0, placement.priority
        if placement.priority > -999:
            return 1, -placement.priority
        return 2, 0

    def post_process(self):
        if self._uses_layered_output():
            self._get_layered_output_rows()
            return

        self._flush_df_parts()
        # remove invalid entries
        self.df = self.df.drop_duplicates()
        self.df = self.df.drop(self.df[(self.df.menu == -1) & (self.df.z == -1)].index)

        duplicate_cell = self.df.duplicated(subset=['xidx', 'yidx'], keep=False)
        min_positive_priority = (
            self.df.loc[self.df.priority > 0]
            .groupby(by=['xidx', 'yidx'])
            .priority
            .min()
            .rename('_min_positive_priority')
        )
        df_with_min_priority = self.df.join(min_positive_priority, on=['xidx', 'yidx'])
        has_positive_priority = df_with_min_priority._min_positive_priority > 0
        indices_to_drop = df_with_min_priority.loc[
            duplicate_cell & (
                (df_with_min_priority.priority == -999) |
                (
                    has_positive_priority & (
                        (df_with_min_priority.priority > df_with_min_priority._min_positive_priority) |
                        ((df_with_min_priority.priority < 0) & (df_with_min_priority.priority > -999))
                    )
                )
            )
        ].index.tolist()

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

        duplicate_rank_indices = self._get_duplicate_cm_rank_indices_to_drop()
        if len(duplicate_rank_indices) > 0:
            self.df = self.df.drop(duplicate_rank_indices)

    def _get_duplicate_cm_rank_indices_to_drop(self):
        duplicate_rows = self.df.loc[
            self.df.duplicated(subset=['xidx', 'yidx', 'priority', 'name'], keep=False)
        ]
        if len(duplicate_rows) == 0:
            return []

        ranked_rows = duplicate_rows.copy()
        ranked_rows['_cm_rank'] = [
            self._get_cm_type_rank(name, menu, cat1, cat2)
            for name, menu, cat1, cat2 in duplicate_rows[['name', 'menu', 'cat1', 'cat2']].itertuples(index=False, name=None)
        ]
        ranked_rows = ranked_rows.dropna(subset=['_cm_rank'])
        if len(ranked_rows) == 0:
            return []

        ranked_rows['_source_order'] = np.arange(len(ranked_rows))
        ranked_rows = ranked_rows.sort_values(
            by=['xidx', 'yidx', 'priority', 'name', '_cm_rank', '_source_order'],
            kind='mergesort',
        )
        keep_first_ranked = (
            ranked_rows.groupby(by=['xidx', 'yidx', 'priority', 'name']).cumcount() == 0
        )

        return ranked_rows.loc[~keep_first_ranked].index.tolist()

    def _get_cm_type_rank(self, name, menu, cat1, cat2):
        for cidx, cm_type in enumerate(self.config[name]['cm_types']):
            if (
                cm_type.get('menu') == menu
                and cm_type.get('cat1') == cat1
                and ('cat2' not in cm_type or cm_type['cat2'] == cat2)
            ):
                return cidx

        return np.nan

    def write_to_file(self, output_file_name):
        if self._uses_layered_output():
            self._get_layered_output_dataframe().to_csv(output_file_name)
            return

        self._flush_df_parts()
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
        if self._uses_layered_output():
            return self._get_layered_output_dataframe()

        self._flush_df_parts()
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

    def _uses_layered_output(self):
        return hasattr(self, "idx_bbox") and self.idx_bbox is not None and hasattr(self, "placements")

    def _get_layered_output_rows(self):
        from terrain_extraction.osm_extraction.output_rows import (
            append_extent_marker,
            normalize_output_coordinates,
            placements_to_output_rows,
            validate_output_rows,
        )

        bounds = tuple(self.idx_bbox)
        internal_rows = placements_to_output_rows(tuple(getattr(self, "placements", ())), include_internal=True)
        rows_with_extent = append_extent_marker(internal_rows, bounds=bounds, include_internal=True)
        validate_output_rows(rows_with_extent, bounds=bounds)
        return normalize_output_coordinates(rows_with_extent, bounds=bounds)

    def _get_layered_output_dataframe(self):
        from terrain_extraction.osm_extraction.output_rows import NORMALIZED_OUTPUT_ROW_COLUMNS

        rows = self._get_layered_output_rows()
        return pandas.DataFrame.from_records(rows, columns=NORMALIZED_OUTPUT_ROW_COLUMNS)

    def _uses_new_debug_export(self):
        return getattr(self, "grid_index", None) is not None and (
            hasattr(self, "placements") or hasattr(self, "output_rows") or hasattr(self, "features")
        )

    def _get_debug_export_geometries(self, crs: CRS | None = None):
        from terrain_extraction.osm_extraction.debug_export import build_debug_layers
        from terrain_extraction.osm_extraction.output_rows import placements_to_output_rows

        grid_index = getattr(self, "grid_index", None)
        if grid_index is None:
            return {}

        output_rows = tuple(getattr(self, "output_rows", ()) or ())
        if not output_rows:
            output_rows = placements_to_output_rows(tuple(getattr(self, "placements", ())), include_internal=True)

        debug_export = build_debug_layers(
            features=tuple(getattr(self, "features", ())),
            topology=getattr(self, "topology", None),
            routing=getattr(self, "routing", None),
            occupancy=getattr(self, "occupancy", None),
            placements=tuple(getattr(self, "placements", ())),
            output_rows=output_rows,
            grid_index=grid_index,
            bounds=tuple(getattr(self, "idx_bbox", (0, 0, grid_index.width - 1, grid_index.height - 1))),
            stats=getattr(self, "stats", None),
        )
        self.debug_export_diagnostics = debug_export.diagnostics
        final_rows = debug_export.layers.get("final_rows")
        if final_rows is None or final_rows.empty:
            return {}

        if crs is not None and final_rows.crs is not None:
            final_rows = final_rows.to_crs(epsg=crs.to_epsg())

        geometry_dict = {}
        for name, group in final_rows.groupby("name"):
            geometry_dict[name] = list(group.geometry)
        return geometry_dict
    
    def get_geometries(self, crs: CRS | None = None):
        if self._uses_new_debug_export():
            return self._get_debug_export_geometries(crs=crs)

        self._flush_df_parts()
        if crs is None:
            crs = CRS.from_epsg(4326)
        grid_gdf = self.gdf.to_crs(epsg=crs.to_epsg())
        sgdf = self.sub_square_grid_gdf
        dgdf = self.sub_square_grid_diagonal_gdf


        geometry_dict = {}
        for name in self.df.name.unique():
            if name in ['default_ground', 'default_foliage']:
                continue
            geometry_dict[name] = []
            is_building = False
            for process in self.config[name]['process']:
                if process.endswith('outline'):
                    is_building = True
                    break
            
            if not is_building:
                df = self.df[self.df.name == name].merge(grid_gdf.loc[:, ['xidx', 'yidx', 'geometry']], on=['xidx', 'yidx'])
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
                    for group_name, group in self.df[self.df.name == name].groupby(by=["menu", "cat1", "cat2", "direction"]):
                        outline, is_diagonal = get_building_outline_by_df_entry(building_type, *group_name)
                        outline = affinity.rotate(outline, self.bbox.get_rotation_angle(), origin=(0,0))
                        merged_group = group.merge(dgdf if is_diagonal else sgdf, on=['xidx', 'yidx'], suffixes=(None, '_y'))
                        merged_group.geometry = merged_group.geometry.apply(lambda x: x.centroid)
                        merged_group.geometry = merged_group.geometry.apply(lambda x, outline=outline: affinity.translate(outline, xoff=x.x, yoff=x.y))
                        building_gdf = geopandas.GeoDataFrame(
                            merged_group.drop(columns='geometry'),
                            geometry=list(merged_group.geometry),
                            crs=self.bbox.crs_projected,
                        ).to_crs(epsg=crs.to_epsg())
                        geometry = union_all(building_gdf.geometry)
                        if geometry.geom_type == 'MultiPolygon':
                            geometry_dict[name].extend(list(geometry.geoms))
                        elif geometry.geom_type == 'Polygon':
                            geometry_dict[name].append(geometry)

                    # building_tiles = get_building_tiles(building_type, 'cold_war')
                    # building_geometries = []
                    # for _, row in building_tiles.iterrows():
                    #     p0 = np.array([0,0])
                    #     if row['is_diagonal']:
                    #         p1 = p0 + np.array([0.5, -0.5]) * 8 * row['width']
                    #         p2 = p1 + np.array([0.5, 0.5]) * 8 * row['height']
                    #         p3 = p2 + np.array([-0.5, 0.5]) * 8 * row['width']
                    #     else:
                    #         p1 = p0 + np.array([0.5, 0]) * 8 * row['width']
                    #         p2 = p1 + np.array([0, 0.5]) * 8 * row['height']
                    #         p3 = p2 + np.array([-0.5, 0]) * 8 * row['width']
                        
                    #     building_geometries.append(affinity.rotate(Polygon([p0, p1, p2, p3]), self.bbox.get_rotation_angle(), origin=(0,0)))
                    # building_tiles = building_tiles.assign(building_geometry=building_geometries)
                    # building_tiles = building_tiles.assign(cat2=[get_building_cat2(building_type, row['row'], row['col'], 'cold_war') for _, row in building_tiles.iterrows()])
                    # for is_diagonal in [True, False]:
                    #     if is_diagonal:
                    #         df = self.df[self.df.name == name].merge(dgdf, on=['xidx', 'yidx'], suffixes=(None, '_y'))
                    #     else:
                    #         df = self.df[self.df.name == name].merge(sgdf, on=['xidx', 'yidx'], suffixes=(None, '_y'))
                    #     tiles = building_tiles[building_tiles['is_diagonal'] == is_diagonal]
                    #     df = df.merge(tiles, on=['menu', 'cat1', 'cat2'])
                    #     df.geometry = df.geometry.apply(lambda x: x.centroid)
                    #     df.geometry = df.apply(lambda x: affinity.translate(x.building_geometry, xoff=x.geometry.x, yoff=x.geometry.y), axis=1)
                    #     df = geopandas.GeoDataFrame(df).set_crs(self.bbox.crs_projected).to_crs(epsg=crs.to_epsg())
                    #     geometry = union_all(df.geometry)
                    #     if geometry.geom_type == 'MultiPolygon':
                    #         geometry_dict[name].extend(list(geometry.geoms))
                    #     elif geometry.geom_type == 'Polygon':
                    #         geometry_dict[name].append(geometry)
                except Exception:
                    pass

                
        return geometry_dict


# df2 = self.df[self.df.name == name].merge(building_tiles, on=['menu', 'cat1', 'cat2'])
# df2[df2.is_diagonal == is_diagonal].merge(dgdf, on=['xidx', 'yidx'])    



