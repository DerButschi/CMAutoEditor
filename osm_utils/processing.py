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

import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString, Polygon, MultiPolygon
from shapely.ops import split, substring, snap
from shapely.affinity import scale, rotate, translate
from shapely import unary_union
import geopandas
import pandas
import logging
from profiles.general import road_tiles, rail_tiles, stream_tiles, fence_tiles
from profiles.cold_war import buildings
from .path_search import search_path, _get_closest_node_in_gdf, _remove_nodes_from_gdf
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from osm_utils.geometry import find_concave_vertices, find_chords, find_subdividing_chords, rectangulate_polygon
import itertools
from copy import deepcopy


DRAW_DEBUG_PLOTS = False

road_direction_dict = {
    (0, 1): 'u',
    (0, -1): 'd',
    (1, 0): 'r',
    (-1, 0): 'l',

    (1, 1): 'ur',
    (1, -1): 'dr',
    (-1, 1): 'ul',
    (-1, -1): 'dl',
}

opposite_road_direction_dict = {
    'u': 'd',
    'd': 'u',
    'r': 'l',
    'l': 'r',

    'ur': 'dl',
    'ul': 'dr',
    'dr': 'ul',
    'dl': 'ur'
}

def _direction_from_square_to_square(square1, square2):
    diff_tuple = (square2[0] - square1[0], square2[1] - square1[1])
    if diff_tuple in road_direction_dict:
        direction = road_direction_dict[diff_tuple]
    else:
        direction = None
    
    return direction, diff_tuple


def draw_line_graph(graph: nx.MultiGraph, show: bool = False):
    plt.figure()
    plt.axis('equal')
    for edge in graph.edges:
        ls = graph.get_edge_data(*edge)['ls']
        plt.plot(ls.xy[0], ls.xy[1], '-')
    
    for node in graph.nodes:
        plt.plot(node[0], node[1], 'ko')

    if show:
        plt.show()

def draw_square_graph(graph: nx.MultiGraph, gdf: geopandas.GeoDataFrame = None, show: bool = False):
    plt.figure()
    plt.axis('equal')
    for edge in graph.edges:
        squares = graph.get_edge_data(*edge)['squares']
        if gdf is not None:
            square_centers = [gdf.loc[(gdf.xidx == sq[0]) & (gdf.yidx == sq[1]), ['x', 'y']].values[0] for sq in squares]
            square_geometries = [gdf.loc[(gdf.xidx == sq[0]) & (gdf.yidx == sq[1])].geometry.values[0] for sq in squares]
            for polygon in square_geometries:
                plt.plot(polygon.exterior.xy[0], polygon.exterior.xy[1], '-r')

            plt.plot([sq[0] for sq in square_centers], [sq[1] for sq in square_centers], '-+')
        else:
            plt.plot([sq[0] for sq in square_centers], [sq[1] for sq in square_centers], '-')
    
    for node in graph.nodes:
        if gdf is not None:
            node_coord = gdf.loc[(gdf.xidx == node[0]) & (gdf.yidx == node[1]), ['x', 'y']].values[0]
            plt.plot(node_coord[0], node_coord[1], 'ko')
        else:
            plt.plot(node[0], node[1], 'ko')

    if show:
        plt.show()

def get_grid_cells_to_fill(gdf, geometry):
    within = gdf.index.isin(gdf.sindex.query(geometry, predicate='contains'))
    intersecting = gdf.index.isin(gdf.sindex.query(geometry, predicate='intersects'))
    # within = gdf.geometry.within(geometry)
    # intersecting = gdf.geometry.intersects(geometry)
    is_border = np.bitwise_and(intersecting, ~within)
    gdf_border = gdf.loc[is_border]
    gdf_border_largest_square_area = gdf_border.loc[gdf_border.geometry.intersection(geometry).area > 32]
    is_largest_square_area = gdf.index.isin(gdf_border_largest_square_area.index)
    to_fill = np.bitwise_or(within, is_largest_square_area)

    return to_fill

def extract_connection_directions_from_node(node_id, other_node_id, road_graph, loop=None):
    directions = []
    edge_data = road_graph.get_edge_data(node_id, other_node_id)
    for key, data in edge_data.items():
        squares = data['squares']
        node_square = node_id
        if squares[0] != node_square:
            squares = squares[::-1]

        # in case of loops
        if node_id == other_node_id:
            if loop == 'end' and node_id == other_node_id:
                directions.append(road_direction_dict[(squares[-2][0] - squares[0][0], squares[-2][1] - squares[0][1])])
            elif loop == 'start' and node_id == other_node_id:
                directions.append(road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])])
            else:
                directions.append(road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])])
                directions.append(road_direction_dict[(squares[-2][0] - squares[0][0], squares[-2][1] - squares[0][1])])
        else:
            directions.append(road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])])


        # if loop is not None:
        #     if loop == 'end' and node_id == other_node_id:
        #         directions.append(road_direction_dict[(squares[-2][0] - squares[0][0], squares[-2][1] - squares[0][1])])
        #     elif loop == 'start' and node_id == other_node_id:
        #         directions.append(road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])])
        # else:
        #     directions.append(road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])])
        #     if node_id == other_node_id:
        #         directions.append(road_direction_dict[(squares[-2][0] - squares[0][0], squares[-2][1] - squares[0][1])])
    
    return directions

def extract_valid_tiles_from_node(node_id, other_node_id, road_graph, edge_graphs):
    # direction_to_other_node = extract_connection_direction_from_node(node_id, other_node_id, road_graph)
    valid_connections = []
    if (node_id, other_node_id) in edge_graphs:
        graph = edge_graphs[(node_id, other_node_id)]['graph']
        if nx.has_path(graph, 'start', 'end'):
            # paths = nx.all_shortest_paths(graph, 'start', 'end', weight='cost')
            paths = nx.all_shortest_paths(graph, 'start', 'end')
            for path in paths:
                valid_connections.append(path[1][0])
    elif (other_node_id, node_id) in edge_graphs:
        graph = nx.reverse_view(edge_graphs[(other_node_id, node_id)]['graph'])
        if nx.has_path(graph, 'end', 'start'):
            paths = nx.all_shortest_paths(graph, 'end', 'start')
            for path in paths:
                valid_connections.append(path[1][0])

    return valid_connections

def extract_all_valid_node_tiles(node_id, square_graph, edge_graphs):
    edges = list({edge for edge in square_graph.edges(node_id)})
    multi_edges = []
    for edge in edges:
        edge_data = square_graph.get_edge_data(*edge)
        for edge_key in edge_data:
            multi_edges.append((edge[0], edge[1], edge_key))

    valid_tiles = None
    for edge in multi_edges:
        start_node = None
        if edge in edge_graphs:
            graph = edge_graphs[edge]['graph']
            if nx.has_path(graph, 'start', 'end'):
                start_node = 'start'
                end_node = 'end'
        elif (edge[1], edge[0], edge[2]) in edge_graphs:
            graph = nx.reverse_view(edge_graphs[(edge[1], edge[0], edge[2])]['graph'])
            if nx.has_path(graph, 'end', 'start'):
                start_node = 'end'
                end_node = 'start'

        if start_node is not None:
            tiles = []        
            for nb in nx.neighbors(graph, start_node):
                if nx.has_path(graph, nb, end_node):
                    tiles.append(nb[0])
            if len(tiles) > 0:
                if valid_tiles is None:
                    valid_tiles = set(tiles)
                else:
                    valid_tiles = valid_tiles.intersection(set(tiles))

    return list(valid_tiles) if valid_tiles is not None else []
    
def extract_required_directions(node_id, neighbors, road_graph, loop=None):
    required_directions = []
    if len(neighbors) > 0:
        for neighbor in neighbors:
            directions_out = extract_connection_directions_from_node(node_id, neighbor, road_graph, loop)

            required_directions.extend(directions_out)

    return list(set(required_directions))

def extract_directions(node, graph):
    directions = []
    for node1, node2, edge_key, edge_data in graph.edges(node, keys=True, data=True):
        squares = edge_data['squares']
        if edge_data['from_node_to_node'][0] == node:
            direction, _ = _direction_from_square_to_square(squares[0], squares[1])
            directions.append(direction)
        if edge_data['from_node_to_node'][1] == node:
            direction, _ = _direction_from_square_to_square(squares[-1], squares[-2])
            directions.append(direction)
        
    return list(set(directions))

def fix_tile_for_node(node_id, edge_graphs, tile):
    for node_pair in edge_graphs:
        if node_id in node_pair:
            graph = edge_graphs[node_pair]['graph']
            nodes_to_delete = []
            if node_id == node_pair[0]:
                for graph_node in nx.neighbors(graph, 'start'):
                    if graph_node[0] != tile:
                        nodes_to_delete.append(graph_node)
            else:
                for graph_node in nx.neighbors(nx.reverse_view(graph), 'end'):
                    if graph_node[0] != tile:
                        nodes_to_delete.append(graph_node)

            graph.remove_nodes_from(nodes_to_delete)

def get_matched_cm_type(config, element_entry):
    cm_types = config[element_entry['name']]['cm_types']

    if 'tags' in element_entry['element'].properties:
        tags = element_entry['element'].properties['tags']
    else: 
        tags = element_entry['element'].properties

    matched_cm_type = None
    for cm_type in cm_types:
        matched = False
        for key, value in cm_type['tags']:
            if key in tags and tags[key] == value:
                matched = True
        
        if matched:
            matched_cm_type = cm_type
            break

    return matched_cm_type

def assign_type_from_tag(osm_processor, config, element_entry):
    grid_cells = get_grid_cells_to_fill(osm_processor.gdf, element_entry['geometry'])
    sub_df = osm_processor._get_sub_df(grid_cells)

    cm_type = get_matched_cm_type(config, element_entry)
    if cm_type is not None:
        for key in cm_type:
            if key in sub_df.columns:
                sub_df[key] = cm_type[key]

        osm_processor._append_to_df(sub_df)


def create_line_graph(osm_processor, config, name):
    logger = logging.getLogger('osm2cm')
    lines = osm_processor.network_graphs[name]['lines']
    lines_intersecting = lines.geometry.sindex.query_bulk(lines.geometry, 'intersects')

    line_graph = nx.MultiGraph()

    for line_idx in range(len(lines)):
        ls = lines.iloc[line_idx].geometry
        other_line_indices = lines_intersecting[1][np.where(np.bitwise_and(lines_intersecting[0] == line_idx, lines_intersecting[1] != line_idx))[0]]
        if len(other_line_indices) > 0:
            intersection_points = lines.iloc[other_line_indices].intersection(ls).unique()
            filtered_intersection_points = []
            for p in intersection_points:
                if type(p) == Point:
                    filtered_intersection_points.append(p)
                elif type(p) == MultiPoint:
                    filtered_intersection_points.extend([g for g in p.geoms])
                elif type(p) == LineString or type(p) == MultiLineString:
                    filtered_intersection_points.extend([g for g in p.boundary.geoms])
                else:
                    logger.warning('Intersection geometry of type {} is not handled.'.format(type(p)))
            
            intersection_points = filtered_intersection_points
            intersection_points.extend([p for p in ls.boundary.geoms if p not in intersection_points])
            intersection_points = sorted(intersection_points, key=lambda p: ls.project(p))
        else:
            intersection_points = [p for p in ls.boundary.geoms]

        if len(intersection_points) > 0:
            ls_splits = split(ls, MultiPoint(intersection_points))
            ls_splits = [geom for geom in ls_splits.geoms]
        else:
            ls_splits = [ls]

        for lidx, ls_split in enumerate(ls_splits):
            if len(ls_split.coords) == 0:
                # TODO: check how this can happen
                logger.warn('LineString split is empty.')
                continue
            if type(ls_split) == LineString:
                p1 = Point(ls_split.coords[0])
                p2 = Point(ls_split.coords[-1])
                line_graph.add_edge((p1.x, p1.y), (p2.x, p2.y), ls=ls_split, element_idx=lines.iloc[line_idx].element_idx, from_node_to_node=[(p1.x, p1.y), (p2.x, p2.y)])
            else:
                logger.warn('LineString split was of type {}.'.format(type(ls_split)))

    if DRAW_DEBUG_PLOTS:
        draw_line_graph(line_graph, show=True)
    osm_processor.network_graphs[name]['line_graph'] = line_graph


def create_octagon_graph(osm_processor, config, name):
    line_graph = osm_processor.network_graphs[name]['line_graph']
    gdf = osm_processor.octagon_gdf
                
    square_graph = line_graph_to_square_graph(line_graph, grid_gdf=osm_processor.gdf, snap_gdf=gdf, snap_to_grid=True)
    handle_square_graph_duplicate_edges(square_graph)

    go_on = True
    while go_on:
        go_on = False
        for edge in square_graph.edges:
            np_squares = np.array(square_graph.get_edge_data(*edge)['squares'])
            sharp_angle_idx = None
            for idx in range(len(np_squares)-2):
                if np.linalg.norm(np_squares[idx+2] - np_squares[idx]) == 1:
                    sharp_angle_idx = idx + 1
                    break
            if sharp_angle_idx is not None:
                squares = square_graph.get_edge_data(*edge)['squares']
                squares.pop(sharp_angle_idx)
                square_graph.get_edge_data(*edge)['squares'] = squares
                go_on = True

    if DRAW_DEBUG_PLOTS:
        draw_square_graph(square_graph, gdf=gdf, show=True)
    osm_processor.network_graphs[name]['square_graph'] = square_graph

def create_square_graph(osm_processor, config: dict, name: str):
    line_graph = osm_processor.network_graphs[name]['line_graph']
    gdf = osm_processor.gdf

    square_graph = line_graph_to_square_graph(line_graph, grid_gdf=gdf, snap_gdf=gdf)
    handle_square_graph_duplicate_edges(square_graph)

    if DRAW_DEBUG_PLOTS:
        draw_square_graph(square_graph, gdf=gdf, show=True)
    osm_processor.network_graphs[name]['square_graph'] = square_graph

def line_graph_to_square_graph(line_graph: nx.MultiGraph, grid_gdf: geopandas.GeoDataFrame, snap_gdf: geopandas.GeoDataFrame, snap_to_grid=False) -> nx.MultiGraph:
    square_graph = nx.MultiGraph()

    for edge in line_graph.edges:
        edge_data = line_graph.edges[edge]
        ls = edge_data['ls']
        if snap_to_grid:
            ls_coords = np.array(ls.coords)
            ls_coords[:,0] = ls_coords[:,0] - grid_gdf.x.min()
            ls_coords[:,1] = ls_coords[:,1] - grid_gdf.y.min()
            ls_coords = np.round(ls_coords / 8 ) * 8
            ls_coords[:, 0] = ls_coords[:, 0] + grid_gdf.x.min()
            ls_coords[:, 1] = ls_coords[:, 1] + grid_gdf.y.min()
            ls = LineString(ls_coords)
        
        squares = snap_gdf.loc[snap_gdf.sindex.query(ls, predicate='intersects')]
        ls_intersection = squares.geometry.intersection(ls)
        if ls_intersection.is_empty.all():
            continue
        intersection_mid_points = []
        for idx in range(len(ls_intersection.values)):
            g = ls_intersection.values[idx]
            if type(g) == LineString:
                intersection_mid_points.append(g.interpolate(0.5, normalized=True))
            elif type(g) == MultiLineString:
                for gidx in range(len(g.geoms)):
                    intersection_mid_points.append(g.geoms[gidx].interpolate(0.5, normalized=True))
            else:
                raise Exception

        if len(intersection_mid_points) > 1:
            # intersection_mid_points = np.array(intersection_mid_points, dtype=object)
            sorted_intersection_mid_points = sorted(intersection_mid_points, key=lambda mid_point: ls.project(mid_point))
        else:
            sorted_intersection_mid_points = intersection_mid_points

        squares_along_way = []
        for p in sorted_intersection_mid_points:
            snap_gdf_point = snap_gdf.loc[snap_gdf.sindex.query(p, predicate='within')]
            if len(snap_gdf_point) > 0:
                # xidx, yidx = grid_gdf_ls.loc[grid_gdf.contains(point), ['xidx', 'yidx']].values[0]
                xidx, yidx = snap_gdf_point.loc[:, ['xidx', 'yidx']].values[0]
                squares_along_way.append((xidx, yidx))


        if len(squares_along_way) > 1:
            square_graph.add_edge(squares_along_way[0], squares_along_way[-1], squares=squares_along_way, element_idx=edge_data['element_idx'], 
            from_node_to_node=[squares_along_way[0], squares_along_way[-1]])

    return square_graph

def handle_square_graph_duplicate_edges(square_graph: nx.MultiGraph):
    nodes = list(square_graph.nodes)
    for node in nodes:
        edges = list({edge for edge in square_graph.edges(node)})
        squares = []
        for edge in edges:
            edge_data = square_graph.get_edge_data(*edge)
            for key in edge_data:
                squares.extend(edge_data[key]['squares'][1:-1])
        
        unique_squares, square_counts = np.unique(squares, axis=0, return_counts=True)
        if (square_counts > 1).any():
            duplicate_squares = unique_squares[np.where(square_counts > 1)[0]]
            for nb_node in list(square_graph.neighbors(node)):
                # yay MultiGraphs!
                to_remove = []
                to_append = []
                edge_data = square_graph.get_edge_data(node, nb_node)
                # FIXME: This will crash if there are more than 2 edges!
                for edge_key in edge_data:
                    squares_to_nb = edge_data[edge_key]['squares']
                    element_idx = edge_data[edge_key]['element_idx']
                    if squares_to_nb[0] != node:
                        squares_to_nb = squares_to_nb[::-1]
                    square_lines = []
                    prev_node_idx = 0
                    for sq_idx, sq in enumerate(squares_to_nb):
                        if ((duplicate_squares[:,0] == sq[0]) & (duplicate_squares[:,1] == sq[1])).any() or sq_idx == len(squares_to_nb) - 1:
                            square_lines.append(squares_to_nb[prev_node_idx:sq_idx+1])
                            prev_node_idx = sq_idx

                    if len(square_lines) > 1:
                        to_remove.append(edge_key)
                        # square_graph.remove_edge(node, nb_node)
                        for sq_line in square_lines:
                            to_append.append([sq_line[0], sq_line[-1], sq_line, element_idx, [sq_line[0], sq_line[-1]]])

                for edge_key in to_remove:
                    square_graph.remove_edge(node, nb_node, key=edge_key)
                for new_edge_data in to_append:
                    square_graph.add_edge(new_edge_data[0], new_edge_data[1], squares=new_edge_data[2], element_idx=new_edge_data[3], from_node_to_node=new_edge_data[4])

def remove_degree_two_nodes_from_graph(graph: nx.MultiGraph, edge_type: str = 'ls') -> None:
    degree_two_nodes = [node for node, degree in graph.degree if degree == 2]
    for node in degree_two_nodes:
        neighbors = [nb for nb in nx.neighbors(graph, node)]
        pass
        


def assign_type_randomly_in_area(osm_processor, config, element_entry):
    cm_types = config[element_entry['name']]['cm_types']
    n_types = len(cm_types)
    df = osm_processor.df

    grid_cells = get_grid_cells_to_fill(osm_processor.gdf, element_entry['geometry'])
    sub_df = osm_processor._get_sub_df(grid_cells)

    sum_of_weights = sum([cm_type['weight'] if 'weight' in cm_type else 1.0 for cm_type in cm_types])
    probabilities = [cm_type['weight'] / sum_of_weights if 'weight' in cm_type else 1.0 / sum_of_weights for cm_type in cm_types]

    rng = np.random.default_rng()

    type_idx = rng.choice(list(range(n_types)), p=probabilities, size=1)[0]
    cm_type = cm_types[type_idx]
    for key in cm_type:
        if key in sub_df.columns:
            sub_df.loc[:, key] = cm_type[key]

    osm_processor._append_to_df(sub_df)


def assign_type_randomly_for_each_square(osm_processor, config, element_entry):
    cm_types = config[element_entry['name']]['cm_types']
    n_types = len(cm_types)
    df = osm_processor.df

    grid_cells = get_grid_cells_to_fill(osm_processor.gdf, element_entry['geometry'])
    sub_df = osm_processor._get_sub_df(grid_cells)

    sum_of_weights = sum([cm_type['weight'] if 'weight' in cm_type else 1.0 for cm_type in cm_types])
    probabilities = [cm_type['weight'] / sum_of_weights if 'weight' in cm_type else 1.0 / sum_of_weights for cm_type in cm_types]

    rng = np.random.default_rng()
    type_indices = rng.choice(list(range(n_types)), p=probabilities, size=(len(sub_df),))

    # type_indices = np.random.randint(0, n_types, size=(len(sub_df),))
    for type_idx in range(n_types):
        idx = np.where(type_indices == type_idx)[0]
        if len(idx) > 0:
            cm_type = cm_types[type_idx]
            for key in cm_type:
                if type(cm_type[key]) != list and key in sub_df.columns:
                    sub_df.loc[sub_df.index[idx], key] = cm_type[key]

    osm_processor._append_to_df(sub_df)
        

def assign_road_tiles_to_network(osm_processor, config, name):
    assign_tiles_to_network(osm_processor, config, name, road_tiles)

def assign_rail_tiles_to_network(osm_processor, config, name):
    assign_tiles_to_network(osm_processor, config, name, rail_tiles)

def assign_stream_tiles_to_network(osm_processor, config, name):
    assign_tiles_to_network(osm_processor, config, name, stream_tiles)

def assign_fence_tiles_to_network(osm_processor, config, name):
    assign_tiles_to_network(osm_processor, config, name, fence_tiles)

def assign_tiles_to_network(osm_processor, config, name, tile_df):
    logger = logging.getLogger('osm2cm')
    square_graph = osm_processor.network_graphs[name]['square_graph']
    edge_graphs = {}
    for edge in square_graph.edges:
        graph = nx.DiGraph()
        node1_id = edge[0]
        node2_id = edge[1]
        squares = square_graph.edges[edge]['squares']
        if not _check_if_squares_are_valid(squares):
            logging.warn('Squares of edge {} -> {} ({}) are invalid'.format(*edge))
            continue

        other_node1_neighbors = [neighbor for neighbor in nx.neighbors(square_graph, node1_id) if neighbor != node2_id or node1_id == node2_id]
        other_node2_neighbors = [neighbor for neighbor in nx.neighbors(square_graph, node2_id) if neighbor != node1_id or node1_id == node2_id]

        if len(squares) < 2:
            continue

        if squares[0] == node2_id:
            squares = squares[::-1]

        last_tiles = []
        graph.add_node('start')
        graph.add_node('end')
        for i_square in range(len(squares)):
            square = squares[i_square]
            direction_in = None
            direction_out = None
            if i_square < len(squares) - 1:
                direction_out = road_direction_dict[(squares[i_square + 1][0] - square[0], squares[i_square + 1][1] - square[1])]
            if i_square > 0:
                direction_in = road_direction_dict[(squares[i_square - 1][0] - square[0], squares[i_square - 1][1] - square[1])]

            if i_square == 0:
                # if square_graph.degree(edge[0]) == 1:
                #     direction_in = opposite_road_direction_dict[direction_out]
                #     valid_tiles = tile_df[(tile_df[direction_in] == (2,3)) & ~pandas.isnull(tile_df[direction_out]) & (tile_df['n_connections'] == 2)].index.values
                # else:
                    # loop = 'start' if node1_id == node2_id else None
                    # directions_in = extract_required_directions(node1_id, other_node1_neighbors, square_graph, loop)
                    # connection_condition = ~pandas.isnull(tile_df[directions_in[0]]) & (tile_df['n_connections'] == len(directions_in) + 1)
                    # for dir_idx in range(1, len(directions_in)):
                    #     connection_condition = np.bitwise_and(connection_condition, ~pandas.isnull(tile_df[directions_in[dir_idx]]))
                    #
                    # valid_tiles = tile_df[~pandas.isnull(tile_df[direction_out]) & connection_condition].index.values

                directions = extract_directions(node1_id, square_graph)
                if len(directions) == 1:
                    directions.append(opposite_road_direction_dict[directions[0]])
                condition = tile_df['n_connections'] == len(directions)
                for direction in directions:
                    condition = condition & ~pandas.isnull(tile_df[direction])

                valid_tiles = tile_df.loc[condition].index.values
                    
                last_tiles = valid_tiles
                for tile in valid_tiles:
                    graph.add_edge('start', (tile, i_square), cost=tile_df.loc[tile, 'cost'])
            elif 0 < i_square < len(squares) - 1:
                last_direction_out = opposite_road_direction_dict[direction_in]
                new_last_tiles = []
                for last_tile in last_tiles:
                    tile_connection = tile_df.loc[last_tile, last_direction_out]
                    valid_tiles = tile_df.loc[(tile_df[direction_in] == tile_connection) & ~pandas.isnull(tile_df[direction_out]) & (tile_df['n_connections'] == 2)].index.values
                    for tile in valid_tiles:
                        new_last_tiles.append(tile)
                        graph.add_edge((last_tile, i_square - 1), (tile, i_square), cost=tile_df.loc[tile, 'cost'])

                if len(new_last_tiles) == 0:
                    logger.debug('Found no possible continuation of network edge between nodes {} and {} at square index {}.'.format(node1_id, node2_id, i_square))
                    logger.debug('    Last tiles were {}, new square has incoming direction {} and outgoing direction {}.'. format(
                        tile_df.loc[last_tiles, ['direction', 'row', 'col', 'variant']].values, direction_in, direction_out)
                    )
                    break
                last_tiles = np.unique(new_last_tiles)

            else:
                last_direction_out = opposite_road_direction_dict[direction_in]
                for last_tile in last_tiles:
                    tile_connection = tile_df.loc[last_tile, last_direction_out]
                    # if square_graph.degree(edge[1]) == 1:
                    #     direction_out = opposite_road_direction_dict[direction_in]
                    #     valid_tiles = tile_df.loc[(tile_df[direction_in] == tile_connection) & (tile_df[direction_out] == (2,3)) & (tile_df['n_connections'] == 2)].index.values
                    # else:
                        # loop = 'end' if node1_id == node2_id else None
                        # directions_out = extract_required_directions(node2_id, other_node2_neighbors, square_graph, loop)
                        # connection_condition = ~pandas.isnull(tile_df[directions_out[0]]) & (tile_df['n_connections'] == len(directions_out) + 1)
                        # for dir_idx in range(1, len(directions_out)):
                        #     connection_condition = np.bitwise_and(connection_condition, ~pandas.isnull(tile_df[directions_out[dir_idx]]))

                        # valid_tiles = tile_df.loc[(tile_df[direction_in] == tile_connection) & connection_condition].index.values

                    directions = extract_directions(node2_id, square_graph)
                    if len(directions) == 1:
                        directions.append(opposite_road_direction_dict[directions[0]])
                    condition = (tile_df['n_connections'] == len(directions)) & (tile_df[direction_in] == tile_connection)
                    for direction in directions:
                        condition = condition & ~pandas.isnull(tile_df[direction])

                    valid_tiles = tile_df.loc[condition].index.values

                    for tile in valid_tiles:
                        graph.add_edge((last_tile, i_square - 1), (tile, i_square), cost=tile_df.loc[tile, 'cost'])
                        graph.add_edge((tile, i_square), 'end')

        edge_graphs[edge] = {'graph': graph, 'squares': squares}


    valid_tiles_dict = {}
    for node in square_graph.nodes:
        valid_tiles_dict[node] = extract_all_valid_node_tiles(node, square_graph, edge_graphs)

    gdf = osm_processor.gdf

    for edge in edge_graphs:
        node1_id, node2_id, edge_key = edge

        if node1_id not in valid_tiles_dict or len(valid_tiles_dict[node1_id]) == 0:
            continue
        if node2_id not in valid_tiles_dict or len(valid_tiles_dict[node2_id]) == 0:
            continue

        graph = edge_graphs[edge]['graph']
        squares = edge_graphs[edge]['squares']

        if not nx.has_path(graph, 'start', 'end'):
            continue

        valid_start_tiles = valid_tiles_dict[node1_id]
        valid_end_tiles = valid_tiles_dict[node2_id]

        edge_end_idx = list(graph.in_edges('end'))[0][0][1]

        valid_paths = []
        for start_tile in valid_start_tiles:
            for end_tile in valid_end_tiles:
                if (start_tile, 0) not in graph.nodes or (end_tile, edge_end_idx) not in graph.nodes:
                    logger.debug('Start or end tile are not in path. This should not happen!')
                    continue
                if nx.has_path(graph, (start_tile, 0), (end_tile, edge_end_idx)):
                    # only return one valid path!
                    valid_paths.append(nx.shortest_path(graph, (start_tile, 0), (end_tile, edge_end_idx), weight='cost'))
                    # paths = nx.all_shortest_paths(graph, (start_tile, 0), (end_tile, edge_end_idx), weight='cost')
                    # valid_paths = [path for path in paths]
                    # valid_paths.extend([path for path in nx.all_shortest_paths(graph, (start_tile, 0), (end_tile, edge_end_idx), weight='cost')])

        if len(valid_paths) == 0:
            continue

        path = valid_paths[0]

        sub_df = osm_processor._get_sub_df(gdf.xidx.isin([sq[0] for sq in squares]) & gdf.yidx.isin([sq[1] for sq in squares]))
        element_idx = square_graph.get_edge_data(*edge)['element_idx']
        element_entry = osm_processor.matched_elements[element_idx]
        cm_type = get_matched_cm_type(config, element_entry)

        if cm_type is None:
            print('Warning: Could not find matching CM type for path elements between {} and {}.'.format(node1_id, node2_id))
            continue

        for idx in range(len(path)):
            square = squares[idx]
            direction, row, col = tile_df.loc[path[idx][0], ['direction', 'row', 'col']]

            # tile_condition = (sub_df.xidx == square[0]) & (sub_df.yidx == square[1]) & (sub_df.name == name)
            tile_condition = (sub_df.xidx == square[0]) & (sub_df.yidx == square[1])

            sub_df.loc[tile_condition, 'menu'] = cm_type['menu']
            sub_df.loc[tile_condition, 'cat1'] = cm_type['cat1']
            sub_df.loc[tile_condition, 'direction'] = 'Direction {}'.format(direction + 1)
            sub_df.loc[tile_condition, 'cat2'] = 'Road Tile {}'.format(row * 3 + col + 1)
            sub_df.loc[tile_condition, 'priority'] = config[name]['priority']
        
        osm_processor._append_to_df(sub_df)

        valid_tiles_dict[node1_id] = [path[0][0]]
        valid_tiles_dict[node2_id] = [path[-1][0]]

        for square in squares:
            polygon = osm_processor.gdf[(osm_processor.gdf.xidx == square[0]) & (osm_processor.gdf.yidx == square[1])].geometry.values[0]
            occupancy_entry = geopandas.GeoDataFrame({'geometry': [polygon], 'priority': [config[name]['priority']], 'name': name})
            osm_processor.occupancy_gdf = pandas.concat((osm_processor.occupancy_gdf, occupancy_entry), ignore_index=True)

        # if len(node1_neighbors) > 0:
        # fix_tile_for_node(node1_id, edge_graphs, path[0][0])
        # if len(node2_neighbors) > 0:
        # fix_tile_for_node(node2_id, edge_graphs, path[-1][0])
        
        valid_paths = []
        a = 1

    # df = df.drop(df.loc[(df.name == name) & (df.direction == -1)].index)

def remove_duplicate_linestring_coordinates(ls):
    unique_coords, unique_coord_counts = np.unique(ls.coords, axis=0, return_counts=True)
    linestrings = [ls]
    if (unique_coord_counts > 1).any():
        duplicate_idx = np.where(unique_coord_counts > 1)[0]
        duplicate_coords = unique_coords[duplicate_idx][0]
        didx = np.where((ls.xy[0] == duplicate_coords[0]) & (ls.xy[1] == duplicate_coords[1]))[0]
        didx = didx[(didx > 0) & (didx < len(ls.xy[0]) - 1)]
        if len(didx) > 0:
            linestrings = [LineString(ls.coords[0:didx[0]+1]), LineString(ls.coords[didx[0]:])]
        else:
            pass
    return linestrings

def collect_network_data(osm_processor, config, element_entry):
    logger = logging.getLogger('osm2cm')
    name = element_entry['name']
    if element_entry['name'] not in osm_processor.network_graphs:
        osm_processor.network_graphs[name] = {}

    geometry = element_entry['geometry']
    geometry = osm_processor.effective_bbox_polygon.intersection(geometry)

    linestrings = []
    if type(geometry) == LineString:
        ls = geometry 
        linestrings = remove_duplicate_linestring_coordinates(ls)
    elif type(geometry) == MultiLineString:
        for ls in geometry.geoms:
            linestrings.extend(remove_duplicate_linestring_coordinates(ls))
    elif type(geometry) == Polygon:
        exterior_ls = LineString(geometry.exterior.coords)
        # closed rings don't work with the rest of the tooling...
        # linestrings.extend(remove_duplicate_linestring_coordinates(substring(exterior_ls, 0, 0.5, normalized=True)))
        # linestrings.extend(remove_duplicate_linestring_coordinates(substring(exterior_ls, 0.5, 1.0, normalized=True)))
        # for interior in geometry.interiors:
        #     linestrings.extend(remove_duplicate_linestring_coordinates(substring(interior, 0, 0.5, normalized=True)))
        #     linestrings.extend(remove_duplicate_linestring_coordinates(substring(interior, 0.5, 1.0, normalized=True)))
        linestrings.extend(remove_duplicate_linestring_coordinates(exterior_ls))
        for interior in geometry.interiors:
            linestrings.extend(remove_duplicate_linestring_coordinates(interior))

    else:
        logger.warn('Network geometry should be LineString but found {}.'.format(type(geometry)))
        return

    if len(linestrings) > 0:
        element_entry_indices = [element_entry['idx']] * len(linestrings)

        element_gdf = geopandas.GeoDataFrame({'element_idx': element_entry_indices, 'geometry': linestrings})
        if 'lines' not in osm_processor.network_graphs[name]:
            osm_processor.network_graphs[name]['lines'] = element_gdf
        else:
            osm_processor.network_graphs[name]['lines'] = pandas.concat((osm_processor.network_graphs[name]['lines'], element_gdf), ignore_index=True)
        
def create_square_graph_path_search(osm_processor, config, name):
    # search_path(osm_processor.network_graphs[name]['line_graph'], osm_processor.gdf, osm_processor.df)
    search_path(osm_processor, config, name)

def collect_building_outlines(osm_processor, config, element_entry):
    logger = logging.getLogger('osm2cm')

    grid_gdf = osm_processor.sub_square_grid_gdf
    geometry = element_entry['geometry']
    if not geometry.intersects(osm_processor.effective_bbox_polygon):
        return

    if 'raw_outlines' not in osm_processor.building_outlines:
        osm_processor.building_outlines['raw_outlines'] = {}

    min_rot_rectangle = geometry.minimum_rotated_rectangle
    if min_rot_rectangle.area == 0:
        logger.debug('Minimum rotated building outline (element idx {}) has 0 area.'.format(element_entry['idx']))
        return
        
    scaled_min_rot_rectangle = scale(min_rot_rectangle, np.sqrt(geometry.area / min_rot_rectangle.area), np.sqrt(geometry.area / min_rot_rectangle.area))

    llc_rectangle = _get_rectangle_coords_starting_at_lower_left_corner(scaled_min_rot_rectangle)
    llc_rectangle_coords = list(llc_rectangle.exterior.coords)
    base_angle = np.arctan2(llc_rectangle_coords[1][1] - llc_rectangle_coords[0][1], llc_rectangle_coords[1][0] - llc_rectangle_coords[0][0])

    # The coordinate system may be rotated so calculate the angle of the x-axis and correct base angle accoringly
    origin = grid_gdf.loc[(grid_gdf.xidx == grid_gdf.xidx.min()) & (grid_gdf.yidx == grid_gdf.yidx.min()), ['x', 'y']].values[0]
    p_xaxis_max = grid_gdf.loc[(grid_gdf.xidx == grid_gdf.xidx.max()) & (grid_gdf.yidx == grid_gdf.yidx.min()), ['x', 'y']].values[0]
    axis_angle = np.arctan2(p_xaxis_max[1] - origin[1], p_xaxis_max[0] - origin[0])

    base_angle = (base_angle - axis_angle) % (2 * np.pi) - np.pi

    osm_processor.building_outlines['raw_outlines'][element_entry['idx']] = (geometry, base_angle)

    # if np.abs(base_angle) < np.pi / 8:
    #     is_diagonal = False
    # else:
    #     is_diagonal = True

def process_building_outlines(osm_processor, config, name):
    logger = logging.getLogger('osm2cm')
    raw_outlines = osm_processor.building_outlines['raw_outlines']

    diag_grid_gdf = osm_processor.sub_square_grid_diagonal_gdf
    grid_gdf = osm_processor.sub_square_grid_gdf

    diagonal_bounds = [diag_grid_gdf.xidx.min(), diag_grid_gdf.yidx.min(), diag_grid_gdf.xidx.max(), diag_grid_gdf.yidx.max()]
    square_bounds = [grid_gdf.xidx.min(), grid_gdf.yidx.min(), grid_gdf.xidx.max(), grid_gdf.yidx.max()]

    grid_vertices = []
    for g in osm_processor.gdf.geometry.values:
        grid_vertices.append([coord for coord in g.exterior.coords])
    occupancy_vertices = []
    for g in osm_processor.occupancy_gdf.geometry.values:
        occupancy_vertices.append([coord for coord in g.exterior.coords])

    plt.figure()
    plt.axis('equal')
    ax = plt.gca()
    grid_collection = PolyCollection(grid_vertices, closed=False, edgecolor='k')
    occupancy_collection = PolyCollection(occupancy_vertices, closed=False, edgecolor='g', facecolor='g')
    ax.add_collection(grid_collection)
    ax.add_collection(occupancy_collection)
    for element_idx, outline_entry in raw_outlines.items():
        plt.plot(*outline_entry[0].exterior.xy, '-m')
        matched_tiles_candidates = []
        for is_diagonal in [True, False]:
            squares = _get_matched_squares(osm_processor, config[name]['priority'], outline_entry[0], is_diagonal)
            if len(squares) == 0:
                continue

            if is_diagonal:
                matching_gdf = diag_grid_gdf
                bounds = diagonal_bounds
            else:
                matching_gdf = grid_gdf
                bounds = square_bounds

            matched_square_rowcols = np.array([_idx2rowcol(d[0], d[1], bounds, is_diagonal) for d in squares.loc[:,['xidx', 'yidx']].values])
            match_polygon = np.zeros(
                (
                    int(matched_square_rowcols[:,0].max() - matched_square_rowcols[:,0].min() + 1), 
                    int(matched_square_rowcols[:,1].max() - matched_square_rowcols[:,1].min() + 1), 
                )
            )
            for rowcol in matched_square_rowcols:
                match_polygon[int(rowcol[0] - matched_square_rowcols[:,0].min()), int(rowcol[1] - matched_square_rowcols[:,1].min())] = 1

            tiles = list(set([(t[0], t[1]) for t in buildings[buildings.is_diagonal].loc[:,['width', 'height']].values]))
            solution = branch_and_bound(match_polygon, tiles)

            if len(solution[2]) == 0:
                continue

            if is_diagonal:
                area = sum([sol[0][0] * sol[0][1] * 2 * 0.25 * 64 for sol in solution[2]])
            else:
                area = sum([sol[0][0] * sol[0][1] * 0.25 * 64 for sol in solution[2]])

            matched_tiles_solution = []
            occupied_polygons = []
            for sol in solution[2]:
                matched_tiles_solution.append((sol[0], _rowcol2idx(sol[1] + matched_square_rowcols[:,0].min(), sol[2] + matched_square_rowcols[:,1].min(), bounds, is_diagonal)))
                occupied_squares = []
                for s in sol[3]:
                    occupied_idx_square = (_rowcol2idx(s[0] + matched_square_rowcols[:,0].min(), s[1] + matched_square_rowcols[:,1].min(), bounds, is_diagonal))
                    condition = (matching_gdf.xidx == occupied_idx_square[0]) & (matching_gdf.yidx == occupied_idx_square[1])
                    if not condition.any():
                        continue
                    occupied_square = matching_gdf[condition].geometry.values[0]
                    occupied_squares.append(occupied_square)
                occupied_polygons.append(MultiPolygon(occupied_squares).buffer(0))

            intersection_over_union = unary_union(occupied_polygons).intersection(outline_entry[0]).area / unary_union(occupied_polygons + [outline_entry[0]]).area

            matched_tiles_candidates.append((is_diagonal, matched_tiles_solution, intersection_over_union, squares, occupied_polygons))

        if len(matched_tiles_candidates) == 0:
            continue

        matched_tiles_candidates = sorted(matched_tiles_candidates, key=lambda x: -x[2])

        matched_tiles = matched_tiles_candidates[0]
        
        for g in matched_tiles[3].geometry:
            plt.plot(*g.exterior.xy, '-k')
        
        is_diagonal = matched_tiles[0]

        if len(matched_tiles) > 1:
            condition = (buildings.menu == 'Modular Buildings') & (buildings.is_diagonal == is_diagonal)
        else:
            condition = buildings.is_diagonal == is_diagonal

        matched_buildings = []
        for mt in matched_tiles[1]:
            building_candidates = buildings[condition & (buildings.width == mt[0][1]) & (buildings.height == mt[0][0])]
            building = building_candidates.sample(n=1, weights=building_candidates.weight)
            matched_buildings.append(building)

        ############
        if is_diagonal:
            matching_gdf = osm_processor.sub_square_grid_diagonal_gdf
        else:
            matching_gdf = grid_gdf

        for square in matched_tiles[4]:
            occupancy_entry = geopandas.GeoDataFrame({'geometry': [square], 'priority': [config[name]['priority']], 'name': name})
            osm_processor.occupancy_gdf = pandas.concat((osm_processor.occupancy_gdf, occupancy_entry), ignore_index=True)
            plt.plot(*square.exterior.xy, '-r')

        if len(matched_buildings) == 0:
            logger.debug('No matching CM building was found for building {}. Area: {}'.format(element_entry['idx'], geometry.area))

        for bidx, building in enumerate(matched_buildings):
            llc_idx = matched_tiles[1][bidx][1]

            condition = (matching_gdf.xidx == llc_idx[0]) & (matching_gdf.yidx == llc_idx[1])

            sub_df = osm_processor._get_sub_df(condition, matching_gdf)

            direction, row, col, menu, cat1 = building[['direction', 'row', 'col', 'menu', 'cat1']].values[0]

            sub_df['menu'] = menu
            sub_df['cat1'] = cat1
            sub_df['direction'] = 'Direction {}'.format(direction + 1)
            sub_df['cat2'] = 'Building {}'.format(row * 4 + col + 1)
            sub_df['priority'] = config[name]['priority']
            
            osm_processor._append_to_df(sub_df)
    
    plt.show()

def _get_matched_squares(osm_processor, priority, geometry, diagonal):
    if diagonal:
        diag_grid_gdf = osm_processor.sub_square_grid_diagonal_gdf
        diamonds = diag_grid_gdf.geometry
        idx = diamonds.sindex.query(geometry, predicate='intersects')
        intersecting_diamonds = diamonds.iloc[idx].geometry
        oidx = osm_processor.occupancy_gdf.loc[osm_processor.occupancy_gdf.priority <= priority].sindex.query_bulk(intersecting_diamonds.geometry, predicate='intersects')
        area_oidx = []
        for oi in range(len(oidx[0])):
            if intersecting_diamonds.iloc[oidx[0][oi]].intersection(osm_processor.occupancy_gdf.iloc[oidx[1][oi]].geometry).area > 0.0:
                area_oidx.append(oidx[0][oi])

        intersecting_diamonds = intersecting_diamonds.drop(intersecting_diamonds.iloc[area_oidx].index)
        idx = intersecting_diamonds.index
        return diag_grid_gdf.iloc[idx]
    else:
        square_grid_gdf = osm_processor.sub_square_grid_gdf
        squares = square_grid_gdf.geometry
        idx = squares.sindex.query(geometry, predicate='intersects')
        intersecting_squares = intersecting_squares = squares.iloc[idx]
        oidx = osm_processor.occupancy_gdf.loc[osm_processor.occupancy_gdf.priority <= priority].sindex.query_bulk(intersecting_squares.geometry, predicate='intersects')
        area_oidx = []
        for oi in range(len(oidx[0])):
            if intersecting_squares.iloc[oidx[0][oi]].intersection(osm_processor.occupancy_gdf.iloc[oidx[1][oi]].geometry).area > 0.0:
                area_oidx.append(oidx[0][oi])

        intersecting_squares = intersecting_squares.drop(intersecting_squares.iloc[area_oidx].index)
        idx = intersecting_squares.index
        return square_grid_gdf.iloc[idx]

def _get_rectangle_coords_starting_at_lower_left_corner(rectangle):
    # make counter-clock wise (though should alread be)
    exterior = rectangle.exterior
    if exterior.is_ccw:
        coords = list(rectangle.exterior.coords)
    else:
        coords = list(rectangle.exterior.coords)[::-1]
    
    if coords[0] == coords[-1]:
        coords = coords[0:-1]

    # centroid = rectangle.centroid
    # # get coords left of center:
    # left_coords = list({coord for coord in coords if coord[0] < centroid.x})
    # # get lower one:
    # lower_left_coord = sorted(left_coords, key=lambda c: c[1])[0]
    # lower_left_coord_index = coords.index(lower_left_coord)

    # coordinates with min x value
    left_idx = np.where([coord[0] for coord in coords] == np.min([coord[0] for coord in coords]))[0]
    left_coords = np.array(coords)[left_idx]

    # get lower one:
    lower_left_coord = tuple(sorted(left_coords, key=lambda c: c[1])[0].tolist())
    lower_left_coord_index = coords.index(lower_left_coord)

    new_coords = coords[lower_left_coord_index::] + coords[0:lower_left_coord_index]
    return Polygon(new_coords)

def _match_building(rectangle, grid_gdf, is_diagonal, stories=None):
    logger = logging.getLogger('osm2cm')

    building_candidates = buildings.loc[buildings.is_diagonal == is_diagonal]

    p0 = rectangle.exterior.coords[0]

    index_p0, p0idx_on_grid, p0_on_grid = _get_closest_node_in_gdf(grid_gdf, rectangle.exterior.coords[0], return_xy=True, return_gdf_index=True)

    rectangle = translate(rectangle, p0_on_grid[0] - p0[0], p0_on_grid[1] - p0[1])
    p1 = rectangle.exterior.coords[1]
    p2 = rectangle.exterior.coords[2]

    p1_rel = ((p1[0] - p0_on_grid[0]) / 8, (p1[1] - p0_on_grid[1]) / 8)
    p2_rel = ((p2[0] - p0_on_grid[0]) / 8, (p2[1] - p0_on_grid[1]) / 8)

    first_side = np.linalg.norm(p1_rel)
    second_side = np.linalg.norm((p2_rel[0] - p1_rel[0], p2_rel[1] - p1_rel[1]))

    first_side_building_candidates = np.linalg.norm((building_candidates.x1, building_candidates.y1), axis=0)
    second_side_building_candidates = np.linalg.norm((building_candidates.x2 - building_candidates.x1, building_candidates.y2 - building_candidates.y1), axis=0)

    matched_buildings = []
    # first check if the first side is larger than any of the available buildings
    first_side_decomposition = get_side_decomposition(first_side, first_side_building_candidates)
    second_side_decomposition = get_side_decomposition(second_side, second_side_building_candidates)

    x_norms = []
    y_norms = []
    for entry in first_side_decomposition:
        x_norms.extend(int(entry[0]) * [entry[1]])
    for entry in second_side_decomposition:
        y_norms.extend(int(entry[0]) * [entry[1]])

    lower_left_corner_vector = list(p0idx_on_grid)
    for y_norm_idx, y_norm in enumerate(y_norms):
        row_vector = [0, 0]
        for x_norm_idx, x_norm in enumerate(x_norms):
            condition = (first_side_building_candidates == x_norm) & (second_side_building_candidates == y_norm)
            if (first_side_building_candidates < first_side).all():
                condition = condition & (building_candidates.menu == 'Modular Buildings')
            
            if len(building_candidates[condition]) == 0:
                continue
            building_segment = building_candidates[condition].sample(n=1, weights=building_candidates.weight)

            matched_buildings.append((
                (
                    lower_left_corner_vector[0] + row_vector[0], 
                    lower_left_corner_vector[1] + row_vector[1]
                ), 
                building_segment
            ))

            if x_norm_idx == len(x_norms) - 1:
                lower_left_corner_vector[0] += building_segment.x2.values[0] - building_segment.x1.values[0]
                lower_left_corner_vector[1] += building_segment.y2.values[0] - building_segment.y1.values[0]
            else:
                row_vector[0] += building_segment.x1.values[0]
                row_vector[1] += building_segment.y1.values[0]

            a = 1
    # else:
    #     x1_val_candidates = building_candidates.x1.unique()
    #     p1x = x1_val_candidates[np.argmin(np.abs(x1_val_candidates - p1_rel[0]))]
    #     building_candidates = building_candidates.loc[building_candidates.x1 == p1x]

    #     y1_val_candidates = building_candidates.y1.unique()
    #     p1y = y1_val_candidates[np.argmin(np.abs(y1_val_candidates - p1_rel[1]))]
    #     building_candidates = building_candidates.loc[building_candidates.y1 == p1y]

    #     x2_val_candidates = building_candidates.x2.unique()
    #     p2x = x2_val_candidates[np.argmin(np.abs(x2_val_candidates - p2_rel[0]))]
    #     building_candidates = building_candidates.loc[building_candidates.x2 == p2x]

    #     y2_val_candidates = building_candidates.y2.unique()
    #     p2y = y2_val_candidates[np.argmin(np.abs(y2_val_candidates - p2_rel[1]))]
    #     building_candidates = building_candidates.loc[building_candidates.y2 == p2y]

    #     if stories is not None:
    #         building_candidates = building_candidates.loc[building_candidates.stories == stories]

    #     building_segment = building_candidates.sample(n=1, weights=building_candidates.weight)
    #     matched_buildings = [(p0idx_on_grid, building_segment)]

    if len(matched_buildings) == 0:
        logger.debug('  sides: {}, {}, decomposition: {}, {}'.format(first_side, second_side, first_side_decomposition, second_side_decomposition))

    return matched_buildings
    # return index_p0, p0idx_on_grid, p0_on_grid, building_candidates.sample(n=1, weights=building_candidates.weight)

def _check_if_squares_are_valid(squares):
    valid = True
    for idx in range(len(squares)-1):
        valid = valid and (squares[idx + 1][0] - squares[idx][0], squares[idx + 1][1] - squares[idx][1]) in road_direction_dict

    return valid

def get_side_decomposition(side_length, side_candidates):
    # TODO: handle situations where not the *entire* lengths can be filled but a part of it

    # sort available segment lengths in descending order
    unique_segment_lengths = np.sort(np.unique(side_candidates))[::-1]

    # divide side_length by available segment lengths and get those that fit in at least once
    partition = np.divmod(side_length, unique_segment_lengths)

    # side_decomposition = []
    # for sidx, segment_length in enumerate(unique_segment_lengths):
    #     if partition[0][sidx] >= 1 and partition[1][sidx] == 0:
    #         side_decomposition.append((partition[0][sidx], segment_length))
    #         break
    #     elif partition[0][sidx] >= 1 and partition[1][sidx] > 0 and sidx < len(unique_segment_lengths) - 1:
    #         side_decomposition.extend(get_side_decomposition(side_length - partition[0][sidx] * segment_length, unique_segment_lengths[sidx+1:]))

    min_diff = np.inf
    min_diff_list = []
    for factor_tuple in itertools.product(*[range(a,-1,-1) for a in partition[0].astype(int)]):
        diff = side_length - sum([factor_tuple[idx] * unique_segment_lengths[idx] for idx in range(len(factor_tuple))])
        if diff == 0:
            return [(factor_tuple[idx], unique_segment_lengths[idx]) for idx in range(len(factor_tuple))]
        elif diff > 0 and diff < min_diff:
            min_diff = diff
            min_diff_list = [(factor_tuple[idx], unique_segment_lengths[idx]) for idx in range(len(factor_tuple))]

    return min_diff_list

def single_object_random(osm_processor, config, element_entry):
    cm_types = config[element_entry['name']]['cm_types']
    n_types = len(cm_types)
    df = osm_processor.df

    grid_cell = osm_processor.gdf.sindex.query(element_entry["geometry"], predicate='within')
    if len(grid_cell) == 0:
        return
    

    sub_df = osm_processor._get_sub_df(grid_cell)

    sum_of_weights = sum([cm_type['weight'] if 'weight' in cm_type else 1.0 for cm_type in cm_types])
    probabilities = [cm_type['weight'] / sum_of_weights if 'weight' in cm_type else 1.0 / sum_of_weights for cm_type in cm_types]

    rng = np.random.default_rng()
    type_indices = rng.choice(list(range(n_types)), p=probabilities, size=(len(sub_df),))

    # type_indices = np.random.randint(0, n_types, size=(len(sub_df),))
    for type_idx in range(n_types):
        idx = np.where(type_indices == type_idx)[0]
        if len(idx) > 0:
            cm_type = cm_types[type_idx]
            for key in cm_type:
                if type(cm_type[key]) != list and key in sub_df.columns:
                    sub_df.loc[sub_df.index[idx], key] = cm_type[key]

    osm_processor._append_to_df(sub_df)


from typing import List, Tuple

def branch_and_bound(polygon: np.array, tiles: List) -> Tuple[List[List[int]], int]:
    best_solution = (polygon, float('inf'))
    def dfs(polygon, total_cost, used_tiles):
        nonlocal best_solution
        if total_cost >= best_solution[1]:
            return
        best_solution = (polygon, total_cost, used_tiles)
        for i in range(polygon.shape[0]):
            for j in range(polygon.shape[1]):
                if polygon[i, j] == 1:
                    for tile in tiles:
                        if i + tile[0]-1 < polygon.shape[0] and j + tile[1]-1 < polygon.shape[1] and not (polygon[i:i + tile[0], j:j + tile[1]] == 0).any():
                            polygon = np.copy(polygon)
                            polygon[i:i + tile[0], j:j + tile[1]] = 0
                            used_tiles = deepcopy(used_tiles)
                            used_tiles.append((tile, i, j, [(ii, jj) for ii in range(i, i + tile[0]) for jj in range(j, j + tile[1])]))
                            dfs(polygon, len(used_tiles) + 1 + len(np.where(polygon == 1)[0]), used_tiles)

    dfs(polygon, len(np.where(polygon == 1)[0]), [])
    return best_solution

def _idx2rowcol(xidx, yidx, bounds, is_diagonal):
    xmin, ymin, xmax, ymax = bounds
    if is_diagonal:
        origin = (np.ceil(ymax - ymin) / 2 + ymin, -np.ceil(ymax - ymin) / 2 + ymin)
        row = yidx - origin[0] + xidx - origin[1]
        col = -(yidx - origin[0] - xidx + origin[1])
    else:
        row = (yidx - ymin) * 2
        col = (xidx - xmin) * 2
    
    return row, col

def _rowcol2idx(row, col, bounds, is_diagonal):
    xmin, ymin, xmax, ymax = bounds
    if is_diagonal:
        origin = (np.ceil(ymax - ymin) / 2 + ymin, -np.ceil(ymax - ymin) / 2 + ymin)
        xidx = (row + col) / 2 + origin[1]
        yidx = (row - col) / 2 + origin[0]
    else:
        yidx = row / 2 + ymin
        xidx = col / 2 + xmin

    return xidx, yidx
