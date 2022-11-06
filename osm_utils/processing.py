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
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString
from shapely.ops import split
import geopandas
import pandas
import logging
from profiles.general import road_tiles, rail_tiles, stream_tiles

road_direction_dict = {
    (0, 1): 'u',
    (0, -1): 'd',
    (1, 0): 'r',
    (-1, 0): 'l',
}

opposite_road_direction_dict = {
    'u': 'd',
    'd': 'u',
    'r': 'l',
    'l': 'r'
}


def get_grid_cells_to_fill(gdf, geometry):
    within = gdf.index.isin(gdf.sindex.query(geometry, predicate='contains'))
    intersecting = gdf.index.isin(gdf.sindex.query(geometry, predicate='intersects'))
    # within = gdf.geometry.within(geometry)
    # intersecting = gdf.geometry.intersects(geometry)
    is_border = np.bitwise_and(intersecting, ~within)
    is_largest_square_area = gdf.index.isin((gdf.loc[is_border].geometry.intersection(geometry).area > 32).index)
    to_fill = np.bitwise_or(within, is_largest_square_area)

    return to_fill

def extract_connection_direction_from_node(node_id, other_node_id, road_graph):
    squares = road_graph.edges[node_id, other_node_id]['squares']
    node_square = node_id
    if squares[0] != node_square:
        squares = squares[::-1]
    
    return road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])]

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
    edges = [edge for edge in square_graph.edges(node_id)]
    valid_tiles = None
    for edge in edges:
        start_node = None
        if edge in edge_graphs:
            graph = edge_graphs[edge]['graph']
            if nx.has_path(graph, 'start', 'end'):
                start_node = 'start'
                end_node = 'end'
        elif edge[::-1] in edge_graphs:
            graph = nx.reverse_view(edge_graphs[edge[::-1]]['graph'])
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
    
def extract_required_directions(node_id, neighbors, road_graph):
    required_directions = []
    if len(neighbors) > 0:
        for neighbor in neighbors:
            direction_out = extract_connection_direction_from_node(node_id, neighbor, road_graph)

            required_directions.append(direction_out)

    return required_directions

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

    line_graph = nx.Graph()

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
                else:
                    logger.warning('Warning, intersection geometry is of type {}!'.format(type(p)))
            
            intersection_points = filtered_intersection_points
            intersection_points.extend([p for p in ls.boundary.geoms if p not in intersection_points])
            intersection_points = sorted(intersection_points, key=lambda p: ls.project(p))
        else:
            intersection_points = [p for p in ls.boundary.geoms]
        
        ls_splits = split(ls, MultiPoint(intersection_points))
        ls_splits = [geom for geom in ls_splits.geoms]

        for lidx, ls_split in enumerate(ls_splits):
            if type(ls_split) == LineString:
                p1 = Point(ls_split.coords[0])
                p2 = Point(ls_split.coords[-1])
                line_graph.add_edge((p1.x, p1.y), (p2.x, p2.y), ls=ls_split, element_idx=lines.iloc[line_idx].element_idx)
            else:
                print('Warning: linestring split was type {}.'.format(type(ls_split)))

    osm_processor.network_graphs[name]['line_graph'] = line_graph


def create_square_graph(osm_processor, config, name):
    line_graph = osm_processor.network_graphs[name]['line_graph']
    gdf = osm_processor.gdf
    square_graph = nx.Graph()

    for edge in line_graph.edges:
        edge_data = line_graph.edges[edge]
        ls = edge_data['ls']
        
        squares = gdf.loc[gdf.sindex.query(ls, predicate='intersects')]
        ls_intersection = squares.geometry.intersection(ls)
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
            gdf_point = gdf.loc[gdf.sindex.query(p, predicate='within')]
            if len(gdf_point) > 0:
                # xidx, yidx = gdf_ls.loc[gdf.contains(point), ['xidx', 'yidx']].values[0]
                xidx, yidx = gdf_point.loc[:, ['xidx', 'yidx']].values[0]
                squares_along_way.append((xidx, yidx))

        if len(squares_along_way) > 1:
            square_graph.add_edge(squares_along_way[0], squares_along_way[-1], squares=squares_along_way, element_idx=edge_data['element_idx'])
                
    
    nodes = list(square_graph.nodes)
    for node in nodes:
        edges = [edge for edge in square_graph.edges(node)]
        squares = []
        for edge in edges:
            squares.extend(square_graph.edges[edge]['squares'][1:-1])
        
        unique_squares, square_counts = np.unique(squares, axis=0, return_counts=True)
        if (square_counts > 1).any():
            duplicate_squares = unique_squares[np.where(square_counts > 1)[0]]
            for nb_node in list(square_graph.neighbors(node)):
                squares_to_nb = square_graph.edges[(node, nb_node)]['squares']
                element_idx = square_graph.edges[(node, nb_node)]['element_idx']
                if squares_to_nb[0] != node:
                    squares_to_nb = squares_to_nb[::-1]
                square_lines = []
                prev_node_idx = 0
                for sq_idx, sq in enumerate(squares_to_nb):
                    if ((duplicate_squares[:,0] == sq[0]) & (duplicate_squares[:,1] == sq[1])).any() or sq_idx == len(squares_to_nb) - 1:
                        square_lines.append(squares_to_nb[prev_node_idx:sq_idx+1])
                        prev_node_idx = sq_idx

                if len(square_lines) > 1:
                    square_graph.remove_edge(node, nb_node)
                    for sq_line in square_lines:
                        square_graph.add_edge(sq_line[0], sq_line[-1], squares=sq_line, element_idx=element_idx)

    osm_processor.network_graphs[name]['square_graph'] = square_graph



def create_network_graph_old(df, gdf, element, cm_types):
    if len(df) == 0:
        return df

    if name not in road_graphs:
        road_graphs[name] = nx.MultiGraph()

    road_graph = road_graphs[name]

    coords = [(projection(coord[0], coord[1])) for coord in element.geometry()['coordinates']]
    points = [Point(coord[0], coord[1]) for coord in coords]

    ls = LineString(coords)
    # ls_crosses = np.bitwise_or(gdf.geometry.crosses(ls), gdf.geometry.contains(ls))

    # squares = gdf.loc[ls_crosses, ['x', 'y', 'xidx', 'yidx']]
    squares = gdf.loc[gdf.sindex.query(ls, predicate='intersects')]
    ls_intersection = squares.geometry.intersection(ls)
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
        intersection_mid_points = np.array(intersection_mid_points, dtype=object)
        intersection_mid_point_dist = [ls.project(mid_point) for mid_point in intersection_mid_points]
        sorted_indices = np.argsort(intersection_mid_point_dist)
        sorted_intersection_mid_points = intersection_mid_points[sorted_indices]
    else:
        sorted_intersection_mid_points = intersection_mid_points

    squares_along_way = []
    for p in sorted_intersection_mid_points:
        gdf_point = gdf.loc[gdf.sindex.query(p, predicate='within')]
        if len(gdf_point) > 0:
            # xidx, yidx = gdf_ls.loc[gdf.contains(point), ['xidx', 'yidx']].values[0]
            xidx, yidx = gdf_point.loc[:, ['xidx', 'yidx']].values[0]
            squares_along_way.append((xidx, yidx))

    # squares = squares.sort_index()

    # normalized_dists = []
    # s = gdf.loc[ls_crosses]
    # for poly in s.geometry:
    #     	plt.plot(poly.exterior.xy[0], poly.exterior.xy[1], '-k')
    # for idx in range(len(squares)):
    #     normalized_dists.append(ls.project(Point(squares['x'].values[idx], squares['y'].values[idx]), normalized=True))

    # df['dist_along_way'] = normalized_dists

    # gdf_ls = gdf.loc[ls_crosses]
    nodes = []
    for node_idx in range(element.countNodes()):
        node_id = element.nodes()[node_idx].id()
        point = Point(coords[node_idx])
        # gdf_point = gdf_ls.loc[gdf.contains(point)]
        gdf_point = gdf.loc[gdf.sindex.query(point, 'within')]
        if len(gdf_point) > 0:
            # xidx, yidx = gdf_ls.loc[gdf.contains(point), ['xidx', 'yidx']].values[0]
            xidx, yidx = gdf_point.loc[:, ['xidx', 'yidx']].values[0]
            nodes.append((node_id, xidx, yidx))

    # sorted_df = df.sort_values(by='dist_along_way')

    # remove duplicates
    # filtered_squares_along_way = []
    # for sidx in range(1, len(squares_along_way)):
    #     if squares_along_way[sidx] != squares_along_way[sidx - 1]:
    #         filtered_squares_along_way.append(squares_along_way[sidx])

    # squares_along_way = filtered_squares_along_way


    node_idx = 0
    current_squares = []
    current_nodes = []
    for sq_idx, sq in enumerate(squares_along_way):
        df_xidx, df_yidx = sq

        current_squares.append((df_xidx, df_yidx))
        if len(current_squares) > 1 and (np.abs(current_squares[-2][0] - current_squares[-1][0]) > 1 or np.abs(current_squares[-2][1] - current_squares[-1][1]) > 1):
            current_squares = [current_squares[-1]]
            current_nodes = []

        while node_idx < len(nodes) and nodes[node_idx][1] == df_xidx and nodes[node_idx][2] == df_yidx:
            node_id, node_xidx, node_yidx = nodes[node_idx]
            current_nodes.append((node_id, node_xidx, node_yidx))
            if len(current_nodes) == 2:
                road_graph.add_edge(current_nodes[0][0], current_nodes[1][0], squares=current_squares)
                square_diffs = [(current_squares[idx][0] - current_squares[idx-1][0], current_squares[idx][1] - current_squares[idx-1][1]) for idx in range(1, len(current_squares))]
                road_graph.nodes[current_nodes[0][0]]['square'] = current_nodes[0][1:3]
                road_graph.nodes[current_nodes[1][0]]['square'] = current_nodes[1][1:3]

                current_nodes = [current_nodes[1]]
                current_squares = [current_squares[-1]]
            
            node_idx += 1

        if len(current_nodes) == 0:
            current_nodes.append(('generic_{}'.format(globals()['generic_node_cnt']), df_xidx, df_yidx))
            globals()['generic_node_cnt'] += 1

        if sq_idx == len(squares_along_way) - 1 and len(current_squares) > 1 and len(current_nodes) == 1:
            current_nodes.append(('generic_{}'.format(globals()['generic_node_cnt']), df_xidx, df_yidx))
            globals()['generic_node_cnt'] += 1
            road_graph.add_edge(current_nodes[0][0], current_nodes[1][0], squares=current_squares)
            square_diffs = [(current_squares[idx][0] - current_squares[idx-1][0], current_squares[idx][1] - current_squares[idx-1][1]) for idx in range(1, len(current_squares))]
            road_graph.nodes[current_nodes[0][0]]['square'] = current_nodes[0][1:3]
            road_graph.nodes[current_nodes[1][0]]['square'] = current_nodes[1][1:3]
        
    return df


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

def assign_tiles_to_network(osm_processor, config, name, tile_df):
    logger = logging.getLogger('osm2cm')
    square_graph = osm_processor.network_graphs[name]['square_graph']
    edge_graphs = {}
    for edge in square_graph.edges:
        graph = nx.DiGraph()
        node1_id = edge[0]
        node2_id = edge[1]
        squares = square_graph.edges[node1_id, node2_id]['squares']

        other_node1_neighbors = [neighbor for neighbor in nx.neighbors(square_graph, node1_id) if neighbor != node2_id]
        other_node2_neighbors = [neighbor for neighbor in nx.neighbors(square_graph, node2_id) if neighbor != node1_id]

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
                if len(other_node1_neighbors) == 0:
                    direction_in = opposite_road_direction_dict[direction_out]
                    valid_tiles = tile_df[(tile_df[direction_in] == (2,3)) & ~pandas.isnull(tile_df[direction_out]) & (tile_df['n_connections'] == 2)].index.values
                else:
                    directions_in = extract_required_directions(node1_id, other_node1_neighbors, square_graph)
                    connection_condition = ~pandas.isnull(tile_df[directions_in[0]]) & (tile_df['n_connections'] == len(directions_in) + 1)
                    for dir_idx in range(1, len(directions_in)):
                        connection_condition = np.bitwise_and(connection_condition, ~pandas.isnull(tile_df[directions_in[dir_idx]]))

                    valid_tiles = tile_df[~pandas.isnull(tile_df[direction_out]) & connection_condition].index.values
                    
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
                    if len(other_node2_neighbors) == 0:
                        direction_out = opposite_road_direction_dict[direction_in]
                        valid_tiles = tile_df.loc[(tile_df[direction_in] == tile_connection) & (tile_df[direction_out] == (2,3)) & (tile_df['n_connections'] == 2)].index.values
                    else:
                        directions_out = extract_required_directions(node2_id, other_node2_neighbors, square_graph)
                        connection_condition = ~pandas.isnull(tile_df[directions_out[0]]) & (tile_df['n_connections'] == len(directions_out) + 1)
                        for dir_idx in range(1, len(directions_out)):
                            connection_condition = np.bitwise_and(connection_condition, ~pandas.isnull(tile_df[directions_out[dir_idx]]))

                        valid_tiles = tile_df.loc[(tile_df[direction_in] == tile_connection) & connection_condition].index.values

                    for tile in valid_tiles:
                        graph.add_edge((last_tile, i_square - 1), (tile, i_square), cost=tile_df.loc[tile, 'cost'])
                        graph.add_edge((tile, i_square), 'end')

        edge_graphs[(node1_id, node2_id)] = {'graph': graph, 'squares': squares}


    valid_tiles_dict = {}
    for node in square_graph.nodes:
        valid_tiles_dict[node] = extract_all_valid_node_tiles(node, square_graph, edge_graphs)

    gdf = osm_processor.gdf

    for node_pair in edge_graphs:
        node1_id, node2_id = node_pair

        if node1_id not in valid_tiles_dict or len(valid_tiles_dict[node1_id]) == 0:
            continue
        if node2_id not in valid_tiles_dict or len(valid_tiles_dict[node2_id]) == 0:
            continue

        graph = edge_graphs[node_pair]['graph']
        squares = edge_graphs[node_pair]['squares']

        if not nx.has_path(graph, 'start', 'end'):
            continue

        valid_start_tiles = valid_tiles_dict[node1_id]
        valid_end_tiles = valid_tiles_dict[node2_id]

        edge_end_idx = list(graph.in_edges('end'))[0][0][1]

        valid_paths = []
        for start_tile in valid_start_tiles:
            for end_tile in valid_end_tiles:
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
        element_idx = square_graph.get_edge_data(node1_id, node2_id)['element_idx']
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
        
        osm_processor._append_to_df(sub_df)

        valid_tiles_dict[node1_id] = [path[0][0]]
        valid_tiles_dict[node2_id] = [path[-1][0]]


        # if len(node1_neighbors) > 0:
        # fix_tile_for_node(node1_id, edge_graphs, path[0][0])
        # if len(node2_neighbors) > 0:
        # fix_tile_for_node(node2_id, edge_graphs, path[-1][0])
        
        valid_paths = []
        a = 1

    # df = df.drop(df.loc[(df.name == name) & (df.direction == -1)].index)
                        
def collect_network_data(osm_processor, config, element_entry):
    name = element_entry['name']
    if element_entry['name'] not in osm_processor.network_graphs:
        osm_processor.network_graphs[name] = {}

    if type(element_entry['geometry']) == LineString:
        ls = element_entry['geometry'] 
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
        
        element_entry_indices = [element_entry['idx']] * len(linestrings)




        element_gdf = geopandas.GeoDataFrame({'element_idx': element_entry_indices, 'geometry': linestrings})
        if 'lines' not in osm_processor.network_graphs[name]:
            osm_processor.network_graphs[name]['lines'] = element_gdf
        else:
            osm_processor.network_graphs[name]['lines'] = pandas.concat((osm_processor.network_graphs[name]['lines'], element_gdf), ignore_index=True)
    else:
        print('Warning, network geometry should be LineString but found {}.'.format(type(element_entry['geometry'])))
        
    
