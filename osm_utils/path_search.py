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

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, MultiPoint
from shapely.affinity import translate, scale
from profiles.general import fence_tiles
import pandas
import logging

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

direction_dict = dict((v,k) for k,v in road_direction_dict.items())

# def search_path(line_graph, grid_gdf, df):
def search_path(osm_processor, config, name):
    logger = logging.getLogger('osm2cm')
    
    line_graph = osm_processor.network_graphs[name]['line_graph']
    grid_gdf = osm_processor.gdf
    df = osm_processor.df 

    square_graph = nx.MultiGraph()

    # TODO: Check what happens if point can't be placed.
    # create grid graph
    xmax = grid_gdf.xidx.max()
    ymax = grid_gdf.yidx.max()

    grid_graph = nx.grid_2d_graph(xmax+1, ymax+1)
    for edge in grid_graph.edges:
        grid_graph.edges[edge]['cost'] = 1.0

    diagonal_edges1 = [((x, y), (x+1, y+1)) for x in range(xmax) for y in range(ymax)]
    diagonal_edges2 = [((x+1, y), (x, y+1)) for x in range(xmax) for y in range(ymax)]

    grid_graph.add_edges_from(diagonal_edges1, weight=np.sqrt(2))
    grid_graph.add_edges_from(diagonal_edges2, weight=np.sqrt(2))

    occupied_nodes = list({(v[0], v[1]) for v in df.loc[df.priority == 1, ['xidx', 'yidx']].values})

    grid_graph.remove_nodes_from(occupied_nodes)
    grid_subgraphs = [grid_graph.subgraph(c) for c in nx.connected_components(grid_graph)]

    valid_gdf = _remove_nodes_from_gdf(grid_gdf, occupied_nodes)

    line_graph_node_connections = {}
    for node, degree in line_graph.degree:
        line_graph_node_connections[node] = {}
        line_graph_node_connections[node]['connections'] = []
        line_graph_node_connections[node]['degree'] = degree

    # plt.figure()
    # plt.axis('equal')
    # plt.plot(valid_gdf.xidx, valid_gdf.yidx, 'k+')
    for edge in line_graph.edges:
        closest_node_to_start = _get_closest_node_in_gdf(valid_gdf, edge[0])
        # plt.plot(closest_node_to_start[0], closest_node_to_start[1], 'yD', markersize=10)
        unreachable_nodes = []
        for sgraph in grid_subgraphs:
            if closest_node_to_start not in sgraph:
                unreachable_nodes.extend(sgraph.nodes)

        valid_gdf_tmp = _remove_nodes_from_gdf(valid_gdf, unreachable_nodes)

        invalid_nodes = _get_invalid_nodes_around_node(edge[0], line_graph_node_connections, valid_gdf_tmp)
        valid_gdf_tmp = _remove_nodes_from_gdf(valid_gdf_tmp, invalid_nodes)
        invalid_nodes.extend(_get_invalid_nodes_around_node(edge[1], line_graph_node_connections, valid_gdf_tmp))
        valid_gdf_tmp = _remove_nodes_from_gdf(valid_gdf_tmp, invalid_nodes)

        squares = []

        edge_data = line_graph.get_edge_data(*edge)
        ls = edge_data['ls']
        ls_rel = scale(translate(ls, -grid_gdf.x.min(), -grid_gdf.y.min()), 1.0/8.0, 1.0/8.0, origin=(0,0))
        # plt.plot(ls_rel.xy[0], ls_rel.xy[1], ':go')
        ls_valid = []
        for coord in ls.coords:
            p_grid_closest = _get_closest_node_in_gdf(valid_gdf_tmp, coord)
            # if p_grid_closest not in ls_valid:
            #     ls_valid.append(p_grid_closest)
            if len(ls_valid) == 0 or (len(ls_valid) > 0 and p_grid_closest != ls_valid[-1]):
                ls_valid.append(p_grid_closest)

        for idx in range(1, len(ls_valid)):
            # plt.plot(ls_valid[idx][0], ls_valid[idx][1], 'ro')
            # plt.plot(ls_valid[idx-1][0], ls_valid[idx-1][1], 'ro')

            # TODO: get networkx.exception.NodeNotFound error. apparently nodes are in invalid_nodes but are are in p_grid_closest when finding ls_valid
            valid_grid_graph = nx.restricted_view(grid_graph, invalid_nodes, [])
            if ls_valid[idx-1] in valid_grid_graph.nodes and ls_valid[idx] in valid_grid_graph.nodes and nx.has_path(valid_grid_graph, ls_valid[idx-1], ls_valid[idx]):
                path = nx.shortest_path(valid_grid_graph, ls_valid[idx-1], ls_valid[idx])
                ls_path = LineString(path)
                # plt.plot(ls_path.xy[0], ls_path.xy[1], '-b')
            else:
                continue
            invalid_nodes.extend(path[1:-1])

            if idx < len(ls_valid) - 1:
                squares.extend(path[0:-1])
            else:
                squares.extend(path)

            if idx == 1:
                start_node_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])
                start_node_dir_string = road_direction_dict[start_node_direction]
                if edge[0] == line_graph.get_edge_data(*edge)['from_node_to_node'][0]:
                    line_graph_node_connections[edge[0]]['connections'].append(start_node_dir_string)
                else:
                    line_graph_node_connections[edge[1]]['connections'].append(start_node_dir_string)

            # find invalid continuing directions after last point
            direction = (path[-2][0] - path[-1][0], path[-2][1] - path[-1][1])
            dir_string = road_direction_dict[direction]
            if idx < len(ls_valid) - 1:
                direction_strings = fence_tiles.columns[fence_tiles[~pandas.isna(fence_tiles[dir_string]) & (fence_tiles.n_connections == 2)].isna().all()]
                direction_tuples = [direction_dict[direction_string] for direction_string in direction_strings]
                invalid_nodes.extend([(path[-1][0] + dt[0], path[-1][1] + dt[1]) for dt in direction_tuples])
            else:
                if edge[0] == line_graph.get_edge_data(*edge)['from_node_to_node'][0]:
                    line_graph_node_connections[edge[1]]['connections'].append(dir_string)
                else:
                    line_graph_node_connections[edge[0]]['connections'].append(dir_string)

        if len(squares) > 0:
            square_graph.add_edge(squares[0], squares[-1], squares=squares, element_idx=edge_data['element_idx'], from_node_to_node=[squares[0], squares[-1]])
        else:
            logger.warn('Could not find valid squares for edge {} - {}.'.format(edge[0], edge[1]))


    # plt.show()
    osm_processor.network_graphs[name]['square_graph'] = square_graph

def _remove_nodes_from_gdf(gdf, nodes_to_remove):
    points = MultiPoint([Point((node[0] * 8 + gdf.x.min(), node[1] * 8 + gdf.y.min())) for node in nodes_to_remove])
    index = gdf.iloc[gdf.geometry.centroid.sindex.nearest(points, max_distance=0.1)[1]].index
    # invalid_indices = []
    # for node in nodes_to_remove:
    #     idx = gdf.loc[(gdf.xidx == node[0]) & (gdf.yidx == node[1])].index
    #     if len(idx) > 0:
    #         invalid_indices.append(idx[0])

    # assert np.array_equal(sorted(index.to_list()), sorted(invalid_indices))

    # return gdf.loc[~np.isin(gdf.index, invalid_indices)]
    return gdf.drop(index)

def _get_closest_node_in_gdf(gdf, coord, return_xy=False, return_gdf_index=False):
    # idx = gdf.geometry.distance(Point(coord)).argmin()
    # TODO: Add PyGEOS to requirements. Otherwise the interface is nearest(coord) (no Point!)
    idx = list(gdf.sindex.nearest(Point(coord)))[1][0]
    idx_tuple = tuple(gdf.iloc[idx][['xidx', 'yidx']].values.tolist())
    if return_xy and return_gdf_index:
        return idx, idx_tuple, tuple(gdf.iloc[idx][['x', 'y']].values.tolist())
    elif return_xy and not return_gdf_index:
        return idx_tuple, tuple(gdf.iloc[idx][['x', 'y']].values.tolist())
    elif not return_xy and return_gdf_index:
        return idx, tuple(gdf.iloc[idx][['x', 'y']].values.tolist())
    else:
        return idx_tuple

def _get_invalid_nodes_around_node(node_xy, node_connections_dict, gdf):
    degree = node_connections_dict[node_xy]['degree']
    connection_directions = node_connections_dict[node_xy]['connections']
    if degree == 1 or len(connection_directions) == 0:
        return []

    node = _get_closest_node_in_gdf(gdf, node_xy)
    
    # direction_tuples = [direction_dict[connection_direction] for connection_direction in connection_directions]

    condition = fence_tiles.n_connections == node_connections_dict[node_xy]['degree']
    for connection_direction in connection_directions:
        condition = condition & ~pandas.isna(fence_tiles[connection_direction])

    direction_strings = fence_tiles.columns[fence_tiles[condition].isna().all()]
    direction_tuples = [direction_dict[direction_string] for direction_string in direction_strings]
    invalid_nodes = [(node[0] + dt[0], node[1] + dt[1]) for dt in direction_tuples]

    return invalid_nodes

