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

import itertools
import logging
from heapq import heappop, heappush
from itertools import count

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas
from shapely.affinity import scale, translate
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import substring

from profiles.general import fence_tiles, rail_tiles, road_tiles, stream_tiles

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

def custom_weight(graph: nx.Graph, node1, node2, edge_dict, ref_line, tiles, source, target, current_path, dist):
    # Directly discard paths where the new node was already visited before the previous node
    # if len(current_path) > 1 and node2 in current_path[-2::]:
    #     return None
    # calculates the cost to go from current node (node1) to neighbor node (node2)
    # Caculate diff in squares between node1 and node2, obtain "direction" for that
    diff_node1_node2 = (node2[0] - node1[0], node2[1] - node1[1])
    diff_node2_node1 = (node1[0] - node2[0], node1[1] - node2[1])
    direction_node1 = road_direction_dict[diff_node1_node2]
    direction_node2 = road_direction_dict[diff_node2_node1]
    # check if either direction for either node1 or node2 is not in tile set.
    if direction_node1 not in tiles.columns or direction_node2 not in tiles.columns:
        return None
    # check if direction of node1 is already taken
    if 'connections' in graph.nodes[node1] and direction_node1 in graph.nodes[node1]['connections']:
        return None
    # check if direction of node1 is already taken
    if 'connections' in graph.nodes[node2] and direction_node2 in graph.nodes[node2]['connections']:
        return None
    # now check if there is a tile in the tile set that has all the necessary connections
    if 'connections' in graph.nodes[node1] and len(graph.nodes[node1]['connections']) > 0:
        # only source and target node may have more than two connections
        if len(graph.nodes[node1]['connections']) > 1 and not (node1 == source or node1 == target):
            return None
        condition = ~pandas.isnull(tiles[direction_node1])
        for con in graph.nodes[node1]['connections']:
            condition = condition & ~pandas.isnull(tiles[con])

        if not condition.any():
            return None 
    if 'connections' in graph.nodes[node2] and len(graph.nodes[node2]['connections']) > 0:
        # only source and target node may have more than two connections
        if len(graph.nodes[node2]['connections']) > 1 and not (node2 == source or node2 == target):
            return None
        condition = ~pandas.isnull(tiles[direction_node2])
        for con in graph.nodes[node2]['connections']:
            condition = condition & ~pandas.isnull(tiles[con])

        if not condition.any():
            return None 

    # create point object for node and store it so that this operation has to be done only once per node
    if 'point' not in graph.nodes[node1]:
        graph.nodes[node1]['point'] = Point(node1)
    if 'point' not in graph.nodes[node2]:
        graph.nodes[node2]['point'] = Point(node2)

    # point_node1 = graph.nodes[node1]['point']
    point_node2 = graph.nodes[node2]['point']

    # # if all the checks above worked (i.e. there is a valid connection between both nodes),
    # # calculate the cost of this connection as the sum of distances of the nodes to the actual LineString
    # # This should minimize the distance of the path chosen to the LineString
    # cost1 = ref_line.interpolate(ref_line.project(point_node1)).distance(point_node1)
    # cost2 = ref_line.interpolate(ref_line.project(point_node2)).distance(point_node2)
    # return cost1 + cost2
    # ref_line_cut1 = substring(ref_line, 0, ref_line.project(point_node1))
    ref_line_cut2 = substring(ref_line, 0, ref_line.project(point_node2))
    
    # geom1 = point_node1 if len(current_path) == 1 else LineString(current_path)
    # geom2 = LineString(current_path + [node2])

    # np.testing.assert_allclose(np.abs(geom2.hausdorff_distance(ref_line_cut2) - geom1.hausdorff_distance(ref_line_cut1)), max(np.min([Point(*ref_line_cut2.coords[i]).distance(Point(node2)) for i in range(len(ref_line_cut2.coords))]), dist) - dist)

    # return np.abs(geom2.hausdorff_distance(ref_line_cut2) - geom1.hausdorff_distance(ref_line_cut1))
    # return max(np.min([Point(*ref_line_cut2.coords[i]).distance(Point(node2)) for i in range(len(ref_line_cut2.coords))]), dist) - dist
    return np.min([Point(*ref_line_cut2.coords[i]).distance(Point(node2)) for i in range(len(ref_line_cut2.coords))])

def _find_astar_path(start_node, end_node, grid_graph, ls, tiles, allow_vary_first_node=False):
    path_found = False
    connected_start_edges = []
    connected_end_edges = []
    if start_node in grid_graph.nodes and 'edges' in grid_graph.nodes[start_node]:
        connected_start_edges = grid_graph.nodes[start_node]['edges']    
    if end_node in grid_graph.nodes and 'edges' in grid_graph.nodes[end_node]:
        connected_end_edges = grid_graph.nodes[end_node]['edges']    

    idx_mods = list(itertools.product([0,1,-1], [0,1,-1], [0,1,-1], [0,1,-1]))
    idx_mods = sorted(idx_mods, key=lambda x: sum(np.abs(x)))
    for idx_mod in idx_mods:
        mod_start_node = (start_node[0] + idx_mod[0], start_node[1] + idx_mod[1])
        if not mod_start_node in grid_graph.nodes:
            continue
        mod_end_node = (end_node[0] + idx_mod[2], end_node[1] + idx_mod[3])
        if not mod_end_node in grid_graph.nodes:
            continue
        if mod_start_node == mod_end_node:
            continue
        mod_start_edges = grid_graph.nodes[mod_start_node]['edges'] if 'edges' in grid_graph.nodes[mod_start_node] else []
        mod_end_edges = grid_graph.nodes[mod_end_node]['edges'] if 'edges' in grid_graph.nodes[mod_end_node] else []
        
        if (idx_mod[0] != 0 or idx_mod[1] != 0) and ((len(connected_start_edges) > 0 and len(set(mod_start_edges).intersection(connected_start_edges)) == 0) or not allow_vary_first_node):
            continue
        if (idx_mod[2] != 0 or idx_mod[3] != 0) and len(connected_end_edges) > 0 and len(set(mod_end_edges).intersection(connected_end_edges)) == 0:
            continue
        try:
            path = custom_weight_astar_path(grid_graph, mod_start_node, mod_end_node, weight=custom_weight, ref=ls, tiles=tiles)
            if len(path) - 1 <= ls.length * 2:
                path_found = True
                break
        except nx.NetworkXNoPath:
            continue

    if path_found:
        return path
    else:
        return None

def search_path(osm_processor, config, name, tqdm_string):
    logger = logging.getLogger('osm2cm')

    if 'road_tiles' in config[name]['process']:
        tiles = road_tiles
    elif 'stream_tiles' in config[name]['process']:
        tiles = stream_tiles
    elif 'rail_tiles' in config[name]['process']:
        tiles = rail_tiles
    elif 'fence_tiles' in config[name]['process']:
        tiles = fence_tiles

    line_graph = osm_processor.network_graphs[name]['line_graph']
    grid_gdf = osm_processor.gdf
    df = osm_processor.df 

    square_graph = nx.MultiGraph()

    # TODO: Check what happens if point can't be placed.
    # create grid graph
    xmax = grid_gdf.xidx.max()
    ymax = grid_gdf.yidx.max()

    if osm_processor.grid_graph is None:
        grid_graph = nx.grid_2d_graph(xmax+1, ymax+1)
        diagonal_edges1 = [((x, y), (x+1, y+1)) for x in range(xmax) for y in range(ymax)]
        diagonal_edges2 = [((x+1, y), (x, y+1)) for x in range(xmax) for y in range(ymax)]

        grid_graph.add_edges_from(diagonal_edges1, weight=np.sqrt(2))
        grid_graph.add_edges_from(diagonal_edges2, weight=np.sqrt(2))
        osm_processor.grid_graph = grid_graph
    else:
        grid_graph = osm_processor.grid_graph

    edge_list = list(line_graph.edges)
    # sort edge list by position in tags
    enhanced_edge_list = []
    for edge in edge_list:
        element_idx = line_graph.get_edge_data(*edge)['element_idx']
        element_entry = osm_processor.matched_elements[element_idx]
        matched_cm_type = get_matched_cm_type(config, element_entry)
        cm_type_idx = config[name]['cm_types'].index(matched_cm_type)
        enhanced_edge_list.append((edge, cm_type_idx))
    
    enhanced_edge_list = sorted(enhanced_edge_list, key=lambda x: -x[1])
    edge_list = [entry[0] for entry in enhanced_edge_list]

    edges_to_process = list(range(len(edge_list)))
    path_dict = {}
    # plt.figure()
    # plt.axis('equal')
    # for g in grid_gdf.geometry.values:
    #     plt.plot(g.exterior.xy[0], g.exterior.xy[1], '-k', linewidth=0.1)

    while len(edges_to_process) > 0:
        edge_idx = edges_to_process.pop()
        edge = edge_list[edge_idx]
        # count += 1
        # if count > 5:
        #     break
        closest_node_to_start = _get_closest_node_in_gdf(grid_gdf, edge[0])
        closest_node_to_end = _get_closest_node_in_gdf(grid_gdf, edge[1])

        edge_data = line_graph.get_edge_data(*edge)
        ls = edge_data['ls']
        ls_points = []
        for coord in ls.coords:
            p_grid_closest = _get_closest_node_in_gdf(grid_gdf, coord)
            # if p_grid_closest not in ls_valid:
            #     ls_valid.append(p_grid_closest)
            if len(ls_points) == 0 or (len(ls_points) > 0 and p_grid_closest != ls_points[-1]):
                ls_points.append(p_grid_closest)
        
        if len(ls_points) < 2:
            continue
        ls_valid = LineString(ls_points)

        # path = []
        # for i in range(1, len(ls_points)):
        #     if len(path) == 0:
        #         node1 = ls_points[i-1]
        #     else:
        #         node1 = path[-1]
        #     node2 = ls_points[i]
        #     allow_vary_first_node = False if i > 1 else True
        #     path_segment = _find_astar_path(node1, node2, grid_graph, LineString([node1, node2]), tiles, allow_vary_first_node)
        #     if path_segment is not None:
        #         if len(path) == 0:
        #             path.extend(path_segment)
        #         elif len(path_segment) > 1:
        #             path.extend(path_segment[1:])
        #     else:
        #         break

        path = _find_astar_path(ls_points[0], ls_points[-1], grid_graph, LineString(ls_points), tiles, allow_vary_first_node=True)

        if path is None or len(path) == 0:
            logger.debug('No path found from {} to {}.'.format(edge[0], edge[1]))
            # plt.plot(ls.xy[0], ls.xy[1], '-ro')
            continue


        # check if path crosses the path of another edge at node that is not the start or end node
        crosses = False
        for node_idx in range(1, len(path)-1):
            node = path[node_idx]
            node_dict = grid_graph.nodes[node]
            if 'connections' in node_dict and len(node_dict['connections']) > 2:
                crosses = True
                edges_to_process.append(edge_idx)
                other_edge_idx = node_dict['edges'][-1]
                edges_to_process.append(other_edge_idx)

                logger.debug('Edge {} crosses edge {} reprocessing both.'.format(edge_idx, other_edge_idx))

                # remove the other edge from the grid_graph
                other_path = path_dict[other_edge_idx]
                for other_nodes in other_path[0]:
                    indices_to_remove = []
                    for lidx, eidx in enumerate(grid_graph.nodes[other_nodes]['edges']):
                        if eidx == other_edge_idx:
                            indices_to_remove.append(lidx)
                
                    for lidx in indices_to_remove[::-1]:
                        grid_graph.nodes[other_nodes]['edges'].pop(lidx)
                        grid_graph.nodes[other_nodes]['connections'].pop(lidx)

        if crosses:
            continue

        for nidx, node in enumerate(path):
            if nidx < len(path) - 1:
                next_node = path[nidx + 1]
                diff = (next_node[0] - node[0], next_node[1] - node[1])
                direction = road_direction_dict[diff]
                if not 'connections' in grid_graph.nodes[path[nidx]]:
                    grid_graph.nodes[path[nidx]]['connections'] = []
                    grid_graph.nodes[path[nidx]]['edges'] = []
                grid_graph.nodes[path[nidx]]['connections'].append(direction)
                grid_graph.nodes[path[nidx]]['edges'].append(edge_idx)
            if nidx > 0:
                prev_node = path[nidx - 1]
                diff = (prev_node[0] - node[0], prev_node[1] - node[1])
                direction = road_direction_dict[diff]
                if not 'connections' in grid_graph.nodes[path[nidx]]:
                    grid_graph.nodes[path[nidx]]['connections'] = []
                    grid_graph.nodes[path[nidx]]['edges'] = []
                grid_graph.nodes[path[nidx]]['connections'].append(direction)
                grid_graph.nodes[path[nidx]]['edges'].append(edge_idx)

        xy = [grid_gdf.loc[(grid_gdf.xidx == p[0]) & (grid_gdf.yidx == p[1]), ['x', 'y']].values[0] for p in path]
        if len(path) == 1:
            plt.plot(ls.xy[0], ls.xy[1], ':go')
            plt.plot(xy[0][0], xy[0][1], 'go')
            continue
        # plt.plot(ls.xy[0], ls.xy[1], ':ko')
        # plt.text(ls.interpolate(0.5, normalized=True).x, ls.interpolate(0.5, normalized=True).y, str(edge_idx))
        # square_geom = [grid_gdf[(grid_gdf.xidx == p[0]) & (grid_gdf.yidx == p[1])].geometry.values[0] for p in path]
        ls_xy = LineString(xy)
        # plt.plot([xy[i][0] for i in range(len(xy))], [xy[i][1] for i in range(len(xy))], '-b')
        # plt.plot(xy[0][0], xy[0][1], 'bo')
        # plt.plot(xy[-1][0], xy[-1][1], 'bD')
        # plt.text(ls_xy.interpolate(0.5, normalized=True).x, ls_xy.interpolate(0.5, normalized=True).y, str(edge_idx), color='b')

        path_dict[edge_idx] = [path, edge_data['element_idx']]

    # plt.savefig('debug/path_search_{}.svg'.format(name))

    # create square graph from paths
    for edge_idx in path_dict.keys():
        # get path and edge data        
        path = path_dict[edge_idx][0]
        edge_data = path_dict[edge_idx][1]

        # split path at nodes with more than 2 connections
        # paths don't necessarily start at intersections so 0 and len(path)-1 have to be added that case
        n_connections_per_node = [len(grid_graph.nodes[n]['connections']) for n in path]
        intersection_indices = np.where(np.array(n_connections_per_node) > 2)[0].tolist()

        if len(intersection_indices) == 0 or intersection_indices[0] != 0:
            intersection_indices = [0] + intersection_indices
        if intersection_indices[-1] != len(path) - 1:
            intersection_indices = intersection_indices + [len(path) - 1]

        sub_paths = []
        for i in range(1, len(intersection_indices)):
            sub_paths.append(path[intersection_indices[i-1]:intersection_indices[i]+1])

        for sub_path in sub_paths:
            square_graph.add_edge(sub_path[0], sub_path[-1], squares=sub_path, element_idx=edge_data, from_node_to_node=[sub_path[0], sub_path[-1]])

    # grid graph is persistent over all path searches, so remove already used nodes
    if config[name]['priority'] < 1:
        osm_processor.grid_graph = None
    else:
        for edge_idx in path_dict.keys():
            path = path_dict[edge_idx][0]
            grid_graph.remove_nodes_from(path)

    osm_processor.network_graphs[name]['square_graph'] = square_graph

    
# def search_path(line_graph, grid_gdf, df):
def search_path2(osm_processor, config, name):
    logger = logging.getLogger('osm2cm')

    if 'road_tiles' in config[name]['process']:
        tiles = road_tiles
    elif 'stream_tiles' in config[name]['process']:
        tiles = stream_tiles
    elif 'rail_tiles' in config[name]['process']:
        tiles = rail_tiles
    elif 'fence_tiles' in config[name]['process']:
        tiles = fence_tiles

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

    if 'ul' in tiles.columns or 'ur' in tiles.columns or 'dl' in tiles.columns or 'dr' in tiles.columns:
        diagonal_edges1 = [((x, y), (x+1, y+1)) for x in range(xmax) for y in range(ymax)]
        diagonal_edges2 = [((x+1, y), (x, y+1)) for x in range(xmax) for y in range(ymax)]

        grid_graph.add_edges_from(diagonal_edges1, weight=np.sqrt(2))
        grid_graph.add_edges_from(diagonal_edges2, weight=np.sqrt(2))

    if df is not None:
        occupied_nodes = list({(v[0], v[1]) for v in df.loc[df.priority == 1, ['xidx', 'yidx']].values})
    else:
        occupied_nodes = []

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
    count = 0
    for edge in line_graph.edges:
        # count += 1
        # if count > 5:
        #     break
        closest_node_to_start = _get_closest_node_in_gdf(valid_gdf, edge[0])
        # plt.plot(closest_node_to_start[0], closest_node_to_start[1], 'yD', markersize=10)
        unreachable_nodes = []
        for sgraph in grid_subgraphs:
            if closest_node_to_start not in sgraph:
                unreachable_nodes.extend(sgraph.nodes)

        valid_gdf_tmp = _remove_nodes_from_gdf(valid_gdf, unreachable_nodes)

        invalid_nodes = _get_invalid_nodes_around_node(edge[0], line_graph_node_connections, valid_gdf_tmp, tiles)
        valid_gdf_tmp = _remove_nodes_from_gdf(valid_gdf_tmp, invalid_nodes)
        invalid_nodes.extend(_get_invalid_nodes_around_node(edge[1], line_graph_node_connections, valid_gdf_tmp, tiles))
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
                path = custom_weight_astar_path(valid_grid_graph, ls_valid[idx-1], ls_valid[idx], weight=custom_weight, ref=LineString([ls_valid[idx-1], ls_valid[idx]]))
                # if len(paths) > 0:
                #     costs = []
                #     ls = LineString([ls_valid[idx-1], ls_valid[idx]])
                #     for path in paths:
                #         costs.append(sum([ls.interpolate(ls.project(Point(p))).distance(Point(p)) for p in path]))

                #     min_cost_idx = np.argmin(costs)
                #     path = paths[min_cost_idx]
                # else:
                #     path = paths[0]

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
                direction_strings = tiles.columns[tiles[~pandas.isna(tiles[dir_string]) & (tiles.n_connections == 2)].isna().all()]
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

def _get_invalid_nodes_around_node(node_xy, node_connections_dict, gdf, tiles):
    degree = node_connections_dict[node_xy]['degree']
    connection_directions = node_connections_dict[node_xy]['connections']
    if degree == 1 or len(connection_directions) == 0:
        return []

    node = _get_closest_node_in_gdf(gdf, node_xy)
    
    # direction_tuples = [direction_dict[connection_direction] for connection_direction in connection_directions]

    condition = tiles.n_connections == node_connections_dict[node_xy]['degree']
    for connection_direction in connection_directions:
        condition = condition & ~pandas.isna(tiles[connection_direction])

    direction_strings = tiles.columns[tiles[condition].isna().all()]
    direction_tuples = [direction_dict[direction_string] for direction_string in direction_strings]
    invalid_nodes = [(node[0] + dt[0], node[1] + dt[1]) for dt in direction_tuples]

    return invalid_nodes

def custom_weight_astar_path(G, source, target, heuristic=None, weight="weight", ref=None, tiles=None):
    # heavily modified version of networkx astar,
    # original license:

    # Copyright (C) 2004-2019, NetworkX Developers
    # Aric Hagberg <hagberg@lanl.gov>
    # Dan Schult <dschult@colgate.edu>
    # Pieter Swart <swart@lanl.gov>
    # All rights reserved.

    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are
    # met:

    #   * Redistributions of source code must retain the above copyright
    #     notice, this list of conditions and the following disclaimer.

    #   * Redistributions in binary form must reproduce the above
    #     copyright notice, this list of conditions and the following
    #     disclaimer in the documentation and/or other materials provided
    #     with the distribution.

    #   * Neither the name of the NetworkX Developers nor the names of its
    #     contributors may be used to endorse or promote products derived
    #     from this software without specific prior written permission.

    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    # OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.    

    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    # weight = _weight_function(G, weight)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            cost = weight(G, curnode, neighbor, w, ref, tiles, source, target, _get_current_path(curnode, parent, explored), dist)
            if cost is None or cost > 5:
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target)
            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

        # if len(_get_current_path(curnode, parent, explored)) > 1 and LineString(_get_current_path(curnode, parent, explored)).length > ref.length * 5:
        #     raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

def _get_current_path(curnode, parent, explored):
    path = [curnode]
    node = parent
    while node is not None:
        path.append(node)
        node = explored[node]
    path.reverse()
    return path


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
