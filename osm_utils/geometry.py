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

from typing import List, Optional, Tuple, Any
import numpy as np
from shapely.geometry import Polygon, LinearRing, LineString, MultiLineString
from shapely.ops import split
import networkx as nx
import matplotlib.pyplot as plt

DRAW_DEBUG_PLOTS = False

def _ring_cross_product(ring: LinearRing) -> np.array:
    # get vertex coordinate append last point before first and first point after last, 
    # so we can calculate the cross product between last and first line
    vertices = [ring.coords[-2]]
    vertices.extend([coord for coord in ring.coords])
    # vertices.append(ring.coords[0])
    vertices = np.array(vertices)

    # calculate vectors between vertices and the cross product between consecutive vectors
    cross_product = np.cross(np.diff(vertices, axis=0)[1::], np.diff(vertices, axis=0)[:-1])

    return cross_product

def _ccw_ring(ring: LinearRing) -> LinearRing:
    if not ring.is_ccw:
        ring = LinearRing(ring.coords[::-1])

    return ring

def find_concave_vertices(polygon: Polygon) -> List[int]:
    # get perimeter ring, check if it is counter-clockwise and if not make it ccw
    ring = _ccw_ring(polygon.exterior)

    cross_product = _ring_cross_product(ring)

    # concave vertices are those which (in a counter-clockwise ring) have crossproduct > 0.
    concave_vertices = np.array(ring.coords[:-1])[cross_product > 0]

    return concave_vertices
    
def find_chords(polygon: Polygon, concave_vertices: List[tuple[float]], is_diagonal: bool) -> List[LineString]:
    """
    For this case (rectilinear polygons) chords are lines parallel to one of the coordinate axes, spanning the interior of a polygon, connecting two concave vertices
    """
    # if there is only one concave vertex there is nothing to do
    if len(concave_vertices) < 2:
        return [], []

    # we want to check if lines are parallel to a coordinate axis, so define the vectors of the coordinate axes
    if is_diagonal:
        axis1_vector = (1,1)
        axis2_vector = (-1,1)
    else:
        axis1_vector = (1,0)
        axis2_vector = (0,1)

    # loop over all pairs of concave vertices, check if their connection is parallel to a coordinate axis and if so check if the connection is within the polygon
    ls2vtx_dict = {}
    vtx2ls_dict = {}
    chords = []
    for idx1 in range(len(concave_vertices)):
        for idx2 in range(idx1+1, len(concave_vertices)):
            vector = (concave_vertices[idx2][1] - concave_vertices[idx1][1], concave_vertices[idx2][0] - concave_vertices[idx1][0])
            is_horizontal, is_vertical = False, False
            if np.cross(vector, axis1_vector) == 0:
                is_horizontal = True
            if np.cross(vector, axis2_vector) == 0:
                is_vertical = True

            if not (is_horizontal or is_vertical):
                continue

            ls = LineString([concave_vertices[idx1], concave_vertices[idx2]])
            if polygon.contains(ls):
                chords.append((ls, 'h' if is_horizontal else 'v'))
                ls_idx = len(chords) - 1
                ls2vtx_dict[ls_idx] = [idx1, idx2]
                if idx1 not in vtx2ls_dict:
                    vtx2ls_dict[idx1] = []
                if idx2 not in vtx2ls_dict:
                    vtx2ls_dict[idx2] = []

                vtx2ls_dict[idx1].append(ls_idx)
                vtx2ls_dict[idx2].append(ls_idx)


    # finally we have to check if vertices have more than one chord and, if so, wether one of them is a substring of another. In that case, the longer one probably goes through another vertex and 
    # partially coincides with the boundary of the polygon. we don't want this, so we delete the longer one.

    out_chords = []
    for ls_idx in range(len(chords)):
        ls_conincides_with_boundary = False
        ls = chords[ls_idx][0]
        idx1, idx2 = ls2vtx_dict[ls_idx]
        other_ls_indices = [idx for idx in vtx2ls_dict[idx1] if idx != ls_idx]
        other_ls_indices.extend([idx for idx in vtx2ls_dict[idx2] if idx != ls_idx])
        if len(other_ls_indices) > 0:
            for other_ls_idx in other_ls_indices:
                other_ls = chords[other_ls_idx][0]
                if ls.contains(other_ls):
                    ls_conincides_with_boundary = True

        if not ls_conincides_with_boundary:
            out_chords.append(chords[ls_idx])

    horizontal_chords = [c[0] for c in out_chords if c[1] == 'h']
    vertical_chords = [c[0] for c in out_chords if c[1] == 'v']

    return horizontal_chords, vertical_chords

def find_subdividing_chords(horizontal_chords: List[LineString], vertical_chords: List[LineString]) -> List[LineString]:
    if len(horizontal_chords) == 0 and len(vertical_chords) == 0:
        return []
    
    # * Input P a simple orthogonal polygon.
    # * Find the chords of P.
    # * Construct a bipartite graph with edges between vertices in the sets V and H, where each vertex in V corresponds to a vertical chord, 
    #   and each vertex in H corresponds to a horizontal chord.  Draw an edge between vertices v \in V and h \in H iff the chords corresponding to v and h intersect.
    # * Find a maximum matching M of the bipartite graph.
    # * Use M to find a maximum independent set S of vertices of the bipartite graph.  (This set corresponds to a maximum set of nonintersecting chords of P.)
    # * Draw the chords corresponding to S in P.  This subdivides P into |S|+1 smaller polygons, none of which contains a chord.
    # * Using Algorithm 1, rectangulate each of the chordless polygons.
    # * Output the union of the rectangulations of the previous step.

    graph = nx.Graph()
    horizontal_nodes = [(hidx, 0) for hidx in range(len(horizontal_chords))]
    vertical_nodes = [(vidx, 1) for vidx in range(len(vertical_chords))]
    if len(horizontal_chords) > 0:
        graph.add_nodes_from(horizontal_nodes, bipartites=0)
    if len(vertical_chords) > 0:
        graph.add_nodes_from(vertical_nodes, bipartites=1)

    for hidx in range(len(horizontal_chords)):
        for vidx in range(len(vertical_chords)):
            if horizontal_chords[hidx].touches(vertical_chords[vidx]):
                continue
            if horizontal_chords[hidx].intersects(vertical_chords[vidx]):
                graph.add_edge((hidx, 0), (vidx, 1))

    assert nx.is_bipartite(graph)

    maximal_matching = nx.maximal_matching(graph)
    maximal_matching_nodes = []
    for edge in maximal_matching:
        maximal_matching_nodes.extend(edge)

    try:
        maximal_independent_set = nx.maximal_independent_set(graph, maximal_matching_nodes)
    except:
        return []

    chords = []
    for node in maximal_independent_set:
        if node[1] == 0:
            chords.append(horizontal_chords[node[0]])
        else:
            chords.append(vertical_chords[node[0]])

    return chords

def remove_one_square_appendages(polygon: Polygon) -> Polygon:
    ring = _ccw_ring(polygon.exterior)
    cross_product_signs = np.sign(_ring_cross_product(ring))

    indices_to_remove = []
    for idx in range(len(cross_product_signs)):
        seq = [cross_product_signs[idx-2], cross_product_signs[idx-1], cross_product_signs[idx]]
        if seq == [1, -1, -1]:
            indices_to_remove.extend([idx-1, idx])
        if seq == [-1, -1, 1]:
            indices_to_remove.extend([idx-2, idx-1])

    if len(indices_to_remove) > 0:
        ring_coords = [coord for coord in ring.coords][:-1]
        ring_coords = np.delete(ring_coords, indices_to_remove, axis=0)
        polygon = Polygon(ring_coords)
    
    return polygon

def get_axis_parallel_cuts_through_point(bounds: List[float], vertex: Tuple[float], is_diagonal: bool) -> List[LineString]:
    min_x, min_y, max_x, max_y = bounds
    if is_diagonal:
        d1 = min(vertex[0] - min_x, vertex[1] - min_y)
        d2 = min(max_x - vertex[0], max_y - vertex[1])
        d3 = min(vertex[0] - min_x, max_y - vertex[1])
        d4 = min(max_x - vertex[0], vertex[1] - min_y)

        splitting_ls1 = LineString([
            (vertex[0] - d1, vertex[1] - d1),
            (vertex[0] + d2, vertex[1] + d2)
        ])
        splitting_ls2 = LineString([
            (vertex[0] + d4, vertex[1] - d4),
            (vertex[0] - d3, vertex[1] + d3),
        ])
    else:
        splitting_ls1 = LineString([
            (min_x, vertex[1]),
            (max_x, vertex[1])
        ])
        splitting_ls2 = LineString([
            (vertex[0], min_y),
            (vertex[0], max_y)
        ])

    return [splitting_ls1, splitting_ls2]

def is_one_square_wide(polygon: Polygon):
    ring = _ccw_ring(polygon.exterior)
    cross_product_signs = np.sign(_ring_cross_product(ring))

    one_square_wide = False
    for idx in range(len(cross_product_signs)):
        seq = [cross_product_signs[idx-1], cross_product_signs[idx]]
        if seq == [-1, -1]:
            one_square_wide = True
            break

    return one_square_wide


def split_polygon(polygon: Polygon, is_diagonal: bool) -> Polygon:
    concave_vertices = find_concave_vertices(polygon).tolist()
    if len(concave_vertices) == 0:
        return [polygon]
    
    vtx = concave_vertices[0]
    cut_lines = get_axis_parallel_cuts_through_point(polygon.bounds, vtx, is_diagonal)

    valid_splits = []
    semi_valid_splits = []
    invalid_splits = []
    for line in cut_lines:
        sub_polygons = split(polygon, line)
        assert len(sub_polygons.geoms) == 2
        assert type(sub_polygons.geoms[0]) == Polygon
        assert type(sub_polygons.geoms[1]) == Polygon

        polygon1_valid = not is_one_square_wide(sub_polygons.geoms[0])        
        polygon2_valid = not is_one_square_wide(sub_polygons.geoms[1])

        if polygon1_valid and polygon2_valid:
            valid_splits.append([sub_polygons.geoms[0], sub_polygons.geoms[1]])
        elif polygon1_valid or polygon2_valid:
            semi_valid_splits.append([
                (sub_polygons.geoms[0], polygon1_valid, sub_polygons.geoms[0].exterior.length), 
                (sub_polygons.geoms[1], polygon2_valid, sub_polygons.geoms[1].exterior.length)
            ])
        else:
            invalid_splits.append([
                (sub_polygons.geoms[0], polygon1_valid, sub_polygons.geoms[0].exterior.length), 
                (sub_polygons.geoms[1], polygon2_valid, sub_polygons.geoms[1].exterior.length)
            ])

    if len(valid_splits) > 0:
        rng = np.random.default_rng()
        split_polygons = rng.choice(valid_splits, 1)
    elif len(semi_valid_splits) > 0:
        semi_valid_splits = sorted(semi_valid_splits, key=lambda split: sum([p[2] for p in split if not p[1]]))
        split_polygons = [p[0] for p in semi_valid_splits[0]]
    else:
        invalid_splits = sorted(invalid_splits, key=lambda split: sum([p[2] for p in split]))
        split_polygons = [p[0] for p in invalid_splits[0]]
    
    sub_split_polygons = []
    for p in split_polygons:
        sub_split_polygons.extend(split_polygon(p, is_diagonal))

    return sub_split_polygons



def rectangulate_polygon(polygon: Polygon, is_diagonal: bool, orig_geometry: Optional[Polygon]) -> List[Polygon]:
    # Input: P a simple orthogonal polygon with no chords.
    # For each concave vertex, select one of its incident edges.  (Two edges are incident to each concave vertex.)
    # Extend this edge until it hits another such extended edge, or a boundary edge of P.
    # Return the extensions of edges as the rectangulation.

    if DRAW_DEBUG_PLOTS:
        plt.figure()
        plt.axis('equal')
        plt.plot(polygon.exterior.xy[0], polygon.exterior.xy[1], ':ko')
    if orig_geometry is not None:
        plt.plot(orig_geometry.exterior.xy[0], orig_geometry.exterior.xy[1], '-mo')

    polygon = remove_one_square_appendages(polygon)
    if DRAW_DEBUG_PLOTS:
        plt.plot(polygon.exterior.xy[0], polygon.exterior.xy[1], '-ko')
        plt.plot(polygon.exterior.xy[0][0], polygon.exterior.xy[1][0], 'kD')

    concave_vertices = find_concave_vertices(polygon)

    if len(concave_vertices) == 0:
        return [polygon]

    horizontal_chords, vertical_chords = find_chords(polygon, concave_vertices, is_diagonal)
    if DRAW_DEBUG_PLOTS:
        for ls in horizontal_chords:
            plt.plot(ls.xy[0], ls.xy[1], '-.g')
        for ls in vertical_chords:
            plt.plot(ls.xy[0], ls.xy[1], '-.g')

    splitting_chords = find_subdividing_chords(horizontal_chords, vertical_chords)

    polygons = [polygon]
    if len(splitting_chords) > 0:
        for ls in splitting_chords:
            new_polygons = []
            for p in polygons:
                if ls.intersects(p):
                    splits = split(p, ls)
                    new_polygons.extend([np for np in splits.geoms])
                else:
                    new_polygons.append(p)
            polygons = new_polygons

    split_polygons = []
    for p in polygons:
        if DRAW_DEBUG_PLOTS:
            plt.plot(p.exterior.xy[0], p.exterior.xy[1], '-')
        split_polygons.extend(split_polygon(p, is_diagonal))


    if DRAW_DEBUG_PLOTS:
        for p in split_polygons:
            plt.plot(p.exterior.xy[0], p.exterior.xy[1], '-')
                # plt.plot(splitting_ls.xy[0], splitting_ls.xy[1], ':k')

        plt.show()
    






    







