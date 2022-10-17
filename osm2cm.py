from unicodedata import category
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from OSMPythonTools.api import Api
from pyproj import Proj
import pandas
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, Point, MultiPoint
from shapely.ops import split, snap
import matplotlib.pyplot as plt
# from skimage.draw import line, line_aa, line_nd, polygon
import geopandas
import re
import networkx as nx
from sklearn import neighbors

PAGE_N_SQUARES_X = 104
PAGE_N_SQUARES_Y = 60

# osm2cm_dict = {
#     'highway': {
#         'primary': (1, 'road', 'Paved 1'),
#         'secondary': (1, 'road', 'Paved 1'),
#         'residential': (1, 'road', 'Paved 2'),
#         'living_street': (1, 'road', 'Paved 2'),
#         'unclassified': (1, 'road', 'Paved 2'),
#         'footway': (2, 'road', 'Foot Path'),
#         'path': (2, 'road', 'Foot Path')
#     },
#     'water': {
#         'river': (0, 'Ground 2', 'Water')
#     }
# }

config = {
    'road': {
        'tags': [
            ('highway', 'primary'),
            ('highway', 'secondary'),
            ('highway', 'residential'),
            ('highway', 'living_street'),
            ('highway', 'unclassified'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Road', 'cat1': 'Paved 1', 'tags': [('highway', 'primary'), ('highway', 'secondary')]},
                {'menu': 'Road', 'cat1': 'Paved 2', 'tags': [('highway', 'residential'), ('highway', 'living_street'), ('highway', 'unclassified')]}
            ],
            'process': [
                'type_from_tag',
                'add_to_road_graph'
            ],
            'post_process': {
                'road_navigation_graph'
            }
        },
        'pass': 2
    },
    'foot_path': {
        'tags': [
            ('highway', 'footway'),
            ('highway', 'path')
        ],
        'cm_types': {
            'types': [
                {'menu': 'Road', 'cat1': 'Gravel Road', 'tags': [('highway', 'footway'), ('highway', 'footway')]},
            ],
            'process': [
                'type_from_tag',
                'add_to_road_graph'
            ],
            'post_process': {
                'road_pattern'
            }
        },
        'pass': 2,
    },
    'water': {
        'tags': [
            ('water', 'river'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 2', 'cat1': 'Water', 'tags': [('water', 'river')]},
            ],
            'process': [
                'type_from_tag'
            ]
        },
        'pass': 3,
    },
    
    'mixed_forest': {
        'tags': [
            ('landuse', 'forest'),
            ('natural', 'wood')
        ],
        'exclude_tags': {
            'leaf_type': 'broadleaved',
        },
        'cm_types': {
            'types': [
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree E', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree E', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree E', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree H', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree H', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree H', 'cat2': 'density 3', 'weight': 1},
            ],
            'post_process': [
                'type_random_individual',
            ]
        },
        'pass': 1
    },

    'broadleaved_forest': {
        'tags': [
            ('landuse', 'forest'),
            ('natural', 'wood')
        ],
        'required_tags': {
            'leaf_type': 'broadleaved',
        },
        'cm_types': {
            'types': [
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 3', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 1', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 2', 'weight': 2},
                {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 3', 'weight': 1},
            ],
            'post_process': [
                'type_random_individual',
            ]
        },
        'pass': 1
    },

    'mixed_bushes': {
        'tags': [
            ('natural', 'scrub')
        ],
        'cm_types': {
            'types': [
                {'menu': 'Foliage', 'cat1': 'Bush A', 'cat2': 'density 1', 'weight': 1.0},
                {'menu': 'Foliage', 'cat1': 'Bush A', 'cat2': 'density 2', 'weight': 2.0},
                {'menu': 'Foliage', 'cat1': 'Bush A', 'cat2': 'density 3', 'weight': 1.0},
                {'menu': 'Foliage', 'cat1': 'Bush B', 'cat2': 'density 1', 'weight': 1.0},
                {'menu': 'Foliage', 'cat1': 'Bush B', 'cat2': 'density 2', 'weight': 2.0},
                {'menu': 'Foliage', 'cat1': 'Bush B', 'cat2': 'density 3', 'weight': 1.0},
                {'menu': 'Foliage', 'cat1': 'Bush C', 'cat2': 'density 1', 'weight': 1.0},
                {'menu': 'Foliage', 'cat1': 'Bush C', 'cat2': 'density 2', 'weight': 2.0},
                {'menu': 'Foliage', 'cat1': 'Bush C', 'cat2': 'density 3', 'weight': 1.0},
            ],
            'post_process': [
                'type_random_individual',
            ]
        },
        'pass': 1
    },

    'farmland': {
        'tags': [
            ('landuse', 'farmland'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 2', 'cat1': 'Plow NS', 'weight': 1.0},
                {'menu': 'Ground 2', 'cat1': 'Plow EW', 'weight': 1.0},
                {'menu': 'Ground 3', 'cat1': 'Crop 1', 'weight': 1.0},
                {'menu': 'Ground 3', 'cat1': 'Crop 2', 'weight': 1.0},
                {'menu': 'Ground 3', 'cat1': 'Crop 3', 'weight': 1.0},
                {'menu': 'Ground 3', 'cat1': 'Crop 4', 'weight': 1.0},
                {'menu': 'Ground 3', 'cat1': 'Crop 5', 'weight': 1.0},
                {'menu': 'Ground 3', 'cat1': 'Crop 6', 'weight': 1.0},
            ],
            'post_process': [
                'type_random_area'
            ]
        },
        'pass': 0
    },

    'grassland': {
        'tags': [
            ('landuse', 'grass')
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 1', 'cat1': 'Grass T', 'weight': 1.0},
                {'menu': 'Ground 1', 'cat1': 'Grass TY', 'weight': 1.0},
                {'menu': 'Ground 1', 'cat1': 'Weeds', 'weight': 1.0},
                {'menu': 'Ground 1', 'cat1': 'Grass XT', 'weight': 1.0},
                {'menu': 'Ground 1', 'cat1': 'Grass XTY', 'weight': 1.0},
            ],
            'post_process': [
                'type_random_individual'
            ],
        }
    },

    'construction_site': {
        'tags': [
            ('landuse', 'construction')
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 1', 'cat1': 'Dirt', 'weight': 1.0},
            ],
            'post_process': [
                'type_random_individual'
            ]
        }
    },

    # 'residential_buildings': {
    #     'tags': [
    #         ('building', 'house'),
    #         ('building', 'residential'),
    #         ('building', 'detached'),
    #         ('building', 'apartments')
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Ground 2', 'cat1': 'Pavement 1', 'weight': 1.0}
    #         ],
    #         'post_process': [
    #             'type_random_individual'
    #         ]
    #     }
    # }

}

# pattern: page, row, col
pattern2roadtile_dict = {
    146: [(0,0,0), (2,0,0)],
    244: [(0,0,1)],
    150: [(0,0,2)],
    176: [(0,1,0)],
    180: [(0,1,1)],
    178: [(0,1,2)],
    253: [(0,2,0)],
    445: [(0,2,0)],
    509: [(0,2,0)],
    182: [(0,2,1)],
    186: [(0,2,2), (1,2,2), (2,2,2), (3,2,2)],
    147: [(0,3,0)],
    153: [(0,3,1)],
    155: [(0,3,2)],

    56: [(1,0,0), (3,0,0)],
    307: [(1,0,1)],
    57: [(1,0,2)],
    50: [(1,1,0)],
    51: [(1,1,1)],
    58: [(1,1,2)],
    247: [(1,2,0)],
    499: [(1,2,0)],
    503: [(1,2,0)],
    59: [(1,2,1)],
    120: [(1,3,0)],
    240: [(1,3,1)],
    248: [(1,3,2)],

    94: [(2,0,1)],
    210: [(2,0,2)],
    26: [(2,1,0)],
    90: [(2,1,1)],
    154: [(2,1,2)],
    379: [(2,2,0)],
    382: [(2,2,0)],
    383: [(2,2,0)],
    218: [(2,2,1)],
    402: [(2,3,0)],
    306: [(2,3,1)],
    434: [(2,3,2)],

    409: [(3,0,1)],
    312: [(3,0,2)],
    152: [(3,1,0)],
    408: [(3,1,1)],
    184: [(3,1,2)],
    478: [(3,2,0)],
    415: [(3,2,0)],
    479: [(3,2,0)],
    440: [(3,2,1)],
    60: [(3,3,0)],
    30: [(3,3,1)],
    62: [(3,3,2)],


    18: [(0,0,0), (2,0,0)],
    144: [(0,0,0), (2,0,0)],
    24: [(1,0,0), (3,0,0)],
    48: [(1,0,0), (3,0,0)],
    313: [(3,0,2)],
    124: [(3,3,0)],
    151: [(0,0,0), (2,0,0)],
    403: [(2,3,0)],
    304: [(3,0,2)],
    52: [(3,3,0)],
    208: [(2,0,2)],
    25: [(1,0,2)],
    406: 402,
    31: 26,
    55: 50,
    19: 147,
    22: 150,
    214: 210,
    179: 178,
    121: 56,
    466: 146,
    400: 402,
    88: 120,
    436: 180,
    211: 147,
    310: 306,
    464: 146,
    473: 153,
    123: 62,
    315: 59,
    472: 152,
    377: 52,
    252: 184


}

pin2pin_dict = {
    1: [(6, 2), (20, 2), (7, 2)],
    2: [(13, 2), (16, 2)],
    3: [(13, 1), (12, 2), (8, 2), (18, 2), (14, 2)],
    4: [(10, 2), (13, 2)],
    5: [(6, 2), (12, 2), (20, 2), (19, 2)],
    6: [(1, 2), (5, 2), (11, 2)],
    7: [(1, 2), (18, 2), (2)],
    8: [(3, 2), (18, 1), (17, 2), (13, 2), (19, 2)],
    9: [(15, 2), (18, 2)],
    10: [(4, 2), (11, 2)],
    11: [(6, 2), (10, 2), (17, 2)],
    12: [(3, 2), (6, 2)],
    13: [(3, 1), (4, 2), (8, 2), (18, 2), (2, 2)],
    14: [(3, 2), (20, 2)],
    15: [(9, 2), (16, 2)],
    16: [(2, 2), (15, 2)],
    17: [(8, 2), (11, 2)],
    18: [(3, 2), (8, 1), (13, 2), (9, 2), (7, 2)],
    19: [(8, 2), (5, 2)],
    20: [(1, 2), (14, 2), (5, 2)],
}

start_pin_dict = {
    (0, 1): 3,
    (0, -1): 13,
    (1, 0): 18,
    (-1, 0): 8,
}

end_pin_dict = {
    (0, 1): 13,
    (0, -1): 3,
    (1, 0): 8,
    (-1, 0): 18,
}

start_intersection_dict = {
    (0, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
    (0, -1): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    (1, 0): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    (-1, 0): [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}

end_intersection_dict = {
    (0, -1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
    (0, 1): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    (-1, 0): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    (1, 0): [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
}

square2square_dict = {
    (0, 1): {
        15: 1,
        14: 2,
        13: 3,
        12: 4,
        11: 5,
    },
    (0, -1): {
        1: 15,
        2: 14,
        3: 13,
        4: 12,
        5: 11,
    },
    (1, 0): {
        6: 20,
        7: 19,
        8: 18,
        9: 17,
        10: 16,
    },
    (-1, 0): {
        20: 6,
        19: 7,
        18: 8,
        17: 9,
        16: 10,
    }
}

# def get_road_match_pattern(gdf, idx):
#     tile_xidx = gdf.xidx[idx]
#     tile_yidx = gdf.yidx[idx]
#     tag_category = gdf.category[idx]

#     exponent = 0
#     sum = 0
#     for yidx in range(tile_yidx + 1, tile_yidx - 2, -1):
#         for xidx in range(tile_xidx - 1, tile_xidx + 2):
#             tile = gdf[(gdf.xidx == xidx) & (gdf.yidx == yidx) & (gdf.category == tag_category)]
#             if len(tile) > 0:
#                 if tile.filled.any():
#                     sum += np.power(2, exponent)
            
#             exponent += 1
    
#     return sum

def get_road_match_pattern(gdf, idx):
    tile_xidx = gdf.xidx[idx]
    tile_yidx = gdf.yidx[idx]

    exponent = 0
    sum = 0
    for yidx in range(tile_yidx + 1, tile_yidx - 2, -1):
        for xidx in range(tile_xidx - 1, tile_xidx + 2):
            tile = gdf[(gdf.xidx == xidx) & (gdf.yidx == yidx)]
            if len(tile) > 0:
                sum += np.power(2, exponent)
            
            exponent += 1
    
    return sum


def type_from_tag(way_df, gdf, element, cm_types):
    tags = element.tags()
    for cm_type in cm_types['types']:
        matched = False
        for key, value in cm_type['tags']:
            if key in tags and tags[key] == value:
                matched = True
        
        if matched:
            for key in cm_type:
                if key in way_df.columns:
                    way_df[key] = cm_type[key]
                
    return way_df

def add_to_road_graph(df, gdf, element, cm_types):
    if len(df) == 0:
        return df

    if name not in road_graphs:
        road_graph = nx.Graph
        road_graphs[name] = road_graph

    coords = [(projection(coord[0], coord[1])) for coord in element.geometry()['coordinates']]
    points = [Point(coord[0], coord[1]) for coord in coords]

    ls = LineString(coords)
    ls_crosses = gdf.geometry.crosses(ls)

    squares = gdf.loc[ls_crosses, ['x', 'y', 'xidx', 'yidx']]
    normalized_dists = []
    # s = gdf.loc[ls_crosses]
    # for poly in s.geometry:
    #     	plt.plot(poly.exterior.xy[0], poly.exterior.xy[1], '-k')
    for idx in range(len(squares)):
        normalized_dists.append(ls.project(Point(squares['x'].values[idx], squares['y'].values[idx]), normalized=True))

    df['dist_along_way'] = normalized_dists

    if name not in node_dict:
        node_dict[name] = {}

    gdf_ls = gdf.loc[ls_crosses]
    nodes = []
    for node_idx in range(element.countNodes()):
        node_id = element.nodes()[node_idx].id()
        point = Point(coords[node_idx])
        gdf_point = gdf_ls.loc[gdf.contains(point)]
        if len(gdf_point) > 0:
            xidx, yidx = gdf_ls.loc[gdf.contains(point), ['xidx', 'yidx']].values[0]
            nodes.append((node_id, xidx, yidx))

    sorted_df = df.sort_values(by='dist_along_way')
    go_on = True
    df_pos_idx = 0
    node_idx = 0
    current_edge = None
    while go_on:
        node_id, node_xidx, node_yidx = nodes[node_idx]
        df_xidx, df_yidx = sorted_df.loc[sorted_df.index[df_pos_idx], ['xidx', 'yidx']].values

        if node_xidx == df_xidx and node_yidx == df_yidx:
            if current_edge is None:
                current_edge = [(node_id, node_xidx, node_yidx)]
                node_idx += 1
            else:
                current_edge.append((node_id, node_xidx, node_yidx))
                road_graph.add_edge(current_edge[0][0], current_edge[-1][0], squares=[edge_element[1:3] for edge_element in current_edge])
                road_graph.nodes[current_edge[0][0]]['square'] = current_edge[0][1:3] 
                road_graph.nodes[current_edge[-1][0]]['square'] = current_edge[-1][1:3] 
                current_edge = [current_edge[-1]]
                node_idx += 1
        else:
            df_pos_idx += 1
        
        if df_pos_idx == len(sorted_df) or node_idx == len(nodes):
            go_on = False
        
    return df


def type_random_area(df, gdf, name, cm_types):
    n_types = len(cm_types['types'])
    sub_df = df[df['name'] == name]

    sum_of_weights = sum([cm_type['weight'] if 'weight' in cm_type else 1.0 for cm_type in cm_types['types']])
    probabilities = [cm_type['weight'] / sum_of_weights if 'weight' in cm_type else 1.0 / sum_of_weights for cm_type in cm_types['types']]

    rng = np.random.default_rng()

    for sub_group_name, sub_group in sub_df.groupby(by='id'):
        type_idx = rng.choice(list(range(n_types)), p=probabilities, size=1)[0]
        cm_type = cm_types['types'][type_idx]
        for key in cm_type:
            if key in df.columns:
                df.loc[sub_group.index, key] = cm_type[key]

def type_random_individual(df, gdf, name, cm_types):
    n_types = len(cm_types['types'])
    sub_df = df[df['name'] == name]

    sum_of_weights = sum([cm_type['weight'] if 'weight' in cm_type else 1.0 for cm_type in cm_types['types']])
    probabilities = [cm_type['weight'] / sum_of_weights if 'weight' in cm_type else 1.0 / sum_of_weights for cm_type in cm_types['types']]

    rng = np.random.default_rng()
    type_indices = rng.choice(list(range(n_types)), p=probabilities, size=(len(sub_df),))

    # type_indices = np.random.randint(0, n_types, size=(len(sub_df),))
    for type_idx in range(n_types):
        idx = np.where(type_indices == type_idx)[0]
        if len(idx) > 0:
            cm_type = cm_types['types'][type_idx]
            for key in cm_type:
                if type(cm_type[key]) != list and key in df.columns:
                    df.loc[sub_df.index[idx], key] = cm_type[key]

def cat2_random_individual(df, gdf, name, cm_types):
    sub_df = df[df['name'] == name]
    for cat1, subgroup in sub_df.groupby(by='cat1'):
        cat2 = None
        for cm_type in cm_types['types']:
            if cm_type['cat1'] == cat1:
                cat2 = cm_type['cat2']
                break

        if cat2 is not None:
            cat2_indices = np.random.randint(0, len(cat2), size=(len(subgroup),))
            for cat2_idx in range(len(cat2)):
                idx = np.where(cat2_indices == cat2_idx)[0]
                if len(idx) > 0:
                    df.loc[subgroup.index[idx], 'cat2'] = cat2[cat2_idx]
        
def road_navigation_graph(df, gdf, name, cm_types):
    road_graph = road_graphs[name]
    nodes_to_delete = []
    for node_id in road_graph.nodes:
        if len(list(nx.neighbors(road_graph, node_id))) == 2:
            nodes_to_delete.append(node_id)

    while len(nodes_to_delete) > 0:
        node_id = nodes_to_delete.pop()

        neighbor1, neighbor2 = list(nx.neighbors(road_graph, node_id))
        node = road_graph.nodes[node_id]
        edge1 = road_graph.edges[neighbor1, node_id]
        edge2 = road_graph.edges[neighbor2, node_id]
        new_squares = []
        new_squares.extend(edge1['squares'] if edge1['squares'][-1] == node['square'] else edge1['squares'][::-1])
        new_squares.extend(edge2['squares'] if edge2['squares'][0] == node['square'] else edge1['squares'][::-1])

        road_graph.add_edge(neighbor1, neighbor2, squares=new_squares)
        road_graph.remove_node(node_id)

    node_pos = {}
    for node in road_graph.nodes:
        node_pos[node] = road_graph.nodes[node]['square']

    # plt.figure()
    # plt.axis('equal')
    # nx.draw_networkx(road_graph, pos=node_pos)
    # plt.show()

    edge_graphs = {}
    for edge in road_graph.edges:
        graph = nx.DiGraph()
        node1_id = edge[0]
        node2_id = edge[1]
        squares = edge['squares']
        if squares[0] == road_graph.nodes[node2_id]['square']:
            squares = squares[::-1]

        last_pins = []
        for i_square in range(len(squares)):
            square = squares[i_square]
            if i_square < len(squares) - 1:
                square_diff = (squares[i_square + 1][0] - square[0], squares[i_square + 1][1] - square[1])

            if i_square == 0:
                if len(list(nx.neighbors(road_graph, node1_id))) == 1:
                    input_pins = [start_pin_dict[square_diff]]
                else:
                    input_pins = [start_intersection_dict[square_diff]]
            else:
                input_pins = [square2square_dict[square_diff][pin] for pin in last_pins]

            last_pins = []
            output_pins = list(square2square_dict[square_diff].keys())
            for in_pin in input_pins:
                for connected_pin, weight in pin2pin_dict[in_pin]:
                    if connected_pin in output_pins:
                        graph.add_edge((in_pin, i_square), (connected_pin, i_square), weight=weight)
                        last_pins.append(connected_pin)

            
                    
                        






def road_pattern(df, gdf, name, cm_types):
    sub_df = df[df.name == name]

    print('road elements')
    for group_id, group in sub_df.groupby(by='id'):
        sorted_group = group.sort_values(by='dist_along_way')

        for idx_pos, idx in enumerate(sorted_group.index):
            cut_sorted_group = sorted_group.loc[sorted_group.index[max(idx_pos-2,0):min(idx_pos+3, len(sorted_group.index))]]

            pattern = get_road_match_pattern(cut_sorted_group, idx)

            if pattern not in pattern2roadtile_dict:
                print('[{}] pattern: {} -> {}'.format(group_id, pattern, re.findall('...', bin(pattern)[2:].zfill(9)[::-1])))
            else:
                if type(pattern2roadtile_dict[pattern]) == int:
                    pattern = pattern2roadtile_dict[pattern]
                direction, road_row, road_col = pattern2roadtile_dict[pattern][0]
                df.loc[idx, 'direction'] = 'Direction {}'.format(direction + 1)
                df.loc[idx, 'cat2'] = 'Road Tile {}'.format(road_row * 3 + road_col + 1)

    print('intersections')
    for node_id in node_dict[name]:
        if len(node_dict[name][node_id]) < 2:
            continue

        node = api.query('node/{}'.format(node_id))
        coords = node.geometry()['coordinates']
        point = Point(projection(coords[0], coords[1]))

        squares = gdf.loc[gdf.contains(point), ['xidx', 'yidx']]
        if len(squares) == 0:
            continue

        xidx, yidx = squares.values[0]
        
        pattern_df = None
        way_ids = node_dict[name][node_id]
        first_node_idx = None
        for way_idx, way_id in enumerate(way_ids):
            sorted_df = df.loc[df.id == way_id].sort_values(by='dist_along_way')
            node_idx = sorted_df.loc[(df.xidx == xidx) & (df.yidx == yidx)].index[0]
            node_idx_pos = np.where(sorted_df.index == node_idx)[0][0]
            cut_sorted_df = sorted_df.loc[sorted_df.index[max(node_idx_pos-2,0):min(node_idx_pos+3, len(sorted_df.index))]]

            if way_idx > 0:
                df.drop(node_idx, inplace=True)
            else:
                first_node_idx = node_idx

            if pattern_df is None:
                pattern_df = cut_sorted_df
            else:
                pattern_df = pandas.concat([pattern_df, cut_sorted_df])

        pattern = get_road_match_pattern(pattern_df, first_node_idx)

        if pattern not in pattern2roadtile_dict:
            print('pattern: {} -> {}'.format(pattern, re.findall('...', bin(pattern)[2:].zfill(9)[::-1])))
        else:
            if type(pattern2roadtile_dict[pattern]) == int:
                pattern = pattern2roadtile_dict[pattern]
            direction, road_row, road_col = pattern2roadtile_dict[pattern][0]
            df.loc[first_node_idx, 'direction'] = 'Direction {}'.format(direction + 1)
            df.loc[first_node_idx, 'cat2'] = 'Road Tile {}'.format(road_row * 3 + road_col + 1)



                


overpass = Overpass()
api = Api()
# query = overpassQueryBuilder(bbox=[7.30153, 50.93133, 7.30745, 50.93588], elementType='way')

# bbox = [50.93133, 7.30153, 50.93588, 7.30745] # lat_min, lon_min, lat_max, lon_max
projection = Proj(proj='utm', zone=32, ellps='WGS84')

bbox_utm = [379964.0, 5643796.0, 380804.0-8, 5644444.0-8]
lon_min, lat_min = projection(bbox_utm[0], bbox_utm[1], inverse=True)
lon_max, lat_max = projection(bbox_utm[2], bbox_utm[3], inverse=True)

bbox = [lat_min, lon_min, lat_max, lon_max]


query = overpassQueryBuilder(bbox=bbox, elementType=['way', 'relation'], includeGeometry=True, out='body')

result = overpass.query(query)


n_bins_x = np.ceil((bbox_utm[2] - bbox_utm[0]) / 8).astype(int)
n_bins_y = np.ceil((bbox_utm[3] - bbox_utm[1]) / 8).astype(int)

bins_x = np.linspace(bbox_utm[0], bbox_utm[0] + n_bins_x * 8, n_bins_x + 1)
bins_y = np.linspace(bbox_utm[1], bbox_utm[1] + n_bins_y * 8, n_bins_y + 1)

xarr = []
yarr = []
xiarr = []
yiarr = []
for xidx, x in enumerate(np.linspace(bbox_utm[0] + 4, bbox_utm[0] + 4 + (n_bins_x - 1) * 8, n_bins_x)):
    for yidx, y in enumerate(np.linspace(bbox_utm[1] + 4, bbox_utm[1] + 4 + (n_bins_y - 1) * 8, n_bins_y)):
        xarr.append(x)
        yarr.append(y)
        xiarr.append(xidx)
        yiarr.append(yidx)

geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, cap_style=3)
# gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr, 'filled': [False] * len(xarr),
#                               'pattern': [-1] * len(xarr), 'tile_page': [-1] * len(xarr), 'tile_row': [-1] * len(xarr), 'tile_col': [-1] * len(xarr), 
#                               'z': [-1] * len(xarr), 'category': [-1] * len(xarr), 'type': [-1] * len(xarr), 'sub_type': [-1] * len(xarr)
#                              }, geometry=geometry)

gdf = geopandas.GeoDataFrame({
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
    'dist_along_way': [-1] * len(xarr),
    }, geometry=geometry)


n_pages_x, n_x_remain = np.divmod(n_bins_x, PAGE_N_SQUARES_X)
n_pages_y, n_y_remain = np.divmod(n_bins_y, PAGE_N_SQUARES_Y)
n_x_remain = (np.floor(n_x_remain / 2) * 2).astype(int)
n_y_remain = (np.floor(n_y_remain / 2) * 2).astype(int)

grid_polygons = [
    LineString([
        (bbox_utm[0], bbox_utm[1]),
        (bbox_utm[2], bbox_utm[1]),
        (bbox_utm[2], bbox_utm[3]),
        (bbox_utm[0], bbox_utm[3]),
        (bbox_utm[0], bbox_utm[1])
        ]).buffer(0.1)
]
for i_page_x in range(int(n_pages_x)):
    grid_polygons.append(
        LineString([
            (bbox_utm[0] + (i_page_x + 1) * PAGE_N_SQUARES_X * 8, bbox_utm[1]),
            (bbox_utm[0] + (i_page_x + 1) * PAGE_N_SQUARES_X * 8, bbox_utm[3]),
        ]).buffer(0.1)
    )
for i_page_y in range(int(n_pages_y)):
    grid_polygons.append(
        LineString([
            (bbox_utm[0], bbox_utm[1] + (i_page_y + 1) * PAGE_N_SQUARES_Y * 8),
            (bbox_utm[2], bbox_utm[1] + (i_page_y + 1) * PAGE_N_SQUARES_Y * 8),
        ]).buffer(0.1)
    )

grid_polygons = MultiPolygon(grid_polygons)
# plt.figure()
# plt.axis('equal')


# df = pandas.DataFrame(columns=['x', 'y', 'z', 'menu', 'cat1', 'cat2', 'direction', 'id', 'name'])
df = None

road_graphs = {}

for element in result.elements():
    if element.type() not in ('relation', 'way'):
        continue
    element_tags = element.tags()
    if element.tags() is None:
        print(element.tags(), element.id())
        continue

    matched = False
    for name in config:
        # if config['name']['pass'] != i_pass:
        #     continue
        excluded = False
        if 'exclude_tags' in config[name]:
            for element_tag_key in element_tags:
                if element_tag_key in config[name]['exclude_tags'] and config[name]['exclude_tags'][element_tag_key] == element_tags[element_tag_key]:
                    excluded = True

        if 'required_tags' in config[name]:
            for required_tag_key in config[name]['required_tags']:
                if not (required_tag_key in element_tags and config[name]['required_tags'][required_tag_key] == element_tags[required_tag_key]):
                    excluded = True

        if excluded:
            continue

        for tag_key, tag_value in config[name]['tags']:
            if tag_key in element_tags and element_tags[tag_key] == tag_value:
                to_fill = None
                matched = True
                element_geometry = element.geometry()
                if element_geometry['type'] == 'Polygon':
                    exterior_coords = [(projection(coord[0], coord[1])) for coord in element_geometry['coordinates'][0]]
                    interiors = []
                    for interior_idx in range(1, len(element_geometry['coordinates'])):
                        interior_coords = [(projection(coord[0], coord[1])) for coord in element_geometry['coordinates'][interior_idx]]
                        interiors.append(interior_coords)

                    polygon = Polygon(exterior_coords, holes=interiors)
                    within = gdf.geometry.within(polygon)
                    intersecting = gdf.geometry.intersects(polygon)
                    is_border = np.bitwise_and(intersecting, ~within)
                    is_largest_square_area = gdf.loc[is_border].geometry.intersection(polygon).area > 32

                    to_fill = np.bitwise_or(within, is_largest_square_area)
                elif element_geometry['type'] == 'LineString':
                    coords = [(projection(coord[0], coord[1])) for coord in element.geometry()['coordinates']]
                    ls = LineString(coords)
                    to_fill = gdf.geometry.crosses(ls)
                else:
                    raise Exception('geometry {} not yet covered'.format(element_geometry['type']))

                if to_fill is not None:
                    element_df = gdf.loc[to_fill, ['xidx', 'yidx', 'z', 'menu', 'cat1', 'cat2', 'direction', 'id', 'name']].copy(deep=True)
                    element_df['id'] = element.id()
                    element_df['name'] = name

                    cm_types = config[name]['cm_types']
                    if 'process' in cm_types:
                        for func_name in cm_types['process']:
                            element_df = globals()[func_name](element_df, gdf, element, cm_types)

                    if df is None:
                        df = element_df
                    else:
                        df = pandas.concat([df, element_df], ignore_index=True)

    if not matched:
        print(element.tags(), element.id())

node_pos = {}
for node in road_graph.nodes:
    node_pos[node] = road_graph.nodes[node]['square']

plt.figure()
plt.axis('equal')
nx.draw_networkx(road_graph, pos=node_pos)
plt.show()

for name in config:
    cm_types = config[name]['cm_types']
    if 'post_process' in cm_types:
        for func_name in cm_types['post_process']:
            globals()[func_name](df, gdf, name, cm_types)


# for way in result.ways():
#     if way.tags() is None:
#         continue
#     if 'water' in way.tags():
#         tag_category = way.tags()['water']
#         coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates'][0]]
#         polygon = Polygon(coords)

#         intersects = gdf.geometry.intersects(polygon)

#         not_filled_with_lower_category = ~((gdf.category != -1) & (gdf.category < osm2cm_dict['water'][tag_category][0]) & (gdf.filled))
#         to_fill = intersects & not_filled_with_lower_category
        
#         gdf['filled'] = np.bitwise_or(gdf['filled'], to_fill)
#         indices = np.where(to_fill)[0]
#         gdf.loc[to_fill, 'category'] = osm2cm_dict['water'][tag_category][0]
#         gdf.loc[to_fill, 'type'] = osm2cm_dict['water'][tag_category][1]
#         gdf.loc[to_fill, 'sub_type'] = osm2cm_dict['water'][tag_category][2]
#         for idx in indices:
#             plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-b')

#         a = 1
        
#     if 'landuse' in way.tags():
#         if way.tags()['landuse'] == 'forest':
#             if 'type' in way.tags() and way.tags['type'] == 'multipolygon':
#                 continue
#             coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates'][0]]
#             polygon = Polygon(coords)
#             intersects = gdf.geometry.intersects(polygon)
#             gdf['filled'] = np.bitwise_or(gdf['filled'], intersects)
#             gdf.loc[intersects, 'category'] = 0
#             gdf.loc[intersects, 'type'] = 'Foliage'

#             tree_dict = {
#                 0: 'Tree A',
#                 1: 'Tree B',
#                 2: 'Tree C',
#                 3: 'Tree D',
#                 4: 'Tree E',
#                 5: 'Tree F',
#                 6: 'Tree G',
#                 7: 'Tree H',
#             }

#             sub_type = [tree_dict[tidx] for tidx in np.random.randint(0, 8, size=(len(intersects[intersects]),))]
#             gdf.loc[intersects, 'sub_type'] = sub_type
#             indices = np.where(intersects)[0]
#             for idx in indices:
#                 plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-g')
#         elif way.tags()['landuse'] == 'farmland':
#             if 'type' in way.tags() and way.tags['type'] == 'multipolygon':
#                 continue
#             coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates'][0]]
#             polygon = Polygon(coords)
#             intersects = gdf.geometry.intersects(polygon)

#             farmland_dict = {
#                 0: ('Ground 2', 'Plow NS'),
#                 1: ('Ground 2', 'Plow EW'),
#                 2: ('Ground 3', 'Crop 1'),
#                 3: ('Ground 3', 'Crop 2'),
#                 4: ('Ground 3', 'Crop 3'),
#                 5: ('Ground 3', 'Crop 4'),
#                 6: ('Ground 3', 'Crop 5'),
#                 7: ('Ground 3', 'Crop 6'),
#             }

#             farmland_type = farmland_dict[np.random.randint(0, 8)]

#             gdf['filled'] = np.bitwise_or(gdf['filled'], intersects)
#             gdf.loc[intersects, 'category'] = 0
#             gdf.loc[intersects, 'type'] = farmland_type[0]
#             gdf.loc[intersects, 'sub_type'] = farmland_type[1]
#             indices = np.where(intersects)[0]
#             for idx in indices:
#                 plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-y')

#     if 'natural' in way.tags():
#         if way.tags()['natural'] == 'scrub':
#             if 'type' in way.tags() and way.tags['type'] == 'multipolygon':
#                 continue
#             coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates'][0]]
#             polygon = Polygon(coords)
#             intersects = gdf.geometry.intersects(polygon)
#             gdf['filled'] = np.bitwise_or(gdf['filled'], intersects)
#             gdf.loc[intersects, 'category'] = 0
#             gdf.loc[intersects, 'type'] = 'Foliage'

#             tree_dict = {
#                 0: 'Bush A',
#                 1: 'Bush B',
#                 2: 'Bush C',
#             }

#             sub_type = [tree_dict[tidx] for tidx in np.random.randint(0, 3, size=(len(intersects[intersects]),))]
#             gdf.loc[intersects, 'sub_type'] = sub_type
#             indices = np.where(intersects)[0]
#             for idx in indices:
#                 plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-c')



#     if 'highway' in way.tags():
#         tag_category = way.tags()['highway']
#         if way.tags()['highway'] not in osm2cm_dict['highway']:
#             continue
#         coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates']]
#         ls = LineString(coords)

#         crosses = geometry.crosses(ls)

#         not_filled_with_lower_category = ~((gdf.category != -1) & (gdf.category < osm2cm_dict['highway'][tag_category][0]) & (gdf.filled))
#         to_fill = crosses & not_filled_with_lower_category
        
#         gdf['filled'] = np.bitwise_or(gdf['filled'], to_fill)
#         indices = np.where(to_fill)[0]
#         gdf.loc[to_fill, 'category'] = osm2cm_dict['highway'][tag_category][0]
#         gdf.loc[to_fill, 'type'] = osm2cm_dict['highway'][tag_category][1]
#         gdf.loc[to_fill, 'sub_type'] = osm2cm_dict['highway'][tag_category][2]
#         for idx in indices:
#             plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-k')


# rindices = np.where(gdf.filled & (gdf['type'] == 'road'))[0]
# for ridx in rindices:
#     pattern = get_road_match_pattern(gdf, ridx)
#     entry = None
#     if pattern in pattern2roadtile_dict:
#         entries = pattern2roadtile_dict[pattern]
#         if type(entries) == int:
#             entries = pattern2roadtile_dict[entries]
#         if len(entries) > 1:
#             entry = entries[np.random.randint(0, len(entries))]
#         else:
#             entry = entries[0]

#     gdf.iloc[ridx, 5] = pattern
#     if entry is not None:
#         gdf.iloc[ridx, 6] = entry[0]
#         gdf.iloc[ridx, 7] = entry[1]
#         gdf.iloc[ridx, 8] = entry[2]
        
# unmatched_patterns = gdf[(gdf.pattern != -1) & (gdf.tile_page == -1)].pattern.unique()
# for p in unmatched_patterns: 
#     print('pattern: {} -> {}'.format(p, re.findall('...', bin(p)[2:].zfill(9)[::-1])))
#     indices = np.where(gdf.pattern == p)[0]
#     for idx in indices:
#         plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-r')



# plt.show()

df_out = df.rename(columns={"xidx": "x", "yidx": "y"})
df_out.to_csv('osm_test_objects2.csv')

# gdf_out = gdf[(gdf.tile_page != -1) & (gdf.tile_row != -1) & (gdf.tile_col != -1)]
# gdf_out = gdf[gdf['type'] != -1]

# gdf_out.x = gdf_out.xidx
# gdf_out.y = gdf_out.yidx
# gdf_out.to_csv('osm_test_roads.csv', columns=['x', 'y', 'z', 'tile_page', 'tile_row', 'tile_col', 'category', 'type', 'sub_type'])

# df = pandas.DataFrame(waypoints)


# df['x'] = pandas.cut(df['utm_x'], bins=bins_x, retbins=True, labels=list(range(n_bins_x)))[0]
# df['y'] = pandas.cut(df['utm_y'], bins=bins_y, retbins=True, labels=list(range(n_bins_y)))[0]

# df = df[~pandas.isna(df.x) & ~pandas.isna(df.y)]

# plt.figure()
# plt.plot(df.x, df.y, 'o')
# plt.show()

# df.to_csv('osm.csv')

a = 1






