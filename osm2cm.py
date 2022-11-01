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
from profile.general import road_tiles

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
    # 'road': {
    #     'tags': [
    #         ('highway', 'primary'),
    #         ('highway', 'secondary'),
    #         ('highway', 'residential'),
    #         ('highway', 'living_street'),
    #         ('highway', 'unclassified'),
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Road', 'cat1': 'Paved 1', 'tags': [('highway', 'primary'), ('highway', 'secondary')]},
    #             {'menu': 'Road', 'cat1': 'Paved 2', 'tags': [('highway', 'residential'), ('highway', 'living_street'), ('highway', 'unclassified')]}
    #         ],
    #         'process': [
    #             'type_from_tag',
    #             'add_to_road_graph'
    #         ],
    #         'post_process': {
    #             'road_navigation_graph'
    #         }
    #     },
    #     'pass': 2
    # },
    # 'foot_path': {
    #     'tags': [
    #         ('highway', 'footway'),
    #         ('highway', 'path')
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Road', 'cat1': 'Gravel Road', 'tags': [('highway', 'footway'), ('highway', 'footway')]},
    #         ],
    #         'process': [
    #             'type_from_tag',
    #             'add_to_road_graph'
    #         ],
    #         'post_process': {
    #             'road_pattern'
    #         }
    #     },
    #     'pass': 2,
    # },
    # 'water': {
    #     'tags': [
    #         ('water', 'river'),
    #         # ('water', 'pond'),
    #         # ('water', 'fishpond'),
    #         # ('water', 'wastewater'),
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Ground 2', 'cat1': 'Water', 'tags': [('water', 'river')]},
    #             {'menu': 'Ground 2', 'cat1': 'Deep Ford', 'tags': [('water', 'pond'), ('water', 'fishpond'), ('water', 'wastewater')]},
    #         ],
    #         'process': [
    #             'type_from_tag'
    #         ]
    #     },
    #     'pass': 3,
    # },

    'wetland': {
        'tags': [
            ('natural', 'wetland'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 2', 'cat1': 'Marsh', 'tags': [('natural', 'wetland')]},
            ],
            'process': [
                'type_from_tag'
            ]
        },
        'pass': 3,
    },

    'garden': {
        'tags': [
            ('leisure', 'garden'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 3', 'cat1': 'Crop 1', 'weight': 1.0},
                {'menu': 'Ground 2', 'cat1': 'Plow NS', 'weight': 0.5},
                {'menu': 'Ground 2', 'cat1': 'Plow EW', 'weight': 0.5},
                {'menu': 'Ground 1', 'cat1': 'Grass', 'weight': 1.0},
                {'menu': 'Ground 1', 'cat1': 'Flowers', 'weight': 1.0},
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 1', 'weight': 0.5},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 1', 'weight': 0.5},
            ],
            'post_process': [
                'type_random_individual',
            ]
        }

    },

    # 'mixed_forest': {
    #     'tags': [
    #         ('landuse', 'forest'),
    #         ('natural', 'wood')
    #     ],
    #     'exclude_tags': {
    #         'leaf_type': 'broadleaved',
    #     },
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree E', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree E', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree E', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree H', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree H', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree H', 'cat2': 'density 3', 'weight': 1},
    #         ],
    #         'post_process': [
    #             'type_random_individual',
    #         ]
    #     },
    #     'pass': 1
    # },

    # 'broadleaved_forest': {
    #     'tags': [
    #         ('landuse', 'forest'),
    #         ('natural', 'wood')
    #     ],
    #     'required_tags': {
    #         'leaf_type': 'broadleaved',
    #     },
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree B', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree C', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree F', 'cat2': 'density 3', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 1', 'weight': 1},
    #             {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 2', 'weight': 2},
    #             {'menu': 'Foliage', 'cat1': 'Tree G', 'cat2': 'density 3', 'weight': 1},
    #         ],
    #         'post_process': [
    #             'type_random_individual',
    #         ]
    #     },
    #     'pass': 1
    # },

    # 'mixed_bushes': {
    #     'tags': [
    #         ('natural', 'scrub')
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Foliage', 'cat1': 'Bush A', 'cat2': 'density 1', 'weight': 1.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush A', 'cat2': 'density 2', 'weight': 2.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush A', 'cat2': 'density 3', 'weight': 1.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush B', 'cat2': 'density 1', 'weight': 1.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush B', 'cat2': 'density 2', 'weight': 2.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush B', 'cat2': 'density 3', 'weight': 1.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush C', 'cat2': 'density 1', 'weight': 1.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush C', 'cat2': 'density 2', 'weight': 2.0},
    #             {'menu': 'Foliage', 'cat1': 'Bush C', 'cat2': 'density 3', 'weight': 1.0},
    #         ],
    #         'post_process': [
    #             'type_random_individual',
    #         ]
    #     },
    #     'pass': 1
    # },

    # 'farmland': {
    #     'tags': [
    #         ('landuse', 'farmland'),
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Ground 2', 'cat1': 'Plow NS', 'weight': 1.0},
    #             {'menu': 'Ground 2', 'cat1': 'Plow EW', 'weight': 1.0},
    #             {'menu': 'Ground 3', 'cat1': 'Crop 1', 'weight': 1.0},
    #             {'menu': 'Ground 3', 'cat1': 'Crop 2', 'weight': 1.0},
    #             {'menu': 'Ground 3', 'cat1': 'Crop 3', 'weight': 1.0},
    #             {'menu': 'Ground 3', 'cat1': 'Crop 4', 'weight': 1.0},
    #             {'menu': 'Ground 3', 'cat1': 'Crop 5', 'weight': 1.0},
    #             {'menu': 'Ground 3', 'cat1': 'Crop 6', 'weight': 1.0},
    #         ],
    #         'post_process': [
    #             'type_random_area'
    #         ]
    #     },
    #     'pass': 0
    # },

    'orchard': {
        'tags': [
            ('landuse', 'orchard'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Foliage', 'cat1': 'Tree A', 'cat2': 'density 4', 'weight': 1},
                {'menu': 'Foliage', 'cat1': 'Tree D', 'cat2': 'density 4', 'weight': 1},
            ],
            'post_process': [
                'type_random_area'
            ]
        },
        'pass': 0
    },

    # 'grassland': {
    #     'tags': [
    #         ('landuse', 'grass')
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Ground 1', 'cat1': 'Grass T', 'weight': 1.0},
    #             {'menu': 'Ground 1', 'cat1': 'Grass TY', 'weight': 1.0},
    #             {'menu': 'Ground 1', 'cat1': 'Weeds', 'weight': 1.0},
    #             {'menu': 'Ground 1', 'cat1': 'Grass XT', 'weight': 1.0},
    #             {'menu': 'Ground 1', 'cat1': 'Grass XTY', 'weight': 1.0},
    #         ],
    #         'post_process': [
    #             'type_random_individual'
    #         ],
    #     }
    # },

    # 'meadow': {
    #     'tags': [
    #         ('landuse', 'meadow'),
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Ground 1', 'cat1': 'Grass', 'tags': [('landuse', 'meadow')]},
    #         ],
    #         'process': [
    #             'type_from_tag'
    #         ]
    #     },
    #     'pass': 3,
    # },

    # 'construction_site': {
    #     'tags': [
    #         ('landuse', 'construction')
    #     ],
    #     'cm_types': {
    #         'types': [
    #             {'menu': 'Ground 1', 'cat1': 'Dirt', 'weight': 1.0},
    #         ],
    #         'post_process': [
    #             'type_random_individual'
    #         ]
    #     }
    # },

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
    # },

    'pitch': {
        'tags': [
            ('leisure', 'pitch'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 1', 'cat1': 'Dirt Red', 'tags': [('leisure', 'pitch')]}
            ],
            'process': [
                'type_from_tag',
            ],
        }
    },

    'gravel_beach': {
        'tags': [
            ('natural', 'beach'),
        ],
        'required_tags': {
            'surface': 'gravel',
        },
        'cm_types': {
            'types': [
                {'menu': 'Ground 2', 'cat1': 'Gravel', 'tags': [('natural', 'beach')]}
            ],
            'process': [
                'type_from_tag',
            ],
        }
    },

    'playground': {
        'tags': [
            ('leisure', 'playground'),
        ],
        'cm_types': {
            'types': [
                {'menu': 'Ground 1', 'cat1': 'Sand', 'tags': [('leisure', 'playground')]}
            ],
            'process': [
                'type_from_tag',
            ],
        }
    },


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
    1: [(6, 0.7), (20, 0.7), (7, 1.1)],
    2: [(13, 1.1), (16, 1.1)],
    3: [(13, 1), (12, 1.1), (8, 1.1), (18, 1.1), (14, 1.1)],
    4: [(10, 1.1), (13, 1.1)],
    5: [(6, 0.7), (12, 1.1), (20, 0.7), (19, 1.1)],
    6: [(1, 0.7), (5, 0.7), (11, 0.7)],
    7: [(1, 1.1), (18, 1.1)],
    8: [(3, 1.1), (18, 1), (17, 1.1), (13, 1.1), (19, 1.1)],
    9: [(15, 1.1), (18, 1.1)],
    10: [(4, 1.1), (11, 0.7)],
    11: [(6, 0.7), (10, 0.7), (17, 1.1), (16, 0.7)],
    12: [(3, 1.1), (6, 1.1)],
    13: [(3, 1), (4, 1.1), (8, 1.1), (18, 1.1), (2, 1.1)],
    14: [(3, 1.1), (20, 1.1)],
    15: [(9, 1.1), (16, 0.7)],
    16: [(2, 1.1), (15, 0.7), (11, 0.7)],
    17: [(8, 1.1), (11, 1.1)],
    18: [(3, 1.1), (8, 1), (13, 1.1), (9, 1.1), (7, 1.1)],
    19: [(8, 1.1), (5, 1.1)],
    20: [(1, 0.7), (14, 1.1), (5, 0.7)],
}

start_pin_dict = {
    (0, 1): [3],
    (0, -1): [13],
    (1, 0): [18],
    (-1, 0): [8],
}

end_pin_dict = {
    (0, 1): [13],
    (0, -1): [3],
    (1, 0): [8],
    (-1, 0): [18],
}

start_intersection_dict = {
    (0, 1): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20],
    (0, -1): [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    (-1, 0): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    (1, 0): [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
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


# direction, row, col
pin2roadtile_dict = {
    (3, 13): [(0, 0, 0), (2, 0, 0)],
    (1, 6): [(0, 0, 1)],
    (5, 6): [(0, 0, 1)],
    (3, 12): [(0, 0, 2)],
    (3, 8): [(0, 1, 0)],
    (4, 10): [(0, 1, 1)],
    (3, 8, 13): [(0, 1, 2)],
    (1, 5, 6, 20): [(0, 2, 0)],
    (4, 9, 13): [(0, 2, 1)],
    (3, 8, 13, 18): [(0, 2, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2)],
    (3, 14): [(0, 3, 0)],
    (2, 16): [(0, 3, 1)],
    (2, 13, 16): [(0, 3, 2)],

    (8, 18): [(1, 0, 0), (3, 0, 0)],
    (6, 11): [(1, 0, 1)],
    (10, 11): [(1, 0, 1)],
    (8, 17): [(1, 0, 2)],
    (8, 13): [(1, 1, 0)],
    (9, 15): [(1, 1, 1)],
    (8, 13, 18): [(1, 1, 2)],
    (5, 6, 10, 11): [(1, 2, 0)],
    (9, 15, 18): [(1, 2, 1)],
    (8, 19): [(1, 3, 0)],
    (1, 7): [(1, 3, 1)],
    (1, 7, 18): [(1, 3, 2)],

    (11, 16): [(2, 0, 1)],
    (15, 16): [(2, 0, 1)],
    (2, 13): [(2, 0, 2)],
    (13, 18): [(2, 1, 0)],
    (14, 20): [(2, 1, 1)],
    (3, 13, 18): [(2, 1, 2)],
    (10, 11, 15, 16): [(2, 2, 0)],
    (3, 14, 19): [(2, 2, 1)],
    (4, 13): [(2, 3, 0)],
    (6, 12): [(2, 3, 1)],
    (3, 6, 12): [(2, 3, 2)],

    (1, 20): [(3, 0, 1)],
    (4, 20): [(3, 0, 1)],
    (7, 18): [(3, 0, 2)],
    (3, 18): [(3, 1, 0)],
    (5, 19): [(3, 1, 1)],
    (3, 8, 18): [(3, 1, 2)],
    (1, 15, 16, 20): [(3, 2, 0)],
    (5, 8, 19): [(3, 2, 1)],
    (9, 18): [(3, 3, 0)],
    (11, 17): [(3, 3, 1)],
    (8, 11, 17): [(3, 3, 2)],

}

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

def extract_connection_direction_from_node(node_id, other_node_id, road_graph):
    squares = road_graph.edges[node_id, other_node_id]['squares']
    node_square = road_graph.nodes[node_id]['square']
    if squares[0] != node_square:
        squares = squares[::-1]
    
    return road_direction_dict[(squares[1][0] - squares[0][0], squares[1][1] - squares[0][1])]

def extract_valid_tiles_from_node(node_id, other_node_id, road_graph, edge_graphs):
    # direction_to_other_node = extract_connection_direction_from_node(node_id, other_node_id, road_graph)
    valid_connections = []
    if (node_id, other_node_id) in edge_graphs:
        graph = edge_graphs[(node_id, other_node_id)]['graph']
        if nx.has_path(graph, 'start', 'end'):
            paths = nx.all_shortest_paths(graph, 'start', 'end', weight='cost')
            for path in paths:
                valid_connections.append(path[1][0])
    elif (other_node_id, node_id) in edge_graphs:
        graph = nx.reverse_view(edge_graphs[(other_node_id, node_id)]['graph'])
        if nx.has_path(graph, 'end', 'start'):
            paths = nx.all_shortest_paths(graph, 'end', 'start', weight='cost')
            for path in paths:
                valid_connections.append(path[1][0])

    return valid_connections
    



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
        neighbors = [neighbor for neighbor in nx.neighbors(road_graph, node_id) if neighbor != node_id]
        if len(neighbors) == 2:
            nodes_to_delete.append(node_id)

    while len(nodes_to_delete) > 0:
        node_id = nodes_to_delete.pop()

        neighbors = list(nx.neighbors(road_graph, node_id))
        if len(neighbors) == 2:
            neighbor1, neighbor2 = neighbors
            node = road_graph.nodes[node_id]
            edge1 = road_graph.get_edge_data(neighbor1, node_id)[0]
            edge2 = road_graph.get_edge_data(neighbor2, node_id)[0]
        else:
            neighbor1 = neighbors[0]
            neighbor2 = neighbors[0]
            node = road_graph.nodes[node_id]
            edge1 = road_graph.get_edge_data(neighbor1, node_id)[0]
            edge2 = road_graph.get_edge_data(neighbor1, node_id)[1]

        new_squares = []
        new_squares.extend(edge1['squares'] if edge1['squares'][-1] == node['square'] else edge1['squares'][::-1])
        new_squares.extend(edge2['squares'][1::] if edge2['squares'][0] == node['square'] else edge2['squares'][::-1][1::])

        square_diffs = [(new_squares[idx][0] - new_squares[idx-1][0], new_squares[idx][1] - new_squares[idx-1][1]) for idx in range(1, len(new_squares))]
        assert (-1, -1) not in square_diffs

        road_graph.add_edge(neighbor1, neighbor2, squares=new_squares)
        road_graph.remove_node(node_id)

    road_graph = nx.Graph(road_graph)

    for edge in road_graph.edges:
        squares = road_graph.edges[edge[0], edge[1]]['squares']
        filtered_squares = []
        for sidx in range(1, len(squares)):
            if squares[sidx] != squares[sidx - 1]:
                filtered_squares.append(squares[sidx])

        road_graph.edges[edge[0], edge[1]]['squares'] = filtered_squares


    # node_pos = {}
    # for node in road_graph.nodes:
    #     node_pos[node] = road_graph.nodes[node]['square']

    # plt.figure()
    # plt.axis('equal')
    # nx.draw_networkx(road_graph, pos=node_pos)
    # plt.show()



    edge_graphs = {}
    for edge in road_graph.edges:
        graph = nx.DiGraph()
        node1_id = edge[0]
        node2_id = edge[1]
        squares = road_graph.edges[node1_id, node2_id]['squares']

        other_node1_neighbors = [neighbor for neighbor in nx.neighbors(road_graph, node1_id) if neighbor != node2_id]
        other_node2_neighbors = [neighbor for neighbor in nx.neighbors(road_graph, node2_id) if neighbor != node1_id]

        if len(squares) < 2:
            continue

        if squares[0] == road_graph.nodes[node2_id]['square']:
            squares = squares[::-1]

        # last_output_pins = []
        # last_direction = None
        last_tiles = []
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
                    valid_tiles = road_tiles[(road_tiles[direction_in] == (2,3)) & ~pandas.isnull(road_tiles[direction_out]) & (road_tiles['n_connections'] == 2)].index.values
                else:
                    directions_in = extract_required_directions(node1_id, other_node1_neighbors, road_graph)
                    connection_condition = ~pandas.isnull(road_tiles[directions_in[0]]) & (road_tiles['n_connections'] == len(directions_in) + 1)
                    for dir_idx in range(1, len(directions_in)):
                        connection_condition = np.bitwise_and(connection_condition, ~pandas.isnull(road_tiles[directions_in[dir_idx]]))

                    valid_tiles = road_tiles[~pandas.isnull(road_tiles[direction_out]) & connection_condition].index.values
                    
                last_tiles = valid_tiles
                for tile in valid_tiles:
                    graph.add_edge('start', (tile, i_square), cost=road_tiles.loc[tile, 'cost'])
            elif 0 < i_square < len(squares) - 1:
                last_direction_out = opposite_road_direction_dict[direction_in]
                new_last_tiles = []
                for last_tile in last_tiles:
                    tile_connection = road_tiles.loc[last_tile, last_direction_out]
                    valid_tiles = road_tiles.loc[(road_tiles[direction_in] == tile_connection) & ~pandas.isnull(road_tiles[direction_out]) & (road_tiles['n_connections'] == 2)].index.values
                    for tile in valid_tiles:
                        new_last_tiles.append(tile)
                        graph.add_edge((last_tile, i_square - 1), (tile, i_square), cost=road_tiles.loc[tile, 'cost'])
                
                last_tiles = np.unique(new_last_tiles)
            else:
                last_direction_out = opposite_road_direction_dict[direction_in]
                for last_tile in last_tiles:
                    tile_connection = road_tiles.loc[last_tile, last_direction_out]
                    if len(other_node2_neighbors) == 0:
                        direction_out = opposite_road_direction_dict[direction_in]
                        valid_tiles = road_tiles.loc[(road_tiles[direction_in] == tile_connection) & (road_tiles[direction_out] == (2,3)) & (road_tiles['n_connections'] == 2)].index.values
                    else:
                        directions_out = extract_required_directions(node2_id, other_node2_neighbors, road_graph)
                        connection_condition = ~pandas.isnull(road_tiles[directions_out[0]]) & (road_tiles['n_connections'] == len(directions_out) + 1)
                        for dir_idx in range(1, len(directions_out)):
                            connection_condition = np.bitwise_and(connection_condition, ~pandas.isnull(road_tiles[directions_out[dir_idx]]))

                        valid_tiles = road_tiles.loc[(road_tiles[direction_in] == tile_connection) & connection_condition].index.values

                    for tile in valid_tiles:
                        graph.add_edge((last_tile, i_square - 1), (tile, i_square), cost=road_tiles.loc[tile, 'cost'])
                        graph.add_edge((tile, i_square), 'end')

        edge_graphs[(node1_id, node2_id)] = {'graph': graph, 'squares': squares}

        #     if i_square < len(squares) - 1:
        #         direction = (squares[i_square + 1][0] - square[0], squares[i_square + 1][1] - square[1])
        #     else:
        #         direction = (square[0] - squares[i_square - 1][0], square[1] - squares[i_square - 1][1])

        #     if i_square == 0:
        #         if len(list(nx.neighbors(road_graph, node1_id))) == 1:
        #             input_pins = start_pin_dict[direction]
        #         else:
        #             input_pins = start_intersection_dict[direction]

        #         for pin in input_pins:
        #             graph.add_edge('start', (pin, i_square))
        #     else:
        #         input_pins = []
        #         for pin in last_output_pins:
        #             in_pin = square2square_dict[last_direction][pin]
        #             graph.add_edge((pin, i_square - 1), (in_pin, i_square))
        #             input_pins.append(in_pin)


        #     output_pins = []
        #     if i_square < len(squares) - 1:
        #         target_pins = list(square2square_dict[direction].keys())
        #     else:
        #         if len(list(nx.neighbors(road_graph, node2_id))) == 1:
        #             target_pins = end_pin_dict[direction]
        #         else:
        #             target_pins = end_intersection_dict[direction]

        #     print(input_pins)
        #     for in_pin in input_pins:
        #         for connected_pin, weight in pin2pin_dict[in_pin]:
        #             if connected_pin in target_pins:
        #                 graph.add_edge((in_pin, i_square), (connected_pin, i_square), weight=weight)
        #                 graph.nodes[(in_pin, i_square)]['square'] = square
        #                 output_pins.append(connected_pin)

        #     if i_square == len(squares) - 1:
        #         for pin in output_pins:
        #             graph.add_edge((pin, i_square), 'end')

        #     last_output_pins = np.unique(output_pins)
        #     last_direction = direction

        # edge_graphs[(node1_id, node2_id)] = {'graph': graph, 'squares': squares}

    for node_pair in edge_graphs:
        node1_id, node2_id = node_pair
        graph = edge_graphs[node_pair]['graph']
        squares = edge_graphs[node_pair]['squares']

        if not nx.has_path(graph, 'start', 'end'):
            continue

        node1_neighbors = [neighbor for neighbor in nx.neighbors(road_graph, node1_id) if neighbor != node2_id]
        node2_neighbors = [neighbor for neighbor in nx.neighbors(road_graph, node2_id) if neighbor != node1_id]

        valid_start_tiles = []
        valid_end_tiles = []
        for neighbor in node1_neighbors:
            valid_start_tiles.extend(extract_valid_tiles_from_node(node1_id, neighbor, road_graph, edge_graphs))
        for neighbor in node2_neighbors:
            valid_end_tiles.extend(extract_valid_tiles_from_node(node2_id, neighbor, road_graph, edge_graphs))

        valid_start_tiles = np.unique(valid_start_tiles)
        valid_end_tiles = np.unique(valid_end_tiles)

        paths = list(nx.all_shortest_paths(graph, 'start', 'end', weight='cost'))
        valid_paths = []
        valid_starts = False
        valid_ends = False
        for path in paths:
            valid_start = False
            valid_end = False
            if len(node1_neighbors) == 0 or (len(node1_neighbors) > 0 and path[1][0] in valid_start_tiles):
                valid_start = True
            if len(node2_neighbors) == 0 or (len(node2_neighbors) > 0 and path[-2][0] in valid_end_tiles):
                valid_end = True

            valid_starts = valid_starts or valid_start
            valid_ends = valid_ends or valid_end
            if valid_start and valid_end:
                valid_paths.append(path)

        if len(valid_paths) == 0:
            if valid_starts and not valid_ends:
                reason_string = 'no valid end was found'    
            elif not valid_starts and valid_ends:
                reason_string = 'no valid start was found'
            else:
                reason_string = 'no valid start or end were found'
            print('Could not find valid path from {} to {} because {}.'.format(node1_id, node2_id, reason_string))
            continue

        print('yeah!')
        path = valid_paths[0][1:-1]
        for idx in range(len(path)):
            square = squares[idx]
            direction, row, col = road_tiles.loc[path[idx][0], ['direction', 'row', 'col']]

            tile_condition = (df.xidx == square[0]) & (df.yidx == square[1]) & (df.name == name)

            df.loc[tile_condition, 'direction'] = 'Direction {}'.format(direction + 1)
            df.loc[tile_condition, 'cat2'] = 'Road Tile {}'.format(row * 3 + col + 1)
        
        # if len(node1_neighbors) > 0:
        fix_tile_for_node(node1_id, edge_graphs, path[0][0])
        # if len(node2_neighbors) > 0:
        fix_tile_for_node(node2_id, edge_graphs, path[-1][0])
        
        valid_paths = []
        a = 1

    df = df.drop(df.loc[(df.name == name) & (df.direction == -1)].index)
                        






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

# bbox_utm = [379964.0, 5643796.0, 380804.0-8, 5644444.0-8] # overath
bbox_utm = [379877.0, 5643109.0, 381461.0, 5645022.0] # overath extended
# bbox_utm = [550894, 5586630, 553442, 5589362] # doellbach
lon_min, lat_min = projection(bbox_utm[0], bbox_utm[1], inverse=True)
lon_max, lat_max = projection(bbox_utm[2], bbox_utm[3], inverse=True)

bbox = [lat_min, lon_min, lat_max, lon_max]


query = overpassQueryBuilder(bbox=bbox, elementType=['way', 'relation'], includeGeometry=True, out='body')

result = overpass.query(query)


n_bins_x = np.floor((bbox_utm[2] - bbox_utm[0]) / 8).astype(int)
n_bins_y = np.floor((bbox_utm[3] - bbox_utm[1]) / 8).astype(int)

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
generic_node_cnt = 0
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
                    to_fill = np.bitwise_or(gdf.geometry.crosses(ls), gdf.geometry.contains(ls))
                elif element_geometry['type'] == 'MultiPolygon':
                    polygons = []
                    for polygon_idx in range(len(element_geometry['coordinates'])):
                        polygon_coordinates = element_geometry['coordinates'][polygon_idx]
                        exterior_coords = [(projection(coord[0], coord[1])) for coord in polygon_coordinates[0]]
                        interiors = []
                        for interior_idx in range(1, len(polygon_coordinates)):
                            interior_coords = [(projection(coord[0], coord[1])) for coord in polygon_coordinates[interior_idx]]
                            interiors.append(interior_coords)
                        
                        polygons.append(Polygon(exterior_coords, holes=interiors))

                    multipolygon = MultiPolygon(polygons)
                    within = gdf.geometry.within(polygon)
                    intersecting = gdf.geometry.intersects(polygon)
                    is_border = np.bitwise_and(intersecting, ~within)
                    is_largest_square_area = gdf.loc[is_border].geometry.intersection(polygon).area > 32

                    to_fill = np.bitwise_or(within, is_largest_square_area)
                else:
                    raise Exception('geometry {} of element {}/{} not yet covered'.format(element_geometry['type'], element.type(), element.id()))

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

# node_pos = {}
# for node in road_graphs['road'].nodes:
#     node_pos[node] = road_graphs['road'].nodes[node]['square']

# plt.figure()
# plt.axis('equal')
# nx.draw_networkx(road_graphs['road'], pos=node_pos)
# plt.show()

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
df = pandas.concat((
    df, 
    pandas.DataFrame({
        'xidx': [gdf.xidx.max()], 
        'yidx': [gdf.yidx.max()], 
        'z': [-1], 
        'menu': [-1], 
        'cat1': [-1], 
        'cat2': [-1], 
        'direction': [-1], 
        'id': [-1], 
        'name': [-1]
    })
))

df_out = df.rename(columns={"xidx": "x", "yidx": "y"})
# df_out.to_csv('osm_test_objects2.csv')
df_out.to_csv('overath_extended_osm_roads.csv')

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






