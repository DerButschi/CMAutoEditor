from unicodedata import category
from OSMPythonTools.overpass import Overpass, overpassQueryBuilder
from pyproj import Proj
import pandas
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString, Point, MultiPoint
from shapely.ops import split, snap
import matplotlib.pyplot as plt
from skimage.draw import line, line_aa, line_nd, polygon
import geopandas
import re

PAGE_N_SQUARES_X = 104
PAGE_N_SQUARES_Y = 60

osm2cm_dict = {
    'highway': {
        'primary': (0, 'road', 'Paved 1'),
        'secondary': (0, 'road', 'Paved 1'),
        'residential': (0, 'road', 'Paved 2'),
        'living_street': (0, 'road', 'Paved 2'),
        'unclassified': (0, 'road', 'Paved 2'),
        'footway': (1, 'road', 'Foot Path'),
        'path': (1, 'road', 'Foot Path')
    }
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
    400: 402

}

def get_road_match_pattern(gdf, idx):
    tile_xidx = gdf.xidx[idx]
    tile_yidx = gdf.yidx[idx]
    tag_category = gdf.category[idx]

    exponent = 0
    sum = 0
    for yidx in range(tile_yidx + 1, tile_yidx - 2, -1):
        for xidx in range(tile_xidx - 1, tile_xidx + 2):
            tile = gdf[(gdf.xidx == xidx) & (gdf.yidx == yidx) & (gdf.category == tag_category)]
            if len(tile) > 0:
                if tile.filled.any():
                    sum += np.power(2, exponent)
            
            exponent += 1
    
    return sum




overpass = Overpass()
# query = overpassQueryBuilder(bbox=[7.30153, 50.93133, 7.30745, 50.93588], elementType='way')

# bbox = [50.93133, 7.30153, 50.93588, 7.30745] # lat_min, lon_min, lat_max, lon_max
projection = Proj(proj='utm', zone=32, ellps='WGS84')

lon_min, lat_min = projection(379964.0, 5643796.0, inverse=True)
lon_max, lat_max = projection(380804.0-8, 5644444.0-8, inverse=True)

bbox = [lat_min, lon_min, lat_max, lon_max]


query = overpassQueryBuilder(bbox=bbox, elementType='way', includeGeometry=True, out='body')

result = overpass.query(query)


bbox_utm = []
bbox_utm.extend(projection(bbox[1], bbox[0]))
bbox_utm.extend(projection(bbox[3], bbox[2]))

n_bins_x = np.ceil((bbox_utm[2] - bbox_utm[0]) / 8).astype(int)
n_bins_y = np.ceil((bbox_utm[3] - bbox_utm[1]) / 8).astype(int)

bins_x = np.linspace(bbox_utm[0], bbox_utm[0] + n_bins_x * 8, n_bins_x + 1)
bins_y = np.linspace(bbox_utm[1], bbox_utm[1] + n_bins_y * 8, n_bins_y + 1)

xarr = []
yarr = []
xiarr = []
yiarr = []
for xidx, x in enumerate(np.linspace(bbox_utm[0] + 4, bbox_utm[0] + n_bins_x * 8, n_bins_x)):
    for yidx, y in enumerate(np.linspace(bbox_utm[1] + 4, bbox_utm[1] + n_bins_y * 8, n_bins_y)):
        xarr.append(x)
        yarr.append(y)
        xiarr.append(xidx)
        yiarr.append(yidx)

geometry = geopandas.points_from_xy(xarr, yarr).buffer(4, cap_style=3)
gdf = geopandas.GeoDataFrame({'x': xarr, 'y': yarr, 'xidx': xiarr, 'yidx': yiarr, 'filled': [False] * len(xarr),
                              'pattern': [-1] * len(xarr), 'tile_page': [-1] * len(xarr), 'tile_row': [-1] * len(xarr), 'tile_col': [-1] * len(xarr), 
                              'z': [-1] * len(xarr), 'category': [-1] * len(xarr), 'type': [-1] * len(xarr), 'sub_type': [-1] * len(xarr)
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
plt.figure()
plt.axis('equal')
# plt.plot([bbox_utm[0], bbox_utm[2], bbox_utm[2], bbox_utm[0], bbox_utm[0]], [bbox_utm[1], bbox_utm[1], bbox_utm[3], bbox_utm[3], bbox_utm[1]])
# for poly in grid_polygons:
#     plt.plot(poly.exterior.xy[0], poly.exterior.xy[1], '-k')


for way in result.ways():
    if way.tags() is not None and 'highway' in way.tags():
        tag_category = way.tags()['highway']
        if way.tags()['highway'] not in osm2cm_dict['highway']:
            continue
        coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates']]
        ls = LineString(coords)

        crosses = geometry.crosses(ls)

        not_filled_with_lower_category = ~((gdf.category != -1) & (gdf.category < osm2cm_dict['highway'][tag_category][0]) & (gdf.filled))
        to_fill = crosses & not_filled_with_lower_category
        
        gdf['filled'] = np.bitwise_or(gdf['filled'], to_fill)
        indices = np.where(to_fill)[0]
        gdf.loc[to_fill, 'category'] = osm2cm_dict['highway'][tag_category][0]
        gdf.loc[to_fill, 'type'] = osm2cm_dict['highway'][tag_category][1]
        gdf.loc[to_fill, 'sub_type'] = osm2cm_dict['highway'][tag_category][2]
        for idx in indices:
            plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-k')




        # plt.plot(ls.xy[0], ls.xy[1], ':b')
        # plt.plot(ls_snap.xy[0], ls_snap.xy[1], '-ok')

# plt.show()
# waypoints = {'utm_x': [bbox_utm[2]], 'utm_y': [bbox_utm[3]], 'id': [-1], 'type': ['dummy'], 'z': [-1]}
# highways = []
# for way in result.ways():
#     if way.tags() is not None and ('highway' in way.tags() or 'railway' in way.tags()):
#         # highways.append(way.tags()['highway'])
#         if ('highway' in way.tags() and way.tags()['highway'] in ('residential', 'secondary', 'primary', 'living_street', 'unclassified', 'footway', 'path')) or ('railway' in way.tags()):
#             coords = [(projection(coord[0], coord[1])) for coord in way.geometry()['coordinates']]
#             ls = LineString(coords)
#             grid_split = split(ls, grid_polygons)
#             processed_coords = []
#             for geom in grid_split.geoms:
#                 plt.plot([geom.xy[0][pidx] for pidx in range(len(geom.xy[0]))], [geom.xy[1][pidx] for pidx in range(len(geom.xy[1]))])
#                 processed_coords.extend((geom.xy[0][pidx], geom.xy[1][pidx]) for pidx in range(len(geom.xy[0])))

#             road_type = way.tags()['highway'] if 'highway' in way.tags() else 'railway'
#             for coord in processed_coords:
#                 waypoints['utm_x'].append(coord[0])
#                 waypoints['utm_y'].append(coord[1])
#                 waypoints['type'].append(road_type)
#                 waypoints['id'].append(way.id())
#                 waypoints['z'].append(-1)

#             a = 1
# print(np.unique(highways))
# plt.show()

rindices = np.where(gdf.filled)[0]
for ridx in rindices:
    pattern = get_road_match_pattern(gdf, ridx)
    entry = None
    if pattern in pattern2roadtile_dict:
        entries = pattern2roadtile_dict[pattern]
        if type(entries) == int:
            entries = pattern2roadtile_dict[entries]
        if len(entries) > 1:
            entry = entries[np.random.randint(0, len(entries))]
        else:
            entry = entries[0]

    gdf.iloc[ridx, 5] = pattern
    if entry is not None:
        gdf.iloc[ridx, 6] = entry[0]
        gdf.iloc[ridx, 7] = entry[1]
        gdf.iloc[ridx, 8] = entry[2]
        
unmatched_patterns = gdf[(gdf.pattern != -1) & (gdf.tile_page == -1)].pattern.unique()
for p in unmatched_patterns: 
    print('pattern: {} -> {}'.format(p, re.findall('...', bin(p)[2:].zfill(9)[::-1])))
    indices = np.where(gdf.pattern == p)[0]
    for idx in indices:
        plt.plot(geometry[idx].exterior.xy[0], geometry[idx].exterior.xy[1], '-r')

plt.show()

gdf_out = gdf[(gdf.tile_page != -1) & (gdf.tile_row != -1) & (gdf.tile_col != -1)]

gdf_out.x = gdf_out.xidx
gdf_out.y = gdf_out.yidx
gdf_out.to_csv('osm_test_roads.csv', columns=['x', 'y', 'z', 'tile_page', 'tile_row', 'tile_col', 'category', 'type', 'sub_type'])

# df = pandas.DataFrame(waypoints)


# df['x'] = pandas.cut(df['utm_x'], bins=bins_x, retbins=True, labels=list(range(n_bins_x)))[0]
# df['y'] = pandas.cut(df['utm_y'], bins=bins_y, retbins=True, labels=list(range(n_bins_y)))[0]

# df = df[~pandas.isna(df.x) & ~pandas.isna(df.y)]

# plt.figure()
# plt.plot(df.x, df.y, 'o')
# plt.show()

# df.to_csv('osm.csv')

a = 1






