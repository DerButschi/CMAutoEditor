from geopandas import GeoDataFrame
import os
from shapely import union_all

def load_raw_data_idx():  # from https://geoportal.geoportal-th.de/hoehendaten/Uebersichten/Stand_2014-2019.zip, .shp-file, converted with QGIS
    return GeoDataFrame.from_file(os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'data_sources', 'thuringia_dgm1', 'raw_data_index.geojson'))

def create_url_gdf(raw_data_idx):
    # https://geoportal.geoportal-th.de/hoehendaten/DGM/dgm_2014-2019/dgm1_561_5609_1_th_2014-2019.zip?
    url_base = 'https://geoportal.geoportal-th.de/hoehendaten/DGM/dgm_2014-2019/'
    url_dict = {
        'url': [],
        'geometry': []
    }
    for _, entry in raw_data_idx.iterrows():
        url_dict['url'].append(url_base + 'dgm1_{}_1_th_2014-2019.zip?'.format(entry['NAME']))
        url_dict['geometry'].append(entry['geometry'].geoms[0])

    return GeoDataFrame(url_dict)


if __name__ == '__main__':
    raw_data_idx = load_raw_data_idx()
    gdf = create_url_gdf(raw_data_idx)
    gdf.to_file(os.path.join('cm_terrain_extractor_app','terrain_extraction', 'data_sources', 'thuringia_dgm1', 'thuringia_dgm1.geojson'), driver="GeoJSON")
    print([coord for coord in union_all(gdf.geometry).envelope.exterior.coords])
