import requests
import os
import zipfile
import pandas
from shapely import Polygon
import geopandas
from shapely import union_all


def get_meta_data():
    r = requests.get("https://arcgis-geojson.s3.eu-de.cloud-object-storage.appdomain.cloud/dgm1/lgln-opengeodata-dgm1.geojson", stream=True)

    meta_data_path = os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'data_sources', 'lower_saxony_dgm1', 'lgln-opengeodata-dgm1.geojson')
    with open(meta_data_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=4096):
            fd.write(chunk)

    gdf = geopandas.read_file(meta_data_path)
    gdf = gdf.rename(columns={'dgm1': 'url'})
    gdf = gdf.drop([col for col in gdf.columns if col not in ['url', 'geometry']], axis=1)

    return gdf

if __name__ == '__main__':
    gdf = get_meta_data()
    gdf.to_file(os.path.join("cm_terrain_extractor_app", "terrain_extraction", "data_sources", "lower_saxony_dgm1", 'lower_saxony_dgm1.geojson'), driver="GeoJSON")
    print([coord for coord in union_all(gdf.geometry).envelope.exterior.coords])


