import requests
import xml.etree.ElementTree as ET
from geopandas import GeoDataFrame
from shapely import Polygon, union_all

import os

def download_metadata_file():
    r = requests.get('https://geodaten.bayern.de/odd/a/dgm/dgm1/meta/metalink/09.meta4', stream=True)
    with open(os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'raw_data_indices', 'bavaria_meta.xml'), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=4096):
            fd.write(chunk)

def url2polygon(url):
    fname = os.path.basename(url).split('.')[0]
    x0 = float(fname.split('_')[0]) * 1000
    y0 = float(fname.split('_')[1]) * 1000
    polygon = Polygon([
        (x0 - 0.5, y0 - 0.5),
        (x0 + 1000 - 0.5, y0 - 0.5),
        (x0 + 1000 - 0.5, y0 + 1000 - 0.5),
        (x0 - 0.5, y0 + 1000 - 0.5),
    ])
    return polygon

def parse_metadata():
    file_dict = {
        'url': [],
        'geometry': []
    }
    root = ET.parse(os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'raw_data_indices', 'bavaria_meta.xml')).getroot()
    for child in root:
        if 'name' in child.attrib and child.attrib['name'].endswith('.tif'):
            for child2 in child:
                if 'url' in child2.tag:
                    url = child2.text
                    polygon = url2polygon(url)
                    file_dict['url'].append(url)
                    file_dict['geometry'].append(polygon)
                    break

    return file_dict

if __name__ == '__main__':
    download_metadata_file()
    file_dict = parse_metadata()

    gdf = GeoDataFrame(file_dict)
    gdf.set_crs(epsg=25832)
    gdf.to_file(os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'data_sources', 'bavaria_dgm1', 'bavaria_dgm1.geojson'), driver="GeoJSON")
    print([coord for coord in union_all(gdf.geometry).envelope.exterior.coords])
