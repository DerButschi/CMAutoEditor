import requests
from bs4 import BeautifulSoup
import re
import os
import geopandas
from pyproj.transformer import Transformer
from shapely import Polygon, MultiPolygon, union_all


def download_product_page():
    r = requests.get('https://geoservices.ign.fr/rgealti', stream=True)
    with open(os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'raw_data_indices', 'rgealti.html'), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=4096):
            fd.write(chunk)

def get_urls():
    with open(os.path.join('cm_terrain_extractor_app', 'terrain_extraction', 'raw_data_indices', 'rgealti.html')) as fp:
        soup = BeautifulSoup(fp, 'html.parser')
        urls = [t.string for t in soup.find_all(href=re.compile('RGEALTI_2-0_5M'))]

    return urls

def match_url_to_geometry(urls):
    departements_gdf = geopandas.GeoDataFrame.from_file("cm_terrain_extractor_app\\terrain_extraction\\raw_data_indices\\departements.geojson")
    gdf_dict = {
        'url': [],
        'geometry': []
    }

    transformer = Transformer.from_crs('epsg:{}'.format(4326), 'epsg:{}'.format(2154), always_xy=True)


    for url in urls:
        dep_code = url.split('_')[-2][1:]
        if dep_code in ['971', '972', '973']:
            continue
        dep_code = dep_code[1:]

        entries = departements_gdf[departements_gdf['code'] == dep_code]
        if len(entries) != 1:
            continue

        geometry = entries['geometry'].values[0]
        if geometry.geom_type == 'Polygon':
            polygon_xy = transformer.transform([coord[0] for coord in geometry.exterior.coords], [coord[1] for coord in geometry.exterior.coords])
            projected_geometry = Polygon([(polygon_xy[0][i], polygon_xy[1][i]) for i in range(len(polygon_xy[0]))])
        else:
            projected_polygons = []
            for polygon in geometry.geoms:
                polygon_xy = transformer.transform([coord[0] for coord in polygon.exterior.coords], [coord[1] for coord in polygon.exterior.coords])
                projected_polygon = Polygon([(polygon_xy[0][i], polygon_xy[1][i]) for i in range(len(polygon_xy[0]))])
                projected_polygons.append(projected_polygon)
                projected_geometry = MultiPolygon(projected_polygons)


        gdf_dict['url'].append(url)
        gdf_dict['geometry'].append(projected_geometry)
    
    return geopandas.GeoDataFrame(gdf_dict)
        
    a = 1



if __name__ == '__main__':
    download_product_page()
    urls = get_urls()
    gdf = match_url_to_geometry(urls)
    gdf.set_crs(epsg=2154)
    gdf.to_file(os.path.join("cm_terrain_extractor_app\\terrain_extraction\\data_sources\\rge_alti", 'rge_alti.geojson'), driver="GeoJSON")
    print([coord for coord in union_all(gdf.geometry).envelope.exterior.coords])

    a = 1