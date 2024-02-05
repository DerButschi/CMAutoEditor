import requests
import os
import zipfile
import pandas
from shapely import Polygon
import geopandas
from shapely import union_all


def get_meta_data():
    r = requests.get("https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/dgm1_meta.zip", stream=True)

    with open(os.path.join("C:\\Users\\der_b\\Downloads\\cm_terrain_extraction\\dgm1_meta.zip"), 'wb') as fd:
        for chunk in r.iter_content(chunk_size=4096):
            fd.write(chunk)

    with zipfile.ZipFile(os.path.join("C:\\Users\\der_b\\Downloads\\cm_terrain_extraction\\dgm1_meta.zip"), 'r') as zip_ref:
        zip_ref.extract('dgm1_xyz.csv', os.path.join("C:\\Users\\der_b\\Downloads\\cm_terrain_extraction", os.path.splitext("dgm1_meta")[0]))

    df = pandas.read_csv(os.path.join("C:\\Users\\der_b\\Downloads\\cm_terrain_extraction", os.path.splitext("dgm1_meta")[0], 'dgm1_xyz.csv'), skiprows=5, delimiter=";")

    return df

def get_polygons(df):
    geometries = []
    for fname in df['Kachelname'].values:
        x0 = float(fname.split('_')[2]) * 1000
        y0 = float(fname.split('_')[3]) * 1000
        polygon = Polygon([
            (x0 - 0.5, y0 - 0.5),
            (x0 + 1000 - 0.5, y0 - 0.5),
            (x0 + 1000 - 0.5, y0 + 1000 - 0.5),
            (x0 - 0.5, y0 + 1000 - 0.5),
        ])
        geometries.append(polygon)

    return geometries

# "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/dgm1_32_280_5652_1_nw.xyz.gz"
def create_gdf(df, polygons):
    df = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_xyz/dgm1_xyz/" + df.loc[:,['Kachelname']] + ".xyz.gz"
    df = df.rename(columns={'Kachelname': 'url'})
    gdf = geopandas.GeoDataFrame(
        df,
        geometry=polygons
    )
    return gdf

if __name__ == '__main__':
    df = get_meta_data()
    polygons = get_polygons(df)
    gdf = create_gdf(df, polygons)
    gdf.set_crs(epsg=25832)
    gdf.to_file(os.path.join("terrain_extraction\\data_sources\\nrw_dgm1", 'nrw_dgm1.geojson'), driver="GeoJSON") 
    print([coord for coord in union_all(gdf.geometry).envelope.exterior.coords])
    a = 1


