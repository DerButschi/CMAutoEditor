import geopandas

def load_gdfs(file_names):
    gdfs = []
    for file_name in file_names:
        gdf = geopandas.GeoDataFrame.from_file(file_name)
        gdfs.append(gdf)

    return gdfs

if __name__ == '__main__':
    gdfs = load_gdfs(
        ['terrain_extraction\\raw_data_indices\\poland_2018.geojson',
         'terrain_extraction\\raw_data_indices\\poland_2019.geojson',
         'terrain_extraction\\raw_data_indices\\poland_2020.geojson',
         'terrain_extraction\\raw_data_indices\\poland_2021.geojson',
         'terrain_extraction\\raw_data_indices\\poland_2022.geojson',
         'terrain_extraction\\raw_data_indices\\poland_2023.geojson'
        ]
    )
    a = 1