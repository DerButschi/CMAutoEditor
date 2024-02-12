from ftplib import FTP
import json
import os
import pandas
import geopandas

#             ftp = FTP('ftp.eorc.jaxa.jp')
#             ftp.login()
#             ftp.cwd('pub/ALOS/ext1/AW3D30/release_v2012_single_format/')

# ftp = FTP('ftp.eorc.jaxa.jp')
# ftp.login()
# ftp.cwd('pub/ALOS/ext1/AW3D30/release_v2303/')
# listing = ftp.mlsd()

# file_dict = {}
# cnt = 0
# for item in listing:
#     # if cnt > 5:
#     #     break
#     # cnt += 1
#     print(item[0])
#     if item[1]['type'] == 'dir':
#         file_dict[item[0]] = []
#         ftp.cwd(item[0])
#         dir_listing = ftp.mlsd()
#         for dir_item in dir_listing:
#             if dir_item[1]['type'] == 'file' and dir_item[0].endswith('.zip'):
#                 file_dict[item[0]].append(dir_item[0])
        
#         ftp.cwd('..')

# ftp.close()
# with open(os.path.join("terrain_extraction", "data_sources", "aw3d30", "aw3d30_file.json"), 'w', encoding="utf-8") as f:
#     json.dump(file_dict, f)

aw3d30_geojson_dict = {
    'x': [],
    'y': [],
    'folder': [],
    'file_name': [],
}
with open(os.path.join("terrain_extraction", "data_sources", "aw3d30", "aw3d30_file.json")) as f:
    aw3d30_dict = json.load(f)
    for folder, file_names in aw3d30_dict.items():
        for file_name in file_names:
            lat = int(file_name[1:4])
            if file_name[0] == 'S':
                lat = -lat 
            lon = int(file_name[5:8])
            if file_name[4] == 'W':
                lon = -lon
            lat += 0.5
            lon += 0.5

            aw3d30_geojson_dict['folder'].append(folder)
            aw3d30_geojson_dict['file_name'].append(file_name)
            aw3d30_geojson_dict['x'].append(lon)
            aw3d30_geojson_dict['y'].append(lat)

df = pandas.DataFrame(aw3d30_geojson_dict)
gdf = geopandas.GeoDataFrame(
    df,
    geometry=geopandas.points_from_xy(df.x, df.y).buffer(0.5, cap_style=3)
)
gdf.set_crs(epsg=4326)

gdf.to_file(os.path.join("terrain_extraction", "data_sources", "aw3d30", "aw3d30_file.geojson"), driver="GeoJSON")  

