import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import os
import PIL
import numpy as np
import pandas
import shapely
from shapely import Point
from pyproj.crs import CRS
from terrain_extraction.data_sources.rge_alti.data_source import FranceDataSource

from terrain_extraction.projection_utils import transform_polygon
from terrain_extraction.bbox_utils import BoundingBox
from terrain_extraction.data_sources.hessen_dgm1.data_source import HessenDataSource
from terrain_extraction.data_sources.aw3d30.data_source import AW3D30DataSource
from terrain_extraction.data_sources.nrw_dgm1.data_source import NRWDataSource
from terrain_extraction.elevation_map import cut_out_bounding_box

# data_sources = [HessenDataSource(), AW3D30DataSource()]
data_sources = [HessenDataSource(), NRWDataSource(), FranceDataSource()]


if 'alos_coordinates' not in st.session_state:
    st.session_state['alos_coordinates'] = None

if 'alos_images_loaded' not in st.session_state:
    st.session_state['alos_images_loaded'] = []

def get_alos_image_array(lon, lat):
    download_dir = 'C:\\Users\\der_b\\Downloads\\alos'

    alos_str = '{}{}{}{}_{}{}{}{}'.format(
        'N' if lat >= 0 else 'S',
        str(int(lat / 5) * 5).zfill(3),
        'E' if lon >= 0 else 'W', 
        str(int(lon / 5) * 5).zfill(3),
        'N' if lat >= 0 else 'S',
        str(int(lat / 5 + 1) * 5).zfill(3),
        'E' if lon >= 0 else 'W', 
        str(int(lon / 5 + 1) * 5).zfill(3),
    )
    alos_file_str = 'ALPSMLC30_{}{}{}{}_DSM'.format(
        'N' if lat >= 0 else 'S',
        str(int(lat)).zfill(3),
        'E' if lon >= 0 else 'W', 
        str(int(lon)).zfill(3),
    )
    img = None
    if os.path.isfile(os.path.join(download_dir, alos_str, alos_file_str + '.tif')):
        # with PIL.Image.open(os.path.join(download_dir, alos_str, alos_file_str)) as im:
        #     img_array = np.array(im)
        if not os.path.isfile(os.path.join(download_dir, alos_str, alos_file_str + '.png')):
            with PIL.Image.open(os.path.join(download_dir, alos_str, alos_file_str + '.tif')) as im:
                im.convert("RGB").save(os.path.join(download_dir, alos_str, alos_file_str + '.png'))
        print('file://{}'.format(os.path.join(download_dir, alos_str, alos_file_str + '.png')).replace('\\', '/'))
        img = folium.raster_layers.ImageOverlay(
            name='ALOS',
            image='{}'.format(os.path.join(download_dir, alos_str, alos_file_str + '.png')).replace('\\', '/'),
            bounds=[[int(lat), int(lon)], [int(lat)+1, int(lon)+1]],
            interactive=True,
            cross_origin=False,
            zindex=1,
        )

    return img

def update_bounding_box(points):
    polygon = shapely.Polygon(points)
    
    # projected_epsg_code = get_epsg_code_from_bbox(polygon.envelope)

    # projected_polygon = transform_polygon(polygon, 4326, projected_epsg_code)
    # projected_box = projected_polygon.minimum_rotated_rectangle
    # projected_box = make_polygon_counter_clockwise(projected_box)
    # projected_origin_idx = find_idx_closest_polygon_node(projected_box, transform_point(Point(points[0]), 4326, projected_epsg_code))
    # projected_box = permute_polygon_to_idx(projected_box, projected_origin_idx)

    # box = transform_polygon(projected_box, projected_epsg_code, 4326)
    # box = make_polygon_counter_clockwise(box)

    # origin_idx = find_idx_closest_polygon_node(box, Point(points[0]))
    # box = permute_polygon_to_idx(box, origin_idx)

    # df = pandas.DataFrame({
    #         'longitude': [coord[0] for coord in box.exterior.coords][:-1],
    #         'latitude': [coord[1] for coord in box.exterior.coords][:-1],
    #         'origin': [True, False, False, False]
    # })

    bounding_box = BoundingBox(polygon)

    if 'bbox' not in st.session_state:
        st.session_state['bbox'] = []

    st.session_state['bbox'] = bounding_box.get_coordinates(xy=False)
    st.session_state['bbox_object'] = bounding_box
    st.session_state['projected_bbox_object'] = bounding_box.get_box(bounding_box.crs_projected)
    st.session_state['len_x'] = np.round(bounding_box.get_length_xaxis())
    st.session_state['len_y'] = np.round(bounding_box.get_length_yaxis())
    st.session_state['bbox_origin'] = 0

    return bounding_box.get_dataframe()

def find_data_sources_in_bbox():
    available_data_sources = []
    for data_source in data_sources:
        # projected_bbox = transform_polygon(st.session_state['bbox_object'], 4326, data_source.epsg_code)
        if data_source.intersects_bounding_box(st.session_state['bbox_object']):
            available_data_sources.append(data_source)

    st.session_state['data_sources'] = available_data_sources

def get_data_source_label(data_source):
    return '{} - {}, {}'.format(data_source.name, data_source.model_type, data_source.resolution)

@st.cache_data
def dataframe2csv(df: pandas.DataFrame):
    return df.to_csv().encode('utf-8')

def extract_data_in_bbox():
    data_source = st.session_state['selected_data_source']
    bounding_box = st.session_state['bbox_object']
    # projected_bbox = transform_polygon(st.session_state['bbox_object'], 4326, data_source.epsg_code)

    os.makedirs('C:\\Users\\der_b\\Downloads\\cm_terrain_extraction', exist_ok=True)
    with st.status('Extracting elevation data', expanded=True) as status:
        elevation_data = data_source.get_data(bounding_box, 'C:\\Users\\der_b\\Downloads\\cm_terrain_extraction')
        st.session_state['elevation_in_bbox'] = elevation_data
        path_to_png = data_source.get_png(bounding_box, 'C:\\Users\\der_b\\Downloads\\cm_terrain_extraction')
        status.update(label="Elevation data extracted!", state="complete", expanded=False)

def draw_sidebar(df):
    with st.sidebar:
        edited_df = st.data_editor(
            df,
            column_config = {
                "x": st.column_config.NumberColumn(
                    "Longitude [°]",
                    min_value=-180.0,
                    max_value=180.0
                ),
                "y": st.column_config.NumberColumn(
                    "Latitude [°]",
                    min_value=-90.0,
                    max_value=90.0
                )
            },
            on_change=bbox_on_change,
            args=[df]
        )

        if 'len_x' in st.session_state:
            area = np.round(st.session_state['len_x'] * st.session_state['len_y'] / 1e6, decimals=1)
            delta = np.round(st.session_state['len_x'] * st.session_state['len_y'] / 1e6 - 16, decimals=1)
            if delta > 0:
                st.metric(label="Selected Area", value='{} km²'.format(area), delta='{} km²'.format(delta), delta_color="inverse")
                st.session_state['selected_area_valid'] = False
            else:
                st.metric(label="Selected Area", value='{} km²'.format(area))
                st.session_state['selected_area_valid'] = True
        else:
            st.session_state['selected_area_valid'] = False

        st.button('Find available data sources', disabled=not st.session_state['selected_area_valid'], on_click=find_data_sources_in_bbox)

        # data_sources_labels = []
        # if 'data_sources' in st.session_state:
        #     for data_source in st.session_state['data_sources']:
        #         data_sources_labels.append(
        #             '{} - {}, {}'.format(data_source.name, data_source.model_type, data_source.resolution)
        #         )
        
        selected_data_source = st.selectbox('Data sources', st.session_state['data_sources'] if 'data_sources' in st.session_state else [], format_func=get_data_source_label)
        st.session_state['selected_data_source'] = selected_data_source
        st.button('Extract elevation data', disabled=selected_data_source is None, on_click=extract_data_in_bbox)

        if 'elevation_in_bbox' in st.session_state:
            st.download_button('Download elevation data', dataframe2csv(st.session_state['elevation_in_bbox']), file_name='elevation_data.csv')

    
    return edited_df

def bbox_on_change(bbox):
    a = 1
# img = folium.raster_layers.ImageOverlay(
#     name="Mercator projection SW",
#     image="C:/Users/der_b/Downloads/alos/N050E005_N055E010/ALPSMLC30_N050E007_DSM.png",
#     bounds=[[-82, -180], [82, 180]],
#     opacity=0.6,
#     interactive=True,
#     cross_origin=False,
#     zindex=1,
# ).add_to(map)

# folium.WmsTileLayer(
#     url="https://www.wms.nrw.de/geobasis/wms_nw_dhm-uebersicht",
#     name="test",
#     fmt="image/png",
#     layers="nw_dhm-uebersicht_planung_2019-2022",
#     attr=u"dgm1",
#     transparent=True,
#     overlay=True,
#     control=True,
# ).add_to(map)


# folium.LayerControl().add_to(map)
# with PIL.Image.open('C:\\Users\\der_b\\Downloads\\alos\\N050E005_N055E010\\ALPSMLC30_N050E007_DSM.tif') as im:
#     img_array = np.array(im)

#     img = folium.raster_layers.ImageOverlay(
#         name='alos',
#         image=img_array,
#         bounds=[[50, 7], [51, 8]],
#         # mercator_project=True,
#         # opacity=0.6,
#         interactive=True,
#         cross_origin=False,
#         zindex=1,
#     )
#     img.add_to(map)

# folium.LayerControl().add_to(map)

fg1 = folium.FeatureGroup(name="Alos")
download_dir = 'C:\\Users\\der_b\\Downloads\\alos'
#         os.makedirs(download_dir, exist_ok=True)
#         # with tempfile.TemporaryDirectory() as tempdirname:
if 'alos_coordinates' in st.session_state and st.session_state['alos_coordinates'] is not None:
    lon, lat = st.session_state['alos_coordinates']
    alos_str = '{}{}{}{}_{}{}{}{}'.format(
        'N' if lat >= 0 else 'S',
        str(int(lat / 5) * 5).zfill(3),
        'E' if lon >= 0 else 'W', 
        str(int(lon / 5) * 5).zfill(3),
        'N' if lat >= 0 else 'S',
        str(int(lat / 5 + 1) * 5).zfill(3),
        'E' if lon >= 0 else 'W', 
        str(int(lon / 5 + 1) * 5).zfill(3),
    )
#         if not os.path.isfile(os.path.join(download_dir, alos_str + '.zip')):
#             ftp = FTP('ftp.eorc.jaxa.jp')
#             ftp.login()
#             ftp.cwd('pub/ALOS/ext1/AW3D30/release_v2012_single_format/')
#             with open(os.path.join(download_dir, alos_str + '.zip'), 'wb') as alos_zip_handle:
#                 print('Downloading file {} to {}.'.format(alos_str + '.zip', download_dir))
#                 ftp.retrbinary('RETR {}'.format(alos_str + '.zip'), alos_zip_handle.write)
            
#             with zipfile.ZipFile(os.path.join(download_dir, alos_str + '.zip'), 'r') as zip_ref:
#                 zip_ref.extractall(download_dir)

    image_array = get_alos_image_array(lon, lat)        
    # img = folium.raster_layers.ImageOverlay(
    #     name='ALOS',
    #     image=image_array,
    #     bounds=[[int(lat), int(lon)], [int(lat)+1, int(lon)+1]],
    #     # mercator_project=True,
    #     # opacity=0.6,
    #     # interactive=True,
    #     # cross_origin=False,
    #     # zindex=1,
    # )
    if image_array is not None:
        fg1.add_child(image_array)
    #     a = 1
    # a = 1

# if st_data['last_active_drawing'] is not None:
#     if st_data['last_active_drawing']['geometry']['type'] == 'Point':
#         # os.makedirs(download_dir, exist_ok=True)
#         # # with tempfile.TemporaryDirectory() as tempdirname:
#         lon, lat = st_data['last_active_drawing']['geometry']['coordinates']
#         if 'alos_coordinates' in st.session_state and st.session_state is not None:
#             st.session_state['alos_coordinates'] = (lon, lat)

# # st.write(st_data)
# to_delete = []
# if st_data['all_drawings'] is not None:
#     if st_data['all_drawings'][0]['geometry']['type'] == 'Point':
#         st_data['all_drawings'] = None


if __name__ == '__main__':
    if 'center' not in st.session_state:
        st.session_state['center'] = {'lat': 0, 'lon': 0}
    if 'zoom' not in st.session_state:
        st.session_state['zoom'] = 1

    map = folium.Map(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", 
                     attr=(
                            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
                            'contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
                     ))
    
    bbox_fg = folium.FeatureGroup('bbox')
    draw = Draw(draw_options={
        'polyline': False,
        'circle': False,
        'marker': False,
        'circlemarker': False

    })
    draw.add_to(map)
    folium.plugins.Geocoder().add_to(map)
    folium.plugins.MeasureControl().add_to(map)
    folium.plugins.Fullscreen().add_to(map)

    if 'elevation_in_bbox' in st.session_state:
        bbox_fg.add_child(
            folium.raster_layers.ImageOverlay(
                name='Elevation data',
                image='C:\\Users\\der_b\\Downloads\\cm_terrain_extraction\\current_height_map.png',
                bounds=st.session_state['bbox'],
                interactive=False,
                cross_origin=False,
                opacity=0.9,
                zindex=1,
            )
        )
    # bbox_fg.add_child(folium.GeoJson(geopandas.GeoDataFrame.from_file("terrain_extraction/data_sources/hessen.geojson")))

    if 'bbox' in st.session_state:
        bbox = st.session_state['bbox']
        line1 = folium.vector_layers.PolyLine([bbox[0], bbox[1]], color='red')
        line2 = folium.vector_layers.PolyLine([bbox[1], bbox[2]], color='red')
        line3 = folium.vector_layers.PolyLine([bbox[2], bbox[3]], color='red')
        line4 = folium.vector_layers.PolyLine([bbox[0], bbox[3]], color='red')
        line1_text = folium.plugins.PolyLineTextPath(line1, "CM W<->E axis, {}m".format(st.session_state['len_x']), center=True, offset=20, color='red', attributes={'font-size': 18, 'fill': 'red'})
        line4_text = folium.plugins.PolyLineTextPath(line4, "CM N<->S axis, {}m".format(st.session_state['len_y']), center=True, offset=-7, color='red', attributes={'font-size': 18, 'fill': 'red'})
        bbox_fg.add_child(line1)
        bbox_fg.add_child(line2)
        bbox_fg.add_child(line3)
        bbox_fg.add_child(line4)
        bbox_fg.add_child(line1_text)
        bbox_fg.add_child(line4_text)
        bbox_fg.add_child(folium.vector_layers.CircleMarker(st.session_state['bbox'][0], color='red', radius=5))

    # folium.LayerControl().add_to(map)

    st_data = st_folium(
        map,
        center=st.session_state['center'],
        zoom=st.session_state['zoom'],
        feature_group_to_add=bbox_fg,
    )

    df = pandas.DataFrame({
        'longitude': [None, None, None, None],
        'latitude': [None, None, None, None],
        'origin': [False, False, False, False]

    })
    if st_data['last_active_drawing'] is not None:
        coordinates = st_data['last_active_drawing']['geometry']['coordinates']
        df = update_bounding_box(coordinates[0])
    
    draw_sidebar(df)

    st.write(st.session_state)         

a = 1
