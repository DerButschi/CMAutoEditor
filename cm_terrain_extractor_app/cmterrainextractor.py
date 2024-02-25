import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
import os
import numpy as np
import pandas
import shapely
import sys

from terrain_extraction.data_sources.rge_alti.data_source import FranceDataSource

from terrain_extraction.bbox_utils import BoundingBox
from terrain_extraction.data_sources.hessen_dgm1.data_source import HessenDataSource
from terrain_extraction.data_sources.aw3d30.data_source import AW3D30DataSource
from terrain_extraction.data_sources.nrw_dgm1.data_source import NRWDataSource
from terrain_extraction.data_sources.netherlands_dtm05.data_source import NetherlandsDataSource
from terrain_extraction.data_sources.bavaria_dgm1.data_source import BavariaDataSource
from terrain_extraction.data_sources.thuringia_dgm1.data_source import ThuringiaDataSource

# data_sources = [HessenDataSource(), AW3D30DataSource()]
data_sources = [HessenDataSource(), 
                NRWDataSource(), 
                FranceDataSource(), 
                NetherlandsDataSource(), 
                BavariaDataSource(), 
                ThuringiaDataSource(),
                AW3D30DataSource()]

st.session_state['selectable_data_sources'] = [ds for ds in data_sources]
# data_sources = [NetherlandsDataSource()]

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    data_cache_path = os.path.join(os.path.dirname(sys.executable), 'data_cache')
else:
    data_cache_path = 'data_cache'

def update_bounding_box(points):
    polygon = shapely.Polygon(points)
    if polygon.minimum_rotated_rectangle.geom_type == 'LineString':
        return

    bounding_box = BoundingBox(polygon)

    if 'bbox' not in st.session_state:
        st.session_state['bbox'] = []

    st.session_state['bbox'] = bounding_box.get_coordinates(xy=False)
    st.session_state['bbox_object'] = bounding_box
    st.session_state['projected_bbox_object'] = bounding_box.get_box(bounding_box.crs_projected)
    st.session_state['len_x'] = bounding_box.get_length_xaxis()
    st.session_state['len_y'] = bounding_box.get_length_yaxis()
    st.session_state['bbox_origin'] = 0
    if 'elevation_in_bbox' in st.session_state:
        del st.session_state['elevation_in_bbox']

    st.rerun()
    # return bounding_box.get_dataframe()

def find_data_sources_in_bbox(status_update_area):
    with status_update_area.container(border=True):
        with st.spinner('Searching for data sources in the selected area...'):
            available_data_sources = []
            for data_source in st.session_state['selectable_data_sources']:
                if data_source.intersects_bounding_box(st.session_state['bbox_object']):
                    available_data_sources.append(data_source)

            st.session_state['data_sources'] = available_data_sources
    status_update_area.empty()

def get_data_source_label(data_source):
    return '{} - {}, {}'.format(data_source.name, data_source.model_type, data_source.resolution)

@st.cache_data
def dataframe2csv(df: pandas.DataFrame):
    return df.to_csv().encode('utf-8')

def extract_data_in_bbox(status_update_area):
    with status_update_area.container():
        data_source = st.session_state['selected_data_source']
        bounding_box = st.session_state['bbox_object']
        st.session_state['currently_processing_data'] = (
            'Extracting data from {}'.format(data_source.name),
            data_source.name
        )

        os.makedirs(data_cache_path, exist_ok=True)
        with st.status('Extracting elevation data', expanded=True) as status:
            elevation_data = data_source.get_data(bounding_box, data_cache_path)
            st.session_state['elevation_in_bbox'] = elevation_data
            path_to_png = data_source.get_png(bounding_box, data_cache_path)
            status.update(label="Elevation data extracted!", state="complete", expanded=False)

        del st.session_state['currently_processing_data']
    status_update_area.empty()

def permute_bbox():
    bounding_box = st.session_state['bbox_object']
    bounding_box.cycle_origin()
    st.session_state['bbox'] = bounding_box.get_coordinates(xy=False)
    st.session_state['projected_bbox_object'] = bounding_box.get_box(bounding_box.crs_projected)
    st.session_state['len_x'] = bounding_box.get_length_xaxis()
    st.session_state['len_y'] = bounding_box.get_length_yaxis()

def update_bbox_from_df():
    df = st.session_state['edited_df'].copy()
    del st.session_state['edited_df']
    if (~df.isnull().any()).all():
        update_bounding_box(list(zip(df.x.values, df.y.values)))

def draw_sidebar(status_update_area):
    max_len_x_axis = 4160 # m
    max_len_y_axis = 4160 # m
    max_area = 18000000 # km2
    len_x_axis = None
    len_y_axis = None
    delta_len_x = None
    delta_len_y = None
    area = None
    delta_area = None
    if 'len_x' in st.session_state:
        len_x_axis = st.session_state['len_x']
        len_y_axis = st.session_state['len_y']
        area = len_x_axis * len_y_axis
        delta_len_x = len_x_axis - max_len_x_axis
        delta_len_y = len_y_axis - max_len_y_axis
        delta_area = area - max_area
        # area = np.round(st.session_state['len_x'] * st.session_state['len_y'] / 1e6, decimals=1)
        # delta = np.round(st.session_state['len_x'] * st.session_state['len_y'] / 1e6 - 16, decimals=1)
        if delta_len_x > 0 or delta_len_y > 0 or delta_area > 0:
            st.session_state['selected_area_valid'] = False
        else:
            st.session_state['selected_area_valid'] = True
    else:
        st.session_state['selected_area_valid'] = False

    with st.sidebar:
        with st.container(border=True):
            if 'bbox_object' in st.session_state:
                df = st.session_state['bbox_object'].get_dataframe()
            else:
                df = pandas.DataFrame({
                    'x': [None, None, None, None],
                    'y': [None, None, None, None]
                })

            st.session_state['edited_df'] = st.data_editor(
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
            )
            st.button('Cycle bounding box origin', disabled=not st.session_state['selected_area_valid'], on_click=permute_bbox)
            col1, col2, col3 = st.columns(3)
            with col1:
                if len_x_axis is not None and delta_len_x <= 0:
                    st.metric(label='Length W\u2194E', value='{} m'.format(np.round(len_x_axis).astype(int)))
                elif len_x_axis is not None and delta_len_x > 0:
                    st.metric(label='Length W\u2194E', value='{} m'.format(np.round(len_x_axis).astype(int)), delta='{} m'.format(np.round(delta_len_x).astype(int)), delta_color="inverse")
                else:
                    st.metric(label='Length W\u2194E', value='-')    
            with col2:
                if len_y_axis is not None and delta_len_y <= 0:
                    st.metric(label='Length S\u2194N', value='{} m'.format(np.round(len_y_axis).astype(int)))
                elif len_y_axis is not None and delta_len_y > 0:
                    st.metric(label='Length S\u2194N', value='{} m'.format(np.round(len_y_axis).astype(int)), delta='{} m'.format(np.round(delta_len_y).astype(int)), delta_color="inverse")
                else:
                    st.metric(label='Length S\u2194N', value='-')    
            with col3:
                if area is not None and delta_area <= 0:
                    st.metric(label='Selected Area', value='{} km²'.format(np.round(area / 1e6, decimals=1)))
                elif area is not None and delta_area > 0:
                    st.metric(label='Selected Area', value='{} km²'.format(np.round(area / 1e6, decimals=1)), delta='{} km²'.format(np.round(delta_area / 1e6, decimals=1)), delta_color="inverse")
                else:
                    st.metric(label='Selected Area', value='-')    

        with st.container(border=True):
            st.button('Find available data sources', disabled=not st.session_state['selected_area_valid'], on_click=find_data_sources_in_bbox, args=[status_update_area])
            selected_data_source = st.selectbox('Data sources', st.session_state['data_sources'] if 'data_sources' in st.session_state else [], format_func=get_data_source_label)

            st.session_state['selected_data_source'] = selected_data_source
            st.button('Extract elevation data', disabled=selected_data_source is None, on_click=extract_data_in_bbox, args=[status_update_area])

        with st.container(border=True):
            st.download_button(
                'Download elevation .csv-file', 
                dataframe2csv(st.session_state['elevation_in_bbox']) if 'elevation_in_bbox' in st.session_state else 'dummy', 
                file_name='elevation_data.csv',
                disabled=not ('elevation_in_bbox' in st.session_state)
            )

def elevations_tab():
    st.header('Extraction of Elevation Data')
    st.markdown(
        "Select an area for which to extract elevation data."
    )

    START_LOCATION = [0,0]
    START_ZOOM = 2

    if "map_center" not in st.session_state:
        st.session_state["map_center"] = START_LOCATION
    if "map_zoom" not in st.session_state:
        st.session_state["map_zoom"] = START_ZOOM
    if "map_key" not in st.session_state:
        st.session_state["map_key"] = 0

    # if 'center' not in st.session_state:
    #     st.session_state['center'] = {'lat': 0, 'lon': 0}
    # if 'zoom' not in st.session_state:
    #     st.session_state['zoom'] = 1

    map = folium.Map(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", 
                     attr=(
                            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
                            'contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
                     ),
                     )
    
    bbox_fg = folium.FeatureGroup('bbox')
    draw = Draw(draw_options={
        'polyline': False,
        'circle': False,
        'marker': False,
        'circlemarker': False

    })
    draw.add_to(map)
    folium.plugins.Geocoder(postion='bottomleft').add_to(map)
    folium.plugins.MeasureControl().add_to(map)
    folium.plugins.Fullscreen().add_to(map)

    if 'elevation_in_bbox' in st.session_state:
        if 'height_map_layer' not in st.session_state:
        # if 'zoom_cache' in st.session_state:
            st.session_state['map_zoom'] = st.session_state['zoom_cache']
            st.session_state['map_center'] = st.session_state['center_cache']
            st.session_state['map_key'] += 1
            st.session_state['height_map_layer'] = True

        bbox_fg.add_child(
            folium.raster_layers.ImageOverlay(
                name='Elevation data',
                image=os.path.join(data_cache_path, 'current_height_map.png'),
                bounds=st.session_state['bbox'],
                # interactive=False,
                # cross_origin=False,
                opacity=0.9,
                # zindex=1,
            )
        )
    # bbox_fg.add_child(folium.GeoJson(geopandas.GeoDataFrame.from_file("terrain_extraction/data_sources/hessen.geojson")))

    if 'bbox' in st.session_state:
        bbox = st.session_state['bbox']
        line1 = folium.vector_layers.PolyLine([bbox[0], bbox[1]], color='red')
        line2 = folium.vector_layers.PolyLine([bbox[1], bbox[2]], color='red')
        line3 = folium.vector_layers.PolyLine([bbox[2], bbox[3]], color='red')
        line4 = folium.vector_layers.PolyLine([bbox[0], bbox[3]], color='red')
        line1_text = folium.plugins.PolyLineTextPath(line1, "CM W\u2194E axis", center=True, offset=20, color='red', attributes={'font-size': 16, 'fill': 'red'})
        line4_text = folium.plugins.PolyLineTextPath(line4, "CM S\u2194N axis", center=True, offset=-7, color='red', attributes={'font-size': 16, 'fill': 'red'})
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
        center=st.session_state['map_center'],
        zoom=st.session_state['map_zoom'],
        feature_group_to_add=bbox_fg,
        width=1200,
        key=st.session_state['map_key']
    )
    status_update_area = st.empty()

    draw_sidebar(status_update_area)

    if st_data['last_active_drawing'] is not None:
        coordinates = np.array(st_data['last_active_drawing']['geometry']['coordinates'])
        if 'drawn_coordinates' not in st.session_state or \
            not (st.session_state['drawn_coordinates'].shape == coordinates.shape) or \
            not (st.session_state['drawn_coordinates'] == coordinates).all():
            st.session_state['drawn_coordinates'] = coordinates
            update_bounding_box(coordinates[0])

    if 'zoom' in st_data:
        st.session_state['zoom_cache'] = st_data['zoom']
    if 'center' in st_data:
        st.session_state['center_cache'] = st_data['center']
        # del st_data['last_active_drawing']
        # if not('bbox_object' in st.session_state and BoundingBox(Polygon(coordinates[0])).equals(st.session_state['bbox_object'])):
        #     update_bounding_box(coordinates[0])
    
    if 'edited_df' in st.session_state:
        if 'bbox_object' in st.session_state:
            df1 = st.session_state['edited_df']
            df2 = st.session_state['bbox_object'].get_dataframe()
            if not (df1 == df2).all().all():
                update_bbox_from_df()
        else:
            update_bbox_from_df()

def options_tab():
    st.markdown(
        "Select which data sources should be queried for available elevation data."
    )
    data_source_dict = st.data_editor(
        {
            'Name': [ds.name for ds in data_sources],
            'Country/Region': [ds.country for ds in data_sources],
            'Type': [ds.model_type for ds in data_sources],
            'Resolution': [ds.resolution for ds in data_sources],
            'Format': [ds.data_type for ds in data_sources],
            'Include in Search': [ds in st.session_state['selectable_data_sources'] for ds in data_sources]
        },
        column_order=['Name', 'Country/Region', 'Type', 'Resolution', 'Format', 'Include in Search'],
        disabled=['Name', 'Type', 'Resolution', 'Format']
    )
    selected_data_source_names = []
    for didx, ds_selected in enumerate(data_source_dict['Include in Search']):
        if ds_selected:
            selected_data_source_names.append(data_source_dict['Name'][didx])

    st.session_state['selectable_data_sources'] = [ds for ds in data_sources if ds.name in selected_data_source_names]

    file_sizes = 0
    for dir_path, dir_name, file_names in os.walk(data_cache_path):
        for fname in file_names:
            file_sizes += os.path.getsize(os.path.join(dir_path, fname))

    sizes = ['KB', 'MB', 'GB', 'TB']
    size_str = ''
    factor = 1
    for sz in sizes:
        factor *= 1024
        size_str = sz
        if file_sizes / factor < 1024:
            break
    
    st.button('Clear Cache ({} {})'.format(np.round(file_sizes / factor, decimals=2), size_str))
    a = 1

if __name__ == '__main__':
    st.set_page_config(
        page_title='CM Terrain Extractor',
        layout="wide", 
        menu_items={'Report a bug': "https://github.com/DerButschi/CMAutoEditor/issues/new/choose"})    



    tab1, tab2 = st.tabs(['Elevations', 'Options'])

    with tab1:
        elevations_tab()

    with tab2:
        options_tab()

    if 'currently_processing_data' in st.session_state:
        st.warning('{} was interrupted before it was finished. This may lead to corrupt data. If you encounter issues with the data, go to the options tab and clear the {} cache'.format(
            *st.session_state['currently_processing_data']
        ))
        del st.session_state['currently_processing_data']

    # st.write(st_data)

