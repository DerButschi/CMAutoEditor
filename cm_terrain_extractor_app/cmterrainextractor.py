import json
import os
import warnings

import numpy as np
import pandas
import streamlit as st
from shapely import Polygon
from streamlit_folium import st_folium
from terrain_extraction.bbox_utils import BoundingBox
from terrain_extraction.data_sources.aw3d30.data_source import AW3D30DataSource
from terrain_extraction.data_sources.bavaria_dgm1.data_source import BavariaDataSource
from terrain_extraction.data_sources.hessen_dgm1.data_source import HessenDataSource
from terrain_extraction.data_sources.lower_saxony_dgm1.data_source import LowerSaxonyDataSource
from terrain_extraction.data_sources.netherlands_dtm05.data_source import NetherlandsDataSource
from terrain_extraction.data_sources.nrw_dgm1.data_source import NRWDataSource
from terrain_extraction.data_sources.rge_alti.data_source import FranceDataSource
from terrain_extraction.data_sources.thuringia_dgm1.data_source import ThuringiaDataSource
from terrain_extraction.osm_processor import OSMProcessor
from terrain_extraction.osm_utils.io import get_bounding_box

from cm_terrain_extractor_app.app_core.actions import (
    download_osm_data as download_osm_data_action,
)
from cm_terrain_extractor_app.app_core.actions import (
    extract_elevation_data as extract_elevation_data_action,
)
from cm_terrain_extractor_app.app_core.actions import (
    find_data_sources_in_bbox as find_data_sources_in_bbox_action,
)
from cm_terrain_extractor_app.app_core.actions import (
    load_osm_config,
    load_osm_data_from_uploaded_bytes,
)
from cm_terrain_extractor_app.app_core.actions import (
    process_osm_data as process_osm_data_action,
)
from cm_terrain_extractor_app.app_core.exports import (
    dataframe_to_csv_bytes,
    suggest_elevation_filename,
    suggest_osm_filename,
)
from cm_terrain_extractor_app.app_core.resources import (
    find_default_osm_configs,
    prepare_runtime_environment,
    resolve_resources,
)
from cm_terrain_extractor_app.app_core.state import (
    clear_elevation_result,
    clear_osm_processing_result,
)
from cm_terrain_extractor_app.app_core.validation import (
    MAX_LEN_X_METERS,
    MAX_LEN_Y_METERS,
    MAX_SELECTED_AREA_SQUARE_METERS,
    compute_bbox_metrics,
    is_selected_area_valid,
    update_state_from_bbox,
)
from cm_terrain_extractor_app.map_view.drawing import (
    drawing_to_bounding_box,
    extract_last_active_drawing,
)
from cm_terrain_extractor_app.map_view.folium_map import build_folium_map
from cm_terrain_extractor_app.streamlit_ui.session_adapter import get_state

warnings.filterwarnings("ignore", category=DeprecationWarning)

resources = resolve_resources()
prepare_runtime_environment(resources)
executable_path = str(resources.config_dir)
data_cache_path = str(resources.data_cache_path)
state = get_state(resources)

# data_sources = [HessenDataSource(), AW3D30DataSource()]
data_sources = [HessenDataSource(), 
                NRWDataSource(), 
                FranceDataSource(), 
                NetherlandsDataSource(), 
                BavariaDataSource(), 
                ThuringiaDataSource(),
                AW3D30DataSource(),
                LowerSaxonyDataSource(),
                ]

if not state.selectable_data_sources:
    state.selectable_data_sources = list(data_sources)

# data_sources = [NetherlandsDataSource()]

# DEBUG_MODE = 'OSM_PROCESSOR'
DEBUG_MODE = None
if DEBUG_MODE == 'OSM_PROCESSOR' and state.osm_output is None:
    import pickle
    with open(os.path.join('test_objects', 'osm_processor_20240322.pkl'), 'rb') as pkl_file:
        osm_processor: OSMProcessor = pickle.load(pkl_file)
        state.osm_output = osm_processor.get_output()
        state.osm_geometries = osm_processor.get_geometries()
        state.osm_config_file = 'default_osm_config.json'
        with open(state.osm_config_file) as config_file_handle:
            state.osm_config = json.load(config_file_handle)

        state.osm_profile = osm_processor.profile
        bounding_box = osm_processor.bbox
        state.bbox_coordinates = bounding_box.get_coordinates(xy=False)
        state.bbox_object = bounding_box
        state.projected_bbox_object = bounding_box.get_box(bounding_box.crs_projected)
        state.len_x = bounding_box.get_length_xaxis()
        state.len_y = bounding_box.get_length_yaxis()
        state.bbox_origin = 0

def update_bounding_box(points):
    polygon = Polygon(points)
    if polygon.minimum_rotated_rectangle.geom_type == 'LineString':
        return

    bounding_box = BoundingBox(polygon)
    update_state_from_bbox(state, bounding_box)

    st.rerun()
    # return bounding_box.get_dataframe()

def find_data_sources_in_bbox(status_update_area):
    with status_update_area.container(border=True), st.spinner('Searching for data sources in the selected area...'):
        state.available_data_sources = find_data_sources_in_bbox_action(
            bbox=state.bbox_object,
            selectable_sources=state.selectable_data_sources,
        )
    status_update_area.empty()

def get_data_source_label(data_source):
    return f'{data_source.name} - {data_source.model_type}, {data_source.resolution}'

def extract_data_in_bbox(status_update_area):
    with status_update_area.container():
        data_source = state.selected_data_source
        bounding_box = state.bbox_object
        state.currently_processing_data = (
            f'Extracting data from {data_source.name}',
            data_source.name
        )

        with st.status('Extracting elevation data', expanded=True) as status:
            state.elevation_in_bbox, state.height_map_png = extract_elevation_data_action(
                data_source=data_source,
                bbox=bounding_box,
                data_cache_path=resources.data_cache_path,
            )
            status.update(label="Elevation data extracted!", state="complete", expanded=False)

        state.currently_processing_data = None
    status_update_area.empty()

def permute_bbox():
    bounding_box = state.bbox_object
    bounding_box.cycle_origin()
    metrics = compute_bbox_metrics(bounding_box)
    state.bbox_coordinates = bounding_box.get_coordinates(xy=False)
    state.projected_bbox_object = bounding_box.get_box(bounding_box.crs_projected)
    state.len_x = metrics["len_x"]
    state.len_y = metrics["len_y"]
    state.selected_area_valid = is_selected_area_valid(state.len_x, state.len_y)
    state.bbox_origin = (state.bbox_origin + 1) % 4

def update_bbox_from_df():
    df = st.session_state['edited_df'].copy()
    del st.session_state['edited_df']
    if (~df.isnull().any()).all():
        update_bounding_box(list(zip(df.x.values, df.y.values, strict=False)))

def draw_sidebar(status_update_area):  # noqa: C901, PLR0915
    len_x_axis = None
    len_y_axis = None
    delta_len_x = None
    delta_len_y = None
    area = None
    delta_area = None
    if state.len_x is not None and state.len_y is not None:
        len_x_axis = state.len_x
        len_y_axis = state.len_y
        area = len_x_axis * len_y_axis
        delta_len_x = len_x_axis - MAX_LEN_X_METERS
        delta_len_y = len_y_axis - MAX_LEN_Y_METERS
        delta_area = area - MAX_SELECTED_AREA_SQUARE_METERS
        # area = np.round(state.len_x * state.len_y / 1e6, decimals=1)
        # delta = np.round(state.len_x * state.len_y / 1e6 - 16, decimals=1)
        state.selected_area_valid = is_selected_area_valid(len_x_axis, len_y_axis)
    else:
        state.selected_area_valid = False

    with st.sidebar:
        with st.container(border=True):
            map_mode_options = ["Bounding Box Selection", "Elevations", "OpenStreetMap"]
            map_mode = st.radio(
                "",
                map_mode_options,
                captions=[
                    "Select the outline of the Combat Mission map.",
                    "Extract elevation data.",
                    "Extract map content from OpenStreetMap."
                ],
                index=map_mode_options.index(state.map_mode)
                if state.map_mode in map_mode_options
                else 0,
                label_visibility="collapsed"
            )
            if map_mode != state.map_mode:
                state.map_mode = map_mode
                # st.rerun()

        if state.map_mode == 'Bounding Box Selection':
            with st.container(border=True):
                if state.bbox_object is not None:
                    df = state.bbox_object.get_dataframe()
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
                st.button('Cycle bounding box origin', disabled=not state.selected_area_valid, on_click=permute_bbox)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if len_x_axis is not None and delta_len_x <= 0:
                        st.metric(label='Length W\u2194E', value=f'{np.round(len_x_axis).astype(int)} m')
                    elif len_x_axis is not None and delta_len_x > 0:
                        st.metric(label='Length W\u2194E', value=f'{np.round(len_x_axis).astype(int)} m', delta=f'{np.round(delta_len_x).astype(int)} m', delta_color="inverse")
                    else:
                        st.metric(label='Length W\u2194E', value='-')    
                with col2:
                    if len_y_axis is not None and delta_len_y <= 0:
                        st.metric(label='Length S\u2194N', value=f'{np.round(len_y_axis).astype(int)} m')
                    elif len_y_axis is not None and delta_len_y > 0:
                        st.metric(label='Length S\u2194N', value=f'{np.round(len_y_axis).astype(int)} m', delta=f'{np.round(delta_len_y).astype(int)} m', delta_color="inverse")
                    else:
                        st.metric(label='Length S\u2194N', value='-')    
                with col3:
                    if area is not None and delta_area <= 0:
                        st.metric(label='Selected Area', value=f'{np.round(area / 1e6, decimals=1)} km²')
                    elif area is not None and delta_area > 0:
                        st.metric(label='Selected Area', value=f'{np.round(area / 1e6, decimals=1)} km²', delta=f'{np.round(delta_area / 1e6, decimals=1)} km²', delta_color="inverse")
                    else:
                        st.metric(label='Selected Area', value='-')    

        if state.map_mode == 'Elevations':
            with st.container(border=True):
                if not state.selected_area_valid:
                    st.markdown(":red[Please select a valid bounding box first.]")
                st.button('Find available data sources', disabled=not state.selected_area_valid, on_click=find_data_sources_in_bbox, args=[status_update_area])
                selected_index = (
                    state.available_data_sources.index(state.selected_data_source)
                    if state.selected_data_source in state.available_data_sources
                    else 0 if state.available_data_sources else None
                )
                selected_data_source = st.selectbox(
                    'Data sources',
                    state.available_data_sources,
                    index=selected_index,
                    format_func=get_data_source_label,
                )

                if selected_data_source != state.selected_data_source:
                    state.selected_data_source = selected_data_source
                    clear_elevation_result(state)
                st.button('Extract elevation data', disabled=selected_data_source is None, on_click=extract_data_in_bbox, args=[status_update_area])

            with st.container(border=True):
                st.download_button(
                    'Download elevation .csv-file', 
                    dataframe_to_csv_bytes(state.elevation_in_bbox) if state.elevation_in_bbox is not None else 'dummy',
                    file_name=suggest_elevation_filename(state),
                    disabled=state.elevation_in_bbox is None,
                )
        if state.map_mode == 'OpenStreetMap':
            title_dict = {
                'black_sea': 'Black Sea',
                'cold_war': 'Cold War',
                'fortress_italy': 'Fortress Italy',
                'shock_force_2': 'Shock Force 2'
            }
            with st.container(border=True):
                profile_str = st.selectbox(
                    "Select Combat Mission Title",
                    options=['black_sea', 'cold_war', 'fortress_italy', 'shock_force_2'],
                    index=list(title_dict).index(state.osm_profile)
                    if state.osm_profile in title_dict
                    else list(title_dict).index("cold_war"),
                    format_func=lambda x: title_dict[x],
                )
                config_files = [path.name for path in find_default_osm_configs(resources)]
                default_config_files = {
                    'black_sea': 'default_osm_config_cmbs.json',
                    'cold_war': 'default_osm_config_cmcw.json',
                    'fortress_italy': 'default_osm_config_cmfi.json',
                    'shock_force_2': 'default_osm_config_cmsf2.json',
                }
                config_file = st.selectbox(
                    "Select configuration file",
                    options=config_files,
                    index=config_files.index(default_config_files[profile_str]) if default_config_files[profile_str] in config_files else 0
                )
                osm_settings_changed = (
                    config_file != state.osm_config_file or profile_str != state.osm_profile
                )
                state.osm_config_file = config_file
                state.osm_profile = profile_str
                state.osm_config = load_osm_config(config_path=resources.config_dir / config_file)
                if osm_settings_changed:
                    clear_osm_processing_result(state)

            with st.container(border=True):
                with st.container(border=True):
                    if not state.selected_area_valid:
                        st.markdown(":red[Please select a valid bounding box first.]")
                    st.button('Download OpenStreeMap data', disabled=not state.selected_area_valid, on_click=get_osm_data, args=[status_update_area])
                st.markdown('-OR-')
                with st.container(border=True):
                    osm_file = st.file_uploader('Import OpenStreetMap file', type='geojson')
                    if osm_file is not None:
                        state.osm_data = load_osm_data_from_uploaded_bytes(
                            data=osm_file.getvalue(),
                            filename=osm_file.name,
                        )
                        state.osm_bbox_object = get_bounding_box(state.osm_data)
                        clear_osm_processing_result(state)
                with st.container(border=True):
                    processing_enabled = True
                    if not state.selected_area_valid:
                        st.markdown(":red[Please select a valid bounding box first.]")
                        processing_enabled = False
                    # if not 'osm_file' in st.session_state:
                    if state.osm_data is None:
                        st.markdown(":red[Please import or download OpenStreetMap data first.]")
                        processing_enabled = False
                    st.button('Process OpenStreeMap data', disabled=not processing_enabled, on_click=process_osm_data, args=[status_update_area])
            with st.container(border=True):
                st.download_button(
                    'Download OpenStreetMap .csv-file', 
                    dataframe_to_csv_bytes(state.osm_output) if state.osm_output is not None else 'dummy',
                    file_name=suggest_osm_filename(state),
                    disabled=state.osm_output is None,
                )


def process_osm_data(status_update_area):
    with status_update_area.container(), st.status('Processing OpenStreetMap data...'):
        st.write('Processing data...')
        state.osm_output, state.osm_geometries = process_osm_data_action(
            osm_data=state.osm_data,
            bbox=state.bbox_object,
            config_path=resources.config_dir / state.osm_config_file,
            profile=state.osm_profile,
        )
        st.write('Processing complete.')
    status_update_area.empty()
    # osm_processor.write_to_file(args.output_file)

def get_osm_data(status_update_area):
    state.osm_data = download_osm_data_action(
        bbox=state.bbox_object,
        config=load_osm_config(config_path=resources.config_dir / state.osm_config_file),
    )
    state.osm_bbox_object = get_bounding_box(state.osm_data)
    clear_osm_processing_result(state)

def map_view_tab():  # noqa: C901, PLR0915
    if state.map_mode == 'Bounding Box Selection':
        header = 'Bounding Box Selection'
        sub_header = 'Select the outline of the Combat Mission map by drawing a rectangle or polygon.'
    elif state.map_mode == 'Elevations':
        header = 'Extraction of Elevation Data'
        sub_header = 'Check which data sources are available for your selected outline and extract the data.'
    else:
        header = 'Extraction of OpenStreetMap Data'
        sub_header = 'Extract map content from OpenStreetMap for your selected outline.'

    st.header(header)
    st.markdown(
        sub_header
    )

    # if 'center' not in st.session_state:
    #     st.session_state['center'] = {'lat': 0, 'lon': 0}
    # if 'zoom' not in st.session_state:
    #     st.session_state['zoom'] = 1

    if state.elevation_in_bbox is not None and 'height_map_layer' not in st.session_state:
        # if 'zoom_cache' in st.session_state:
        if 'zoom_cache' in st.session_state:
            state.map_zoom = st.session_state['zoom_cache']
        if 'center_cache' in st.session_state:
            center_cache = st.session_state['center_cache']
            state.map_center = (center_cache['lat'], center_cache['lng'])
        state.map_key += 1
        st.session_state['height_map_layer'] = True

    # tags = []
    # if st.session_state['map_mode'] == 'OpenStreetMap' and 'osm_config' in st.session_state:
    #     tags = list(st.session_state['osm_config'].keys())
    # if len(tags) > 0:
    #     folium.plugins.TagFilterButton(tags[0:2]).add_to(map)

    # folium.LayerControl().add_to(map)

    map_obj = build_folium_map(state=state, resources=resources)

    st_data = st_folium(
        map_obj,
        center=state.map_center,
        zoom=state.map_zoom,
        width=1200,
        key=state.map_key,
    )

    drawing = extract_last_active_drawing(st_data)
    if drawing is not None and state.map_mode == 'Bounding Box Selection':
        coordinates = np.array(drawing['geometry']['coordinates'])
        if 'drawn_coordinates' not in st.session_state or \
            st.session_state['drawn_coordinates'].shape != coordinates.shape or \
            not (st.session_state['drawn_coordinates'] == coordinates).all():
            st.session_state['drawn_coordinates'] = coordinates
            update_state_from_bbox(state, drawing_to_bounding_box(drawing))
            st.rerun()

    if 'zoom' in st_data:
        st.session_state['zoom_cache'] = st_data['zoom']
    if 'center' in st_data:
        st.session_state['center_cache'] = st_data['center']
        # del st_data['last_active_drawing']
        # if not('bbox_object' in st.session_state and BoundingBox(Polygon(coordinates[0])).equals(st.session_state['bbox_object'])):
        #     update_bounding_box(coordinates[0])
    
    if 'edited_df' in st.session_state:
        if state.bbox_object is not None:
            df1 = st.session_state['edited_df']
            df2 = state.bbox_object.get_dataframe()
            if not (df1 == df2).all().all():
                update_bbox_from_df()
        else:
            update_bbox_from_df()

    # st.write(st_data)

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
            'Include in Search': [ds in state.selectable_data_sources for ds in data_sources]
        },
        column_order=['Name', 'Country/Region', 'Type', 'Resolution', 'Format', 'Include in Search'],
        disabled=['Name', 'Type', 'Resolution', 'Format']
    )
    selected_data_source_names = []
    for didx, ds_selected in enumerate(data_source_dict['Include in Search']):
        if ds_selected:
            selected_data_source_names.append(data_source_dict['Name'][didx])

    state.selectable_data_sources = [ds for ds in data_sources if ds.name in selected_data_source_names]

    file_sizes = 0
    for dir_path, _dir_name, file_names in os.walk(data_cache_path):
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
    
    st.button(f'Clear Cache ({np.round(file_sizes / factor, decimals=2)} {size_str})')

if __name__ == '__main__':
    st.set_page_config(
        page_title='CM Terrain Extractor',
        layout="wide", 
        menu_items={'Report a bug': "https://github.com/DerButschi/CMAutoEditor/issues/new/choose"})    

    status_update_area = st.empty()
    draw_sidebar(status_update_area)


    tab1, tab2 = st.tabs(['Map View', 'Options'])

    with tab1:
        map_view_tab()

    with tab2:
        options_tab()

    if state.currently_processing_data is not None:
        st.warning('{} was interrupted before it was finished. This may lead to corrupt data. If you encounter issues with the data, go to the options tab and clear the {} cache'.format(
            *state.currently_processing_data
        ))
        state.currently_processing_data = None

    # st.write(st_data)

