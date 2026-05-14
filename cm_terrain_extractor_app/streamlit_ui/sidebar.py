from __future__ import annotations

import hashlib
import sys
from typing import Any

import streamlit as st

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
    AppResources,
    find_default_osm_configs,
)
from cm_terrain_extractor_app.app_core.state import (
    OSM_DATA_SOURCE_DOWNLOADED,
    AppState,
    clear_elevation_result,
    clear_osm_processing_result,
)
from cm_terrain_extractor_app.app_core.validation import (
    compute_bbox_metrics,
    is_selected_area_valid,
    update_state_from_uploaded_osm_data,
)
from cm_terrain_extractor_app.streamlit_ui.widgets import (
    format_data_source_label,
    render_bbox_editor,
    render_bbox_metrics,
)


def render_sidebar(
    *,
    state: AppState,
    resources: AppResources,
    status_update_area: Any,
) -> None:
    if state.len_x is not None and state.len_y is not None:
        state.selected_area_valid = is_selected_area_valid(state.len_x, state.len_y)
    else:
        state.selected_area_valid = False

    with st.sidebar:
        _render_mode_selector(state)
        if state.map_mode == "Bounding Box Selection":
            _render_bbox_controls(state)
        if state.map_mode == "Elevations":
            _render_elevation_controls(state, resources, status_update_area)
        if state.map_mode == "OpenStreetMap":
            _render_osm_controls(state, resources, status_update_area)


def _render_mode_selector(state: AppState) -> None:
    with st.container(border=True):
        map_mode_options = ["Bounding Box Selection", "Elevations", "OpenStreetMap"]
        map_mode = st.radio(
            "",
            map_mode_options,
            captions=[
                "Select the outline of the Combat Mission map.",
                "Extract elevation data.",
                "Extract map content from OpenStreetMap.",
            ],
            index=map_mode_options.index(state.map_mode)
            if state.map_mode in map_mode_options
            else 0,
            label_visibility="collapsed",
        )
        if map_mode != state.map_mode:
            state.map_mode = map_mode


def _render_bbox_controls(state: AppState) -> None:
    with st.container(border=True):
        render_bbox_editor(state)
        st.button(
            "Cycle bounding box origin",
            disabled=not state.selected_area_valid,
            on_click=_permute_bbox,
            args=[state],
        )
        render_bbox_metrics(state)


def _render_elevation_controls(
    state: AppState,
    resources: AppResources,
    status_update_area: Any,
) -> None:
    with st.container(border=True):
        if not state.selected_area_valid:
            st.markdown(":red[Please select a valid bounding box first.]")
        st.button(
            "Find available data sources",
            disabled=not state.selected_area_valid,
            on_click=_find_data_sources_in_bbox,
            args=[state, status_update_area],
        )
        selected_index = (
            state.available_data_sources.index(state.selected_data_source)
            if state.selected_data_source in state.available_data_sources
            else 0 if state.available_data_sources else None
        )
        selected_data_source = st.selectbox(
            "Data sources",
            state.available_data_sources,
            index=selected_index,
            format_func=format_data_source_label,
        )

        if selected_data_source != state.selected_data_source:
            state.selected_data_source = selected_data_source
            clear_elevation_result(state)
        st.button(
            "Extract elevation data",
            disabled=selected_data_source is None,
            on_click=_extract_data_in_bbox,
            args=[state, resources, status_update_area],
        )

    with st.container(border=True):
        st.download_button(
            "Download elevation .csv-file",
            dataframe_to_csv_bytes(state.elevation_in_bbox)
            if state.elevation_in_bbox is not None
            else "dummy",
            file_name=suggest_elevation_filename(state),
            disabled=state.elevation_in_bbox is None,
        )


def _render_osm_controls(
    state: AppState,
    resources: AppResources,
    status_update_area: Any,
) -> None:
    title_dict = {
        "black_sea": "Black Sea",
        "cold_war": "Cold War",
        "fortress_italy": "Fortress Italy",
        "shock_force_2": "Shock Force 2",
    }
    with st.container(border=True):
        profile_str = st.selectbox(
            "Select Combat Mission Title",
            options=["black_sea", "cold_war", "fortress_italy", "shock_force_2"],
            index=list(title_dict).index(state.osm_profile)
            if state.osm_profile in title_dict
            else list(title_dict).index("cold_war"),
            format_func=lambda value: title_dict[value],
        )
        config_files = [path.name for path in find_default_osm_configs(resources)]
        default_config_files = {
            "black_sea": "default_osm_config_cmbs.json",
            "cold_war": "default_osm_config_cmcw.json",
            "fortress_italy": "default_osm_config_cmfi.json",
            "shock_force_2": "default_osm_config_cmsf2.json",
        }
        config_file = st.selectbox(
            "Select configuration file",
            options=config_files,
            index=config_files.index(default_config_files[profile_str])
            if default_config_files[profile_str] in config_files
            else 0,
        )
        osm_settings_changed = config_file != state.osm_config_file or profile_str != state.osm_profile
        state.osm_config_file = config_file
        state.osm_profile = profile_str
        state.osm_config = load_osm_config(config_path=resources.config_dir / config_file)
        if osm_settings_changed:
            clear_osm_processing_result(state)

    with st.container(border=True):
        with st.container(border=True):
            if not state.selected_area_valid:
                st.markdown(":red[Please select a valid bounding box first.]")
            st.button(
                "Download OpenStreeMap data",
                disabled=not state.selected_area_valid,
                on_click=_get_osm_data,
                args=[state, resources],
            )
        st.markdown("-OR-")
        with st.container(border=True):
            osm_file = st.file_uploader("Import OpenStreetMap file", type="geojson")
            if osm_file is not None:
                _handle_uploaded_osm_file(state, resources, osm_file)
        with st.container(border=True):
            processing_enabled = True
            if not state.selected_area_valid:
                st.markdown(":red[Please select a valid bounding box first.]")
                processing_enabled = False
            if state.osm_data is None:
                st.markdown(":red[Please import or download OpenStreetMap data first.]")
                processing_enabled = False
            st.button(
                "Process OpenStreeMap data",
                disabled=not processing_enabled,
                on_click=_process_osm_data,
                args=[state, resources, status_update_area],
            )
    with st.container(border=True):
        st.download_button(
            "Download OpenStreetMap .csv-file",
            dataframe_to_csv_bytes(state.osm_output) if state.osm_output is not None else "dummy",
            file_name=suggest_osm_filename(state),
            disabled=state.osm_output is None,
        )


def _find_data_sources_in_bbox(state: AppState, status_update_area: Any) -> None:
    with status_update_area.container(
        border=True
    ), st.spinner("Searching for data sources in the selected area..."):
        state.available_data_sources = find_data_sources_in_bbox_action(
            bbox=state.bbox_object,
            selectable_sources=state.selectable_data_sources,
        )
    status_update_area.empty()


def _extract_data_in_bbox(
    state: AppState,
    resources: AppResources,
    status_update_area: Any,
) -> None:
    with status_update_area.container():
        data_source = state.selected_data_source
        bounding_box = state.bbox_object
        state.currently_processing_data = (
            f"Extracting data from {data_source.name}",
            data_source.name,
        )

        with st.status("Extracting elevation data", expanded=True) as status:
            state.elevation_in_bbox, state.height_map_png = extract_elevation_data_action(
                data_source=data_source,
                bbox=bounding_box,
                data_cache_path=resources.data_cache_path,
            )
            status.update(label="Elevation data extracted!", state="complete", expanded=False)

        state.currently_processing_data = None
    status_update_area.empty()


def _process_osm_data(
    state: AppState,
    resources: AppResources,
    status_update_area: Any,
) -> None:
    with status_update_area.container(), st.status("Processing OpenStreetMap data..."):
        st.write("Processing data...")
        state.osm_output, state.osm_geometries = process_osm_data_action(
            osm_data=state.osm_data,
            bbox=state.bbox_object,
            config_path=resources.config_dir / state.osm_config_file,
            profile=state.osm_profile,
        )
        st.write("Processing complete.")
    status_update_area.empty()


def _handle_uploaded_osm_file(
    state: AppState,
    resources: AppResources,
    osm_file: Any,
) -> None:
    upload_bytes = osm_file.getvalue()
    upload_signature = _uploaded_osm_file_signature(
        filename=getattr(osm_file, "name", None),
        data=upload_bytes,
    )
    if upload_signature == state.osm_uploaded_file_signature:
        return

    osm_data = load_osm_data_from_uploaded_bytes(
        data=upload_bytes,
        filename=getattr(osm_file, "name", None),
    )
    update_state_from_uploaded_osm_data(
        state,
        osm_data=osm_data,
        osm_bbox_object=_get_bounding_box(osm_data, resources),
        upload_signature=upload_signature,
    )


def _uploaded_osm_file_signature(*, filename: str | None, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    return f"{filename or ''}:{len(data)}:{digest}"


def _get_osm_data(
    state: AppState,
    resources: AppResources,
) -> None:
    state.osm_data = download_osm_data_action(
        bbox=state.bbox_object,
        config=load_osm_config(config_path=resources.config_dir / state.osm_config_file),
    )
    state.osm_data_source = OSM_DATA_SOURCE_DOWNLOADED
    state.osm_uploaded_file_signature = None
    state.osm_bbox_object = _get_bounding_box(state.osm_data, resources)
    clear_osm_processing_result(state)


def _permute_bbox(state: AppState) -> None:
    bounding_box = state.bbox_object
    bounding_box.cycle_origin()
    metrics = compute_bbox_metrics(bounding_box)
    state.bbox_coordinates = bounding_box.get_coordinates(xy=False)
    state.projected_bbox_object = bounding_box.get_box(bounding_box.crs_projected)
    state.len_x = metrics["len_x"]
    state.len_y = metrics["len_y"]
    state.selected_area_valid = is_selected_area_valid(state.len_x, state.len_y)
    state.bbox_origin = (state.bbox_origin + 1) % 4


def _get_bounding_box(osm_data: dict, resources: AppResources) -> Any:
    _ensure_legacy_app_path(resources)
    from terrain_extraction.osm_utils.io import get_bounding_box

    return get_bounding_box(osm_data)


def _ensure_legacy_app_path(resources: AppResources) -> None:
    app_root = str(resources.app_root)
    if app_root not in sys.path:
        sys.path.append(app_root)
