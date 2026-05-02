from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.app_core.state import AppState
from cm_terrain_extractor_app.app_core.validation import (
    MAX_LEN_X_METERS,
    MAX_LEN_Y_METERS,
    MAX_SELECTED_AREA_SQUARE_METERS,
)


def format_data_source_label(data_source: Any) -> str:
    return f"{data_source.name} - {data_source.model_type}, {data_source.resolution}"


def format_cache_size_label(cache_path: Path) -> str:
    file_sizes = 0
    if cache_path.exists():
        for dir_path, _dir_name, file_names in os.walk(cache_path):
            for file_name in file_names:
                file_sizes += os.path.getsize(Path(dir_path) / file_name)

    sizes = ["KB", "MB", "GB", "TB"]
    factor = 1
    for size in sizes:
        factor *= 1024
        if file_sizes / factor < 1024:
            return f"{np.round(file_sizes / factor, decimals=2)} {size}"
    return f"{np.round(file_sizes / factor, decimals=2)} {sizes[-1]}"


def render_bbox_editor(state: AppState) -> None:
    if state.bbox_object is not None:
        dataframe = state.bbox_object.get_dataframe()
    else:
        dataframe = pd.DataFrame(
            {
                "x": [None, None, None, None],
                "y": [None, None, None, None],
            }
        )

    st.session_state["edited_df"] = st.data_editor(
        dataframe,
        column_config={
            "x": st.column_config.NumberColumn(
                "Longitude [\u00b0]",
                min_value=-180.0,
                max_value=180.0,
            ),
            "y": st.column_config.NumberColumn(
                "Latitude [\u00b0]",
                min_value=-90.0,
                max_value=90.0,
            ),
        },
    )


def render_bbox_metrics(state: AppState) -> None:
    len_x_axis = state.len_x
    len_y_axis = state.len_y
    area = len_x_axis * len_y_axis if len_x_axis is not None and len_y_axis is not None else None
    delta_len_x = len_x_axis - MAX_LEN_X_METERS if len_x_axis is not None else None
    delta_len_y = len_y_axis - MAX_LEN_Y_METERS if len_y_axis is not None else None
    delta_area = area - MAX_SELECTED_AREA_SQUARE_METERS if area is not None else None

    col1, col2, col3 = st.columns(3)
    with col1:
        _metric_with_limit(
            label="Length W\u2194E",
            value=len_x_axis,
            suffix="m",
            delta=delta_len_x,
        )
    with col2:
        _metric_with_limit(
            label="Length S\u2194N",
            value=len_y_axis,
            suffix="m",
            delta=delta_len_y,
        )
    with col3:
        _metric_with_limit(
            label="Selected Area",
            value=area / 1e6 if area is not None else None,
            suffix="km\u00b2",
            delta=delta_area / 1e6 if delta_area is not None else None,
            decimals=1,
        )


def render_options_tab(
    *,
    state: AppState,
    resources: AppResources,
    data_sources: list[Any],
) -> None:
    st.markdown("Select which data sources should be queried for available elevation data.")
    data_source_dict = st.data_editor(
        {
            "Name": [data_source.name for data_source in data_sources],
            "Country/Region": [data_source.country for data_source in data_sources],
            "Type": [data_source.model_type for data_source in data_sources],
            "Resolution": [data_source.resolution for data_source in data_sources],
            "Format": [data_source.data_type for data_source in data_sources],
            "Include in Search": [
                data_source in state.selectable_data_sources for data_source in data_sources
            ],
        },
        column_order=[
            "Name",
            "Country/Region",
            "Type",
            "Resolution",
            "Format",
            "Include in Search",
        ],
        disabled=["Name", "Type", "Resolution", "Format"],
    )

    selected_data_source_names = [
        data_source_dict["Name"][index]
        for index, is_selected in enumerate(data_source_dict["Include in Search"])
        if is_selected
    ]
    state.selectable_data_sources = [
        data_source for data_source in data_sources if data_source.name in selected_data_source_names
    ]

    st.button(f"Clear Cache ({format_cache_size_label(resources.data_cache_path)})")


def render_interrupted_processing_warning(state: AppState) -> None:
    if state.currently_processing_data is None:
        return
    st.warning(
        "{} was interrupted before it was finished. This may lead to corrupt data. "
        "If you encounter issues with the data, go to the options tab and clear the {} cache".format(
            *state.currently_processing_data
        )
    )
    state.currently_processing_data = None


def _metric_with_limit(
    *,
    label: str,
    value: float | None,
    suffix: str,
    delta: float | None,
    decimals: int = 0,
) -> None:
    if value is None:
        st.metric(label=label, value="-")
        return

    rounded_value = np.round(value, decimals=decimals)
    value_text = f"{rounded_value.astype(int) if decimals == 0 else rounded_value} {suffix}"
    if delta is not None and delta > 0:
        rounded_delta = np.round(delta, decimals=decimals)
        delta_text = f"{rounded_delta.astype(int) if decimals == 0 else rounded_delta} {suffix}"
        st.metric(label=label, value=value_text, delta=delta_text, delta_color="inverse")
    else:
        st.metric(label=label, value=value_text)
