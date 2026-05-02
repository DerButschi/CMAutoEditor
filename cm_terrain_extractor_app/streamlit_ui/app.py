from __future__ import annotations

import sys
import warnings

import streamlit as st

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.streamlit_ui.map_page import render_map_page
from cm_terrain_extractor_app.streamlit_ui.session_adapter import get_state
from cm_terrain_extractor_app.streamlit_ui.sidebar import render_sidebar
from cm_terrain_extractor_app.streamlit_ui.widgets import (
    render_interrupted_processing_warning,
    render_options_tab,
)


def render_app(resources: AppResources) -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    st.set_page_config(
        page_title="CM Terrain Extractor",
        layout="wide",
        menu_items={
            "Report a bug": "https://github.com/DerButschi/CMAutoEditor/issues/new/choose"
        },
    )

    state = get_state(resources)
    data_sources = _build_data_sources(resources)
    if not state.selectable_data_sources:
        state.selectable_data_sources = list(data_sources)

    status_update_area = st.empty()
    render_sidebar(
        state=state,
        resources=resources,
        status_update_area=status_update_area,
    )

    map_tab, options_tab = st.tabs(["Map View", "Options"])
    with map_tab:
        render_map_page(state=state, resources=resources)
    with options_tab:
        render_options_tab(
            state=state,
            resources=resources,
            data_sources=data_sources,
        )

    render_interrupted_processing_warning(state)


def _build_data_sources(resources: AppResources) -> list[object]:
    _ensure_legacy_app_path(resources)
    from terrain_extraction.data_sources.aw3d30.data_source import AW3D30DataSource
    from terrain_extraction.data_sources.bavaria_dgm1.data_source import BavariaDataSource
    from terrain_extraction.data_sources.hessen_dgm1.data_source import HessenDataSource
    from terrain_extraction.data_sources.lower_saxony_dgm1.data_source import (
        LowerSaxonyDataSource,
    )
    from terrain_extraction.data_sources.netherlands_dtm05.data_source import (
        NetherlandsDataSource,
    )
    from terrain_extraction.data_sources.nrw_dgm1.data_source import NRWDataSource
    from terrain_extraction.data_sources.rge_alti.data_source import FranceDataSource
    from terrain_extraction.data_sources.thuringia_dgm1.data_source import (
        ThuringiaDataSource,
    )

    return [
        HessenDataSource(),
        NRWDataSource(),
        FranceDataSource(),
        NetherlandsDataSource(),
        BavariaDataSource(),
        ThuringiaDataSource(),
        AW3D30DataSource(),
        LowerSaxonyDataSource(),
    ]


def _ensure_legacy_app_path(resources: AppResources) -> None:
    app_root = str(resources.app_root)
    if app_root not in sys.path:
        sys.path.append(app_root)
