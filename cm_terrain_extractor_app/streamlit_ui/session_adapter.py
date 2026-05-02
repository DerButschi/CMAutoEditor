from __future__ import annotations

import streamlit as st

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.app_core.state import AppState

STATE_KEY = "cm_terrain_extractor_state"


def get_state(resources: AppResources) -> AppState:
    _ = resources
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = AppState()
    return st.session_state[STATE_KEY]


def reset_state(resources: AppResources) -> AppState:
    _ = resources
    st.session_state[STATE_KEY] = AppState()
    return st.session_state[STATE_KEY]
