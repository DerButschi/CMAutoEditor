from __future__ import annotations

import folium

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.app_core.state import AppState
from cm_terrain_extractor_app.map_view.folium_layers import (
    add_bbox_layer,
    add_draw_control,
    add_elevation_overlay,
    add_osm_bbox_layer,
    add_osm_debug_layers,
    add_osm_geometry_layers,
)

OPENTOPOMAP_TILES = "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
OPENTOPOMAP_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: '
    '&copy; <a href="https://opentopomap.org">OpenTopoMap</a> '
    '(<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
)


def build_folium_map(*, state: AppState, resources: AppResources) -> folium.Map:
    map_obj = folium.Map(
        location=state.map_center,
        zoom_start=state.map_zoom,
        tiles=OPENTOPOMAP_TILES,
        attr=OPENTOPOMAP_ATTRIBUTION,
    )

    if state.map_mode == "Bounding Box Selection":
        add_draw_control(map_obj)

    folium.plugins.Geocoder(position="bottomleft").add_to(map_obj)
    folium.plugins.MeasureControl().add_to(map_obj)
    folium.plugins.Fullscreen().add_to(map_obj)

    add_elevation_overlay(map_obj, state, resources)
    add_bbox_layer(map_obj, state)
    add_osm_bbox_layer(map_obj, state)
    add_osm_geometry_layers(map_obj, state)
    if add_osm_debug_layers(map_obj, state):
        folium.LayerControl(collapsed=False).add_to(map_obj)
    return map_obj
