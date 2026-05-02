from __future__ import annotations

from pathlib import Path
from typing import Any

import folium
from folium.plugins import Draw

from cm_terrain_extractor_app.app_core.resources import AppResources
from cm_terrain_extractor_app.app_core.state import AppState


def add_draw_control(map_obj: folium.Map) -> None:
    draw = Draw(
        draw_options={
            "polyline": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
        }
    )
    draw.add_to(map_obj)


def add_bbox_layer(map_obj: folium.Map, state: AppState) -> None:
    if state.bbox_coordinates is None:
        return

    coordinates = state.bbox_coordinates
    line1 = folium.vector_layers.PolyLine([coordinates[0], coordinates[1]], color="red")
    line2 = folium.vector_layers.PolyLine([coordinates[1], coordinates[2]], color="red")
    line3 = folium.vector_layers.PolyLine([coordinates[2], coordinates[3]], color="red")
    line4 = folium.vector_layers.PolyLine([coordinates[0], coordinates[3]], color="red")
    line1_text = folium.plugins.PolyLineTextPath(
        line1,
        "CM W\u2194E axis",
        center=True,
        offset=20,
        color="red",
        attributes={"font-size": 16, "fill": "red"},
    )
    line4_text = folium.plugins.PolyLineTextPath(
        line4,
        "CM S\u2194N axis",
        center=True,
        offset=-7,
        color="red",
        attributes={"font-size": 16, "fill": "red"},
    )

    for layer in (
        line1,
        line2,
        line3,
        line4,
        line1_text,
        line4_text,
        folium.vector_layers.CircleMarker(coordinates[0], color="red", radius=5),
    ):
        layer.add_to(map_obj)


def add_elevation_overlay(
    map_obj: folium.Map,
    state: AppState,
    resources: AppResources,
) -> None:
    if state.elevation_in_bbox is None or state.bbox_coordinates is None:
        return

    image_path = state.height_map_png or resources.data_cache_path / "current_height_map.png"
    folium.raster_layers.ImageOverlay(
        name="Elevation data",
        image=str(image_path),
        bounds=state.bbox_coordinates,
        opacity=0.9,
    ).add_to(map_obj)


def add_osm_bbox_layer(map_obj: folium.Map, state: AppState) -> None:
    if state.osm_bbox_object is None:
        return

    coordinates = state.osm_bbox_object.get_coordinates(xy=False)
    line1 = folium.vector_layers.PolyLine([coordinates[0], coordinates[1]], color="red", dash_array="6")
    line2 = folium.vector_layers.PolyLine([coordinates[1], coordinates[2]], color="red", dash_array="6")
    line3 = folium.vector_layers.PolyLine([coordinates[2], coordinates[3]], color="red", dash_array="6")
    line4 = folium.vector_layers.PolyLine([coordinates[0], coordinates[3]], color="red", dash_array="6")
    line1_text = folium.plugins.PolyLineTextPath(
        line1,
        "OSM data",
        center=True,
        offset=20,
        color="red",
        attributes={"font-size": 16, "fill": "red"},
    )

    for layer in (line1, line2, line3, line4, line1_text):
        layer.add_to(map_obj)


def add_osm_geometry_layers(map_obj: folium.Map, state: AppState) -> None:
    if state.osm_geometries is None:
        return

    grouped_layers: dict[int, list[folium.MacroElement]] = {}
    for key, geometries in state.osm_geometries.items():
        visualization = _visualization_for_key(state, key)
        priority = _priority_for_key(state, key)
        grouped_layers.setdefault(priority, [])
        for geometry in geometries:
            folium_geometry = _shapely_to_folium(geometry, visualization, key)
            if folium_geometry is not None:
                grouped_layers[priority].append(folium_geometry)

    for priority in sorted(grouped_layers, reverse=True):
        for folium_geometry in grouped_layers[priority]:
            folium_geometry.add_to(map_obj)


def _visualization_for_key(state: AppState, key: str) -> dict[str, Any] | None:
    if state.osm_config is None or key not in state.osm_config:
        return None
    return state.osm_config[key].get("visualization")


def _priority_for_key(state: AppState, key: str) -> int:
    if state.osm_config is None or key not in state.osm_config:
        return -999
    return int(state.osm_config[key].get("priority", -999))


def _shapely_to_folium(
    geometry: Any,
    visualization: dict[str, Any] | None,
    tooltip: str | None = None,
) -> folium.vector_layers.Polygon | None:
    if visualization is None:
        return None

    color = visualization.get("color", "#3388ff")
    opacity = visualization.get("opacity", 1.0)
    if geometry.geom_type == "Polygon":
        locations = [_lon_lat_to_lat_lon(geometry.exterior.coords)]
    elif geometry.geom_type == "MultiPolygon":
        locations = [_lon_lat_to_lat_lon(polygon.exterior.coords) for polygon in geometry.geoms]
    else:
        return None

    return folium.vector_layers.Polygon(
        locations=locations,
        stroke=False,
        fill_color=color,
        fill_opacity=opacity,
        tooltip=tooltip,
        tags=[tooltip],
    )


def _lon_lat_to_lat_lon(coordinates: Any) -> list[list[float]]:
    return [[coord[1], coord[0]] for coord in coordinates]


def default_height_map_path(resources: AppResources) -> Path:
    return resources.data_cache_path / "current_height_map.png"
