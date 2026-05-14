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


def add_osm_debug_layers(map_obj: folium.Map, state: AppState) -> bool:
    layers_added = False
    if _has_raw_osm_features(state.osm_data):
        raw_group = folium.FeatureGroup(name="Raw OSM features", overlay=True, show=False)
        folium.GeoJson(
            state.osm_data,
            name="Raw OSM features",
            style_function=lambda _feature: {
                "color": "#6b7280",
                "weight": 2,
                "fillColor": "#9ca3af",
                "fillOpacity": 0.12,
            },
            marker=folium.CircleMarker(radius=3, color="#6b7280", fill=True, fill_opacity=0.8),
        ).add_to(raw_group)
        raw_group.add_to(map_obj)
        layers_added = True

    debug_layers = state.osm_debug_layers or {}
    layers_added = _add_debug_geojson_layer(
        map_obj,
        debug_layers.get("source_features"),
        name="Matched OSM features",
        style={"color": "#16a34a", "weight": 3, "fillColor": "#22c55e", "fillOpacity": 0.18},
    ) or layers_added

    network_group = _network_debug_group(debug_layers)
    if network_group is not None:
        network_group.add_to(map_obj)
        layers_added = True

    layers_added = _add_debug_geojson_layer(
        map_obj,
        debug_layers.get("final_rows"),
        name="Output CSV tiles",
        style={"color": "#0891b2", "weight": 1, "fillColor": "#06b6d4", "fillOpacity": 0.32},
    ) or layers_added
    return layers_added


def _has_raw_osm_features(osm_data: dict | None) -> bool:
    return isinstance(osm_data, dict) and bool(osm_data.get("features"))


def _network_debug_group(debug_layers: dict) -> folium.FeatureGroup | None:
    group = folium.FeatureGroup(name="Internal OSM networks", overlay=True, show=False)
    layer_specs = (
        ("topology_edges", {"color": "#f97316", "weight": 3, "fillOpacity": 0.0}),
        ("routed_paths", {"color": "#2563eb", "weight": 2, "dashArray": "4", "fillOpacity": 0.0}),
        ("topology_nodes", {"color": "#7c3aed", "weight": 2, "fillColor": "#7c3aed", "fillOpacity": 0.8}),
        ("route_anchors", {"color": "#db2777", "weight": 2, "fillColor": "#db2777", "fillOpacity": 0.8}),
    )
    added = False
    for layer_name, style in layer_specs:
        added = _add_debug_geojson_layer(group, debug_layers.get(layer_name), name=layer_name, style=style) or added
    return group if added else None


def _add_debug_geojson_layer(
    target: folium.Map | folium.FeatureGroup,
    layer: Any,
    *,
    name: str,
    style: dict[str, Any],
) -> bool:
    if layer is None or getattr(layer, "empty", True):
        return False

    folium.GeoJson(
        _geojson_data(layer),
        name=name,
        style_function=lambda _feature, style=style: style,
        marker=folium.CircleMarker(
            radius=4,
            color=style.get("color", "#3388ff"),
            fill=True,
            fill_color=style.get("fillColor", style.get("color", "#3388ff")),
            fill_opacity=style.get("fillOpacity", 0.8),
        ),
    ).add_to(target)
    return True


def _geojson_data(layer: Any) -> Any:
    if hasattr(layer, "to_json"):
        return layer.to_json()
    return layer


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
