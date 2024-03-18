from typing import Dict
import streamlit as st
import folium
from shapely import GeometryCollection
from pyproj.crs import CRS
from terrain_extraction.projection_utils import transform_point, transform_linestring, transform_polygon

def shapely2folium(geometry: GeometryCollection, visualization_dict: Dict, tooltip=None):
    if visualization_dict is None:
        return
    color = visualization_dict['color'] if 'color' in visualization_dict else '#3388ff'
    opacity = visualization_dict['opacity'] if 'opacity' in visualization_dict else 1.0
    if geometry.geom_type == 'Polygon':
        locations = [[coord[1], coord[0]] for coord in geometry.exterior.coords]
        folium_geometry = folium.vector_layers.Polygon(
            stroke=False,
            locations=locations,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=tooltip,
            tags=[tooltip]
        )
        return folium_geometry
    elif geometry.geom_type == 'MultiPolygon':
        locations = []
        for polygon in geometry.geoms:
            locations.append([[coord[1], coord[0]] for coord in polygon.exterior.coords])
        folium_geometry = folium.vector_layers.Polygon(
            stroke=False,
            locations=locations,
            fill_color=color,
            fill_opacity=opacity,
            tooltip=tooltip,
            tags=[tooltip]
        )
        return folium_geometry
    else:
        return None




    