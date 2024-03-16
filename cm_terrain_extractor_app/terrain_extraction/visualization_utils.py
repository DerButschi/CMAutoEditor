import streamlit as st
import folium
from shapely import GeometryCollection
from pyproj.crs import CRS
from terrain_extraction.projection_utils import transform_point, transform_linestring, transform_polygon

def shapely2folium(geometry: GeometryCollection):
    if geometry.geom_type == 'Polygon':
        locations = [[coord[1], coord[0]] for coord in geometry.exterior.coords]
        folium_geometry = folium.vector_layers.Polygon(
            locations=locations
        )
        return folium_geometry
    elif geometry.geom_type == 'MultiPolygon':
        locations = []
        for polygon in geometry.geoms:
            locations.append([[coord[1], coord[0]] for coord in polygon.exterior.coords])
        folium_geometry = folium.vector_layers.Polygon(
            locations=locations
        )
        return folium_geometry
    else:
        return None




    