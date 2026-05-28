from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely import union_all
from shapely.geometry import Polygon, box, shape
from shapely.geometry.base import BaseGeometry

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for import_path in (REPO_ROOT, APP_DIR):
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))

from terrain_extraction.osm_extraction.diagnostics import json_safe  # noqa: E402

DEFAULT_PORT = 8087
DEFAULT_VISIBLE_LAYERS = frozenset(
    {
        "original_osm",
        "source_features",
        "routed_paths",
        "building_footprints",
        "final_rows",
    }
)


LAYER_STYLES: Mapping[str, Mapping[str, Any]] = {
    "original_osm": {"color": "#64748b", "weight": 2, "fillColor": "#94a3b8", "fillOpacity": 0.12},
    "source_features": {"color": "#16a34a", "weight": 3, "fillColor": "#22c55e", "fillOpacity": 0.18},
    "topology_edges": {"color": "#f97316", "weight": 3, "fillOpacity": 0.0},
    "topology_nodes": {"color": "#7c3aed", "weight": 2, "fillColor": "#7c3aed", "fillOpacity": 0.85},
    "routed_paths": {"color": "#2563eb", "weight": 2, "fillOpacity": 0.0, "dashArray": "4"},
    "raster_spines": {"color": "#0f766e", "weight": 1, "fillColor": "#14b8a6", "fillOpacity": 0.25},
    "route_anchors": {"color": "#db2777", "weight": 2, "fillColor": "#db2777", "fillOpacity": 0.8},
    "anchor_candidates": {"color": "#9333ea", "weight": 1, "fillColor": "#c084fc", "fillOpacity": 0.18},
    "selected_anchor_plans": {"color": "#be123c", "weight": 2, "fillColor": "#fb7185", "fillOpacity": 0.22},
    "connection_bits": {"color": "#475569", "weight": 1, "fillColor": "#64748b", "fillOpacity": 0.28},
    "tile_required_dirs": {"color": "#ea580c", "weight": 1, "fillColor": "#fb923c", "fillOpacity": 0.28},
    "selected_tiles": {"color": "#0284c7", "weight": 1, "fillColor": "#38bdf8", "fillOpacity": 0.24},
    "tile_failures": {"color": "#dc2626", "weight": 2, "fillColor": "#ef4444", "fillOpacity": 0.34},
    "building_footprints": {"color": "#a16207", "weight": 2, "fillColor": "#facc15", "fillOpacity": 0.24},
    "road_validation": {"color": "#b91c1c", "weight": 2, "fillColor": "#f87171", "fillOpacity": 0.32},
    "final_rows": {"color": "#0891b2", "weight": 1, "fillColor": "#06b6d4", "fillOpacity": 0.32},
}


class _NullProgress:
    def progress(self, *args: object, **kwargs: object) -> _NullProgress:
        return self


class _FeatureCollection:
    def __init__(self, fixture: Mapping[str, Any]) -> None:
        self.features = [
            SimpleNamespace(
                properties=feature.get("properties", {}),
                geometry=feature.get("geometry"),
            )
            for feature in fixture.get("features", [])
        ]

    def __getitem__(self, key: str) -> list[SimpleNamespace]:
        if key != "features":
            raise KeyError(key)
        return self.features


@dataclass
class DebugElement:
    stable_id: str
    layer: str
    properties: Mapping[str, Any]
    geometry: BaseGeometry


@dataclass
class DebugSession:
    raw_osm: Mapping[str, Any]
    layers: Mapping[str, gpd.GeoDataFrame]
    output_rows: list[dict[str, Any]]
    diagnostics: Mapping[str, Any]
    source_label: str | None = None
    created_at: float = field(default_factory=time.time)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _elements: dict[str, DebugElement] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        prepared_layers = {}
        for layer_name, layer in self.layers.items():
            prepared = _safe_layer(layer)
            if prepared.empty:
                continue
            prepared = prepared.reset_index(drop=True).copy()
            prepared["_debug_id"] = [f"{layer_name}:{index}" for index in range(len(prepared))]
            prepared_layers[layer_name] = prepared
        self.layers = prepared_layers
        self._elements = {
            element.stable_id: element
            for layer_name, layer in self.layers.items()
            for element in _elements_from_layer(layer_name, layer)
        }

    @property
    def output_row_count(self) -> int:
        return len(self.output_rows)

    @classmethod
    def from_layers(
        cls,
        *,
        raw_osm: Mapping[str, Any],
        layers: Mapping[str, gpd.GeoDataFrame],
        output_rows: Iterable[Mapping[str, Any]],
        diagnostics: Mapping[str, Any],
        source_label: str | None = None,
    ) -> DebugSession:
        return cls(
            raw_osm=raw_osm,
            layers=dict(layers),
            output_rows=[dict(row) for row in output_rows],
            diagnostics=json_safe(dict(diagnostics)),
            source_label=source_label,
        )

    def layer_metadata(self) -> list[dict[str, Any]]:
        metadata = []
        for layer_name, layer in self.layers.items():
            metadata.append(
                {
                    "name": layer_name,
                    "label": _layer_label(layer_name),
                    "group": _layer_group(layer_name),
                    "feature_count": int(len(layer)),
                    "bounds": _layer_bounds(layer),
                    "default_visible": layer_name in DEFAULT_VISIBLE_LAYERS,
                    "style": dict(LAYER_STYLES.get(layer_name, _style_for_group(_layer_group(layer_name)))),
                }
            )
        return metadata

    def feature_collection(
        self,
        layer_name: str,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        limit: int = 5000,
    ) -> dict[str, Any]:
        layer = self.layers.get(layer_name)
        if layer is None:
            return _empty_feature_collection(layer_name)

        filtered = _filter_layer(layer, bbox)
        limited = len(filtered) > limit
        if limited:
            filtered = filtered.iloc[:limit].copy()
        features = [_feature_from_row(row) for _, row in filtered.iterrows()]
        return {
            "type": "FeatureCollection",
            "layer": layer_name,
            "features": features,
            "feature_count": int(len(layer)),
            "returned_count": len(features),
            "limited": limited,
        }

    def selection_details(self, stable_id: str) -> dict[str, Any]:
        element = self._elements.get(stable_id)
        if element is None:
            raise KeyError(f"Unknown debug element: {stable_id}")
        properties = dict(json_safe(element.properties))
        return {
            "stable_id": stable_id,
            "layer": element.layer,
            "geometry": _geometry_summary(element.geometry),
            "properties": properties,
            "related": self._related_records(element),
        }

    def table_rows(self, layer_name: str = "final_rows", *, limit: int = 1000) -> list[dict[str, Any]]:
        collection = self.feature_collection(layer_name, limit=limit)
        return [
            {
                "stable_id": feature["id"],
                "layer": layer_name,
                **feature["properties"],
            }
            for feature in collection["features"]
        ]

    def _related_records(self, element: DebugElement) -> dict[str, list[dict[str, Any]]]:
        same_feature: list[dict[str, Any]] = []
        same_cell: list[dict[str, Any]] = []
        final_rows: list[dict[str, Any]] = []
        props = dict(element.properties)
        feature_id = _first_present(props, ("feature_id", "_feature_id"))
        edge_id = props.get("edge_id")
        node_id = props.get("node_id")
        cell = _cell_key(props)
        config_name = props.get("config_name") or props.get("name")

        for candidate in self._elements.values():
            if candidate.stable_id == element.stable_id:
                continue
            candidate_props = dict(candidate.properties)
            candidate_feature_id = _first_present(candidate_props, ("feature_id", "_feature_id"))
            candidate_cell = _cell_key(candidate_props)
            candidate_config_name = candidate_props.get("config_name") or candidate_props.get("name")
            linked_by_feature = feature_id is not None and str(candidate_feature_id) == str(feature_id)
            linked_by_edge = edge_id is not None and str(candidate_props.get("edge_id")) == str(edge_id)
            linked_by_node = node_id is not None and str(candidate_props.get("node_id")) == str(node_id)
            linked_by_cell = cell is not None and candidate_cell == cell
            linked_by_config = (
                candidate.layer == "final_rows"
                and config_name is not None
                and str(candidate_config_name) == str(config_name)
            )

            if linked_by_feature or linked_by_edge or linked_by_node:
                same_feature.append(_related_summary(candidate))
            if linked_by_cell:
                same_cell.append(_related_summary(candidate))
            if candidate.layer == "final_rows" and (linked_by_feature or linked_by_cell or linked_by_config):
                final_rows.append(_related_summary(candidate))

        return {
            "same_feature": same_feature[:50],
            "same_cell": same_cell[:50],
            "final_rows": final_rows[:50],
        }


def build_debug_session(
    *,
    fixture: str | Path | None = None,
    osm_data: Mapping[str, Any] | None = None,
    profile: str = "cold_war",
    config: str | Path = "default_osm_config.json",
    seed: int | None = None,
) -> DebugSession:
    if fixture is None and osm_data is None:
        raise ValueError("Either fixture or osm_data must be provided")
    fixture_data = load_fixture(fixture) if fixture is not None else dict(osm_data or {})
    if not isinstance(fixture_data, Mapping):
        raise ValueError("OSM debug input must be a GeoJSON mapping")

    from terrain_extraction import osm_processor as osm_processor_module
    from terrain_extraction.osm_processor import OSMProcessor

    osm_processor_module.st.progress = lambda *args, **kwargs: _NullProgress()

    bbox = _bbox_from_fixture(fixture_data)
    processor = OSMProcessor(profile=profile, bbox=bbox, path_to_config=str(config))
    if seed is not None:
        processor.pipeline.context.seed = seed
        processor.pipeline.context.rng = __import__("numpy").random.default_rng(seed)

    processor.preprocess_osm_data(_FeatureCollection(fixture_data))
    processor.run_processors()
    processor.post_process()
    output_df = processor.get_output()
    debug_layers = dict(processor.get_debug_layers(crs=CRS.from_epsg(4326)))
    layers = {"original_osm": _raw_osm_layer(fixture_data), **debug_layers}
    diagnostics = {
        "diagnostics": processor.get_extraction_diagnostics(),
        "debug_export_diagnostics": getattr(processor, "debug_export_diagnostics", {}),
    }
    return DebugSession.from_layers(
        raw_osm=fixture_data,
        layers=layers,
        output_rows=_records(output_df),
        diagnostics=diagnostics,
        source_label=str(fixture) if fixture is not None else None,
    )


def load_fixture(fixture: str | Path) -> dict[str, Any]:
    fixture_path = Path(fixture)
    if not fixture_path.suffix:
        fixture_path = REPO_ROOT / "tests" / "cm_terrain_extractor" / "osm_extraction" / "fixtures" / f"{fixture_path}.geojson"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _bbox_from_fixture(fixture_data: Mapping[str, Any]) -> Any:
    geometries = [
        shape(feature["geometry"])
        for feature in fixture_data.get("features", [])
        if isinstance(feature, Mapping) and feature.get("geometry") is not None
    ]
    if not geometries:
        raise ValueError("Fixture contains no geometries")
    minx, miny, maxx, maxy = union_all(geometries).bounds
    pad = max(maxx - minx, maxy - miny, 0.0002) * 0.35
    polygon = Polygon(
        [
            (minx - pad, miny - pad),
            (maxx + pad, miny - pad),
            (maxx + pad, maxy + pad),
            (minx - pad, maxy + pad),
        ]
    )
    from terrain_extraction.bbox_utils import BoundingBox

    return BoundingBox(polygon, crs=CRS.from_epsg(4326))


def _raw_osm_layer(raw_osm: Mapping[str, Any]) -> gpd.GeoDataFrame:
    rows = []
    geometries = []
    for index, feature in enumerate(raw_osm.get("features", ())):
        if not isinstance(feature, Mapping) or feature.get("geometry") is None:
            continue
        geometry = shape(feature["geometry"])
        if geometry.is_empty:
            continue
        properties = dict(feature.get("properties", {}) or {})
        rows.append(
            {
                "source_index": index,
                "feature_id": properties.get("id", index),
                "properties": json_safe(properties),
            }
        )
        geometries.append(geometry)
    return gpd.GeoDataFrame(rows, geometry=geometries, crs="EPSG:4326")


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records"))


def _safe_layer(layer: Any) -> gpd.GeoDataFrame:
    if layer is None or getattr(layer, "empty", True):
        return gpd.GeoDataFrame(geometry=[], crs=getattr(layer, "crs", None))
    geometry = getattr(layer, "geometry", None)
    if geometry is None:
        return gpd.GeoDataFrame(layer)
    mask = ~geometry.is_empty & ~geometry.isna()
    if not mask.any():
        return gpd.GeoDataFrame(geometry=[], crs=getattr(layer, "crs", None))
    return layer.loc[mask].copy()


def _filter_layer(
    layer: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float] | None,
) -> gpd.GeoDataFrame:
    if bbox is None:
        return layer
    viewport = box(*bbox)
    mask = layer.geometry.intersects(viewport)
    return layer.loc[mask].copy()


def _feature_from_row(row: pd.Series) -> dict[str, Any]:
    properties = {
        str(key): json_safe(value)
        for key, value in row.items()
        if key != "geometry" and key != "_debug_id"
    }
    return {
        "type": "Feature",
        "id": row["_debug_id"],
        "properties": properties,
        "geometry": json.loads(gpd.GeoSeries([row.geometry], crs="EPSG:4326").to_json())["features"][0]["geometry"],
    }


def _elements_from_layer(layer_name: str, layer: gpd.GeoDataFrame) -> Iterable[DebugElement]:
    for _, row in layer.iterrows():
        yield DebugElement(
            stable_id=str(row["_debug_id"]),
            layer=layer_name,
            properties={
                str(key): json_safe(value)
                for key, value in row.items()
                if key != "geometry" and key != "_debug_id"
            },
            geometry=row.geometry,
        )


def _empty_feature_collection(layer_name: str) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "layer": layer_name,
        "features": [],
        "feature_count": 0,
        "returned_count": 0,
        "limited": False,
    }


def _layer_bounds(layer: gpd.GeoDataFrame) -> list[float] | None:
    if layer.empty:
        return None
    bounds = layer.total_bounds
    if pd.isna(bounds).any():
        return None
    return [float(value) for value in bounds]


def _layer_group(layer_name: str) -> str:
    if layer_name == "original_osm":
        return "original"
    if layer_name == "source_features":
        return "matched"
    if layer_name == "final_rows":
        return "final_output"
    return "diagnostics"


def _layer_label(layer_name: str) -> str:
    return layer_name.replace("_", " ").title()


def _style_for_group(group: str) -> Mapping[str, Any]:
    if group == "original":
        return LAYER_STYLES["original_osm"]
    if group == "matched":
        return LAYER_STYLES["source_features"]
    if group == "final_output":
        return LAYER_STYLES["final_rows"]
    return {"color": "#334155", "weight": 1, "fillColor": "#94a3b8", "fillOpacity": 0.2}


def _geometry_summary(geometry: BaseGeometry) -> dict[str, Any]:
    return {
        "type": geometry.geom_type,
        "bounds": [round(float(value), 7) for value in geometry.bounds],
        "area": round(float(getattr(geometry, "area", 0.0)), 6),
        "length": round(float(getattr(geometry, "length", 0.0)), 6),
    }


def _related_summary(element: DebugElement) -> dict[str, Any]:
    return {
        "stable_id": element.stable_id,
        "layer": element.layer,
        "properties": dict(json_safe(element.properties)),
    }


def _first_present(values: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = values.get(key)
        if value is not None and value != "":
            return value
    return None


def _cell_key(values: Mapping[str, Any]) -> tuple[float, float] | None:
    x_value = _first_present(values, ("xidx", "x", "_cell_xidx"))
    y_value = _first_present(values, ("yidx", "y", "_cell_yidx"))
    if x_value is None or y_value is None:
        return None
    try:
        return (float(x_value), float(y_value))
    except (TypeError, ValueError):
        return None


_CURRENT_SESSION: DebugSession | None = None
_ROUTES_REGISTERED = False


def set_current_session(session: DebugSession) -> None:
    global _CURRENT_SESSION
    _CURRENT_SESSION = session


def current_session() -> DebugSession:
    if _CURRENT_SESSION is None:
        raise RuntimeError("No OSM debug session is loaded")
    return _CURRENT_SESSION


def register_routes() -> None:
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    from nicegui import app

    @app.get("/api/osm-debug/layers")
    def api_layers() -> list[dict[str, Any]]:
        return current_session().layer_metadata()

    @app.get("/api/osm-debug/layers/{layer_name}")
    def api_layer(
        layer_name: str,
        bbox: str | None = None,
        limit: int = 5000,
    ) -> dict[str, Any]:
        return current_session().feature_collection(layer_name, bbox=_parse_bbox(bbox), limit=limit)

    @app.get("/api/osm-debug/selection/{stable_id:path}")
    def api_selection(stable_id: str) -> dict[str, Any]:
        return current_session().selection_details(stable_id)

    @app.get("/api/osm-debug/table/{layer_name}")
    def api_table(layer_name: str = "final_rows", limit: int = 1000) -> list[dict[str, Any]]:
        return current_session().table_rows(layer_name, limit=limit)

    _ROUTES_REGISTERED = True


def _parse_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must have four comma-separated numbers")
    return (parts[0], parts[1], parts[2], parts[3])


def build_page(  # noqa: C901, PLR0915 - NiceGUI page assembly is a linear UI declaration.
    *,
    initial_fixture: str | Path | None = None,
    profile: str = "cold_war",
    config: str | Path = "default_osm_config.json",
    seed: int | None = 123,
) -> None:
    from nicegui import run as nicegui_run
    from nicegui import ui

    register_routes()
    state = {
        "fixture": str(initial_fixture or ""),
        "profile": profile,
        "config": str(config),
        "seed": seed,
    }
    drawn_layers: dict[str, Any] = {}

    ui.add_head_html(
        """
        <style>
        .osm-debug-shell { height: 100vh; overflow: hidden; }
        .osm-debug-map { height: calc(100vh - 230px); min-height: 430px; }
        .debug-json { max-height: 42vh; overflow: auto; }
        </style>
        """
    )

    async def load_from_state() -> None:
        status.set_text("Processing OSM data...")
        try:
            session = await nicegui_run.io_bound(
                build_debug_session,
                fixture=state["fixture"],
                profile=state["profile"],
                config=state["config"],
                seed=state["seed"],
            )
            set_current_session(session)
            status.set_text(
                f"Loaded {Path(state['fixture']).name}: {len(session.layers)} layers, {session.output_row_count} CSV rows"
            )
            await refresh_ui()
        except Exception as exc:  # pragma: no cover - exercised through browser smoke
            status.set_text(f"{type(exc).__name__}: {exc}")

    async def load_upload(event: Any) -> None:
        payload = json.loads((await event.file.text()).lstrip("\ufeff"))
        session = await nicegui_run.io_bound(
            build_debug_session,
            osm_data=payload,
            profile=state["profile"],
            config=state["config"],
            seed=state["seed"],
        )
        set_current_session(session)
        state["fixture"] = event.file.name
        fixture_input.value = event.file.name
        status.set_text(f"Loaded upload {event.file.name}: {len(session.layers)} layers")
        await refresh_ui()

    async def refresh_ui() -> None:
            metadata = current_session().layer_metadata()
            table.options["rowData"] = current_session().table_rows("final_rows", limit=1000)
            table.update()
            layer_column.clear()
            with layer_column:
                for layer in metadata:
                    ui.checkbox(
                        f"{layer['label']} ({layer['feature_count']})",
                        value=bool(layer["default_visible"]),
                        on_change=lambda event, layer=layer: toggle_layer(layer, bool(event.value)),
                    ).props(f'data-testid="layer-toggle-{layer["name"]}"')
            await map_element.initialized()
            for leaflet_layer in tuple(drawn_layers.values()):
                map_element.remove_layer(leaflet_layer)
            drawn_layers.clear()
            await _draw_default_layers()

    async def _draw_default_layers() -> None:
        for layer in current_session().layer_metadata():
            if not layer["default_visible"]:
                continue
            draw_layer(layer)
        bounds = _combined_bounds(current_session().layer_metadata())
        if bounds is not None:
            map_element.run_map_method("fitBounds", [[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    async def refresh_existing_session() -> None:
        session = current_session()
        source = session.source_label or state["fixture"] or "loaded data"
        status.set_text(f"Loaded {Path(source).name}: {len(session.layers)} layers, {session.output_row_count} CSV rows")
        await refresh_ui()

    def draw_layer(layer: Mapping[str, Any]) -> None:
        if layer["name"] in drawn_layers:
            return
        collection = current_session().feature_collection(str(layer["name"]), limit=2000)
        if not collection["features"]:
            return
        leaflet_layer = map_element.generic_layer(
            name="geoJSON",
            args=[
                collection,
                {"style": layer["style"]},
            ],
        )
        leaflet_layer.run_method(
            ":eachLayer",
            """
            function(featureLayer) {
                const props = featureLayer.feature.properties || {};
                featureLayer.bindPopup(`<b>${featureLayer.feature.id}</b><pre>${JSON.stringify(props, null, 2)}</pre>`);
            }
            """,
        )
        drawn_layers[str(layer["name"])] = leaflet_layer

    def toggle_layer(layer: Mapping[str, Any], visible: bool) -> None:
        layer_name = str(layer["name"])
        if visible:
            draw_layer(layer)
            return
        leaflet_layer = drawn_layers.pop(layer_name, None)
        if leaflet_layer is not None:
            map_element.remove_layer(leaflet_layer)

    async def show_selected_row() -> None:
        selected = await table.get_selected_row()
        if not selected:
            rows = table.options.get("rowData", [])
            selected = rows[0] if rows else None
        if not selected:
            return
        details = current_session().selection_details(selected["stable_id"])
        details_json.set_content(json.dumps(details, indent=2, sort_keys=True))
        details_drawer.value = True

    with ui.column().classes("osm-debug-shell w-full gap-2 p-3"):
        with ui.row().classes("items-end gap-3 w-full"):
            fixture_input = ui.input("Fixture or GeoJSON path", value=state["fixture"]).classes("w-96")
            profile_input = ui.input("Profile", value=state["profile"]).classes("w-40")
            config_input = ui.input("Config", value=state["config"]).classes("w-72")
            seed_input = ui.number("Seed", value=state["seed"]).classes("w-28")

            async def apply_inputs() -> None:
                state["fixture"] = str(fixture_input.value or "")
                state["profile"] = str(profile_input.value or "cold_war")
                state["config"] = str(config_input.value or "default_osm_config.json")
                state["seed"] = None if seed_input.value in ("", None) else int(seed_input.value)
                await load_from_state()

            ui.button("Run", icon="play_arrow", on_click=apply_inputs).props('data-testid="run-osm-debug"')
            ui.upload(label="Upload GeoJSON", auto_upload=True, on_upload=load_upload).props("accept=.geojson,.json")
        status = ui.label("Ready").props('data-testid="debug-status"')
        with ui.row().classes("w-full gap-3"):
            with ui.column().classes("w-80"):
                ui.label("Layers").classes("text-lg font-semibold")
                layer_column = ui.column().classes("gap-1")
            with ui.column().classes("grow"):
                map_element = ui.leaflet(center=(0, 0), zoom=13, options={"preferCanvas": True}).classes(
                    "osm-debug-map w-full"
                )
                table = ui.aggrid(
                    {
                        "columnDefs": [
                            {"field": "stable_id"},
                            {"field": "name"},
                            {"field": "x"},
                            {"field": "y"},
                            {"field": "menu"},
                            {"field": "cat1"},
                            {"field": "cat2"},
                            {"field": "direction"},
                            {"field": "priority"},
                        ],
                        "rowData": [],
                        "rowSelection": {"mode": "singleRow", "checkboxes": True},
                    },
                    auto_size_columns=True,
                ).classes("h-48 w-full")
                ui.button("Show Selection", icon="info", on_click=show_selected_row).props(
                    'data-testid="show-selection"'
                )
    with ui.drawer(side="right", value=False).classes("w-[34rem]") as details_drawer:
        ui.label("Selection Details").classes("text-lg font-semibold")
        details_json = ui.code("{}", language="json").classes("debug-json w-full")

    if _CURRENT_SESSION is not None:
        ui.timer(0.1, refresh_existing_session, once=True)
    elif initial_fixture:
        ui.timer(0.1, load_from_state, once=True)


def _combined_bounds(metadata: Iterable[Mapping[str, Any]]) -> list[float] | None:
    bounds_values = [item["bounds"] for item in metadata if item.get("default_visible") and item.get("bounds")]
    if not bounds_values:
        return None
    return [
        min(bounds[0] for bounds in bounds_values),
        min(bounds[1] for bounds in bounds_values),
        max(bounds[2] for bounds in bounds_values),
        max(bounds[3] for bounds in bounds_values),
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone NiceGUI debugger for CMAutoEditor OSM extraction")
    parser.add_argument("--fixture", default=None, help="GeoJSON fixture/path to load on startup")
    parser.add_argument("--profile", default="cold_war")
    parser.add_argument("--config", default="default_osm_config.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    from nicegui import ui

    _patch_nicegui_process_pool_permission_fallback()
    register_routes()
    if args.fixture:
        set_current_session(
            build_debug_session(
                fixture=args.fixture,
                profile=args.profile,
                config=args.config,
                seed=args.seed,
            )
        )

    @ui.page("/")
    def index(
        fixture: str | None = None,
        profile: str | None = None,
        config: str | None = None,
        seed: int | None = None,
    ) -> None:
        build_page(
            initial_fixture=fixture or args.fixture,
            profile=profile or args.profile,
            config=config or args.config,
            seed=args.seed if seed is None else seed,
        )

    ui.run(host=args.host, port=args.port, reload=False, show=False)


def _patch_nicegui_process_pool_permission_fallback() -> None:
    from nicegui import run as nicegui_run

    original_setup = nicegui_run.setup

    def setup_without_required_process_pool() -> None:
        try:
            original_setup()
        except PermissionError as exc:
            nicegui_run.process_pool = None
            logging.warning("NiceGUI ProcessPoolExecutor disabled: %s", exc)

    nicegui_run.setup = setup_without_required_process_pool


if __name__ in {"__main__", "__mp_main__"}:
    main()
