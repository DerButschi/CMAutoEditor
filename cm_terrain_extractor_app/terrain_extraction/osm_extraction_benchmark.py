from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from pyproj.crs import CRS
from shapely import union_all
from shapely.geometry import Polygon, shape

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "cm_terrain_extractor" / "osm_extraction" / "fixtures"

LINEAR_PROCESSES = {"road_tiles", "rail_tiles", "stream_tiles", "fence_tiles"}


@dataclass(frozen=True)
class FixtureRunResult:
    stats: dict[str, Any]
    output_rows: list[dict[str, Any]]
    post_process_rows: list[dict[str, Any]]


class _NullProgress:
    def progress(self, *args: object, **kwargs: object) -> _NullProgress:
        return self


class _FeatureCollection:
    def __init__(self, fixture: dict[str, Any]) -> None:
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


@contextmanager
def _legacy_seed(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return

    original_default_rng = np.random.default_rng
    seed_sequence = np.random.SeedSequence(seed)

    def seeded_default_rng(seed_arg: object = None) -> np.random.Generator:
        if seed_arg is not None:
            return original_default_rng(seed_arg)
        return original_default_rng(seed_sequence.spawn(1)[0])

    np.random.seed(seed)
    np.random.default_rng = seeded_default_rng
    try:
        yield
    finally:
        np.random.default_rng = original_default_rng


def resolve_fixture_path(fixture: str | Path) -> Path:
    fixture_path = Path(fixture)
    if fixture_path.suffix:
        return fixture_path
    return FIXTURE_DIR / f"{fixture_path}.geojson"


def load_fixture(fixture: str | Path) -> dict[str, Any]:
    fixture_path = resolve_fixture_path(fixture)
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def run_fixture(
    fixture: str | Path,
    *,
    profile: str,
    config: str | Path,
    seed: int | None = None,
) -> FixtureRunResult:
    fixture_path = resolve_fixture_path(fixture)
    fixture_data = load_fixture(fixture_path)

    from terrain_extraction import osm_processor as osm_processor_module
    from terrain_extraction.osm_processor import OSMProcessor

    osm_processor_module.st.progress = lambda *args, **kwargs: _NullProgress()

    bbox = _bbox_from_fixture(fixture_data)
    processor = OSMProcessor(profile=profile, bbox=bbox, path_to_config=str(config))
    if seed is not None:
        processor.pipeline.context.seed = seed
        processor.pipeline.context.rng = np.random.default_rng(seed)

    timings: dict[str, float | None] = {
        "feature_matching": None,
        "projection": None,
        "grid_init": None,
        "area_rasterization": None,
        "network_noding": None,
        "network_routing": None,
        "tile_assignment": None,
        "building_fitting": None,
        "output_assembly": None,
        "debug_export": None,
    }
    total_start = time.perf_counter()

    with _legacy_seed(seed):
        preprocess_start = time.perf_counter()
        processor.preprocess_osm_data(_FeatureCollection(fixture_data))
        timings["preprocess_osm_data"] = time.perf_counter() - preprocess_start

        processors_start = time.perf_counter()
        processor.run_processors()
        timings["run_processors"] = time.perf_counter() - processors_start

        post_process_start = time.perf_counter()
        processor.post_process()
        timings["post_process"] = time.perf_counter() - post_process_start

        output_start = time.perf_counter()
        output_df = processor.get_output()
        timings["output_assembly"] = time.perf_counter() - output_start

    timings["total"] = time.perf_counter() - total_start
    post_process_df = processor.df if processor.df is not None else pd.DataFrame()
    stats = _build_stats(
        fixture_path=fixture_path,
        fixture_data=fixture_data,
        processor=processor,
        output_df=output_df,
        post_process_df=post_process_df,
        timings=timings,
        seed=seed,
    )

    return FixtureRunResult(
        stats=stats,
        output_rows=_records(output_df),
        post_process_rows=_records(post_process_df),
    )


def _bbox_from_fixture(fixture_data: dict[str, Any]) -> Any:
    geometries = [
        shape(feature["geometry"])
        for feature in fixture_data.get("features", [])
        if feature.get("geometry") is not None
    ]
    if len(geometries) == 0:
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


def _build_stats(
    *,
    fixture_path: Path,
    fixture_data: dict[str, Any],
    processor: Any,
    output_df: pd.DataFrame,
    post_process_df: pd.DataFrame,
    timings: dict[str, float | None],
    seed: int | None,
) -> dict[str, Any]:
    counts = {
        "fixture_features": len(fixture_data.get("features", [])),
        "matched_elements": len(getattr(processor, "matched_elements", [])),
        "grid_cells": 0 if processor.gdf is None else int(len(processor.gdf)),
        "rows_after_post_process": int(len(post_process_df)),
        "output_rows": int(len(output_df)),
        "rows_by_name": _value_counts(post_process_df, "name"),
    }
    quality = {
        "rows_outside_map": _count_rows_outside_map(processor, output_df),
        "duplicate_cells_after_post_process": _count_duplicate_cells(post_process_df),
        "linear_connected_components": _count_linear_components(processor.config, post_process_df),
        "building_linear_collisions": _count_building_linear_collisions(processor.config, post_process_df),
        "source_building_linear_intersections": _count_source_building_linear_intersections(fixture_data),
    }
    quality["reported_collision_cells"] = max(
        quality["building_linear_collisions"],
        quality["source_building_linear_intersections"],
    )

    return {
        "fixture": str(fixture_path),
        "profile": processor.profile,
        "config": str(getattr(processor, "path_to_congih", "")),
        "seed": seed,
        "timings": timings,
        "counts": counts,
        "quality": quality,
        "diagnostics": {
            "legacy_stage_timings_are_approximate": True,
            "seeded_legacy_default_rng_in_benchmark_only": seed is not None,
        },
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if len(df) == 0:
        return []
    return json.loads(df.to_json(orient="records"))


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if len(df) == 0 or column not in df:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts().sort_index().items()}


def _count_rows_outside_map(processor: Any, output_df: pd.DataFrame) -> int:
    if len(output_df) == 0:
        return 0
    max_x = int(processor.idx_bbox[2] - processor.idx_bbox[0])
    max_y = int(processor.idx_bbox[3] - processor.idx_bbox[1])
    outside = ~(
        output_df["x"].between(0, max_x)
        & output_df["y"].between(0, max_y)
    )
    return int(outside.sum())


def _count_duplicate_cells(df: pd.DataFrame) -> int:
    if len(df) == 0 or not {"xidx", "yidx", "menu"}.issubset(df.columns):
        return 0
    layer_df = df.loc[:, ["xidx", "yidx", "menu"]].copy()
    layer_df["_baseline_layer"] = layer_df["menu"].map(_coarse_layer)
    return int(layer_df.duplicated(subset=["xidx", "yidx", "_baseline_layer"], keep=False).sum())


def _coarse_layer(menu: object) -> str:
    menu_text = str(menu)
    if menu_text.startswith("Ground"):
        return "ground"
    if menu_text in {"Foliage", "Brush"}:
        return "foliage"
    if menu_text in {"Roads", "Walls/Fences"}:
        return "linear"
    if "Building" in menu_text:
        return "building"
    if menu_text.startswith("Flavor Objects"):
        return "point_object"
    return menu_text


def _count_linear_components(config: dict[str, Any], df: pd.DataFrame) -> int:
    cells = _cells_for_processes(config, df, LINEAR_PROCESSES)
    if len(cells) == 0:
        return 0

    remaining = set(cells)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            cell = stack.pop()
            for neighbor in _neighbors(cell):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _count_building_linear_collisions(config: dict[str, Any], df: pd.DataFrame) -> int:
    linear_cells = _cells_for_processes(config, df, LINEAR_PROCESSES)
    building_cells = _cells_for_buildings(config, df)
    return len(linear_cells & building_cells)


def _count_source_building_linear_intersections(fixture_data: dict[str, Any]) -> int:
    buildings = []
    linear_features = []
    for feature in fixture_data.get("features", []):
        properties = feature.get("properties") or {}
        tags = properties.get("tags", properties)
        geometry = shape(feature["geometry"])
        if "building" in tags:
            buildings.append(geometry)
        elif any(key in tags for key in ("highway", "railway", "waterway", "barrier")):
            linear_features.append(geometry)

    return sum(
        1
        for building in buildings
        for linear_feature in linear_features
        if building.intersects(linear_feature)
    )


def _cells_for_processes(config: dict[str, Any], df: pd.DataFrame, processes: set[str]) -> set[tuple[int, int]]:
    if len(df) == 0 or not {"xidx", "yidx", "name"}.issubset(df.columns):
        return set()
    names = {
        name
        for name, entry in config.items()
        if any(process in processes for process in entry.get("process", []))
    }
    rows = df[df["name"].isin(names)]
    return _cell_set(rows)


def _cells_for_buildings(config: dict[str, Any], df: pd.DataFrame) -> set[tuple[int, int]]:
    if len(df) == 0 or not {"xidx", "yidx", "name"}.issubset(df.columns):
        return set()
    names = {
        name
        for name, entry in config.items()
        if any(str(process).endswith("outline") for process in entry.get("process", []))
    }
    rows = df[df["name"].isin(names)]
    return _cell_set(rows)


def _cell_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    cells = set()
    for xidx, yidx in df[["xidx", "yidx"]].itertuples(index=False, name=None):
        cells.add((int(xidx), int(yidx)))
    return cells


def _neighbors(cell: tuple[int, int]) -> Iterator[tuple[int, int]]:
    xidx, yidx = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield xidx + dx, yidx + dy


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline OSM extraction baseline fixtures.")
    parser.add_argument("--fixture", required=True, help="Fixture name or path to a GeoJSON FeatureCollection.")
    parser.add_argument("--profile", default="cold_war", help="Combat Mission profile name.")
    parser.add_argument("--config", default="default_osm_config.json", help="Path to the OSM config JSON.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic benchmark replay.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional path for metrics JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_fixture(args.fixture, profile=args.profile, config=args.config, seed=args.seed)
    metrics_json = json.dumps(result.stats, indent=2, sort_keys=True)

    if args.json_output is not None:
        args.json_output.write_text(metrics_json + "\n", encoding="utf-8")
    else:
        print(metrics_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
