from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

APP_DIR = Path(__file__).parents[3] / "cm_terrain_extractor_app"
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))


def test_pipeline_records_progress_and_wraps_legacy_processor() -> None:
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext, ExtractionPipeline

    events: list[tuple[str, float, str | None]] = []
    context = ExtractionContext.create(
        profile="cold_war",
        bbox=object(),
        config_path="default_osm_config.json",
        seed=123,
        progress=lambda stage, value, message=None: events.append((stage, value, message)),
    )
    processor = _LegacyProcessor()

    result = ExtractionPipeline(context).run_legacy(processor, osm_data={"features": []})

    assert processor.calls == ["preprocess_osm_data", "run_processors", "post_process", "get_output"]
    assert result.output_rows == ({"x": 1, "y": 2, "name": "forest"},)
    assert result.stats.counts["output_rows"] == 1
    assert [event[0] for event in events] == [
        "preprocess_osm_data",
        "run_processors",
        "post_process",
        "output_assembly",
    ]
    assert events[-1] == ("output_assembly", 1.0, "Legacy output rows assembled")


def test_context_initializes_deterministic_rng_and_noop_progress() -> None:
    from terrain_extraction.osm_extraction.pipeline import ExtractionContext

    first = ExtractionContext.create(profile="cold_war", bbox=object(), config_path="cfg.json", seed=99)
    second = ExtractionContext.create(profile="cold_war", bbox=object(), config_path="cfg.json", seed=99)

    assert first.rng.integers(0, 10_000, size=5).tolist() == second.rng.integers(
        0,
        10_000,
        size=5,
    ).tolist()
    assert first.progress("stage", 0.5, None) is None


def test_osm_processor_imports_without_streamlit_and_holds_pipeline(monkeypatch) -> None:
    config_path = "default_osm_config.json"

    for module_name in ["terrain_extraction.osm_processor", "streamlit"]:
        sys.modules.pop(module_name, None)

    original_import = builtins.__import__

    def import_without_streamlit(name: str, *args: object, **kwargs: object) -> object:
        if name == "streamlit":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_streamlit)

    module = importlib.import_module("terrain_extraction.osm_processor")
    processor = module.OSMProcessor("cold_war", SimpleNamespace(), config_path)

    assert processor.pipeline.context.profile == "cold_war"
    assert processor.pipeline.context.config_path == config_path
    assert hasattr(module.st, "progress")


class _LegacyProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def preprocess_osm_data(self, osm_data: object) -> None:
        self.calls.append("preprocess_osm_data")

    def run_processors(self) -> None:
        self.calls.append("run_processors")

    def post_process(self) -> None:
        self.calls.append("post_process")

    def get_output(self) -> pd.DataFrame:
        self.calls.append("get_output")
        return pd.DataFrame([{"x": 1, "y": 2, "name": "forest"}])
