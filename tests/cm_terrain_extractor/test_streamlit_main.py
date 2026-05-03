from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_direct_streamlit_run_bootstrap_prioritizes_package_parent(monkeypatch) -> None:
    app_root = Path("cm_terrain_extractor_app").resolve()
    script_path = app_root / "streamlit_main.py"
    bootstrap_source = script_path.read_text(encoding="utf-8").split(
        "from cm_terrain_extractor_app.app_core.resources",
        maxsplit=1,
    )[0]
    monkeypatch.delitem(sys.modules, "cm_terrain_extractor_app", raising=False)
    monkeypatch.setattr(
        sys,
        "path",
        [
            str(app_root),
            *[
                path
                for path in sys.path
                if path and Path(path).resolve() not in {app_root, app_root.parent}
            ],
        ],
    )

    exec(
        compile(bootstrap_source, str(script_path), "exec"),
        {"__file__": str(script_path), "__name__": "streamlit_main_bootstrap_test"},
    )

    spec = importlib.util.find_spec("cm_terrain_extractor_app")

    assert Path(sys.path[0]).resolve() == app_root.parent
    assert spec is not None
    assert spec.submodule_search_locations is not None
