from __future__ import annotations

import sys
from pathlib import Path


def _prepare_direct_streamlit_run_imports() -> None:
    app_root = Path(__file__).resolve().parent
    package_parent = str(app_root.parent)
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)


_prepare_direct_streamlit_run_imports()

from cm_terrain_extractor_app.app_core.resources import (  # noqa: E402
    prepare_import_environment,
    prepare_runtime_environment,
    resolve_resources,
)
from cm_terrain_extractor_app.streamlit_ui.app import render_app  # noqa: E402


def main() -> None:
    resources = resolve_resources()
    prepare_runtime_environment(resources)
    prepare_import_environment(resources)
    render_app(resources)


if __name__ == "__main__":
    main()
