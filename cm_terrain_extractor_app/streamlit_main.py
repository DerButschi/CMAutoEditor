from __future__ import annotations

from cm_terrain_extractor_app.app_core.resources import (
    prepare_import_environment,
    prepare_runtime_environment,
    resolve_resources,
)
from cm_terrain_extractor_app.streamlit_ui.app import render_app


def main() -> None:
    resources = resolve_resources()
    prepare_runtime_environment(resources)
    prepare_import_environment(resources)
    render_app(resources)


if __name__ == "__main__":
    main()
