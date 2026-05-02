import os

import streamlit.runtime.scriptrunner.magic_funcs  # noqa: F401
import streamlit.web.bootstrap as bootstrap

from cm_terrain_extractor_app.app_core.resources import (
    prepare_runtime_environment,
    resolve_resources,
)

if __name__ == "__main__":
    resources = resolve_resources()
    prepare_runtime_environment(resources)
    os.chdir(resources.executable_root)
    # for dirpath, dirnames, filenames in os.walk(current_location):
    #     print(dirpath, filenames)

    flag_options = {
        "server.port": 8501,
        "global.developmentMode": False,
    }

    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    bootstrap.run(
        str(resources.app_root / "cmterrainextractor.py"),
        False,
        # "streamlit run",
        [],
        flag_options
        )
