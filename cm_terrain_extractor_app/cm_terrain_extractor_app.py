import os

import streamlit.runtime.scriptrunner.magic_funcs  # noqa: F401
import streamlit.web.bootstrap as bootstrap

from cm_terrain_extractor_app.app_core.resources import (
    prepare_import_environment,
    prepare_runtime_environment,
    resolve_resources,
    streamlit_entrypoint_path,
)

if __name__ == "__main__":
    resources = resolve_resources()
    prepare_runtime_environment(resources)
    prepare_import_environment(resources)
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
        str(streamlit_entrypoint_path(resources)),
        False,
        # "streamlit run",
        [],
        flag_options,
    )
