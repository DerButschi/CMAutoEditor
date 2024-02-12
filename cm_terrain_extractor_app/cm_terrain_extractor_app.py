import os
import streamlit.web.bootstrap as bootstrap
# It was throwing an error when I was running exe file below import was for that only
import streamlit.runtime.scriptrunner.magic_funcs

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    current_location = os.getcwd()
    # for dirpath, dirnames, filenames in os.walk(current_location):
    #     print(dirpath, filenames)

    path = [os.path.join(current_location, 'dll')]
    if 'PATH' in os.environ:
        for p in os.environ['PATH'].split(os.pathsep):
            if p == '':
                path.append(current_location)
            else:
                path.append(p)
    path_str = os.pathsep.join(path)
    os.environ['PATH'] = path_str
    print('PATH variable: {}'.format(os.environ['PATH']))

    flag_options = {
        "server.port": 8501,
        "global.developmentMode": False,
    }

    bootstrap.load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True
    bootstrap.run(
        "./cm_terrain_extractor_app/cmterrainextractor.py",
        False,
        # "streamlit run",
        [],
        flag_options
        )
    