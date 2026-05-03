# CM Terrain Extractor Packaging

This page records the source and packaged execution contract for the Streamlit
CM Terrain Extractor after the UI split.

## Entry Points

Source mode uses the split Streamlit entrypoint:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\streamlit_main.py
```

The compatibility wrapper remains available:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py
```

Packaged mode keeps `cm_terrain_extractor_app\cm_terrain_extractor_app.py` as
the executable launcher. The launcher resolves resources, prepares DLL and
import paths, changes the working directory to the executable directory, and
then bootstraps `streamlit_main.py`.

## Resource Paths

In source mode, `resolve_resources()` uses:

| Resource | Path |
| --- | --- |
| App root | `cm_terrain_extractor_app/` from the source tree |
| Executable root | Current working directory |
| Config directory | Current working directory |
| Data cache | `<current working directory>\data_cache` |
| DLL directory | `<app root>\dll` |

In packaged mode, `resolve_resources()` uses:

| Resource | Path |
| --- | --- |
| App root | `<PyInstaller _MEIPASS>\cm_terrain_extractor_app` |
| Executable root | Directory containing `cm_terrain_extractor_app.exe` |
| Config directory | Directory containing `cm_terrain_extractor_app.exe` |
| Data cache | `<executable root>\data_cache` |
| DLL directory | `<PyInstaller _MEIPASS>\dll`, falling back to `<executable root>\dll` |

`prepare_runtime_environment(resources)` prepends the resolved DLL directory to
`PATH` once and registers it with `os.add_dll_directory()` on Windows when the
directory exists. `prepare_import_environment(resources)` prepends both the
package parent and app root to `sys.path`; this keeps
`cm_terrain_extractor_app.*` imports and legacy `terrain_extraction.*` imports
available in packaged and source execution.

## Configs, Cache, and Profiles

OSM config discovery reads `*.json` files from `resources.config_dir`. In source
mode this is the launch working directory, preserving the existing editable JSON
config behavior. In packaged mode this is the executable directory, so editable
OSM configs should live next to the executable.

Elevation cache and generated height-map previews use `resources.data_cache_path`.
The directory is created by backend actions when extraction runs.

Profiles are bundled from the repository `profiles/` directory into the
PyInstaller app under `profiles/`. The spec also collects `profiles` submodules
as hidden imports because OSM processing imports profile modules dynamically.

## PyInstaller Build

Build from the repository root with the approved Conda environment:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m PyInstaller cm_terrain_extractor_app.spec --noconfirm --clean
```

The spec bundles:

- Streamlit static assets.
- `streamlit_folium` frontend assets.
- The full `cm_terrain_extractor_app/` source tree as data.
- The repository `profiles/` directory.
- `gdal.dll` from the approved Conda environment.
- Hidden imports for the split `app_core`, `map_view`, and `streamlit_ui`
  packages.
- Explicit terrain data-source hidden imports, including Lower Saxony.
- Explicit `skimage.measure` and `skimage.transform` hidden imports.
- Excludes for `skimage.io`, `skimage.io._plugins`, and `skimage.viewer`
  to avoid pulling optional image IO/viewer plugin paths into analysis.
- `upx=False` for the executable stage.

## Milestone 9 Verification

On 2026-05-02, the source Streamlit smoke command returned HTTP 200 on port
`18559`.

On 2026-05-02, the PyInstaller command above was attempted in the approved
Conda environment. The sandboxed attempt failed while replacing
`build\cm_terrain_extractor_app\base_library.zip`. The unrestricted retry
advanced through analysis but failed before executable creation inside
PyInstaller's scikit-image hook isolation:

```text
PyInstaller.isolated._parent.SubprocessDiedError:
Child process died calling _is_package() with args=('skimage.io._plugins', ...)
```

A narrower retry after avoiding broad terrain submodule collection reached the
same class of failure. A temporary `skimage.io` exclusion only moved the failure
to the `skimage.filters` hook, so that exclusion was not kept.

Packaged executable launch is therefore unverified in this environment. The
remaining blocker appears to be the local PyInstaller/scikit-image hook
interaction in the Conda environment, not the Streamlit split entrypoint itself.

## Milestone 10 Verification

On 2026-05-02, `hooks/` was inspected and contains only the custom Streamlit
hook; there is no custom scikit-image hook and no broad
`skimage.io._plugins` collection in the app spec. The app spec now keeps
`skimage.measure` and `skimage.transform` explicit, excludes `skimage.io`,
`skimage.io._plugins`, and `skimage.viewer`, and sets `upx=False`.

The requested clean rebuild was attempted:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m PyInstaller cm_terrain_extractor_app.spec --noconfirm --clean
```

The sandboxed run failed during build-directory cleanup with a Windows
permission error. The unrestricted retry advanced past the previous
`skimage.io._plugins` failure point, but still failed during PyInstaller
analysis while the contrib hook for `skimage.filters` collected
`skimage.filters.rank.tests`:

```text
PyInstaller.isolated._parent.SubprocessDiedError:
Child process died calling _collect_submodules()
with args=('skimage.filters.rank.tests', 'warn once')
```

The skimage IO/plugin excludes therefore reduce the original blocker but do not
fully solve packaging in this Conda environment. Packaged executable launch
remains unverified.
