# Streamlit Refactoring Status

This is the living status ledger for the CMTerrainExtractor Streamlit refactor.

Authoritative contract: `docs/plans/streamlit_refactoring_contract.md`.

Execution plan: `docs/plans/streamlit_refactoring_exec_plan.md`.

Future agents and humans must update this document after each milestone. Record progress, decisions, contract-change requests, technical debt, validation commands, results, blockers, and residual risk. If execution would violate the contract, stop and ask the user before continuing.

## Progress by Milestone

| Milestone | Name | Status | Owner | Last updated | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Inventory and behavior snapshot | Done | Codex | 2026-05-02 | Created `docs/plans/streamlit_refactoring_inventory.md`; app code untouched. |
| 2 | Introduce test harness and first regression tests | Done | Codex | 2026-05-02 | Added `tests/cm_terrain_extractor/` with expected failures for missing `app_core` and `map_view` modules. |
| 3 | Extract resource and path handling | Done | Codex | 2026-05-02 | Added `app_core.resources`; app and launcher now share source/packaged path and DLL setup. |
| 4 | Introduce `AppState` and session adapter | Done | Codex | 2026-05-02 | Added `AppState`, session adapter, and moved durable app state behind `STATE_KEY`; temporary widget/map keys recorded as debt. |
| 5 | Extract validation and export helpers | Done | Codex | 2026-05-02 | Added Streamlit-free validation/export helpers; CSV bytes and download filenames remain compatible. |
| 6 | Extract backend actions | Done | Codex | 2026-05-02 | Added `app_core.actions` and offline smoke tests for elevation, config, upload bytes, OSM download wrapper, and OSM processing wrapper. |
| 7 | Extract drawing parser and Folium map/layers | Done | Codex | 2026-05-02 | Added `map_view` drawing parser and Folium map/layer builders; coordinate order is covered by regression tests. |
| 8 | Split Streamlit UI modules | Not started | Unassigned | 2026-05-02 | Streamlit imports must remain in UI/entrypoint modules only. |
| 9 | Stabilize launcher and PyInstaller spec | Not started | Unassigned | 2026-05-02 | Manual packaged-run status must be recorded. |
| 10 | Final integration, docs, and validation | Not started | Unassigned | 2026-05-02 | Must record final tests, ruff, import checks, manual smoke checks, and residual risks. |

## Decisions Made

| Date | Decision | Rationale | Approved by |
| --- | --- | --- | --- |
| 2026-05-02 | Use `docs/plans/streamlit_refactoring_contract.md` as the authoritative contract. | The user requested a contract document whose content must not be changed unless approved. | User |
| 2026-05-02 | Use `docs/plans/streamlit_refactoring_exec_plan.md` as the milestone execution plan. | The user requested an execPlan-style workflow with decision-complete milestones. | User |
| 2026-05-02 | Use `docs/plans/streamlit_refactoring_status.md` as the living status ledger. | The user requested status tracking for progress, decisions, debt, and validation results. | User |
| 2026-05-02 | Treat the existing `test/` directory as fixture data, not an active pytest suite. | The plan states the repository currently contains no test suite; future tests go under `tests/cm_terrain_extractor/`. | User |
| 2026-05-02 | Milestone 1 is documentation-only. | The execution plan explicitly forbids Python code changes in M1; inventory is sufficient to guide M2-M8. | Codex |
| 2026-05-02 | Configure pytest with `pythonpath = ["."]`. | The approved `pytest.exe` entrypoint starts from the Conda `Scripts` directory, so repository packages were not importable without explicitly adding the repo root. | Codex |
| 2026-05-02 | Source-mode resources use the current working directory for JSON configs and `data_cache`. | This preserves the current `cmterrainextractor.py` source behavior where config lookup used `executable_path = "."` and cache writes used `data_cache` relative to the Streamlit working directory. | Codex |
| 2026-05-02 | Packaged resources use `sys.executable` parent for configs/cache and `_MEIPASS` for bundled app/DLL files. | This preserves the launcher/PyInstaller split where user-editable configs and cache live beside the executable while bundled app files are loaded from the PyInstaller extraction root. | Codex |
| 2026-05-02 | Seed `selectable_data_sources` through the `AppState` object in the Streamlit entrypoint. | The current terrain data-source modules import Streamlit indirectly, so importing them from `app_core.state` or `streamlit_ui.session_adapter` would violate the intended boundary earlier than the backend extraction milestone can fix. | Codex |
| 2026-05-02 | Keep elevation and OSM download filenames fixed as `elevation_data.csv` and `osm_data.csv`. | Milestone 5 extracted filename helpers without changing current Streamlit download behavior. | Codex |
| 2026-05-02 | Keep `app_core.actions` free of direct Streamlit/Folium imports and use lazy legacy imports for OSMnx and `OSMProcessor`. | This makes action tests offline and fakeable while preserving the existing terrain/OSM implementation ownership. | Codex |
| 2026-05-02 | Render OSM Shapely geometries inside `map_view.folium_layers` instead of importing `terrain_extraction.visualization_utils`. | The legacy visualization helper imports Streamlit, which would violate the M7 `map_view` boundary. | Codex |

## Contract Changes Requested or Approved

| Date | Request | Status | Resolution |
| --- | --- | --- | --- |
| 2026-05-02 | Initial contract document created. | Approved by request | Contract is authoritative once added. |

No further contract changes have been requested or approved.

## Technical Debt Ledger

| Date | Milestone | Debt | Reason | Paydown target |
| --- | --- | --- | --- | --- |
| 2026-05-02 | Planning docs | No implementation debt recorded. | Docs-only setup. | Not applicable. |
| 2026-05-02 | Milestone 1 | No application debt introduced. Inventory records existing unknowns around `bbox_origin`, `height_map_layer`, cache clearing, invalidation, source-mode paths, and Streamlit-cached OSM IO. | M1 changed only planning docs. | Resolve through M3-M8 tests and extraction work. |
| 2026-05-02 | Milestone 2 | No production-code debt introduced. Pytest and ruff cache writes reported sandbox permission issues; ruff validation used `--no-cache`. | M2 added only tests and pytest import-path configuration. | Recheck cache behavior in unrestricted local runs if warnings become noisy. |
| 2026-05-02 | Milestone 3 | `cmterrainextractor.py` still keeps compatibility globals `resources`, `executable_path`, and `data_cache_path`; existing large UI functions carry ruff complexity suppressions. | M3 centralizes resource handling without performing the planned UI/module split. Ruff validation would otherwise fail on pre-existing monolithic Streamlit complexity. | Pay down during M8 Streamlit UI split. |
| 2026-05-02 | Milestone 4 | Temporary Streamlit session keys remain for `edited_df`, `drawn_coordinates`, `zoom_cache`, `center_cache`, and `height_map_layer`. | These are widget/map rerun coordination keys, not durable app state; removing them cleanly belongs with the M7 map parser and M8 UI split. | Replace with map/UI-local adapters during M7/M8. |
| 2026-05-02 | Milestone 4 | `cmterrainextractor.py` still carries compatibility globals and complexity suppressions. | M4 moved durable state without decomposing the monolithic UI file. | Pay down during M8 Streamlit UI split. |
| 2026-05-02 | Milestone 5 | `cmterrainextractor.py` still renders bbox metrics and download widgets inline. | M5 extracted pure validation/export behavior only; the UI split is explicitly scheduled for M8. | Move rendering code into `streamlit_ui` during M8. |
| 2026-05-02 | Milestone 6 | `terrain_extraction.osm_processor` and `terrain_extraction.osm_utils.io` still import Streamlit internally, so `app_core.actions` uses lazy wrapper imports and the UI still imports `get_bounding_box` for uploaded/downloaded OSM data. | M6 extracted backend action ownership without rewriting terrain algorithms or changing existing OSM processing internals. | Revisit during M8 UI split or a later terrain-internal cleanup if direct framework imports remain a blocker. |
| 2026-05-02 | Milestone 7 | `map_view.drawing` adds the app package directory to `sys.path` before lazily importing the legacy `terrain_extraction.bbox_utils.BoundingBox`. | Existing terrain modules use absolute `terrain_extraction.*` imports, while pytest imports the app as `cm_terrain_extractor_app`; normalizing the terrain package imports is broader than M7. | Revisit during M8/M9 package stabilization or a later terrain-internal import cleanup. |
| 2026-05-02 | Milestone 7 | Temporary Streamlit session keys remain for `height_map_layer`, `zoom_cache`, `center_cache`, `drawn_coordinates`, and `edited_df`. | M7 moved map construction and drawing parsing while intentionally leaving rerun/widget coordination in the Streamlit edge. | Pay down during M8 Streamlit UI split. |

## Test and Validation Results

| Date | Milestone | Command or check | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-02 | Planning docs | `Get-ChildItem docs\plans` | Passed | Confirmed contract, exec plan, status, and source outline exist under `docs/plans/`. |
| 2026-05-02 | Planning docs | `Select-String -Path docs\plans\streamlit_refactoring_*.md -Pattern "authoritative","contract","TDD","pytest","ruff"` | Passed | Confirmed required keywords are present across the generated planning docs. |
| 2026-05-02 | Milestone 1 | `Get-Content docs\plans\streamlit_refactoring_inventory.md \| Select-String -Pattern "session_state","cmterrainextractor.py","Folium","PyInstaller","CSV"` | Passed | Confirmed the inventory includes required behavior categories. |
| 2026-05-02 | Milestone 1 | `Get-Content docs\plans\streamlit_refactoring_status.md \| Select-String -Pattern "Milestone 1","Inventory"` | Passed | Confirmed the status ledger records M1 completion and inventory notes. |
| 2026-05-02 | Milestone 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -v` | Failed as expected | Collected 3 tests. Failures are expected missing target modules: `cm_terrain_extractor_app.map_view` and `cm_terrain_extractor_app.app_core`. Pytest also warned that `.pytest_cache` could not be written in the sandbox. |
| 2026-05-02 | Milestone 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check tests\cm_terrain_extractor\__init__.py tests\cm_terrain_extractor\test_exports.py tests\cm_terrain_extractor\test_state.py tests\cm_terrain_extractor\test_drawing.py` | Failed due environment/cache | Ruff could not initialize `.ruff_cache` because the sandbox denied cache writes. |
| 2026-05-02 | Milestone 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check --no-cache tests\cm_terrain_extractor\__init__.py tests\cm_terrain_extractor\test_exports.py tests\cm_terrain_extractor\test_state.py tests\cm_terrain_extractor\test_drawing.py` | Passed | Retry with cache disabled reported `All checks passed!`. |
| 2026-05-02 | Milestone 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_resources.py -v --basetemp=tmp_pytest_codex` | Failed as expected | After sandbox temp-directory failures, an unrestricted run collected 4 tests and failed with expected `ModuleNotFoundError: No module named 'cm_terrain_extractor_app.app_core'`. |
| 2026-05-02 | Milestone 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_resources.py -v --basetemp=tmp_pytest_codex` | Passed | 4 resource tests passed after implementing `AppResources`, source/packaged path resolution, config discovery, and idempotent DLL path setup. |
| 2026-05-02 | Milestone 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\resources.py cm_terrain_extractor_app\cmterrainextractor.py cm_terrain_extractor_app\cm_terrain_extractor_app.py tests\cm_terrain_extractor\test_resources.py --no-cache` | Passed | Ruff cache writes are still avoided with `--no-cache`; safe mechanical fixes were applied in the touched Streamlit wrapper. |
| 2026-05-02 | Milestone 3 | Temporary source smoke: `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py --server.headless=true --server.port=18501 --browser.gatherUsageStats=false` | Passed | Hidden Streamlit process returned HTTP 200 and was stopped. Full interactive checks for config selection/elevation/OSM remain future manual smoke scope. |
| 2026-05-02 | Milestone 4 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_state.py -v --basetemp=tmp_pytest_codex` | Failed as expected | After extending M4 tests, collected 6 tests and failed with expected `ModuleNotFoundError: No module named 'cm_terrain_extractor_app.app_core.state'`. |
| 2026-05-02 | Milestone 4 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_state.py -v --basetemp=tmp_pytest_codex` | Passed | 6 state/session tests passed after implementing `AppState`, invalidation helpers, and the Streamlit session adapter. Pytest cache warnings persist in the sandbox. |
| 2026-05-02 | Milestone 4 | `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['cm_terrain_extractor_app/cmterrainextractor.py','cm_terrain_extractor_app/app_core/state.py','cm_terrain_extractor_app/streamlit_ui/session_adapter.py']]"` | Passed | Used AST parsing after `py_compile` hit sandbox permission errors writing `__pycache__`. |
| 2026-05-02 | Milestone 4 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\state.py cm_terrain_extractor_app\streamlit_ui\session_adapter.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_state.py --no-cache` | Passed | Ruff cache writes are still avoided with `--no-cache`; `draw_sidebar` keeps a temporary complexity suppression until the M8 UI split. |
| 2026-05-02 | Milestone 4 | Temporary source smoke: `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py --server.headless=true --server.port=18505 --browser.gatherUsageStats=false` | Passed | Hidden Streamlit process returned HTTP 200 and was stopped. Full interactive bbox drawing/metrics was not automated in this milestone. |
| 2026-05-02 | Milestone 5 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_validation.py tests\cm_terrain_extractor\test_exports.py -v --basetemp=tmp_pytest_codex` | Failed as expected | After adding M5 tests, collected 6 tests and failed with expected missing modules: `cm_terrain_extractor_app.app_core.validation` and `cm_terrain_extractor_app.app_core.exports`. |
| 2026-05-02 | Milestone 5 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_validation.py tests\cm_terrain_extractor\test_exports.py -v --basetemp=tmp_pytest_codex` | Passed | 6 validation/export tests passed after implementing bbox limits, bbox metric/state update helpers, CSV bytes, and filename helpers. Pytest cache warnings persist in the sandbox. |
| 2026-05-02 | Milestone 5 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_state.py -v --basetemp=tmp_pytest_codex` | Passed | 6 nearby state/session tests passed, covering the invalidation helpers used by `update_state_from_bbox`. Pytest cache warnings persist in the sandbox. |
| 2026-05-02 | Milestone 5 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\validation.py cm_terrain_extractor_app\app_core\exports.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_validation.py tests\cm_terrain_extractor\test_exports.py --no-cache` | Passed | Ruff cache writes are still avoided with `--no-cache`. |
| 2026-05-02 | Milestone 5 | `Select-String -Path cm_terrain_extractor_app\app_core\validation.py,cm_terrain_extractor_app\app_core\exports.py -Pattern "streamlit","streamlit_folium"` | Passed | No forbidden Streamlit imports matched in the new core modules. |
| 2026-05-02 | Milestone 5 | Temporary source smoke: `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py --server.headless=true --server.port=18515 --browser.gatherUsageStats=false` | Passed | Hidden Streamlit process returned HTTP 200 and was stopped. Interactive bbox drawing and CSV download checks remain manual smoke scope. |
| 2026-05-02 | Milestone 6 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_actions_smoke.py -v --basetemp=tmp_pytest_codex` | Failed as expected | After adding M6 tests, collected 6 tests and failed with expected missing module `cm_terrain_extractor_app.app_core.actions`; sandbox temp cleanup also reported stale permission errors from prior runs. |
| 2026-05-02 | Milestone 6 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_actions_smoke.py -v` | Passed | 6 action smoke tests passed using fakes/fixtures only; no network access required. Pytest cache warning persists in the sandbox. |
| 2026-05-02 | Milestone 6 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_validation.py tests\cm_terrain_extractor\test_exports.py tests\cm_terrain_extractor\test_state.py -q` | Passed | 12 nearby state/export/validation tests passed after rewiring the Streamlit callers. Pytest cache warnings persist in the sandbox. |
| 2026-05-02 | Milestone 6 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\actions.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_actions_smoke.py --no-cache` | Passed | Ruff cache writes are still avoided with `--no-cache`; import sorting was fixed manually after sandbox write permissions blocked `ruff --fix`. |
| 2026-05-02 | Milestone 6 | `Select-String -Path cm_terrain_extractor_app\app_core\actions.py -Pattern "streamlit","streamlit_folium","folium"` | Passed | No direct forbidden framework imports or references matched in `app_core.actions`. |
| 2026-05-02 | Milestone 6 | `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['cm_terrain_extractor_app/app_core/actions.py','cm_terrain_extractor_app/cmterrainextractor.py','tests/cm_terrain_extractor/test_actions_smoke.py']]"` | Passed | AST parse smoke passed without writing `__pycache__`. |
| 2026-05-02 | Milestone 6 | Temporary source smoke: `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py --server.headless=true --server.port=18526 --browser.gatherUsageStats=false` | Passed | Hidden Streamlit process returned HTTP 200 and was stopped. Live OSM download/upload/process interactions were not manually exercised. |
| 2026-05-02 | Milestone 7 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_drawing.py tests\cm_terrain_extractor\test_folium_layers.py -v` | Failed as expected | After adding M7 tests, collected 7 tests and failed with expected missing module `cm_terrain_extractor_app.map_view`. |
| 2026-05-02 | Milestone 7 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_drawing.py tests\cm_terrain_extractor\test_folium_layers.py -v` | Passed | 7 drawing/layer tests passed after adding `map_view.drawing`, `map_view.folium_map`, and `map_view.folium_layers`. Pytest cache warnings persist in the sandbox. |
| 2026-05-02 | Milestone 7 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\map_view\__init__.py cm_terrain_extractor_app\map_view\drawing.py cm_terrain_extractor_app\map_view\folium_map.py cm_terrain_extractor_app\map_view\folium_layers.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_drawing.py tests\cm_terrain_extractor\test_folium_layers.py --no-cache` | Passed | Ruff cache writes are still avoided with `--no-cache`. |
| 2026-05-02 | Milestone 7 | `Select-String -Path cm_terrain_extractor_app\map_view\*.py -Pattern "streamlit","streamlit_folium"` | Passed | No forbidden Streamlit or `streamlit_folium` imports/references matched in `map_view`. |
| 2026-05-02 | Milestone 7 | `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['cm_terrain_extractor_app/map_view/drawing.py','cm_terrain_extractor_app/map_view/folium_map.py','cm_terrain_extractor_app/map_view/folium_layers.py','cm_terrain_extractor_app/cmterrainextractor.py','tests/cm_terrain_extractor/test_drawing.py','tests/cm_terrain_extractor/test_folium_layers.py']]"` | Passed | AST parse smoke passed without writing `__pycache__`. |
| 2026-05-02 | Milestone 7 | Temporary source smoke: `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py --server.headless=true --server.port=18537 --browser.gatherUsageStats=false` | Passed | Hidden Streamlit process returned HTTP 200 and was stopped. Interactive drawing/elevation/OSM visual checks remain manual smoke scope. |
| 2026-05-02 | Milestone 7 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -q` | Failed due environment/temp setup | 25 tests passed; 4 resource tests errored during pytest temp-dir setup at `.tmp/pytest` before executing. |
| 2026-05-02 | Milestone 7 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -q --basetemp=tmp_pytest_codex` | Failed due environment/temp cleanup | Retry still had 25 tests pass; the same 4 resource tests errored during stale `tmp_pytest_codex` cleanup with Windows permission errors. Full app-specific suite remains unverified in this sandbox after the allowed retry. |
| 2026-05-02 | Milestone 7 | Elevated retry: `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -q --basetemp=tmp_pytest_elevated_m7` | Passed | 29 app-specific tests passed outside the sandbox with a fresh top-level basetemp directory; only deprecation warnings were reported. This supersedes the earlier sandbox temp-directory failures. |

## Known Blockers

| Date | Blocker | Impact | Next action |
| --- | --- | --- | --- |
| 2026-05-02 | No active pytest suite exists for this app. | Refactor cannot safely proceed without adding tests first. | M2 creates `tests/cm_terrain_extractor/` and first failing regression tests. |
| 2026-05-02 | PyInstaller behavior must be verified manually on Windows. | Automated tests cannot fully prove packaged execution. | M9 records manual packaging command, executable launch result, and residual risk. |
| 2026-05-02 | OSM download requires network access. | Unit tests must not depend on live OSMnx calls. | Use fakes/fixtures for unit tests; keep live download as manual or marked integration check. |

## Current Working Assumptions

- The current Streamlit workflow must remain recognizable and usable after each major milestone.
- `cm_terrain_extractor_app/cm_terrain_extractor_app.py` remains the packaged launcher.
- `cm_terrain_extractor_app/cmterrainextractor.py` may remain as a compatibility wrapper until module splitting is complete.
- New tests belong under `tests/cm_terrain_extractor/`.
- Existing files under `test/` are fixture data unless a future milestone proves otherwise.
- The approved Conda environment is `C:\Users\der_b\miniconda3\envs\cm_terrain`.
- Use these validation binaries for future Python work:
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe`

## Update Procedure

After each milestone:

- Change the milestone status to `In progress`, `Blocked`, or `Done`.
- Add the owner and exact date.
- Record decisions and contract-change requests.
- Record technical debt introduced intentionally.
- Paste the exact validation commands and concise results.
- Record manual smoke checks separately from automated tests.
- Record unresolved risks before handing off.
