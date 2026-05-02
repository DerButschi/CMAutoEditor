# Streamlit Refactoring Exec Plan

> **For agentic workers:** This plan is executable milestone by milestone. Use an execution workflow, update `docs/plans/streamlit_refactoring_status.md` after every milestone, and obey `docs/plans/streamlit_refactoring_contract.md` as the authoritative contract.

**Goal:** Refactor the CMTerrainExtractor Streamlit app into explicit UI, state, core action, map rendering, and packaging layers while preserving current behavior.

**Architecture:** Streamlit remains at the edge. `app_core/` holds state, resources, validation, actions, and exports without Streamlit imports. `map_view/` builds Folium maps and parses drawing payloads. `streamlit_ui/` renders the app and coordinates state updates.

**Tech Stack:** Python, Streamlit, `streamlit-folium`, Folium, Pandas, Shapely, OSMnx, PyInstaller, pytest, ruff.

**Authoritative contract:** `docs/plans/streamlit_refactoring_contract.md`. If a milestone conflicts with the contract, stop and ask the user before editing code.

Reference style: This plan follows an execPlan style suitable for controlled Codex execution and plan-execute workflows. See OpenAI Codex task delegation guidance at `https://platform.openai.com/docs/codex` and shell plan-execute guidance at `https://platform.openai.com/docs/guides/tools-shell`.

## Global Rules

- Do not edit application code before writing the failing test for the behavior being moved or introduced.
- All touched Python code must be covered by tests after the milestone.
- Place new tests under `tests/cm_terrain_extractor/`.
- Use the approved Conda tools:
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe`
- Keep the app runnable or document a rollback path after every milestone.
- Do not change the GUI framework, CSV format, or terrain algorithms.
- Update `docs/plans/streamlit_refactoring_status.md` after each milestone with progress, decisions, debt, blockers, and validation output.

## Dependency Graph

```text
M1 inventory
  -> M2 test harness
    -> M3 resources
      -> M4 AppState/session adapter
        -> M5 validation/exports
          -> M6 backend actions
            -> M7 drawing/map view
              -> M8 Streamlit UI split
                -> M9 packaging stabilization
                  -> M10 final integration
```

M2 may begin after M1 creates enough inventory to identify import paths. M3 through M8 must remain sequential because each milestone depends on contracts stabilized by the prior one. M9 must wait until module movement is complete. M10 must wait until all earlier validations are recorded.

## Do Not Proceed Conditions

- The contract would need to change and the user has not approved it.
- A test cannot be written for touched non-UI code.
- The app cannot remain runnable and no rollback path is available.
- A moved module requires a forbidden import, especially `streamlit` inside `app_core/`.
- A packaging change cannot be verified or documented.
- A milestone reveals that current behavior is ambiguous and cannot be inferred from code or existing fixtures.

## Likely Blockers

- `cmterrainextractor.py` currently performs module-level Streamlit state initialization and path resolution, which can make imports side-effectful.
- Existing imports assume `terrain_extraction` is importable from the app working directory; package-relative imports may expose path issues.
- PyInstaller hidden imports and data files may need updates after module splitting.
- `streamlit_folium` payload coordinate order must be preserved; changing it risks bbox rotation or axis errors.
- OSM download behavior requires network access and should not become a unit-test dependency.
- Fixture data exists under `test/`, but there is no active pytest suite; tests must be introduced carefully without relying on large files.

## Rollback Strategy

- Keep each milestone small and behavior-preserving.
- Prefer creating new modules and adapting callers over deleting old code immediately.
- Keep `cmterrainextractor.py` as a compatibility wrapper until the modular UI is stable.
- If a milestone fails validation, revert only that milestone's files and leave prior milestones intact.
- Record rollback decisions in the status document.

## Milestone 1: Inventory and Behavior Snapshot

**Goal:** Create a durable inventory of current behavior before code movement.

**Files:**

- Create: `docs/plans/streamlit_refactoring_inventory.md`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** No Python code is touched in this milestone. Automated tests are not required, but the inventory must identify which tests will be created in later milestones.

**Tasks:**

- Inventory current `st.session_state` keys and map each key to planned `AppState` fields, owner module, and reset conditions.
- Inventory current top-level functions in `cm_terrain_extractor_app/cmterrainextractor.py` and assign each function to target ownership.
- Inventory current Streamlit widgets and note where state is read or mutated.
- Inventory current Folium layers, draw controls, `st_folium` options, and map-state behavior.
- Inventory current runtime path and PyInstaller behavior from `cm_terrain_extractor_app/cm_terrain_extractor_app.py` and `cm_terrain_extractor_app.spec`.
- Document current data flows for bbox drawing, elevation extraction, OSM download/upload/process, and CSV export.
- Mark unresolved behavior as `Unknown` with the exact file and line window that needs later inspection.

**Validation:**

```powershell
Get-Content docs\plans\streamlit_refactoring_inventory.md | Select-String -Pattern "session_state","cmterrainextractor.py","Folium","PyInstaller","CSV"
Get-Content docs\plans\streamlit_refactoring_status.md | Select-String -Pattern "Milestone 1","Inventory"
```

**Acceptance criteria:**

- No application code changed.
- Inventory is specific enough to guide M2-M8.
- Status document records M1 completion and any unknowns.

## Milestone 2: Introduce Test Harness and First Regression Tests

**Goal:** Establish pytest structure before refactoring Python code.

**Files:**

- Create: `tests/cm_terrain_extractor/__init__.py`
- Create: `tests/cm_terrain_extractor/test_exports.py`
- Create: `tests/cm_terrain_extractor/test_state.py`
- Create: `tests/cm_terrain_extractor/test_drawing.py`
- Modify only if necessary: `pyproject.toml`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Write tests that initially fail because target modules do not exist yet. Do not implement modules in this milestone except minimal empty package files if needed to make pytest collection meaningful.

**Tasks:**

- Add test package under `tests/cm_terrain_extractor/`.
- Write `test_exports.py` with a regression asserting CSV bytes match `df.to_csv().encode("utf-8")`.
- Write `test_state.py` with a regression for `clear_bbox_dependent_results` clearing bbox-dependent outputs while preserving `osm_config_file`, `osm_profile`, `map_mode`, `map_center`, and `map_zoom`.
- Write `test_drawing.py` with a small `last_active_drawing` payload and expected coordinate ring extraction.
- Run the targeted tests and record expected failures caused by missing target modules.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -v
```

Expected before implementation: failures for missing `app_core` or `map_view` modules.

**Acceptance criteria:**

- Pytest discovers the new tests.
- Expected failures are recorded in the status document.
- No production behavior changed.

## Milestone 3: Extract Resource and Path Handling

**Goal:** Centralize runtime path handling without changing GUI behavior.

**Files:**

- Create: `cm_terrain_extractor_app/app_core/__init__.py`
- Create: `cm_terrain_extractor_app/app_core/resources.py`
- Create: `tests/cm_terrain_extractor/test_resources.py`
- Modify: `cm_terrain_extractor_app/cmterrainextractor.py`
- Modify: `cm_terrain_extractor_app/cm_terrain_extractor_app.py`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Write or extend `test_resources.py` before implementation. Tests must cover source-mode paths and fake packaged-mode inputs without requiring PyInstaller.

**Tasks:**

- Implement `AppResources`.
- Implement `resolve_resources()` for source and packaged execution.
- Implement `prepare_runtime_environment(resources)` so DLL path setup happens exactly once.
- Implement `find_default_osm_configs(resources)`.
- Replace duplicated `sys.frozen`, `sys.executable`, `os.getcwd`, and DLL path logic in app/launcher with resource calls.
- Keep launcher behavior compatible with `cm_terrain_extractor_app.spec`.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_resources.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\resources.py cm_terrain_extractor_app\cmterrainextractor.py cm_terrain_extractor_app\cm_terrain_extractor_app.py tests\cm_terrain_extractor\test_resources.py
```

Manual smoke:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\cmterrainextractor.py
```

Confirm the app starts, finds JSON configs, and does not duplicate DLL path entries.

**Acceptance criteria:**

- Resource tests pass.
- Source run still starts.
- Launcher still bootstraps the same app script.
- Status document records any packaging risk.

## Milestone 4: Introduce AppState and Session Adapter

**Goal:** Replace scattered app state with one explicit state object while preserving Streamlit's rerun model.

**Files:**

- Create: `cm_terrain_extractor_app/app_core/state.py`
- Create: `cm_terrain_extractor_app/streamlit_ui/__init__.py`
- Create: `cm_terrain_extractor_app/streamlit_ui/session_adapter.py`
- Modify: `cm_terrain_extractor_app/cmterrainextractor.py`
- Modify: `tests/cm_terrain_extractor/test_state.py`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Extend `test_state.py` before implementation to cover default state, bbox-dependent invalidation, selected-source invalidation, OSM config/profile invalidation, and `mark_map_dirty`.

**Tasks:**

- Implement `AppState` with fields listed in the contract.
- Implement `clear_bbox_dependent_results(state)`, `clear_elevation_result(state)`, `clear_osm_processing_result(state)`, and `mark_map_dirty(state)`.
- Implement `STATE_KEY`, `get_state(resources)`, and `reset_state(resources)` in `streamlit_ui/session_adapter.py`.
- Store only the main `AppState` under `STATE_KEY`; allow temporary legacy keys only where needed and record them as debt.
- Initialize `selectable_data_sources` through the state path rather than unconditional module-level `st.session_state` assignment.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_state.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\state.py cm_terrain_extractor_app\streamlit_ui\session_adapter.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_state.py
```

Manual smoke:

- Start the Streamlit app.
- Confirm initial map mode, source list, and bbox editing still work.
- Draw a bbox and confirm metrics update.

**Acceptance criteria:**

- `AppState` tests pass.
- App starts with equivalent defaults.
- Temporary legacy session keys, if any, are documented in status.

## Milestone 5: Extract Validation and Export Helpers

**Goal:** Move pure bbox validity and CSV export behavior into tested core modules.

**Files:**

- Create: `cm_terrain_extractor_app/app_core/validation.py`
- Create: `cm_terrain_extractor_app/app_core/exports.py`
- Create: `tests/cm_terrain_extractor/test_validation.py`
- Modify: `tests/cm_terrain_extractor/test_exports.py`
- Modify: `cm_terrain_extractor_app/cmterrainextractor.py`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Tests must fail first for missing validation/export functions, then pass after extraction.

**Tasks:**

- Implement `is_selected_area_valid(len_x, len_y)` using the contractual limits.
- Implement `compute_bbox_metrics(bbox_object)` using current `BoundingBox` methods.
- Implement `update_state_from_bbox(state, bbox_object)` and ensure bbox-dependent invalidation happens.
- Implement `dataframe_to_csv_bytes(df)` matching current CSV bytes exactly.
- Implement filename helpers for elevation and OSM downloads.
- Replace `dataframe2csv` and inline bbox validity logic in Streamlit code with core helpers.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_validation.py tests\cm_terrain_extractor\test_exports.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\validation.py cm_terrain_extractor_app\app_core\exports.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_validation.py tests\cm_terrain_extractor\test_exports.py
```

Manual smoke:

- Draw or edit a bbox.
- Confirm W-E length, S-N length, and area validity match prior behavior.
- Download elevation and OSM CSVs when data exists and confirm filenames are stable.

**Acceptance criteria:**

- CSV bytes are compatible with current behavior.
- Bbox validity tests cover valid, too-wide, too-tall, and too-large-area cases.
- No Streamlit imports exist in `app_core/validation.py` or `app_core/exports.py`.

## Milestone 6: Extract Backend Actions

**Goal:** Move elevation and OSM backend operations into Streamlit-free functions.

**Files:**

- Create: `cm_terrain_extractor_app/app_core/actions.py`
- Optional create: `cm_terrain_extractor_app/app_core/errors.py`
- Create: `tests/cm_terrain_extractor/test_actions_smoke.py`
- Modify: `cm_terrain_extractor_app/cmterrainextractor.py`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Write smoke tests with fakes before implementation. Tests must not use network access.

**Tasks:**

- Implement `find_data_sources_in_bbox` using fake source tests and current `intersects_bounding_box` contract.
- Implement `extract_elevation_data` using a fake data source that records `get_data` and `get_png` calls.
- Implement `load_osm_config` with JSON fixtures or temporary files.
- Implement `load_osm_data_from_uploaded_bytes` around existing OSM IO behavior if practical without large fixture dependency.
- Implement `download_osm_data` as a thin wrapper around OSMnx; leave network validation as manual or integration-only.
- Implement `process_osm_data` as a thin wrapper around `OSMProcessor`.
- Move Streamlit status messages out of backend logic and keep them in UI callers.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_actions_smoke.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\app_core\actions.py cm_terrain_extractor_app\app_core\errors.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_actions_smoke.py
```

Manual smoke:

- Find available data sources for a selected bbox.
- Extract elevation for a small known bbox.
- Load an OSM config.
- Upload a small GeoJSON fixture.
- Process OSM data from a fixture if available.

**Acceptance criteria:**

- `app_core/actions.py` imports no Streamlit and no Folium.
- UI still shows status before/after backend calls.
- Existing CSV output remains compatible.
- Network-dependent behavior is not required for unit test pass.

## Milestone 7: Extract Drawing Parser and Folium Map/Layers

**Goal:** Isolate Folium map construction and drawing parsing from Streamlit UI code.

**Files:**

- Create: `cm_terrain_extractor_app/map_view/__init__.py`
- Create: `cm_terrain_extractor_app/map_view/drawing.py`
- Create: `cm_terrain_extractor_app/map_view/folium_map.py`
- Create: `cm_terrain_extractor_app/map_view/folium_layers.py`
- Modify: `tests/cm_terrain_extractor/test_drawing.py`
- Create: `tests/cm_terrain_extractor/test_folium_layers.py`
- Modify: `cm_terrain_extractor_app/cmterrainextractor.py`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Write drawing tests and lightweight layer smoke tests before moving map code.

**Tasks:**

- Implement drawing payload extraction with small GeoJSON-like payloads.
- Convert drawing payloads to `BoundingBox` without changing coordinate order.
- Implement `build_folium_map(state, resources)` with existing OpenTopoMap tile attribution.
- Implement layer helpers for draw control, bbox, elevation overlay, OSM bbox, and OSM geometry layers.
- Preserve geocoder, measure control, fullscreen control, bbox styling, axis labels, OSM dashed bbox, elevation overlay opacity, and geometry priority ordering.
- Keep `st_folium(...)` invocation in Streamlit UI code.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_drawing.py tests\cm_terrain_extractor\test_folium_layers.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\map_view\drawing.py cm_terrain_extractor_app\map_view\folium_map.py cm_terrain_extractor_app\map_view\folium_layers.py cm_terrain_extractor_app\cmterrainextractor.py tests\cm_terrain_extractor\test_drawing.py tests\cm_terrain_extractor\test_folium_layers.py
```

Manual smoke:

- Start the app.
- Draw a rectangle and polygon.
- Confirm bbox appears in red with axis labels.
- Confirm elevation overlay appears after extraction.
- Confirm OSM bbox and geometries appear after processing.
- Confirm map center and zoom are preserved as before.

**Acceptance criteria:**

- `map_view` does not import Streamlit.
- Drawing parser is unit-testable.
- Map builder does not download or process data.
- Visual behavior is preserved closely enough to avoid a user-facing redesign.

## Milestone 8: Split Streamlit UI Modules

**Goal:** Reduce `cmterrainextractor.py` to a small wrapper or replace it with modular Streamlit UI files.

**Files:**

- Create: `cm_terrain_extractor_app/streamlit_main.py`
- Create: `cm_terrain_extractor_app/streamlit_ui/app.py`
- Create: `cm_terrain_extractor_app/streamlit_ui/sidebar.py`
- Create: `cm_terrain_extractor_app/streamlit_ui/map_page.py`
- Create: `cm_terrain_extractor_app/streamlit_ui/widgets.py`
- Optional create: `cm_terrain_extractor_app/streamlit_ui/elevation_page.py`
- Optional create: `cm_terrain_extractor_app/streamlit_ui/osm_page.py`
- Optional create: `cm_terrain_extractor_app/streamlit_ui/status.py`
- Modify: `cm_terrain_extractor_app/cmterrainextractor.py`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** For any pure helper moved out of Streamlit UI, write tests first. For Streamlit-only rendering code, rely on manual smoke checks and keep logic thin.

**Tasks:**

- Implement `render_app(resources)` that configures page, gets `AppState`, renders sidebar, and renders tabs.
- Move sidebar controls to `streamlit_ui/sidebar.py`.
- Move `st_folium` invocation and map-result handling to `streamlit_ui/map_page.py`.
- Keep UI status handling in Streamlit modules.
- Add comments for any remaining `st.rerun()` calls explaining why they are required.
- Reduce `cmterrainextractor.py` to a compatibility wrapper that calls the new entrypoint, or replace launcher target with `streamlit_main.py`.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\streamlit_main.py cm_terrain_extractor_app\streamlit_ui cm_terrain_extractor_app\cmterrainextractor.py
```

Manual smoke:

- Source-run the app through Streamlit.
- Verify tabs, sidebar mode selector, bbox editing, data-source search, elevation extraction, OSM upload/download/process controls, and CSV downloads.
- Confirm no app-core module imports Streamlit:

```powershell
Select-String -Path cm_terrain_extractor_app\app_core\*.py -Pattern "streamlit"
```

Expected: no matches except comments/docstrings if intentionally documented.

**Acceptance criteria:**

- App layout remains recognizable.
- `cmterrainextractor.py` is substantially smaller or a wrapper.
- Streamlit imports are limited to UI and entrypoint files.
- All tests pass.

## Milestone 9: Stabilize Launcher and PyInstaller Spec

**Goal:** Ensure source and packaged execution still work after module splitting.

**Files:**

- Modify: `cm_terrain_extractor_app/cm_terrain_extractor_app.py`
- Modify: `cm_terrain_extractor_app.spec`
- Create: `docs/cm_terrain_extractor_packaging.md`
- Modify: `docs/plans/streamlit_refactoring_status.md`

**TDD requirement:** Automated tests are limited to resource-path behavior from M3. Packaging must have manual smoke checks and documented evidence.

**Tasks:**

- Point the launcher at `streamlit_main.py` or the chosen compatibility wrapper.
- Ensure `prepare_runtime_environment(resources)` is called before bootstrap.
- Update PyInstaller hidden imports and datas for new `app_core`, `map_view`, and `streamlit_ui` modules if needed.
- Document source mode paths, packaged mode paths, DLL handling, config discovery, data cache, profiles, and manual packaging command.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_resources.py -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\cm_terrain_extractor_app.py cm_terrain_extractor_app\app_core\resources.py
```

Manual smoke:

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m streamlit run cm_terrain_extractor_app\streamlit_main.py
```

If packaging is in scope for the execution environment, also run the existing PyInstaller workflow and launch the built executable. Record the exact command and result in the status document.

**Acceptance criteria:**

- Source run works.
- Packaged run works or exact unverified packaging risk is documented.
- App finds OSM configs, data cache, DLLs, profiles, and Streamlit assets.

## Milestone 10: Final Integration, Docs, and Validation

**Goal:** Finish the refactor with tests, docs, and status up to date.

**Files:**

- Modify: `docs/plans/streamlit_refactoring_status.md`
- Modify: `docs/cm_terrain_extractor_packaging.md` if needed
- Modify canonical docs under `docs/` only if behavior, contract, or packaging semantics changed

**TDD requirement:** No new behavior should be introduced in this milestone. Any final bugfix must still start with a failing regression test.

**Tasks:**

- Run full app-specific pytest suite.
- Run ruff on all touched Python files.
- Inspect imports for contract violations.
- Confirm no forbidden framework migration or algorithm rewrite occurred.
- Confirm status document contains final progress, decisions, debt, validations, and residual risks.
- Confirm contract remains unchanged unless user-approved changes were recorded.

**Validation:**

```powershell
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor -v
C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check <all touched Python files>
Select-String -Path cm_terrain_extractor_app\app_core\*.py -Pattern "streamlit","streamlit_folium"
Select-String -Path cm_terrain_extractor_app\map_view\*.py -Pattern "streamlit"
```

Expected import checks:

- No `streamlit` or `streamlit_folium` matches in `app_core/`.
- No `streamlit` matches in `map_view/`, except comments that document payload shapes.

Manual smoke:

- Source-run the app.
- Complete bbox drawing.
- Complete elevation data-source lookup and extraction for a small known area.
- Complete OSM upload/process flow with a fixture.
- Confirm CSV downloads are usable.
- Confirm packaged executable status is recorded.

**Acceptance criteria:**

- All app-specific tests pass.
- Ruff passes on touched files.
- Contract import boundaries hold.
- Status document is complete enough for a future agent or human to resume.
- Any unverified packaging or network-dependent behavior is explicitly recorded as residual risk.
