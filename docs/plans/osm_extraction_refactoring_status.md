# OSM Extraction Refactoring Status

This is the living status ledger for the CMTerrainExtractor OSM extraction refactor.

Authoritative contract: `docs/plans/osm_extraction_refactoring_contract.md`.

Execution plan: `docs/plans/osm_extraction_refactoring_exec_plan.md`.

Source outline: `docs/plans/osm_extraction_refactoring_plan.md`.

Future agents and humans must update this document after each milestone. Record progress, decisions, contract-change requests, technical debt, validation commands, results, blockers, and residual risk. If execution would violate the contract, stop and ask the user before continuing.

## Progress by Milestone

| Milestone | Name | Status | Owner | Last updated | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline fixtures, invariant tests, benchmark runner, and metrics JSON | Done | Codex | 2026-05-05 | Added offline fixtures, invariant tests, and benchmark JSON metrics. |
| 1 | Pipeline skeleton and compatibility wrapper | Done | Codex | 2026-05-05 | Added typed skeleton package, headless-compatible import fallback, progress callback context, deterministic RNG owner, and `OSMProcessor.pipeline` compatibility holder. |
| 2 | Config schema, feature matcher, deterministic seed, and early bug fixes | Done | Codex | 2026-05-05 | Added validated config schema, feature matcher, `path_to_config` compatibility, barn spelling bridge, and non-WGS84 `BoundingBox` fix. |
| 3 | `GridIndex` and dense `OccupancyModel` | Done | Codex | 2026-05-05 | Added affine grid math, lazy debug grid views, dense layered occupancy arrays, and readable conflict decisions. |
| 4 | Area rasterizer | Not started | Unassigned | 2026-05-05 | Moves area/default/point placement to new grid and occupancy model. |
| 5 | Network topology and noding | Not started | Unassigned | 2026-05-05 | Adds topology-first line handling and snap tolerance. |
| 6 | Corridor-limited integer-grid routing | Not started | Unassigned | 2026-05-05 | Replaces full-grid NetworkX routing in the new path. |
| 7 | Tile catalog assignment and intersection solving | Not started | Unassigned | 2026-05-05 | Replaces per-edge NetworkX tile graphs and validates tile labels. |
| 8 | Building fitter v2 | Not started | Unassigned | 2026-05-05 | Adds candidate scoring, road avoidance, and bounded diagnostics. |
| 9 | Layered output rows and conflict validation | Not started | Unassigned | 2026-05-05 | Makes `post_process` validation/compatibility instead of primary conflict handling. |
| 10 | Debug export and quality visualization | Not started | Unassigned | 2026-05-05 | Rebuilds debug geometries from records, placements, and stats. |
| 11 | Legacy quarantine/removal and durable docs | Not started | Unassigned | 2026-05-05 | Removes or quarantines inactive legacy paths after tests prove new paths are active. |

## Decisions Made

| Date | Decision | Rationale | Approved by |
| --- | --- | --- | --- |
| 2026-05-05 | Use `docs/plans/osm_extraction_refactoring_contract.md` as the authoritative contract. | The user requested a contract document whose content must not be changed without approval. | User |
| 2026-05-05 | Use `docs/plans/osm_extraction_refactoring_exec_plan.md` as the milestone execution plan. | The user requested an ExecPlan-style workflow with decision-complete milestones. | User |
| 2026-05-05 | Use `docs/plans/osm_extraction_refactoring_status.md` as the living status ledger. | The user requested status tracking for progress, decisions, debt, validation results, and blockers. | User |
| 2026-05-05 | Preserve `cm_terrain_extractor_app/terrain_extraction/osm_processor.py` as the external compatibility entry point. | The source outline explicitly identifies it as the public boundary to preserve. | User |
| 2026-05-05 | Create the new internal engine under `cm_terrain_extractor_app/terrain_extraction/osm_extraction/`. | The source outline recommends this package split and it keeps migration separate from legacy helpers. | User |
| 2026-05-05 | Put new OSM extraction tests under `tests/cm_terrain_extractor/osm_extraction/`. | The repository now has an active `tests/cm_terrain_extractor/` suite; the old no-test-suite claim is stale. | Codex |
| 2026-05-05 | Treat rail debug/export as a revalidation target rather than a confirmed current defect. | Current `get_geometries` has generic non-building handling, so the outline's rail concern needs a test before code changes. | Codex |
| 2026-05-05 | Keep Milestone 0 fixtures WGS84-only. | The non-WGS84 `BoundingBox` constructor bug is already assigned to Milestone 2, so baseline fixtures avoid that unrelated known defect. | Codex |
| 2026-05-05 | Seed legacy RNG only inside the benchmark runner and record it in diagnostics. | Legacy processors call `np.random.default_rng()` without an injected seed; the runner needs repeatable baseline replay without changing app behavior. | Codex |
| 2026-05-05 | Use `building=shed` for Milestone 0 village/collision fixtures and report source geometry building-line intersections. | Synthetic outline-building tags routed into the legacy building outline collector and aborted the Windows process in Shapely before Python could catch an exception; algorithm replacement is out of scope for Milestone 0. | Codex |
| 2026-05-05 | Keep Milestone 1 algorithm behavior on the legacy path while introducing `ExtractionPipeline.run_legacy`. | The milestone goal is orchestration and compatibility only; routing, placement, and output semantics remain owned by later milestones. | Codex |
| 2026-05-05 | Replace the hard Streamlit import requirement in `osm_processor.py` with a no-op fallback object. | Headless tests must be able to import the compatibility adapter even when Streamlit is unavailable, while existing tests can still monkeypatch `st.progress`. | Codex |
| 2026-05-05 | Accept `{"dummy": true}` CM type entries as valid schema records with `-1` menu/category placeholders. | The existing default foliage config uses a weighted dummy option to mean no placed foliage; failing startup on that current config would break compatibility. | Codex |
| 2026-05-05 | Keep `path_to_congih` as a read-only compatibility alias for `path_to_config`. | Existing diagnostics and callers may still read the misspelled attribute, but new code should use the corrected name. | Codex |
| 2026-05-05 | Implement `GridIndex` with affine unit axes derived from bbox reference points rather than spatial nearest queries. | This matches the legacy representative grid conventions while keeping snapping in regular-grid math for new hot paths. | Codex |
| 2026-05-05 | Keep normal, sub-square, and diagonal GeoDataFrame creation lazy and debug/export-oriented. | Milestone 3 must not make GeoDataFrames the normal snapping or conflict mechanism for new code. | Codex |
| 2026-05-05 | Use dense `OccupancyModel` arrays indexed by `(xidx, yidx)` with `-1` empty sentinels and explicit stronger-priority replacement only when requested. | This preserves deterministic conflict checks, keeps accidental overwrites visible, and aligns with the status assumption that smaller positive priority is the stronger claim. | Codex |

## Contract Changes Requested or Approved

| Date | Request | Status | Resolution |
| --- | --- | --- | --- |
| 2026-05-05 | Initial OSM extraction contract document created. | Approved by request | Contract is authoritative once added. |

No further contract changes have been requested or approved.

## Technical Debt Ledger

| Date | Milestone | Debt | Reason | Paydown target |
| --- | --- | --- | --- | --- |
| 2026-05-05 | Planning docs | No production-code debt introduced. | This setup creates planning documents only. | Not applicable. |
| 2026-05-05 | Current code | Core OSM preprocessing imports Streamlit and creates progress bars. | Existing implementation predates the headless pipeline boundary. | Milestone 1 progress callback and compatibility wrapper. |
| 2026-05-05 | Current code | Current OSM processing uses `matched_elements`, `processing_stages`, GeoDataFrame grids, `occupancy_gdf`, and NetworkX full-grid routing. | These are the main refactor targets identified by the outline and current-code inspection. | Milestones 1 through 9. |
| 2026-05-05 | Current code | Known drift includes `path_to_congih`, barn process spelling mismatch, non-WGS84 `BoundingBox` EPSG bug, duplicate `get_matched_cm_type`, hard-coded shared tile labels, and NetworkX routing. | These were confirmed or revalidated while preparing the planning documents. | Milestones 2, 6, and 7. |
| 2026-05-05 | 0 | Benchmark stage timings are approximate legacy wrapper timings. | Current processing does not expose per-stage structured timings; Milestone 0 records `preprocess_osm_data`, `run_processors`, `post_process`, `output_assembly`, and null placeholders for future stage timings. | Milestone 1 stats/pipeline boundary. |
| 2026-05-05 | 0 | Legacy outline-building synthetic fixtures can abort the Windows process in Shapely during `collect_building_geometries`. | The crash occurs before a catchable Python exception and Milestone 0 cannot refactor building algorithms. | Milestone 8 building fitter v2, with earlier guard tests if needed. |
| 2026-05-05 | 1 | `OSMProcessor` now holds a pipeline skeleton, but public processing methods still execute legacy algorithms directly. | Milestone 1 intentionally adds the compatibility wrapper without changing extraction behavior. | Milestones 2 through 10 activate typed stages behind feature flags. |
| 2026-05-05 | 1 | `preprocess_osm_data` still uses an `st.progress`-compatible object rather than delegating progress through `ExtractionContext`. | Preserving current tests and UI behavior was safer for this skeleton milestone. | Later adapter milestones can route public-method progress through the callback once stage ownership moves into the pipeline. |
| 2026-05-05 | 2 | Legacy preprocessing still uses its local tag index and duplicated `get_matched_cm_type` helpers instead of the new `FeatureMatcher` for active output generation. | Milestone 2 introduced the typed schema and matcher while preserving legacy extraction behavior; swapping active matching belongs with later feature-flag activation. | Milestones 4 through 11, as feature classes move to typed records. |
| 2026-05-05 | 2 | Hard-coded shared tile labels were not changed in this milestone. | Tile assignment semantics are owned by Milestone 7 and need catalog/intersection tests before changing output labels. | Milestone 7 tile catalog assignment and intersection solving. |
| 2026-05-05 | 3 | `GridIndex` and `OccupancyModel` are implemented foundations but are not yet active in the pipeline. | Milestone 3 adds the shared grid/conflict primitives; feature processors move onto them in later milestones. | Milestones 4 through 9. |

## Test and Validation Results

| Date | Milestone | Command or check | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-05 | Planning docs | `Test-Path docs\plans\osm_extraction_refactoring_contract.md; Test-Path docs\plans\osm_extraction_refactoring_exec_plan.md; Test-Path docs\plans\osm_extraction_refactoring_status.md` | Passed before creation with all `False` | Confirmed the three requested docs did not already exist. |
| 2026-05-05 | Planning docs | `Get-Content docs\plans\streamlit_refactoring_contract.md` and `Get-Content docs\plans\streamlit_refactoring_exec_plan.md` | Passed | Used existing planning docs as local style references. |
| 2026-05-05 | Planning docs | `Get-Content docs\plans\osm_extraction_refactoring_plan.md` plus targeted `rg` inspections of current OSM code | Passed | Confirmed source outline and current-state corrections before writing docs. |
| 2026-05-05 | Planning docs | `Get-ChildItem docs\plans\osm_extraction_refactoring_*.md` | Passed | Confirmed contract, exec plan, source outline, and status docs exist under `docs/plans/`. |
| 2026-05-05 | Planning docs | `Select-String -Path docs\plans\osm_extraction_refactoring_*.md -Pattern "authoritative","contract","TDD","pytest","ruff","Dependency Graph","Blockers","Status"` | Passed | Confirmed required keywords are present across the OSM planning docs. |
| 2026-05-05 | Planning docs | `Select-String -Path docs\plans\osm_extraction_refactoring_exec_plan.md -Pattern "Milestone 0","Milestone 11","C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe"` | Failed due invocation | PowerShell parsed the literal Windows path as a regex and rejected `\U` as an escape sequence. |
| 2026-05-05 | Planning docs | `Select-String -SimpleMatch -Path docs\plans\osm_extraction_refactoring_exec_plan.md -Pattern "Milestone 0","Milestone 11","C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe"` | Passed | Equivalent literal search confirmed Milestone 0, Milestone 11, and the approved Conda pytest path are present. |
| 2026-05-05 | 0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_baseline_invariants.py -q` | Failed as expected before implementation | TDD red step: collection failed because `cm_terrain_extractor_app.terrain_extraction.osm_extraction_benchmark` did not exist yet. |
| 2026-05-05 | 0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_baseline_invariants.py -v` | Failed during development | Initial outline-building fixtures reached legacy `collect_building_geometries` and produced a Windows fatal exception in Shapely; fixtures were adjusted to avoid outline processing in this baseline milestone. |
| 2026-05-05 | 0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_baseline_invariants.py -v` | Passed | 13 passed. Warnings are from existing GeoPandas/Pandas/legacy processing paths plus `.pytest_cache` permission warning. |
| 2026-05-05 | 0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe -m cm_terrain_extractor_app.terrain_extraction.osm_extraction_benchmark --fixture crossroads --profile cold_war --config default_osm_config.json --seed 123` | Passed | Emitted metrics JSON with `timings`, `counts`, `quality`, and `diagnostics`; legacy warnings went to stderr. |
| 2026-05-05 | 0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check tests\cm_terrain_extractor\osm_extraction\test_baseline_invariants.py --fix --no-cache` | Failed due environment | Ruff could not rewrite the file in-place: `Zugriff verweigert (os error 5)`. The import order was fixed manually. |
| 2026-05-05 | 0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction_benchmark.py tests\cm_terrain_extractor\osm_extraction --no-cache` | Passed | All checks passed after manual fixes. |
| 2026-05-05 | 1 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_models.py tests\cm_terrain_extractor\osm_extraction\test_pipeline.py -q` | Failed as expected before implementation | TDD red step: new `terrain_extraction.osm_extraction` package did not exist; one `tmp_path` test also exposed the known sandbox temp-directory permission issue before the test was adjusted to use an existing config. |
| 2026-05-05 | 1 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_models.py tests\cm_terrain_extractor\osm_extraction\test_pipeline.py -v --basetemp .pytest-tmp` | Failed due environment | Pytest could not create `E:\Spiele\CMAutoEditor\.pytest-tmp` due `Zugriff verweigert`; the test no longer requires `tmp_path`, so validation continued without basetemp. |
| 2026-05-05 | 1 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_models.py tests\cm_terrain_extractor\osm_extraction\test_pipeline.py -v` | Passed | 5 passed. Warnings are existing dependency deprecations plus `.pytest_cache` permission warning. |
| 2026-05-05 | 1 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py -q` | Passed | 11 passed. Existing GeoPandas/Pandas/protobuf deprecation warnings and `.pytest_cache` permission warning remain. |
| 2026-05-05 | 1 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction cm_terrain_extractor_app\terrain_extraction\osm_processor.py tests\cm_terrain_extractor\osm_extraction --no-cache` | Passed | All checks passed after manual import ordering and `StrEnum` fixes. |
| 2026-05-05 | 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_config_schema.py tests\cm_terrain_extractor\osm_extraction\test_feature_matcher.py -q` | Failed as expected before implementation | TDD red step: `config_schema.py` and `feature_matcher.py` did not exist, `OSMProcessor` exposed only `path_to_congih`, and non-WGS84 `BoundingBox` passed `crs.to_epsg` instead of `crs.to_epsg()`. |
| 2026-05-05 | 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_config_schema.py tests\cm_terrain_extractor\osm_extraction\test_feature_matcher.py -v` | Passed | 8 passed. Existing dependency deprecation warnings and `.pytest_cache` permission warning remain. |
| 2026-05-05 | 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\test_osm_processor.py -q` | Passed | 11 passed. Existing GeoPandas/Pandas/protobuf deprecation warnings and `.pytest_cache` permission warning remain. |
| 2026-05-05 | 2 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction cm_terrain_extractor_app\terrain_extraction\osm_processor.py cm_terrain_extractor_app\terrain_extraction\bbox_utils.py profiles\__init__.py tests\cm_terrain_extractor\osm_extraction tests\cm_terrain_extractor\test_osm_processor.py --no-cache` | Failed during cleanup, then passed | Initial run exposed pre-existing lint in touched `bbox_utils.py` and `profiles/__init__.py`; manual mechanical cleanup was needed because `ruff --fix` failed with Windows `Zugriff verweigert`. Final run: all checks passed. |
| 2026-05-05 | 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_grid_index.py tests\cm_terrain_extractor\osm_extraction\test_occupancy.py -q` | Failed as expected before implementation | TDD red step: `grid_index.py` and `occupancy.py` did not exist yet. |
| 2026-05-05 | 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_grid_index.py tests\cm_terrain_extractor\osm_extraction\test_occupancy.py -v` | Passed | 7 passed. Existing dependency deprecation warnings and `.pytest_cache` permission warning remain. |
| 2026-05-05 | 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_models.py -q` | Passed | 2 passed. Run because `models.py` and package exports were touched. Existing `.pytest_cache` permission warning remains. |
| 2026-05-05 | 3 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check cm_terrain_extractor_app\terrain_extraction\osm_extraction\grid_index.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\occupancy.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\models.py cm_terrain_extractor_app\terrain_extraction\osm_extraction\__init__.py tests\cm_terrain_extractor\osm_extraction --no-cache` | Failed during cleanup, then passed | Initial run found import-order issues; `ruff --fix` failed with Windows `Zugriff verweigert`, so the equivalent import-order diff was applied manually. Final run: all checks passed. |

## Known Blockers

| Date | Blocker | Impact | Next action |
| --- | --- | --- | --- |
| 2026-05-05 | Streamlit progress is still used through an `st.progress`-compatible adapter in `osm_processor.py`. | The compatibility adapter can import headlessly, but public preprocessing has not yet delegated progress through the new pipeline context. | Later pipeline activation milestones. |
| 2026-05-05 | OSM download requires network access. | Unit tests cannot depend on live OSMnx calls. | Use fixture GeoJSON and fakes in Milestone 0. |
| 2026-05-05 | Current output extent marker semantics are not documented. | Output assembly replacement could silently change CSV import behavior. | Capture behavior in Milestone 0 and test explicit replacement in Milestone 9. |
| 2026-05-05 | NetworkX full-grid routing is likely the largest runtime risk. | Large or dense extracts may be slow or memory-heavy until routing is replaced. | Milestone 6 after topology and occupancy foundations. |
| 2026-05-05 | Windows sandbox cache/temp permissions may interrupt pytest or ruff. | Validation can fail before executing tests. | Retry once with `--basetemp` or `--no-cache`; record unverified results after two environment failures. |
| 2026-05-05 | Synthetic outline-building fixtures can abort the process in legacy building outline collection. | Milestone 0 cannot safely baseline `building=house` or `building=barn` outline paths in this environment. | Keep using non-outline building fixtures for baseline; add guarded outline/building-fitter coverage in Milestone 8 or earlier if the implementation touches that path. |

## Current Working Assumptions

- `OSMProcessor` remains the app-facing compatibility class until the user approves a different public boundary.
- The new internal package will be `cm_terrain_extractor_app/terrain_extraction/osm_extraction/`.
- New OSM extraction tests belong under `tests/cm_terrain_extractor/osm_extraction/`.
- Existing tests under `tests/cm_terrain_extractor/` must remain passing unless a contract-approved behavior change updates them.
- Fixture-based tests are preferred over live OSM downloads.
- Random extraction behavior must become deterministic under an explicit seed.
- Smaller numeric priority currently appears to be the stronger positive placement claim; the new code should name this internally as rank if it clarifies semantics.
- Default network snap tolerance starts at `1.0 m` unless fixture results justify a status-recorded change.
- Route deviation starts with `24 m` for major roads, `32 m` for normal routes, and up to `48 m` for minor-route relaxation unless fixture results justify a status-recorded change.
- Use these validation binaries for future Python work:
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\python.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe`
  - `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe`

## Status Update Procedure

After each milestone:

- Change the milestone status to `In progress`, `Blocked`, or `Done`.
- Add the owner and exact date.
- Record decisions and contract-change requests.
- Record technical debt introduced intentionally.
- Paste the exact validation commands and concise results.
- Record manual smoke checks separately from automated tests.
- Record unresolved risks before handing off.
