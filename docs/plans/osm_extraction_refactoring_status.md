# OSM Extraction Refactoring Status

This is the living status ledger for the CMTerrainExtractor OSM extraction refactor.

Authoritative contract: `docs/plans/osm_extraction_refactoring_contract.md`.

Execution plan: `docs/plans/osm_extraction_refactoring_exec_plan.md`.

Source outline: `docs/plans/osm_extraction_refactoring_plan.md`.

Future agents and humans must update this document after each milestone. Record progress, decisions, contract-change requests, technical debt, validation commands, results, blockers, and residual risk. If execution would violate the contract, stop and ask the user before continuing.

## Progress by Milestone

| Milestone | Name | Status | Owner | Last updated | Notes |
| --- | --- | --- | --- | --- | --- |
| 0 | Baseline fixtures, invariant tests, benchmark runner, and metrics JSON | Not started | Unassigned | 2026-05-05 | Must establish OSM-specific TDD fixtures before algorithm changes. |
| 1 | Pipeline skeleton and compatibility wrapper | Not started | Unassigned | 2026-05-05 | Must preserve `OSMProcessor` public surface. |
| 2 | Config schema, feature matcher, deterministic seed, and early bug fixes | Not started | Unassigned | 2026-05-05 | Covers known current drift including barn process spelling and `BoundingBox` EPSG bug. |
| 3 | `GridIndex` and dense `OccupancyModel` | Not started | Unassigned | 2026-05-05 | Introduces affine grid math and explicit conflict policy. |
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

## Known Blockers

| Date | Blocker | Impact | Next action |
| --- | --- | --- | --- |
| 2026-05-05 | Streamlit is imported by `osm_processor.py`. | Core extraction cannot be cleanly headless until the pipeline/progress callback exists. | Milestone 1. |
| 2026-05-05 | OSM download requires network access. | Unit tests cannot depend on live OSMnx calls. | Use fixture GeoJSON and fakes in Milestone 0. |
| 2026-05-05 | Current output extent marker semantics are not documented. | Output assembly replacement could silently change CSV import behavior. | Capture behavior in Milestone 0 and test explicit replacement in Milestone 9. |
| 2026-05-05 | Current building process mapping disagrees on barn process spelling. | Barn outlines may be processed but not reconstructed in debug/export paths. | Fix with config/profile tests in Milestone 2. |
| 2026-05-05 | NetworkX full-grid routing is likely the largest runtime risk. | Large or dense extracts may be slow or memory-heavy until routing is replaced. | Milestone 6 after topology and occupancy foundations. |
| 2026-05-05 | Windows sandbox cache/temp permissions may interrupt pytest or ruff. | Validation can fail before executing tests. | Retry once with `--basetemp` or `--no-cache`; record unverified results after two environment failures. |

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
