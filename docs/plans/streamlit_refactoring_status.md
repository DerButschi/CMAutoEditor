# Streamlit Refactoring Status

This is the living status ledger for the CMTerrainExtractor Streamlit refactor.

Authoritative contract: `docs/plans/streamlit_refactoring_contract.md`.

Execution plan: `docs/plans/streamlit_refactoring_exec_plan.md`.

Future agents and humans must update this document after each milestone. Record progress, decisions, contract-change requests, technical debt, validation commands, results, blockers, and residual risk. If execution would violate the contract, stop and ask the user before continuing.

## Progress by Milestone

| Milestone | Name | Status | Owner | Last updated | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | Inventory and behavior snapshot | Done | Codex | 2026-05-02 | Created `docs/plans/streamlit_refactoring_inventory.md`; app code untouched. |
| 2 | Introduce test harness and first regression tests | Not started | Unassigned | 2026-05-02 | Must establish `tests/cm_terrain_extractor/`; tests should fail before target modules exist. |
| 3 | Extract resource and path handling | Not started | Unassigned | 2026-05-02 | Must preserve launcher and source-run behavior. |
| 4 | Introduce `AppState` and session adapter | Not started | Unassigned | 2026-05-02 | Temporary legacy session keys must be recorded as debt. |
| 5 | Extract validation and export helpers | Not started | Unassigned | 2026-05-02 | CSV bytes must remain compatible with current `df.to_csv().encode("utf-8")`. |
| 6 | Extract backend actions | Not started | Unassigned | 2026-05-02 | Unit tests must use fakes and avoid network dependency. |
| 7 | Extract drawing parser and Folium map/layers | Not started | Unassigned | 2026-05-02 | Coordinate order must be preserved and tested. |
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

## Test and Validation Results

| Date | Milestone | Command or check | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-02 | Planning docs | `Get-ChildItem docs\plans` | Passed | Confirmed contract, exec plan, status, and source outline exist under `docs/plans/`. |
| 2026-05-02 | Planning docs | `Select-String -Path docs\plans\streamlit_refactoring_*.md -Pattern "authoritative","contract","TDD","pytest","ruff"` | Passed | Confirmed required keywords are present across the generated planning docs. |
| 2026-05-02 | Milestone 1 | `Get-Content docs\plans\streamlit_refactoring_inventory.md \| Select-String -Pattern "session_state","cmterrainextractor.py","Folium","PyInstaller","CSV"` | Passed | Confirmed the inventory includes required behavior categories. |
| 2026-05-02 | Milestone 1 | `Get-Content docs\plans\streamlit_refactoring_status.md \| Select-String -Pattern "Milestone 1","Inventory"` | Passed | Confirmed the status ledger records M1 completion and inventory notes. |

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
