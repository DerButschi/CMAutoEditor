# OSM Network Recovery Status

This is the living status ledger for OSM road and linear-network recovery after the typed OSM extraction refactor.

Authoritative contract: `docs/plans/osm_network_recovery_contract.md`.

Execution plan: `docs/plans/osm_network_recovery_exec_plan.md`.

Source plan: `docs/plans/osm_network_recovery_tdd_plan.md`.

Related refactor status: `docs/plans/osm_extraction_refactoring_status.md`.

Future agents and humans must update this document after each milestone. Record progress, decisions, contract-change requests, technical debt, validation commands, results, blockers, and residual risk. If execution would violate the contract, stop and ask the user before continuing.

## Progress by Milestone

| Milestone | Name | Status | Owner | Last updated | Notes |
| --- | --- | --- | --- | --- | --- |
| M0 | Semantic regression harness | Complete | Codex | 2026-05-15 | Added six network recovery fixtures, `run_osm_extraction_fixture`, `ExtractionTestResult`, road graph reconstruction, debug capture, ASCII grids, and an xfailed four-way road case. |
| M1 | Route node vs tile-cell contract | Not started | Unassigned | 2026-05-15 | Remove ambiguous `RouteRecord.cells` semantics or replace with direct cell-path routing. |
| M2 | Raster-spine / line-support extraction | Not started | Unassigned | 2026-05-15 | Add source-line support cells and route/spine diagnostics. |
| M3 | Tile catalog feasibility oracle | Not started | Unassigned | 2026-05-15 | Extend `CompiledTileCatalog` with pre-assignment direction-set feasibility. |
| M4 | `LinearNetworkState` | Not started | Unassigned | 2026-05-15 | Add persistent linear connection state and debug layer. |
| M5 | Tile-aware anchor selection | Not started | Unassigned | 2026-05-15 | Add anchor candidates, selected anchor plans, and split/failure planning. |
| M6 | Tile-feasible routing | Not started | Unassigned | 2026-05-15 | Integrate raster-spine cost, tile feasibility, and linear state into route search. |
| M7 | Priority/stage linear processing | Not started | Unassigned | 2026-05-15 | Route linear features by priority/process stages with explicit interaction policy. |
| M8 | Tile assignment validator/finalizer | Not started | Unassigned | 2026-05-15 | Make tile assignment consume connection state and fail hard on impossible directions. |
| M9 | Output-level road invariants and visual debug | Not started | Unassigned | 2026-05-15 | Add final-row road validation and small-fixture debug rendering. |
| M10 | Building and area revalidation after road repair | Not started | Unassigned | 2026-05-15 | Revalidate building, area, occupancy, and cross-feature behavior after roads stabilize. |
| M11 | Pipeline orchestration consolidation | Not started | Unassigned | 2026-05-15 | Ensure `ExtractionPipeline` owns orchestration and `OSMProcessor` delegates. |
| M12 | Performance pass after correctness | Not started | Unassigned | 2026-05-15 | Optimize only after correctness fixtures and semantic validators pass. |

## Decisions Made

| Date | Decision | Rationale | Approved by |
| --- | --- | --- | --- |
| 2026-05-15 | Use `docs/plans/osm_network_recovery_tdd_plan.md` as the authoritative source plan for network recovery planning. | The user explicitly said the document is authoritative. | User |
| 2026-05-15 | Use `docs/plans/osm_network_recovery_contract.md` as the authoritative recovery contract. | The user requested a contract whose content must not be changed without approval. | User |
| 2026-05-15 | Use `docs/plans/osm_network_recovery_exec_plan.md` as the milestone execution plan. | The user requested an ExecPlan-style workflow with decision-complete milestone handoffs. | User |
| 2026-05-15 | Use `docs/plans/osm_network_recovery_status.md` as the living status ledger. | The user requested milestone progress, decisions, debt, validation results, and blockers. | User |
| 2026-05-15 | Preserve `OSMProcessor` as the public compatibility adapter. | This is required by the existing refactor contract and the recovery plan keeps the typed architecture. | User |
| 2026-05-15 | Put new road recovery fixtures under `tests/cm_terrain_extractor/osm_extraction/fixtures/network_recovery/`. | This keeps network recovery fixtures separate from older general OSM extraction fixtures. | User |
| 2026-05-15 | Keep the current implementation facts in the exec plan as actionable context, not as contract overrides. | The source plan is authoritative, while implementation facts help future agents locate the first changes. | Codex |
| 2026-05-15 | Implement M0 as test harness and fixtures only, with no road algorithm changes or debug production-code changes. | Existing `OSMProcessor` state and `get_debug_layers()` expose enough topology, routing, placement, row, and debug-layer information for the semantic harness. | Codex |
| 2026-05-15 | Track `four_way_crossing` as an explicit xfailed red fixture. | The fixture documents the expected future outcome: one connected, legal 4-way intersection after tile-feasible intersection and persistent linear-state milestones. | Codex |

## Contract Changes Requested or Approved

| Date | Request | Status | Resolution |
| --- | --- | --- | --- |
| 2026-05-15 | Initial OSM network recovery contract document created. | Approved by request | Contract is authoritative once added. |

No further contract changes have been requested or approved.

## Technical Debt Ledger

| Date | Milestone | Debt | Reason | Paydown target |
| --- | --- | --- | --- | --- |
| 2026-05-15 | Planning docs | No production-code debt introduced. | This setup creates planning documents only. | Not applicable. |
| 2026-05-15 | Current code | `RouteRecord` currently exposes ambiguous `cells` alongside `nodes`. | The source plan identifies route node versus road-tile-cell ambiguity as the most concrete likely bug. | Milestone M1. |
| 2026-05-15 | Current code | Router currently records route cells directly from route nodes. | This can blur lattice-node and CM-tile-cell semantics. | Milestone M1. |
| 2026-05-15 | Current code | Raster-spine and line-support diagnostics are not yet first-class recovery artifacts. | Endpoint-only and equal-length route ambiguity needs source-line support. | Milestone M2. |
| 2026-05-15 | Current code | `CompiledTileCatalog` lacks the full pre-assignment feasibility oracle required by the recovery contract. | Routing and anchor selection must reject impossible direction sets before final tile assignment. | Milestone M3. |
| 2026-05-15 | Current code | Accepted routes do not yet update an explicit persistent `LinearNetworkState`. | The old graph kept hidden connection state; the typed path needs an inspectable equivalent. | Milestone M4. |
| 2026-05-15 | Current code | Anchor selection is still too close to nearest-cell snapping. | Legal CM intersections require tile-aware candidate scoring and split/failure planning. | Milestone M5. |
| 2026-05-15 | M0 | The network recovery helper uses a file-level ruff import-order/E402 waiver because tests extend `sys.path` before importing app modules. | Existing local test pattern imports app modules this way; changing global test import setup is outside M0. | Consider a shared test import/conftest cleanup outside network recovery milestones. |

## Test and Validation Results

| Date | Milestone | Command or check | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-15 | Planning docs | `Test-Path docs\plans\osm_network_recovery_contract.md; Test-Path docs\plans\osm_network_recovery_exec_plan.md; Test-Path docs\plans\osm_network_recovery_status.md` | Passed | Returned `True`, `True`, `True`. |
| 2026-05-15 | Planning docs | `Select-String -SimpleMatch -Path docs\plans\osm_network_recovery_*.md -Pattern "authoritative","contract","Milestone M0","Milestone M12","Dependency Graph","Blockers","Validation","LinearNetworkState","RasterSpine","CompiledTileCatalog"` | Passed | Required terms were found across the recovery planning documents. |
| 2026-05-15 | M0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -q` | Failed as expected before helper implementation | Collection failed because `tests.cm_terrain_extractor.osm_extraction.network_recovery_helpers` did not exist yet. This was the intended red harness API test. |
| 2026-05-15 | M0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\pytest.exe tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py -v` | Passed | Final result: 14 passed, 1 xfailed. Xfail is `test_four_way_crossing_has_a_single_legal_four_way_intersection`. Pytest emitted dependency deprecation warnings and a `PytestCacheWarning` because `.pytest_cache` nodeids could not be written in the sandbox. |
| 2026-05-15 | M0 | `C:\Users\der_b\miniconda3\envs\cm_terrain\Scripts\ruff.exe check tests\cm_terrain_extractor\osm_extraction\network_recovery_helpers.py tests\cm_terrain_extractor\osm_extraction\test_network_recovery_harness.py --no-cache` | Passed | `ruff --fix` was attempted once and hit Windows access denied on the helper file; the style issue was then fixed manually and the final check passed. |

## Blockers and Residual Risk

| Date | Milestone | Blocker or risk | Impact | Next action |
| --- | --- | --- | --- | --- |
| 2026-05-15 | Planning docs | None for document creation. | The requested planning docs were created without code changes. | Document validation checks passed. |
| 2026-05-15 | M1 | Existing tests and debug exports may depend on `RouteRecord.cells`. | Refactoring route semantics could touch routing, tile assignment, debug export, and output rows together. | Start M1 with failing unit tests and migrate callers deliberately. |
| 2026-05-15 | M3-M6 | Active profile catalogs may lack all direction sets needed by road fixtures. | Some fixtures may require split-intersection behavior instead of simple single-tile intersections. | Record catalog gaps and implement structured failure or split plans. |
| 2026-05-15 | M7 | Process-pair interaction policy may be under-specified in config. | Road/stream/fence behavior could require a product decision. | Use conservative defaults and request user approval if contract semantics would change. |
| 2026-05-15 | M12 | Sandbox performance results may be CPU-only or otherwise constrained. | Performance numbers could be misleading. | Record environment context and do not treat restricted results as authoritative. |
| 2026-05-15 | M0 | `four_way_crossing` is xfailed, not fixed. | The harness now preserves a reproducible red scenario for later milestones; it does not claim current four-way output is semantically legal. | M4-M8 should turn this into a passing semantic road invariant after linear state, anchor planning, routing, and tile assignment are repaired. |
| 2026-05-15 | M0 | The helper's road graph reconstructs only road-tile config rows from post-process rows and checks orthogonal connectivity plus catalog-backed direction sets for degree 2+. | This is sufficient for M0 diagnostics but not the final authoritative output validator promised in M9. | Keep M9 as the owner of strict final-row validation and visual debug rendering. |
| 2026-05-15 | M0 | Pytest passed despite `.pytest_cache` write warnings in the sandbox. | Cache warnings do not affect harness correctness, but repeated runs may not benefit from pytest cache state. | No code action needed; record if cache writes become hard failures in future validation. |

## Status Update Procedure

After every milestone:

- Update the milestone row with status, owner, date, and short notes.
- Add decisions to `Decisions Made`, including rationale and approver.
- Add contract-change requests to `Contract Changes Requested or Approved`; do not edit the contract without user approval.
- Add technical debt with a reason and paydown target.
- Add exact validation commands and results, including failures due to invocation, sandbox, or environment.
- Add blockers and residual risk that future agents must know before continuing.
- Keep this ledger factual. Do not use it to silently change the authoritative contract.
