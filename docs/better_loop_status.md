# Better Loop Status

Branch: `dev/better_loop`

## What Is Working

- Accepted far false loops are currently zero on the expanded smoke set after
  splitting heading labels:
  - KITTI360 drive0005 900 frames: `loop_accepted_far_false_loop=0`
  - M2DGR hall05 600 frames: `loop_accepted_far_false_loop=0`
- Near opposite/cross-heading revisits are now tracked separately from far false
  loops. Treating them as hard false positives was misleading.
- If a query has a selectable position-level loop candidate, selection is not
  the bottleneck:
  - drive0005: `query_position_selection_failure_count=0`
  - hall05: `query_position_selection_failure_count=0`

## Current Bottleneck

The bottleneck is loop verification, not candidate retrieval or best-loop
selection.

Expanded matrix evidence (`/tmp/n3mapping_verification_evidence_v1_matrix_current_20260625`):

These runs were regenerated with the current `dev/better_loop` binary after
extracting `LoopVerifier`; accepted loop counts and far-false counts stayed
unchanged from the baseline artifacts.

| run | GT position opportunity queries | queries with any position candidate | selectable position queries | missed position candidates |
| --- | ---: | ---: | ---: | ---: |
| drive0005_900 | 204 | 30 | 12 | 18 |
| hall05_600 | 198 | 32 | 29 | 3 |

Pair-level rejected position candidates:

| run | verification rejects | not selected |
| --- | ---: | ---: |
| drive0005_900 | 48 | 2 |
| hall05_600 | 33 | 8 |

Reject-reason breakdown from the current logged candidates:

| run | icp_not_converged | fitness_threshold | inlier_threshold | geometry_gate | loop_referee | not_selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| drive0005_900 | 392 | 1823 | 126 | 3 | 55 | 29 |
| hall05_600 | 115 | 0 | 0 | 1 | 127 | 95 |

This breakdown is pair-level over logged candidates, not a direct true-loop
recall number. It is useful because it shows different dominant failure modes:
KITTI360 is mostly failing before or inside ICP quality gates, while hall05 is
dominated by non-convergence and LoopReferee rejection.

## Verification Evidence v1

The current development direction is evidence-only. Runtime accept/reject
behavior must stay unchanged until a verifier signal separates true recoverable
loops from repeated-structure false positives on both outdoor and indoor smoke
runs.

Implemented evidence contract:

- Replace ambiguous `candidate_residual` diagnostics with explicit transforms:
  - `T_pred_match_query`
  - `T_icp_correction_match`
  - `T_measured_match_query`
- Enforce the frame invariant:

```text
T_measured_match_query = T_icp_correction_match * T_pred_match_query
```

- Export registration termination evidence:
  - `icp_iterations`
  - `icp_optimizer_error`
  - `icp_termination`
- Add global query-level position opportunity metrics so queries with no logged
  candidate are not silently removed from recall accounting.
- Add reject reason summaries by source and heading class in the analyzer.

The important interpretation change: `converged=false` is no longer treated as
proof that a loop is geometrically false. It is only one evidence field. A
future verifier must explain whether the geometry is stable, symmetric, and
unique before any runtime rescue path is considered.

## Rejected Experiment

Tested allowing `non-converged + high inlier` ICP results into LoopReferee.

Result:

- KITTI360 drive0005_700 improved locally.
- M2DGR hall05_600 introduced two real far false loops:
  - `60 -> 1`, GT distance 5.19 m
  - `65 -> 2`, GT distance 9.88 m

Verdict: do not use `non-converged + high inlier` as an accept path. Indoor
repeated structure makes it unsafe.

## Next Step

Do not tune LoopReferee weights yet. Do not add a runtime rescue path yet.

The next useful change is a shadow-only symmetric verifier:

1. Use equal-length causal submaps for query and match.
2. Evaluate forward and reverse registration with the same final transform
   scoring function.
3. Record symmetric support, transform cycle error, and coarse/fine stability.
4. Keep the runtime committed loops unchanged.

Only if that shadow evidence separates KITTI true candidates from hall05
repeated-structure false candidates should it become a runtime decision input.
