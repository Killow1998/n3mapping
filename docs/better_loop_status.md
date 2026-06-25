# Better Loop Status

Branch: `dev/better_loop`

## 2026-06-25 Segment Referee Trial

Current unmerged development state has moved beyond shadow-only evidence:

- `LoopReferee` now uses segment support/consensus as runtime commit evidence.
- Supported but segment-inconsistent loops are rejected.
- Accepted loops are committed as `vertical_neutral` measurements:
  - XY/yaw come from ICP.
  - Z/roll/pitch are neutralized to the current graph prediction.
- Two-frame toy loops without segment support are intentionally rejected.

This is not proven as the final v2 model. It is a testable behavior checkpoint.

### Humble Regression

Command:

```bash
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --packages-select n3mapping --parallel-workers 1 \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ROS_LOG_DIR=/tmp/n3mapping_ros_log colcon test --packages-select n3mapping \
  --event-handlers console_direct+ --parallel-workers 1
colcon test-result --test-result-base build/n3mapping --verbose
```

Result:

```text
274 tests, 0 errors, 0 failures, 0 skipped
```

### M2DGR hall_05 stride5

Artifact:

```text
/tmp/n3mapping_segment_topology_hall05_stride5_20260625
/tmp/n3mapping_segment_topology_matrix_hall05_20260625
```

Compared with the previous segment diagnostic run:

| metric | previous | segment-referee trial |
| --- | ---: | ---: |
| accepted_loop_count | 29 | 18 |
| loop_precision | 0.9655 | 1.0 |
| loop_position_precision | 1.0 | 1.0 |
| accepted_false_loop | 0 | 0 |
| accepted_true_loop | 28 | 18 |
| accepted_opposite_heading_loop | 1 | 0 |
| trajectory_translation_p95_m | 0.0102 | 0.0167 |
| trajectory_z_p95_m | 0.0000045 | 0.0000031 |
| high_residual_z_after_count | 0 | 0 |

Interpretation:

- Positive: accepted loops are cleaner; the previous opposite-heading accepted
  case disappeared, and all accepted loops have segment recommendation
  `consistent`.
- Negative: the model is more conservative and drops 10 accepted true loops.
  The trajectory remains good, but p95 translation is slightly worse.

### M2DGR gate_02 stride5

Artifact:

```text
/tmp/n3mapping_segment_topology_gate02_stride5_20260625
/tmp/n3mapping_segment_topology_matrix_gate02_20260625
```

Result:

| metric | value |
| --- | ---: |
| accepted_loop_count | 1 |
| loop_precision | 0.0 |
| loop_position_precision | 1.0 |
| accepted_false_loop | 0 |
| accepted_cross_heading_loop | 1 |
| trajectory_translation_p95_m | 0.0260 |
| trajectory_z_p95_m | 0.0031 |
| high_residual_z_after_count | 0 |

The single accepted loop is:

```text
query=150 match=64
GT translation=1.80 m
GT yaw diff=45.98 deg
heading_class=cross_heading
segment_consensus_ratio=1.0
segment_translation_median=0.934 m
segment_yaw_median=0.245 rad
edge_mode=vertical_neutral
```

Interpretation:

- This is position-consistent but fails the current strict heading-based
  `loop_precision` definition by a small margin.
- For map consistency, accepting a cross-heading revisit can still be useful if
  the geometry and graph remain consistent.
- For the current matrix gate, this is a failure. The model needs either:
  - an explicit decision that position-level loops are valid loop closures, with
    matrix gates updated accordingly, or
  - a runtime heading-class/observability signal that rejects cross-heading
    loops without also discarding useful opposite-direction revisits.

### Current Verdict

Segment-referee commit is promising but not enough:

- It improves accepted-loop cleanliness on `hall_05`.
- It is too conservative on recall.
- It exposes a mismatch between map-closure semantics and strict heading-based
  loop precision on `gate_02`.

Do not declare the loop subsystem solved from this checkpoint.

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
