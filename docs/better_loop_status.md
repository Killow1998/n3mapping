# Better Loop Status

Branch: `dev/better_loop`

## 2026-06-27 Consensus Referee Gate Checkpoint

Runtime change:

- `LoopConsensusVerifier` now runs for selected `best_loops` before graph commit,
  not only for debug output.
- The commit path rejects two narrow high-risk cases:
  - `consensus_insufficient_planar`: no estimator support while vertical
    hypothesis says the loop is only planar.
  - `consensus_unstable_large_delta`: unstable consensus and consensus-vs-ICP
    translation delta >= 5 m.
- KITTI360/M2DGR eval tools now support `--start_index` so broad tests can use
  GT-selected loop-opportunity windows instead of always starting at frame 0.

Regression artifacts:

```text
/tmp/n3mapping_consensus_referee_matrix_20260627
/tmp/n3mapping_combined_broad_matrix_20260627
/tmp/n3mapping_opportunity_matrix_20260627
```

Key matrix results:

| run | loops | precision | trans p95 m | XY p95 m | Z p95 m | high-Z after | max Z after m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI360 drive0000 start0 900/s5 | 4 | 0.750 | 2.445 | 2.429 | 1.726 | 2 | 0.983 |
| KITTI360 drive0000 start400 600/s10 | 2 | 0.500 | 2.181 | 2.180 | 0.123 | 0 | 0.407 |
| KITTI360 drive0005 900/s5 | 3 | 1.000 | 0.750 | 0.729 | 0.247 | 1 | 0.792 |
| M2DGR hall05 600/s5 | 14 | 0.929 | 0.015 | 0.015 | 0.001 | 0 | 0.162 |
| M2DGR gate02 600/s5 | 0 | n/a | ~0 | ~0 | ~0 | 0 | 0 |

Compared with the previous drive0000 start0 900/s5 run:

```text
accepted loops: 6 -> 4
precision: 0.50 -> 0.75
translation p95: 3.39 m -> 2.45 m
Z p95: 1.93 m -> 1.73 m
```

Current verdict:

```text
Keep the consensus referee gate. It removes part of the descriptor+ICP
straight-through failure without regressing drive0005 or M2DGR hall/gate.
It is not enough: drive0000 start400 still has a false accepted loop, and many
KITTI360 GT-opportunity windows have true loop candidates but zero accepted
loops. The next improvement should target recall/correspondence quality across
GT-selected windows, not another narrow Z or graph residual threshold.
```

## 2026-06-27 Consensus-Estimator Graph-Trial Checkpoint

Added a second shadow graph-trial path:

```text
central ICP measurement -> existing graph_trial_*
consensus-estimated measurement -> consensus_estimator_trial_*
```

Runtime behavior is still unchanged:

- the committed edge still uses the current runtime loop measurement;
- consensus-estimator trial output is debug/eval only;
- no consensus gate or edge-mode change.

### Fields

```text
consensus_estimator_trial_success
consensus_estimator_trial_residual_x_after
consensus_estimator_trial_residual_y_after
consensus_estimator_trial_residual_z_after
consensus_estimator_trial_residual_roll_after
consensus_estimator_trial_residual_pitch_after
consensus_estimator_trial_residual_yaw_after
consensus_estimator_trial_residual_translation_norm_after
consensus_estimator_trial_residual_rotation_norm_after
consensus_estimator_trial_consistency_score
consensus_estimator_trial_recommendation
```

### Regression

Focused Humble regression:

```text
260 tests, 0 errors, 0 failures, 0 skipped
```

### Smoke Matrix

Artifact:

```text
/tmp/n3mapping_consensus_estimator_trial_smoke_20260627
```

| run | accepted loops | precision | trans p95 m | XY p95 m | Z p95 m | high-Z after | estimator trial candidates | estimator trial success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI360 drive0005 330 stride5 | 9 | 1.000 | 0.308 | 0.223 | 0.151 | 5 | 8 | 8 |
| M2DGR hall05 300 stride5 | 7 | 0.857 | 0.014 | 0.014 | 0.0001 | 0 | 4 | 4 |

KITTI separation signal:

| signal | bad-Z-after | corrected-Z |
| --- | ---: | ---: |
| consensus_estimator_trial_residual_z_after_mean | 0.736 | 0.136 |

### Interpretation

- Positive: unlike `estimator_valid`, the graph trial residual from the
  consensus-estimated measurement separates bad-Z-after and corrected-Z cases
  on the 330-frame KITTI smoke.
- Positive: this signal is closer to the final goal than raw heightmap,
  vertical hypothesis, or component-wise consensus, because it evaluates whether
  the alternative measurement can be absorbed by the graph.
- Negative: this is still a short smoke. It is not enough to authorize runtime
  gating or measurement replacement.

Current verdict:

```text
Keep consensus-estimator graph trial as the next behavior proof point.
Before changing runtime behavior, rerun this on longer KITTI360 900-frame and
M2DGR hall/gate matrices. If the separation remains, the next behavior trial
should commit the consensus-estimated measurement only when its graph-trial Z
residual is lower than the central ICP graph-trial residual and precision stays
safe.
```

## 2026-06-27 Robust SE(3) LoopConsensusEstimator Checkpoint

Replaced the component-wise consensus estimator with a robust SE(3) estimator:

1. collect neighbor-pair central `T_match_query` estimates;
2. select a medoid transform using normalized SE(3) residuals;
3. keep inliers around the medoid;
4. average inlier translation and quaternion orientation;
5. report the same `consensus_estimator_*` fields as shadow evidence.

Runtime behavior is still unchanged:

- no consensus gate;
- no edge-mode change;
- no threshold tuning;
- no GT in runtime.

### Regression

Focused Humble regression:

```bash
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --packages-select n3mapping --symlink-install \
  --parallel-workers 1 --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ROS_LOG_DIR=/tmp/n3mapping_ros_log colcon test --packages-select n3mapping \
  --event-handlers console_direct+ --parallel-workers 1 \
  --ctest-args -R 'test_loop_consensus_verifier|test_loop_debug_logger|test_kitti360_eval'
colcon test-result --test-result-base build/n3mapping --verbose
```

Result:

```text
260 tests, 0 errors, 0 failures, 0 skipped
```

### Smoke Matrix

Artifact:

```text
/tmp/n3mapping_consensus_estimator_se3_smoke_20260627
```

Runs:

```text
drive0005_330_stride5
hall05_300_stride5
```

| run | accepted loops | precision | trans p95 m | XY p95 m | Z p95 m | high-Z after | max Z after m | estimator candidates | estimator valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI360 drive0005 330 stride5 | 9 | 1.000 | 0.308 | 0.223 | 0.151 | 5 | 1.687 | 9 | 3 |
| M2DGR hall05 300 stride5 | 7 | 0.857 | 0.014 | 0.014 | 0.0001 | 0 | 0.073 | 6 | 2 |

Compared to the component-wise estimator:

```text
KITTI estimator_valid_count: 0 -> 3
KITTI behavior metrics: unchanged
M2DGR accepted loops: 8 -> 7 on this short smoke
```

KITTI separation signals:

| signal | bad-Z-after | corrected-Z |
| --- | ---: | ---: |
| consensus_estimator_z_mad_mean | 0.465 | 0.822 |
| consensus_estimator_measurement_delta_translation_mean | 1.268 | 0.793 |

### Interpretation

- Positive: robust SE(3) medoid/inlier estimation produces stable estimates on
  some real KITTI loops, where the component-wise estimator produced none.
- Positive: behavior remains unchanged, so this is a clean diagnostic upgrade.
- Negative: two bad-Z-after KITTI loops still receive
  `stable_consensus_measurement`; therefore this estimator is not ready to
  become a commit gate.
- Negative: z-MAD still does not directly separate bad-Z-after from corrected-Z
  cases.
- Useful signal: central-vs-consensus translation delta is higher for bad-Z
  than corrected-Z on this smoke, but the margin is not yet enough for an
  algorithmic rule.

Current verdict:

```text
Keep robust SE(3) LoopConsensusEstimator as shadow evidence.
Do not connect estimator_valid to graph commit.
The next credible behavior change should use the consensus estimate as an
alternative measurement in a shadow graph trial, not as an accept/reject gate.
```

## 2026-06-27 LoopConsensusEstimator Shadow Checkpoint

Implemented a shadow-only consensus measurement estimator on top of
`LoopConsensusVerifier`.

The runtime behavior is intentionally unchanged:

- no graph commit gate from consensus estimator;
- no XY-yaw / vertical-neutral behavior revival;
- no threshold tuning;
- no GT in runtime.

### What Changed

Each converged neighbor-pair ICP refinement now back-solves a central
`T_match_query` estimate:

```text
T_m_q_est = inverse(T_mi_m) * T_mi_qi_icp * inverse(T_q_qi)
```

The estimator then reports median/MAD evidence:

```text
consensus_estimator_valid
consensus_estimator_pair_count
consensus_estimator_inlier_count
consensus_estimator_inlier_ratio
consensus_estimator_translation_median
consensus_estimator_z_median
consensus_estimator_yaw_median
consensus_estimator_translation_mad
consensus_estimator_z_mad
consensus_estimator_yaw_mad
consensus_estimator_measurement_delta_translation
consensus_estimator_measurement_delta_rotation
consensus_estimator_recommendation
```

These fields are written through:

```text
loop_debug.jsonl
accepted_loops.csv
loop_candidates_labeled.csv
matrix_summary.csv/json
```

### Regression

Focused Humble regression:

```bash
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --packages-select n3mapping --symlink-install \
  --parallel-workers 1 --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ROS_LOG_DIR=/tmp/n3mapping_ros_log colcon test --packages-select n3mapping \
  --event-handlers console_direct+ --parallel-workers 1 \
  --ctest-args -R 'test_loop_consensus_verifier|test_loop_debug_logger|test_kitti360_eval'
colcon test-result --test-result-base build/n3mapping --verbose
```

Result:

```text
260 tests, 0 errors, 0 failures, 0 skipped
```

### Smoke Matrix

Artifact:

```text
/tmp/n3mapping_consensus_estimator_smoke_20260627
```

Runs:

```text
drive0005_330_stride5
hall05_300_stride5
```

| run | accepted loops | precision | trans p95 m | XY p95 m | Z p95 m | high-Z after | max Z after m | estimator candidates | estimator valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI360 drive0005 330 stride5 | 9 | 1.000 | 0.308 | 0.223 | 0.151 | 5 | 1.687 | 9 | 0 |
| M2DGR hall05 300 stride5 | 8 | 0.875 | 0.014 | 0.014 | 0.0001 | 0 | 0.060 | 7 | 2 |

Estimator separation signals:

| run | bad-Z estimator z MAD mean | bad-Z estimator central delta mean | corrected-Z estimator z MAD mean | corrected-Z estimator central delta mean |
| --- | ---: | ---: | ---: | ---: |
| KITTI360 drive0005 330 stride5 | 0.353 | 1.319 | 1.039 | 0.898 |

### Interpretation

- Positive: the estimator is now a real measurement-level signal, not just a
  consistency label.
- Positive: it produces non-empty evidence on both KITTI360 and M2DGR.
- Negative: KITTI360 `estimator_valid_count=0`; this cannot be used as a
  runtime gate yet.
- Negative: the current z-MAD signal is inverted for this smoke: bad-Z-after
  loops have lower z-MAD than corrected-Z loops. Directly gating on z-MAD would
  be another hack.
- Mixed: `consensus_estimator_measurement_delta_translation` is higher for
  bad-Z-after loops than corrected-Z loops, but the margin is not enough by
  itself to become a commit rule.

Current verdict:

```text
Keep LoopConsensusEstimator as shadow evidence.
Do not connect it to graph commit behavior yet.
Next proof point should improve the estimator from component-wise medians to a
robust SE(3) consensus estimate and compare it on longer KITTI360 + M2DGR
matrices.
```

## 2026-06-27 LoopConsensusVerifier Shadow Checkpoint

Implemented `LoopConsensusVerifier` as shadow-only evidence for the next
proof point:

```text
single loop pair -> central ICP measurement -> neighborhood correspondence
consensus diagnostics
```

The runtime behavior is intentionally unchanged:

- no graph-trial authority;
- no segment-only authority;
- no vertical/heightmap/XY-yaw gate;
- no LoopReferee weight tuning;
- no graph optimizer changes;
- no GT in runtime.

### Implementation

New files:

```text
include/n3mapping/loop_consensus_verifier.h
src/loop_consensus_verifier.cpp
test/test_loop_consensus_verifier.cpp
```

`LoopConsensusVerifier` evaluates neighbor pairs `(query+d, match+d)` for a
verified central loop measurement. For each valid neighbor pair it predicts the
relative transform:

```text
T_mi_qi_pred = T_mi_m * T_m_q * T_q_qi
```

and records the ICP refinement delta. The summary writes:

```text
consensus_shadow_decision
consensus_shadow_reason
consensus_valid_pair_count
consensus_left_support_count
consensus_right_support_count
consensus_contradiction_count
consensus_median_translation_delta
consensus_mad_translation_delta
consensus_median_rotation_delta
consensus_mad_rotation_delta
```

The first implementation evaluated consensus for every verified candidate.
That made KITTI smoke too slow and stopped around frame 301/450 in
`process_loops_start`. The corrected implementation evaluates consensus only
for the loops selected by the existing pipeline. This preserves behavior while
keeping the shadow diagnostic cost bounded.

### Regression

Focused Humble regression:

```bash
source /opt/ros/humble/setup.bash
MAKEFLAGS=-j1 colcon build --packages-select n3mapping --symlink-install \
  --parallel-workers 1 --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
ROS_LOG_DIR=/tmp/n3mapping_ros_log colcon test --packages-select n3mapping \
  --event-handlers console_direct+ --parallel-workers 1 \
  --ctest-args -R 'test_loop_consensus_verifier|test_loop_debug_logger|test_n3mapping_core'
colcon test-result --test-result-base build/n3mapping --verbose
```

Result:

```text
259 tests, 0 errors, 0 failures, 0 skipped
```

### KITTI360 Smoke

Artifact:

```text
/tmp/n3mapping_loop_consensus_shadow_selected_smoke_20260627/drive0005_330_stride5_online_support
```

Command:

```bash
ros2 run n3mapping n3mapping_kitti360_eval \
  --kitti_root /home/user/DUALoc/KITTI360 \
  --sequence 2013_05_28_drive_0005_sync \
  --mode mapping_loop \
  --calib_mode auto \
  --max_frames 330 \
  --stride 5 \
  --output /tmp/n3mapping_loop_consensus_shadow_selected_smoke_20260627/drive0005_330_stride5_online_support
```

Result:

```text
frames=330
keyframes=329
accepted_loop_count=9
accepted_false_loop=0
loop_precision=1.0
trajectory_translation_p95_m=0.307648823
trajectory_xy_p95_m=0.222619781
trajectory_z_p95_m=0.151083766
optimization_high_residual_z_after_count=5
optimization_max_residual_z_after_m=1.68667993
consensus_candidate_count=9
consensus_commit_count=2
consensus_defer_count=7
consensus_reject_count=0
bad_z_after_consensus_commit_count=1
```

Interpretation:

- Positive: consensus shadow now produces non-trivial evidence instead of all
  `defer`.
- Positive: the smoke crosses the previous frame-301 stall and completes.
- Positive: accepted false loops remain zero in this smoke.
- Negative: one bad-Z-after loop still receives `consensus_shadow_decision =
  commit`; therefore consensus v1 is not ready to become a runtime gate.
- Negative: online mapping naturally has no future `query+d` support, so the
  verifier must treat enough past-side support as usable shadow evidence.

Current verdict:

```text
Keep LoopConsensusVerifier as shadow evidence.
Do not connect it to graph commit behavior yet.
Next proof point should compare this consensus signal across longer KITTI360
and M2DGR hall/gate runs before any runtime behavior change.
```

## 2026-06-25 Risk-Aware Referee + Full-6DoF Checkpoint

Latest local development state replaces the over-strict segment energy gate
with an explicit risk-referee model:

- descriptor-backed verified geometry may commit even when segment evidence is
  limited;
- spatial-only candidates cannot commit without descriptor support;
- large predicted motion with weak segment support is rejected;
- yaw-flip ICP corrections with large segment disagreement are rejected;
- accepted loops keep the normal full-6DoF measurement from
  `LoopClosureManager::applyEdgeModel()`.

This is a better checkpoint than the two rejected alternatives below, but it is
still not the final v2 loop model.

### Regression

Focused Humble tests:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select n3mapping --symlink-install --parallel-workers 1 \
  --cmake-args -DBUILD_TESTING=ON
ROS_LOG_DIR=/tmp/n3mapping_ros_log colcon test --packages-select n3mapping \
  --event-handlers console_direct+ --parallel-workers 1 \
  --ctest-args -R 'test_loop_closure_manager|test_n3mapping_core'
colcon test-result --test-result-base build/n3mapping --verbose
```

Result:

```text
252 tests, 0 errors, 0 failures, 0 skipped
```

### Matrix

Artifact:

```text
/tmp/n3mapping_risk_full6dof_matrix_20260625_230512
```

Compared to the pre-referee evidence baseline
`/tmp/n3mapping_verification_evidence_v1_matrix_current_20260625`:

| run | metric | baseline | risk full-6DoF |
| --- | --- | ---: | ---: |
| KITTI360 drive0005 900 stride5 | accepted_loop_count | 12 | 11 |
| KITTI360 drive0005 900 stride5 | loop_precision | 0.833 | 0.909 |
| KITTI360 drive0005 900 stride5 | accepted_false_loop | 0 | 0 |
| KITTI360 drive0005 900 stride5 | accepted_opposite_heading_loop | 2 | 1 |
| KITTI360 drive0005 900 stride5 | trajectory_translation_p95_m | 12.041 | 3.537 |
| KITTI360 drive0005 900 stride5 | trajectory_xy_p95_m | 4.854 | 1.550 |
| KITTI360 drive0005 900 stride5 | trajectory_z_p95_m | 1.932 | 3.179 |
| KITTI360 drive0005 900 stride5 | optimization_high_residual_z_after_count | 1 | 7 |
| M2DGR hall05 600 stride5 | accepted_loop_count | 29 | 21 |
| M2DGR hall05 600 stride5 | loop_precision | 0.966 | 0.952 |
| M2DGR hall05 600 stride5 | accepted_false_loop | 0 | 0 |
| M2DGR hall05 600 stride5 | trajectory_translation_p95_m | 0.010 | 0.048 |
| M2DGR hall05 600 stride5 | trajectory_xy_p95_m | 0.010 | 0.048 |

Interpretation:

- Positive: the risk referee is the first checkpoint that improves KITTI360
  loop precision and overall/XY trajectory consistency at the same time.
- Positive: the spatial-only false loop failure mode from the segment-veto
  trial is removed.
- Positive: M2DGR remains above the 0.95 loop-precision line and keeps
  far-false loops at zero.
- Negative: full-6DoF restores trajectory correction but reintroduces high
  per-loop Z residuals on KITTI360.
- Negative: M2DGR recall and centimeter-level trajectory error are still worse
  than the baseline.

Current verdict:

```text
Keep the risk-aware referee direction.
Do not declare the loop subsystem solved.
Do not continue direct evidence-to-gate tuning:
  vertical hypothesis, heightmap, segment consistency, graph trial, and XY/yaw
  edge variants have all failed as standalone behavior rules.
Next proof point should move above single-pair ICP:
  multi-submap / segment-level correspondence consensus should decide whether a
  loop measurement is graph-worthy.
```

### Rejected Alternatives

The following behaviors were tested and should not be revived without new
matrix evidence:

1. `segment-energy` as the main accept threshold.
   - KITTI accepted loops dropped to 6 and trajectory worsened.
   - M2DGR accepted loops dropped to 15.
2. `segment-veto` that accepts any verified geometry unless segment is clearly
   inconsistent.
   - KITTI accepted loops rose to 17 but precision fell to 0.588 and trajectory
     degraded badly.
   - M2DGR precision fell to 0.76.
3. all accepted loops forced to `vertical_neutral`.
   - This reduced high-Z residual count, but did not repair trajectory Z and
     removed useful full-6DoF correction.
4. conservative hybrid neutralization using heightmap + vertical-hypothesis
   disagreement.
   - Artifact: `/tmp/n3mapping_hybrid_edge_matrix_20260626`.
   - KITTI360 high-Z-after improved from 7 to 3 and Z p95 improved from
     3.179m to 1.953m versus the risk full-6DoF checkpoint.
   - KITTI360 XY p95 regressed from 1.550m to 4.351m and translation p95
     regressed from 3.537m to 4.770m.
   - M2DGR hall05 stayed safe on far-false loops but accepted loops dropped
     from 21 to 19 and trajectory p95 regressed from 0.048m to 0.053m.
   - Verdict: do not commit this behavior. The evidence is useful, but direct
     vertical-neutral conversion still removes useful full-6DoF correction.
5. narrow opposite-heading place-loop neutralization.
   - Artifact: `/tmp/n3mapping_place_neutral_matrix_20260626`.
   - Behavior tested: same-heading pose loops stayed full-6DoF; only
     runtime opposite-heading loops with segment support used
     `vertical_neutral`.
   - KITTI360 accepted loop count and false-loop count stayed unchanged, but
     trajectory p95 regressed:
     - translation p95: `3.537m -> 3.992m`
     - XY p95: `1.550m -> 1.695m`
     - Z p95: `3.179m -> 3.615m`
   - M2DGR hall05 improved in this smoke (`21 -> 26` accepted loops,
     p95 translation `0.048m -> 0.044m`), but the outdoor regression violates
     the matrix gate.
   - Verdict: do not commit this behavior. Even narrow place-loop
     neutralization is not a reliable global-map improvement yet.
6. configured loop-noise information when ICP information is disabled.
   - Artifact: `/tmp/n3mapping_loop_noise_matrix_20260626`.
   - Behavior tested: when `loop_use_icp_information=false`, replace the
     identity loop information matrix with one derived from
     `loop_noise_position` / `loop_noise_rotation`.
   - KITTI360 regressed sharply:
     - translation p95: `3.537m -> 7.133m`
     - XY p95: `1.550m -> 6.151m`
     - Z p95: `3.179m -> 3.612m`
     - high-Z-after stayed at `7`
   - M2DGR hall05 also regressed despite accepting more loops:
     - accepted loops: `21 -> 25`
     - pose precision: `0.952 -> 0.960`
     - translation p95: `0.048m -> 0.056m`
   - Verdict: do not commit this behavior. Strengthening loop edges through
     configured noise also strengthens bad KITTI loop deformation.
7. correlated loop burst suppression by query segment cooldown.
   - Artifact: `/tmp/n3mapping_correlated_loop_suppression_20260626`.
   - Behavior tested: after one committed loop, suppress additional loop
     commits whose query id falls within the existing descriptor exclusion
     segment length. The intention was to avoid treating many correlated
     revisit edges as independent graph constraints.
   - KITTI360 accepted loops dropped from `11 -> 2`; this reduced high-Z-after
     from `7 -> 1`, but global XY/translation regressed badly:
     - translation p95: `3.537m -> 5.815m`
     - XY p95: `1.550m -> 5.518m`
     - Z p95: `3.179m -> 1.837m`
   - M2DGR hall05 improved in this smoke (`21 -> 3` accepted loops,
     p95 translation `0.048m -> 0.012m`), but the outdoor regression violates
     the matrix gate.
   - Verdict: do not commit this behavior. The problem is not simply loop
     density; KITTI needs enough true loop constraints to correct XY/global
     drift, while still preventing bad vertical measurements from deforming Z.
8. loop correspondence retargeting to nearest pose in the matched segment.
   - Artifact: `/tmp/n3mapping_loop_retarget_experiment_20260626`.
   - Behavior tested: preserve the ICP-implied measured world pose, but anchor
     the graph edge to the nearest keyframe inside the original matched
     segment instead of the descriptor-selected `match_id`.
   - KITTI360 accepted loops dropped from `11 -> 9` and high-Z-after improved
     from `7 -> 5`, but trajectory consistency regressed sharply:
     - translation p95: `3.537m -> 6.227m`
     - XY p95: `1.550m -> 4.954m`
     - Z p95: `3.179m -> 3.772m`
   - M2DGR hall05 improved in this smoke (`21 -> 10` accepted loops,
     p95 translation `0.048m -> 0.016m`), but the outdoor regression violates
     the matrix gate.
   - Verdict: do not commit this behavior. Retargeting the keyframe anchor
     weakens the KITTI loop constraints needed for outdoor XY/global correction
     and does not solve vertical deformation.
9. automatic `XY_YAW` edge mode from
   `vertical_hypothesis_edge_recommendation == planar_xy_yaw`.
   - Artifact: `/tmp/n3mapping_xyyaw_clean_matrix_20260626`.
   - Behavior tested: add a graph-level XY/yaw-only loop factor and use it
     whenever the vertical-hypothesis diagnostic recommended
     `planar_xy_yaw`.
   - Positive: trajectory p95 improved on the smoke runs:
     - KITTI360 drive0005 450 stride5 translation p95 `0.813m`, XY p95
       `0.555m`, Z p95 `0.601m`.
     - M2DGR hall05 600 stride5 translation p95 `0.032m`.
   - Negative: the trigger was not safe enough:
     - KITTI360 loop precision was `0.9` with one false place loop.
     - M2DGR hall05 loop precision was `0.929` with two cross-heading loops.
   - Verdict: keep the explicit XY/yaw graph factor as a substrate for a future
     graph-authority decision, but do not let the vertical-hypothesis diagnostic
     alone select XY/yaw behavior.
10. segment-only graph authority.
    - Artifact: `/tmp/n3mapping_graph_authority_matrix_20260626`.
    - Behavior tested: split place acceptance from graph commit; allow graph
      commit only when segment evidence recommends `commit`, while keeping
      accepted-place candidates in debug output.
    - KITTI360 drive0005 450 stride5 regressed:
      - loop precision: `0.875`
      - accepted far-false loop: `1`
      - the false graph edge was `295 -> 13`, with GT distance `5.94m`.
    - M2DGR hall05 stayed clean in this smoke:
      - loop precision: `1.0`
      - false loops: `0`
      - trajectory translation p95: `0.038m`
    - Failure mode: segment evidence was too weak. A candidate with only two
      valid segment-neighbor pairs could look `consistent` while still being a
      false KITTI place loop.
    - Verdict: do not commit this behavior. Segment consistency is evidence,
      not a graph-authority rule by itself.
11. graph-trial-before-selection.
    - Artifact: `/tmp/n3mapping_graph_trial_selection_matrix_20260627`.
    - Behavior tested: compute graph-trial consistency for every graph-authority
      candidate before selecting the per-query graph edge, then prefer higher
      graph-trial consistency over local ICP fitness.
    - KITTI360 drive0005 450 stride5 regressed further:
      - loop precision: `0.75`
      - accepted far-false loops: `2`
      - trajectory XY p95 stayed near `1.35m`, but false loop rate is
        unacceptable for the v2 goal.
    - M2DGR hall05 also regressed in loop quality:
      - loop precision: `0.92`
      - cross-heading graph edges: `2`
      - false far loops remained `0`
    - Verdict: do not commit this behavior. Graph trial is a useful diagnostic,
      but it is not a reliable per-query selector or commit authority.

Current high-level conclusion from rejected experiments 4-11:

```text
The bottleneck is no longer "find one more local evidence score".
Single-pair ICP measurement is the wrong authority boundary.
The next credible direction is multi-submap correspondence consensus:
  a loop edge should be committed only after a neighborhood of query/match
  submap correspondences agrees on the same relative constraint.
```

## 2026-06-25 Earlier Segment Referee Trial

Earlier unmerged development state moved beyond shadow-only evidence:

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

## 2026-06-27 Graph-Consistency Yaw Gate

Status: kept as the first behavior-changing dev-branch referee gate.

Change:

- Compute graph-trial diagnostics for best loop candidates regardless of
  `loop_debug_enable`.
- Reject a candidate before commit when the shadow graph trial succeeds but
  leaves a near half-turn yaw residual.
- Log the rejected candidate as `reject_reason=graph_inconsistent_yaw`.

Rationale:

The 900-frame KITTI360 + 600-frame M2DGR matrix showed that all accepted
opposite/cross-heading failures had `graph_trial_residual_yaw_after` near pi.
The same gate also removed several true-but-damaging full-6DoF loops whose
shadow graph trial could not absorb rotation consistently. This is not a Z
threshold and does not use GT; it is a graph-consistency referee check.

Baseline matrix:

```text
/tmp/n3mapping_long_matrix_20260627
```

Behavior matrix:

```text
/tmp/n3mapping_graph_yaw_gate_20260627
```

Results:

| run | accepted loops | loop precision | trans p95 m | xy p95 m | z p95 m | high-Z after | max Z after m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI360 drive0005 900 baseline | 11 | 0.909 | 3.537 | 1.550 | 3.179 | 7 | 1.687 |
| KITTI360 drive0005 900 yaw-gate | 5 | 1.000 | 1.055 | 0.746 | 0.867 | 3 | 1.418 |
| M2DGR hall05 600 baseline | 27 | 0.926 | 0.0459 | 0.0458 | 0.00225 | 0 | 0.164 |
| M2DGR hall05 600 yaw-gate | 16 | 0.875 | 0.0217 | 0.0217 | 0.000087 | 0 | 0.0307 |
| M2DGR gate02 600 baseline | 1 | 0.000 | 0.0284 | 0.0274 | 0.00470 | 0 | 0.0378 |
| M2DGR gate02 600 yaw-gate | 0 | n/a | ~0 | ~0 | ~0 | 0 | 0 |

Interpretation:

- Positive: global trajectory consistency improved substantially on KITTI360 and
  M2DGR hall05; the single gate02 false loop was removed.
- Positive: the gate removes the most obvious opposite-heading / near-pi graph
  inconsistency without changing descriptor retrieval or ICP thresholds.
- Caveat: hall05 still has two accepted cross-heading loops after this gate:
  `200 -> 134` and `210 -> 8`. Their graph-trial yaw residual is small, so this
  gate cannot solve same-place / wrong-heading aliasing by itself.
- Caveat: accepted loop count drops. This is acceptable for a dev-branch
  consistency-first trial, but the next model must recover safe recall through
  better correspondence evidence rather than by weakening the gate.

Next step:

Do not tune the yaw threshold. The next behavior target should be a
correspondence-level referee for indoor aliasing:

1. Detect when ICP requires a near-pi roll/pitch/yaw measurement residual to
   explain a corridor/hall match.
2. Distinguish real reverse traversal from wrong-heading aliasing using
   symmetric local submap support or a small sequence-window consistency check.
3. Keep graph consistency as a final commit referee, not as the only decision
   layer.

## 2026-06-27 Graph-Consistency Translation Gate

Status: kept. This is the second behavior-changing graph referee gate after
the yaw gate.

Change:

- Reject a candidate before commit when the shadow graph trial succeeds but
  still leaves a large translation residual.
- Log the rejected candidate as `reject_reason=graph_inconsistent_translation`.

Rationale:

After the yaw gate, the remaining harmful cases were no longer near-pi yaw
failures. The next clean graph-level signal was residual translation after the
same shadow trial. This is still not a raw ICP fitness threshold and does not
use GT; it asks whether the proposed edge can be absorbed by the current graph
without leaving a large residual.

Behavior matrix:

```text
/tmp/n3mapping_graph_yaw_translation_gate_20260627
```

Three-stage result comparison:

| run | accepted loops | precision | trans p95 m | xy p95 m | z p95 m | high-Z after | max Z after m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KITTI360 baseline | 11 | 0.909 | 3.537 | 1.550 | 3.179 | 7 | 1.687 |
| KITTI360 yaw gate | 5 | 1.000 | 1.055 | 0.746 | 0.867 | 3 | 1.418 |
| KITTI360 yaw+translation gate | 3 | 1.000 | 0.750 | 0.729 | 0.247 | 1 | 0.792 |
| M2DGR hall05 baseline | 27 | 0.926 | 0.0459 | 0.0458 | 0.00225 | 0 | 0.164 |
| M2DGR hall05 yaw gate | 16 | 0.875 | 0.0217 | 0.0217 | 0.000087 | 0 | 0.0307 |
| M2DGR hall05 yaw+translation gate | 14 | 0.929 | 0.0203 | 0.0203 | 0.000137 | 0 | 0.0307 |
| M2DGR gate02 baseline | 1 | 0.000 | 0.0284 | 0.0274 | 0.00470 | 0 | 0.0378 |
| M2DGR gate02 yaw gate | 0 | n/a | ~0 | ~0 | ~0 | 0 | 0 |
| M2DGR gate02 yaw+translation gate | 0 | n/a | ~0 | ~0 | ~0 | 0 | 0 |

Interpretation:

- Positive: KITTI360 Z consistency improved again; `z_p95` dropped from
  `0.867 m` after yaw-only gating to `0.247 m`.
- Positive: hall05 precision recovered from `0.875` to `0.929` while trajectory
  p95 improved slightly.
- Positive: gate02 still has no accepted loop, so the previous false loop stays
  removed.
- Caveat: accepted loop count is now conservative. This is acceptable for a
  dev branch while proving the graph referee model, but the next step must
  recover safe recall with better correspondence evidence rather than weakening
  these consistency gates.

Next step:

Stop adding one-off graph thresholds unless a larger matrix shows a new
dominant damage pattern. The next useful algorithm work is recall recovery:

1. Build a sequence-window correspondence check around accepted/rejected loop
   candidates.
2. Accept only if several neighboring query frames support a coherent match
   sequence.
3. Use graph consistency as a final commit referee, not as the only source of
   loop proposals.
