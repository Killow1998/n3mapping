# Global Relocalization Evidence v2 Execution Record

## Run metadata

- Start time: 2026-07-21 15:13:04 Asia/Shanghai (UTC+8)
- Branch baseline: `research/relocalization-evidence-v2` from `origin/dev/better_loop@cab5805`
- Host memory at start: 31 GiB total, 20 GiB available
- Build/test memory limit: `MemoryMax=12G`, `MemorySwapMax=4G`
- Finish time: 2026-07-21 16:35:41 Asia/Shanghai (UTC+8)
- Current verdict: `FALSIFIER` for G0-S; G0-formal remains `DEFERRED_DATA`

## Why this run starts

Establish a trustworthy synthetic relocalization evidence path before any behavior change. The two local Go2W bags are confirmed not to overlap spatially, so they are not a map/query pair. The formal cross-session product Gate is intentionally deferred.

## Authorized scope

- P0-core evaluator/artifact validation needed by the synthetic path.
- P0-S synthetic visibility and occlusion Gate.
- Measurement and test changes only; default runtime lock/reject behavior must remain unchanged.
- No FAST-LIO replay, bag conversion, dataset download, new map generation, loop-authority change, or formal product claim.

## Planned execution

1. Freeze the P0-S contract in the Roadmap.
2. Register and extend the Humble synthetic renderer tests.
3. Add deterministic explicit-pose synthetic queries and auditable artifacts without changing legacy defaults.
4. Add strict artifact validation and paired comparison tooling.
5. Build and run fresh tests under the memory limit.
6. If a map with verified provenance is available, run a bounded synthetic smoke; otherwise end with a tested toolchain and `SHADOW_ONLY` data verdict.

## Exact commands

Dataset readiness artifact created before the bounded baseline:

```text
/tmp/n3mapping_evidence_v2_20260721/dataset_readiness_report.json
```

Legacy-tool S0 baseline command, using the installed binary from the clean `cab5805` build and the existing map candidate only:

```bash
rtk systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=4G \
  /home/user/ros_ws/to_migrate_ws/install/n3mapping/lib/n3mapping/n3mapping_synthetic_relocalization_eval \
  --map /home/user/ros_ws/bagfile/n3mapping_test/n3map.pbstream \
  --output /tmp/n3mapping_evidence_v2_20260721/baseline_legacy \
  --max_queries 10 \
  --query_source global_map \
  --query_pose_xy_jitter_m 0.5 \
  --query_pose_z_jitter_m 0.1 \
  --query_pose_yaw_jitter_deg 30 \
  --query_pose_roll_pitch_jitter_deg 2 \
  --range_min 0.5 \
  --range_max 30 \
  --raycast_azimuth_resolution_deg 1 \
  --raycast_vertical_resolution_deg 1 \
  --occlusion_dilation_bins 1 \
  --occlusion_depth_tolerance 0.3
```

This command does not use `--strict`; the legacy evaluator has relaxed one-frame lock settings, so its output is baseline behavior only.

Execution note:

- The first `systemd-run --user --scope` attempt did not start the process because the sandbox carrier was identified as a kernel thread.
- A named service without the sourced workspace exited 127 because `libn3mapping_core.so` was not on its library path.
- The successful bounded command used a named user service and sourced Humble plus the workspace:

```bash
rtk systemd-run --user --wait --collect \
  --unit=n3mapping-p0s-baseline-env-20260721 \
  -p MemoryMax=12G -p MemorySwapMax=4G \
  /usr/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /home/user/ros_ws/to_migrate_ws/install/setup.bash && exec /home/user/ros_ws/to_migrate_ws/install/n3mapping/lib/n3mapping/n3mapping_synthetic_relocalization_eval --map /home/user/ros_ws/bagfile/n3mapping_test/n3map.pbstream --output /tmp/n3mapping_evidence_v2_20260721/baseline_legacy --max_queries 10 --query_source global_map --query_pose_xy_jitter_m 0.5 --query_pose_z_jitter_m 0.1 --query_pose_yaw_jitter_deg 30 --query_pose_roll_pitch_jitter_deg 2 --range_min 0.5 --range_max 30 --raycast_azimuth_resolution_deg 1 --raycast_vertical_resolution_deg 1 --occlusion_dilation_bins 1 --occlusion_depth_tolerance 0.3'
```

## Result

Implementation is still in progress. The S0 baseline completed successfully in 7.655 seconds of service runtime and consumed 20.696 seconds of CPU time under the declared memory limit.

Baseline artifact root:

```text
/tmp/n3mapping_evidence_v2_20260721/baseline_legacy
```

Observed legacy behavior:

- tested queries: 10
- reported lock events: 10
- pose-accurate locks at 1 m / 10 deg yaw / 5 deg roll-pitch: 4
- lock success rate: 1.0
- pose success rate: 0.4
- median translation error: 3.352257701 m
- p95 translation error: 19.74003416 m
- p95 yaw error: 143.5270778 deg
- examples include approximately 180 deg, 99 deg, and 92 deg wrong-yaw locks

This falsifies any interpretation of legacy `lock_success_rate` as correct relocalization. It is baseline evidence only and does not pass G0-S. No bag playback, conversion, FAST-LIO, or map generation was performed.

The determinism check repeats the same command with only the output directory and unit name changed:

```bash
rtk systemd-run --user --wait --collect \
  --unit=n3mapping-p0s-baseline-repeat-20260721 \
  -p MemoryMax=12G -p MemorySwapMax=4G \
  /usr/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /home/user/ros_ws/to_migrate_ws/install/setup.bash && exec /home/user/ros_ws/to_migrate_ws/install/n3mapping/lib/n3mapping/n3mapping_synthetic_relocalization_eval --map /home/user/ros_ws/bagfile/n3mapping_test/n3map.pbstream --output /tmp/n3mapping_evidence_v2_20260721/baseline_legacy_repeat --max_queries 10 --query_source global_map --query_pose_xy_jitter_m 0.5 --query_pose_z_jitter_m 0.1 --query_pose_yaw_jitter_deg 30 --query_pose_roll_pitch_jitter_deg 2 --range_min 0.5 --range_max 30 --raycast_azimuth_resolution_deg 1 --raycast_vertical_resolution_deg 1 --occlusion_dilation_bins 1 --occlusion_depth_tolerance 0.3'
```

Repeat result: success in 7.469 seconds of service runtime and 20.802 seconds of CPU time. `summary.json`, `config_used.json`, and all per-query behavior columns were identical; only the expected `elapsed_ms` values differed.

## Final implementation and verification

### Fresh Release build

The first clean configure used only `/opt/ros/humble` and failed before compilation because the workspace-provided GTSAM package was not on `CMAKE_PREFIX_PATH`. This was an environment-underlay failure, not a source failure. Its retained log root is:

```text
/tmp/n3mapping_p0s_fresh_20260721_01/log
```

The final source was rebuilt from an absent build directory after explicitly sourcing the current workspace only as a dependency underlay. Exact command:

```bash
rtk systemd-run --user --wait --collect \
  --unit=n3mapping-p0s-fresh-build-20260721-03 \
  -p MemoryMax=12G -p MemorySwapMax=4G \
  --working-directory=/home/user/ros_ws/to_migrate_ws \
  /usr/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /home/user/ros_ws/to_migrate_ws/install/setup.bash && exec /usr/bin/env MAKEFLAGS=-j1 /usr/bin/colcon --log-base /tmp/n3mapping_p0s_fresh_20260721_03/log build --base-paths /home/user/ros_ws/to_migrate_ws/src/n3mapping --build-base /tmp/n3mapping_p0s_fresh_20260721_03/build --install-base /tmp/n3mapping_p0s_fresh_20260721_03/install --packages-select n3mapping --allow-overriding n3mapping --symlink-install --parallel-workers 1 --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release'
```

Result: success. Service runtime 5 minutes 23.100 seconds; CPU time 5 minutes 23.322 seconds. The final producer used below has SHA256:

```text
68accdb817e8751d0f92eb4251db2aaa70d11e8037a5d52019cc9588a26637bf
```

### Full Humble test tree

Exact command:

```bash
rtk systemd-run --user --wait --collect \
  --unit=n3mapping-p0s-fresh-test-20260721-03 \
  -p MemoryMax=12G -p MemorySwapMax=4G \
  --working-directory=/home/user/ros_ws/to_migrate_ws \
  /usr/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /home/user/ros_ws/to_migrate_ws/install/setup.bash && source /tmp/n3mapping_p0s_fresh_20260721_03/install/setup.bash && exec /usr/bin/env ROS_LOG_DIR=/tmp/n3mapping_p0s_fresh_20260721_03/ros_logs PYTHONDONTWRITEBYTECODE=1 /usr/bin/colcon --log-base /tmp/n3mapping_p0s_fresh_20260721_03/test_log test --build-base /tmp/n3mapping_p0s_fresh_20260721_03/build --install-base /tmp/n3mapping_p0s_fresh_20260721_03/install --test-result-base /tmp/n3mapping_p0s_fresh_20260721_03/test_results --packages-select n3mapping --parallel-workers 1 --return-code-on-test-failure --event-handlers console_direct+'
```

Result:

- service success in 7.051 seconds, 10.601 seconds CPU;
- `colcon test-result`: 32 test entries, 0 errors, 0 failures, 0 skipped;
- 263 GTest cases, including synthetic renderer/state tests 14/14;
- strict validator/wrapper/comparator Python tests 33/33;
- `git diff --check` passed.

### Frozen query set

The provisional samplers were run only under `/tmp`; they did not create final evidence and did not use either user-provided Go2W bag. Exact-pose and deterministic-jitter candidates are retained at:

```text
/tmp/n3mapping_evidence_v2_20260721/provisional_exact_v2
/tmp/n3mapping_evidence_v2_20260721/provisional_jitter_v2
```

The final evaluator's `relaxed_smoke` jitter run reproduced the clean `cab5805` legacy behavior exactly for all non-timing summary metrics and query outcomes: 10/10 reported locks, 4/10 pose-accurate locks, the same inaccurate query IDs, and the same translation/yaw/roll-pitch aggregates. The new fields therefore extend measurement without changing the default localization decision path.

The final strict nine-column pose manifest contains four five-frame episodes:

| Episode | Class | Construction |
|---|---|---|
| 100 | centered | exact optimized pose near keyframe 15 |
| 101 | offset | keyframe-45 pose plus 1.0 m map X and 0.5 m map Y |
| 102 | large_yaw | keyframe-90 position with +120 degree yaw |
| 103 | repetitive_ambiguous | exact deterministic pose that produced a wrong lock in the legacy relaxed baseline |

Frozen inputs:

```text
query_pose_manifest.csv  sha256=5a9fc23ce9e66d07c5f02693729b5a44f365dd3ad7c248f0de5c5b81f90eb03f
episode_catalog.csv      sha256=7bdd5c2f50c767efeb1a5184fc1ed306b4f314170dd9ed8aebb30b2cc4e0c68a
synthetic contract       sha256=ce74dd15ab248176d1ff2b6e1230c795032dac9526d72d99802619e364f82ab0
map                      sha256=a69db12c2476ec1eea2781068fe334415e102bf92d64871d3e707e99a9554d75
```

The strict-v2 product profile freezes a five-frame temporal window, one-second frame period, positive one-degree azimuth/vertical raycast, at least 100 final query points, and at least 100 occupied ray bins. All 20 final frames passed renderer validity.

### Final repeated synthetic runs

Both runs used the same command below, changing only `<unit>` and `<run_id>`:

```bash
rtk systemd-run --user --wait --collect \
  --unit=<unit> -p MemoryMax=12G -p MemorySwapMax=4G \
  --working-directory=/home/user/ros_ws/to_migrate_ws \
  /usr/bin/bash -lc 'source /opt/ros/humble/setup.bash && source /home/user/ros_ws/to_migrate_ws/install/setup.bash && source /tmp/n3mapping_p0s_fresh_20260721_03/install/setup.bash && exec /usr/bin/env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B src/n3mapping/tools/n3mapping_synthetic_eval_gate.py --producer /tmp/n3mapping_p0s_fresh_20260721_03/install/n3mapping/lib/n3mapping/n3mapping_synthetic_relocalization_eval --output /home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_p0s/20260721/<run_id> --map /home/user/ros_ws/bagfile/n3mapping_test/n3map.pbstream --query-pose-manifest /home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_p0s/20260721/inputs/query_pose_manifest.csv --frozen-contract /home/user/ros_ws/to_migrate_ws/src/n3mapping/test/fixtures/eval_v2/synthetic_frozen_contract.yaml --dataset-id n3map_unknown_legacy_p0s_four_class_v1 --map-provenance unknown_legacy --split p0s_shadow_four_class_v1 --repo /home/user/ros_ws/to_migrate_ws/src/n3mapping --frame-period-s 1.0 --build-type Release -- --query_source global_map --eval_profile product_default --range_min 0.5 --range_max 30 --fov_azimuth_deg 360 --raycast_azimuth_resolution_deg 1 --raycast_vertical_resolution_deg 1 --occlusion_dilation_bins 1 --occlusion_depth_tolerance 0.3'
```

Run A succeeded in 13.864 seconds of service runtime and 50.115 seconds CPU. Run B succeeded in 14.030 seconds and 50.501 seconds CPU. Both artifact directories pass independent strict validation with zero warnings and their complete checksum manifests pass.

The artifacts recorded the effective cgroup values, not only requested values:

```text
memory.max      12884901888 bytes (12 GiB)
memory.swap.max 4294967296 bytes (4 GiB)
```

Determinism evidence:

```text
raw/resolved_queries.csv  run_a=run_b=d6f09df6676e47446bdc1fc46e7b750a080f81b239b7953b711a4af6c60f68c3
raw/renderer_visibility.csv run_a=run_b=a09d1fec72ef2ace8e479ee149c5941fe5a3a6054419dcb79acc0752328bb5bd
per-query FNV records equal: 20/20
dataset manifest equal: 37c2e364268d1eebcc74e7d7a982f0b078f9b4681f55454332a588a2a2de0de0
resolved config equal: 85a3ba1ae10b366276fd905df75b8a22003865ae5b04fcba2d1375074f91b067
```

The manifests honestly record `dirty=true` because the final measurements were made before the single local evidence commit. They also record the exact producer binary hash and baseline repo SHA, so the measured executable remains identifiable.

### Paired comparison and three-axis verdict

Exact comparison command:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 python3 -B tools/n3mapping_eval_compare.py \
  --baseline /home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_p0s/20260721/run_a \
  --candidate /home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_p0s/20260721/run_b \
  --contract test/fixtures/eval_v2/synthetic_frozen_contract.yaml \
  --output /home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_p0s/20260721/comparison_run_a_vs_run_b
```

Renderer contract: `PASS`.

- 20 planned, 20 valid, 0 invalid;
- map-conditioned point z-buffer enabled on every valid frame;
- visibility accounting conserved;
- repeated resolved query and visibility artifacts are byte-identical.

Evaluator contract: `PASS`.

- both runs strict-valid with atomic `COMPLETE` and complete checksums;
- one attempt per episode, 20 eligible relocalization frames;
- GT/derivation fields are explicit: `gt_runtime_access=true`, `odom_derived_from_gt=true`, `synthetic_from_target_map=true`, `gt_passed_to_localizer=false`;
- evidence ceiling is `SHADOW_ONLY`.

Algorithm observation: `FALSIFIER`.

- correct attempt count: 0/4;
- timeout attempts: 3/4;
- false-lock attempts: 1/4, false-lock rate 0.25;
- the false lock occurred on episode 103 at frame 4 with 11.571829135 m translation error, 89.679023307 degrees yaw error, and 19.710798452 degrees roll-pitch error;
- p95 total frame time: 1079.25454055 ms in run A and 1087.58850025 ms in run B.

The frozen paired comparator therefore reports safety `FAIL`, utility `FAIL`, resource `FAIL`, and overall `FAIL`. The G0-S decision label is `FALSIFIER`, because the contract itself is valid and a deterministic wrong-lock counterexample was observed. It is not a renderer or evaluator invalidity.

### Boundary and follow-up

- G0-formal is `DEFERRED_DATA`: no independent recorded cross-session scan/GT pair exists.
- The two Go2W bag directories were never paired, played, converted, or sent into the evaluator.
- The existing map provenance remains `unknown_legacy`; synthetic query and target use the same map.
- This evidence cannot establish product readiness and cannot authorize P1A or any runtime threshold/decision change.
- A follow-up algorithm experiment is needed only after separate user authorization. The first target should be the deterministic episode-103 false lock; the falsifier is already frozen and reproducible.
