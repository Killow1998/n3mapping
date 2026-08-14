# n3mapping Industrial Closeout Review — 2026-08-14

Recorded in Pacific daylight time (PDT, UTC-07:00).

## Verdict

The architecture-hardening roadmap is complete, but n3mapping does **not** yet
meet an industrial release acceptance standard.

The correct closeout state is:

- **Engineering roadmap: COMPLETE.** Every architecture-hardening work package
  has an evidence-backed PASS or an explicit NO-GO/default-off terminal state.
- **Current backend: suitable for a maintenance freeze and controlled field
  use on the qualified floor7 workflow.** Existing product defaults and
  fail-closed behavior should remain frozen.
- **General industrial release: NO-GO.** Cross-session positive recall is below
  the historical 80% working target, end-to-end recorded-LIO public-dataset
  evidence is absent, large Atlas generation exceeds its own load limit, and
  release governance/artifacts are not established.

This distinction is important: completing a roadmap means the planned
questions were answered. It does not make every answer a product PASS.

## Evidence reviewed

| Surface | Confirmed result | Classification |
| --- | --- | --- |
| Architecture boundaries | ROS-free backend, thin Humble/Noetic wrappers, external LIO contract, explicit session pose domains, transactional graph/map paths, bounded caches | PASS |
| Architecture roadmap | WP-00 through WP-C4, SG-01 through SG-09, and PERF-ME-01 have terminal PASS/NO-GO decisions; rejected behavior remains default-off | COMPLETE |
| Final local Humble Product build | Fresh isolated Release build at `f408013`, research tools OFF, verified build identity; 53/53 CTest targets | PASS |
| Closeout research build | Fresh isolated Release build with research tools; 57/57 CTest targets after evaluator-contract fixes | PASS |
| Static analysis | Project `missingReturn`, boolean bitwise, signedness, and avoidable copy warnings removed; remaining findings are vendored nanoflann/KD adaptor warnings plus an Eigen comma-initializer false positive | PASS WITH VENDOR FINDINGS |
| floor7 real workflow | Mapping, raw-bag localization, map extension, C1/C2/C3/SG evidence and owner visual checks have auditable terminal results | CONDITIONAL PASS for the qualified workflow |
| Loaded-map tracking performance | Target-cache hit rate 93.51%; callback median fell from 211.734 ms to 20.208 ms on the frozen replay, with no tracking or saved-map regression | PASS for that replay only |
| KITTI-360 positive cross-drive | `0005 map -> 0006 query`: 3/5 correct locks, 2/5 no-lock, 0/5 false locks; lock p95 0.554 m and 0.434 deg | FAIL recall target; fail-closed behavior preserved |
| KITTI-360 wrong-map | `0005 map -> 0009 query`: 5/5 no-lock, 0 unexpected locks at symmetric 0.5 m input voxel | PASS for this negative slice |
| M2DGR gate_02 | Same-session held-out half, measured quaternion: 1/5 correct locks, 4/5 no-lock, 0 false locks | FAIL recall target |
| M2DGR hall_05 | Same-session held-out half, trajectory-derived yaw: 0/3 locks, 0 false locks at 0.5 m input voxel | Diagnostic only; FAIL recall target |
| Atlas scale | 100 KITTI keyframes at 0.2 m produced 1,218,130,685 bytes, exceeding the 1 GiB load contract | RELEASE BLOCKER |
| Noetic | No local installation on this Jammy host; clean Noetic/Focal container build and tests passed remotely at `2d4225f` | PASS in remote matrix |
| Remote CI | Recovery commit `2d4225f`, [run 31805997273](https://github.com/Killow1998/n3mapping/actions/runs/31805997273): Humble/Jammy and Noetic/Focal both build/test PASS | PASS |
| Release governance | `main` is unprotected; no release artifact exists; package version remains 1.0.0 | NOT INDUSTRIALIZED |

## Dataset evidence contract

The KITTI-360 and M2DGR tools are backend diagnostics. They supply
`gt_pose_plus_lidar`, not recorded LIO odometry. Their metrics now state both
that contract and `real_lio_safety_filters_applied=false`; real-LIO static-start
and divergence behavior belongs to raw-bag qualification.

The frozen manifests hash every selected cloud, both GT files, and official
KITTI-360 calibration. Benchmark outputs are finalized with `checksums.sha256`
and `COMPLETE`. Even so, the runs remain `formal_gate_ready=false` because:

- no recorded LIO trajectory is bound to the public-dataset episodes;
- KITTI-360 same-session or cross-drive oracle replay is not end-to-end SLAM;
- M2DGR LiDAR-to-IMU/frontend configuration is not frozen;
- `hall_05` has trajectory-derived yaw rather than measured full attitude;
- a GT distance overlap label does not prove visible-surface overlap by itself.

Closeout artifacts:

The compact, machine-readable results are retained in
[`evidence/industrial_closeout_20260814/auxiliary_dataset_results.json`](evidence/industrial_closeout_20260814/auxiliary_dataset_results.json).
The multi-hundred-megabyte maps, Atlases, clouds, and raw datasets are not
duplicated into Git. Paths below record the original execution locations; the
compact JSON is the durable closeout evidence.

| Run | Summary SHA-256 | Result |
| --- | --- | --- |
| `/tmp/n3mapping_closeout_benchmarks_20260814/kitti_0005_map_0006_query_lock` | `fc025304315000fa54d8c8c5cb20853bfea791eb98da36beded4971c25b75d16` | 3 correct / 0 false / 2 no-lock |
| `/tmp/n3mapping_closeout_benchmarks_20260814/kitti_0005_map_0009_query_abstain_100_voxel05` | `8df458fd3905b2e4d3469bf30c1043acbd2cffadc680a34c09ba193a57adff7d` | 5/5 expected abstain |
| `/tmp/n3mapping_closeout_benchmarks_20260814/m2dgr_gate02_lock` | `b7914c3a78f001a3be1c27e6897d139067e8debcfdd3f1dfebaa7df873659826` | 1 correct / 0 false / 4 no-lock |
| `/tmp/n3mapping_closeout_benchmarks_20260814/m2dgr_hall05_lock_3x100_voxel05` | `d31aa899c9c4264c1bbc132cd908c8a49a7ed30dea5d0700ffeb02dafee7a991` | 0 correct / 0 false / 3 no-lock |

The failed 0.2 m negative-map attempt was observed at
`/tmp/n3mapping_closeout_benchmarks_20260814/kitti_0005_map_0009_query_abstain_100`.
It failed during Atlas verification and is not counted as a localization run.

## What the closeout fixes change

- Refuse CMake build directories inside the source repository, including an
  in-tree symlink to an external directory. This protects clean Product
  identity and is covered by three regression cases.
- Make oracle-dataset evaluator behavior explicit and stop applying real-LIO
  safety filters to GT input. This repaired previously hidden research-test
  drift without changing runtime product defaults.
- Refuse an Atlas whose exact protobuf size exceeds the loader's 1 GiB limit
  before publishing it. This makes oversize generation atomic and fail-closed;
  it does not solve the scale limit.
- Remove compiler/static-analysis warnings in legacy ScanContext and tests,
  without changing current HybridScanContext authority.
- Set CMake CMP0074 explicitly so dependency discovery is deterministic and
  warning-free on current CMake.
- Keep CI build/install trees outside the checkout, share one backend source
  list across Humble and Noetic, and link the core's direct TBB dependency
  explicitly. This closed a real Noetic source-list/linker drift exposed by
  the first recovery-branch matrix run.
- Pin the official Node 24 GitHub checkout/cache actions to immutable release
  commits instead of retaining deprecated Node 20 action majors.

## Industrial release blockers

### P0 — must close before a general release

1. **Atlas scalability:** shard/stream/compress the prepared target or define a
   tested map-size budget. Raising the 1 GiB limit is not an acceptable fix;
   parse memory and protobuf limits remain.
2. **End-to-end dataset contract:** freeze sensor timestamps, LiDAR/IMU
   extrinsics, frontend configuration, recorded LIO output, cloud payloads, and
   independent map/query sessions. Re-run positive and negative episodes
   without GT pose as runtime odometry.
3. **Acceptance performance:** demonstrate the predeclared positive recall and
   false-lock budget on more than one environment. Current positive results of
   60%, 20%, and 0% fail that bar.
4. **Release governance:** protect the release branch, identify required CI,
   update the package/release version, and publish a reproducible binary/config
   bundle with its commit and profile hashes.

### P1 — required for sustained industrial operation

- long-duration mapping/localization/map-extension regression on target
  hardware, including restart and resource exhaustion;
- compatibility corpus for older production `pbstream` files;
- crash/power-loss and disk-full tests for map and sidecar publication;
- latency/RSS budgets on each supported computer, not one replay/host;
- sanitizer/fuzz coverage for protobuf and map loading boundaries;
- documented sensor/front-end integration and an operator rollback procedure.

## Recommended stop strategy

Do not resume C1/C2/C3/SG threshold development, builtin-LIO integration,
shadow graph writeback, or cross-session automatic merge. The existing
evidence rejected those promotions or left them unqualified.

Choose one of two honest endpoints:

1. **Maintenance freeze now:** keep the recovery branch as the qualified
   floor7 line, retain all NO-GO/default-off boundaries, fix only reproducible
   correctness/security/build defects, and stop feature development.
2. **Industrial release later:** open a new, tightly scoped release project
   containing only the four P0 gates above. Dataset failure must change the
   release verdict, not trigger another unbounded algorithm roadmap.

The first option is the recommended closeout for the stated goal of ending
n3mapping development.

## Ref cleanup result

The review and exact retired targets are recorded in
[`branch_retirement_review_20260814.md`](branch_retirement_review_20260814.md).
The ledger is present on the remote recovery branch and all temporary
`archive/retired-20260814/*` tags have been deleted locally and remotely. The
durable refs are three remote branches (`main`, recovery, realtime) and two
tags (`archive/humble`, `archive/noetic`).
