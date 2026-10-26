# Large-map follow-up — 2026-10-26

This continues the [crash-containment qualification](large_mapping_crash_20261025.md).
It covers dependency repair, save latency and the failed acquisition's accuracy.
The navigation team's frozen libraries, installed GTSAM, FAST-LIO, SpRobo, ETC
and TGW were not changed. No remote release or deployment is performed.

Evidence root, relative to the parent ROS workspace:
`build/n3mapping/reliability_precision_20261026/`.
Original input directories remain read-only. Truth is an offline observer and
is never published to N3M. This is Humble qualification; no new Noetic runtime
qualification or target-hardware accuracy claim is made.

## Dependency defect and repair

The earlier TSan experiment used uninstrumented TBB and could not distinguish
task publication from a real GTSAM race. This follow-up builds oneTBB 2021.5.0
with ThreadSanitizer and instantiates the actual GTSAM tree-elimination templates.
The GTSAM-free control uses immutable task captures and an atomic result.

| Instrumented probe | Result |
| --- | --- |
| GTSAM-free TBB control | Pass |
| Installed GTSAM 4.1.1 traversal, four threads | TSan race, exit 66 |
| Same traversal with upstream locking patch | 20 runs of 4,001 nodes pass |

The race is between reallocation of a parent's `childFactors` vector and a
child's result write into that vector. Upstream
[PR #1158](https://github.com/borglab/gtsam/pull/1158), merged as
`a0a8010bfea8ac1a74c81b8646b00951841fb52e`, protects both accesses with the same
mutex. Both commits in the official patch are applied to GTSAM 4.1.1 source
`69a3a75195b65356d6e56669e5199d325c7962c9` in an isolated prefix.

Official patch SHA256:
`16c954218cab5b93b0c66f7c1982fa225afc02f17b6502f8ad527eb665e9ec8b`.
oneTBB source archive SHA256:
`e5b57537c741400cf6134b428fc1689a649d7d38d9bb9c1b6d64f092ea28178a`.
Commands, source, reports and controls are retained in `dependency/`.
ASLR is disabled only for these diagnostic children, not system-wide.

This demonstrates a real dependency defect and its repair. It does not prove
that this was the only cause of the original `init_bucket` SIGSEGV, nor does a
template probe qualify every GTSAM path as race-free.

N3M retains one synchronous local execution boundary for all six solve entries.
The selected headers must contain the locking implementation, and the actual
loaded GTSAM build ID must match the configured library. Only then does the arena
use the existing `num_threads` setting. Unknown or mismatched dependencies use
local concurrency 1. An intentional old-library overlay passes the numerical
snapshot regression and logs `policy=serial_containment`. There is no
process-wide throttle or per-solve file access. See the
[execution design](../graph_solver_execution.md).

## Build identity and regression

Runtime revision: **`111b1ac335ad9fa0dce525efadfa78af2995bc54`**.
Save observability is the preceding independent commit `d62304e`.
The clean Release build reports `verified=true`, research tools OFF, and uses
CMake's Release optimization, ccache and four build workers.

| Artifact | SHA256 |
| --- | --- |
| `release/install/n3mapping/lib/n3mapping/n3mapping_node` | `aa5b86148b56ec90519c8a228110ff60a7304266d87758a143b4620d3478e10c` |
| `release/install/n3mapping/lib/libn3mapping_core.so` | `f14bee2496d1b1f47b7157048b7df618ec8e2ebb89ad4514c9f001d3ffc28526` |
| `dependency/gtsam_install/lib/libgtsam.so.4.1.1` | `84be7fb16cd93b8bf782674eae5e161489f08b3c5b964c14b29b632a9f65110b` |

The selected GTSAM GNU build ID is
`a25c1600b852828206fd4f39d54eb6c05dca76f5`.
`build_manifest.json` records the executable identities, profiles, original
tags and unchanged protected-library hashes.

- Final Release: **67/67 CTest targets pass, 41.81 s**.
- Current source built against the original GTSAM: **67/67 pass, 46.37 s**.
- Patched-header build intentionally run with old GTSAM: concurrent graph
  snapshot numerical regression passes; runtime containment is confirmed.
- The initial development `ctest -j4` run caused six logger tests to compete
  for a single default instance directory. Retesting with one test process and
  an explicit private log directory passed. The original failure is retained.
  No assertion was weakened. Tests use `ctest -j1`; algorithm workers still
  have four-core affinity.

Main node and performance comparisons use affinity 0,2,4,6, OpenMP 4,
OpenBLAS 1 and passive OpenMP waiting. Some compilation and functional
diagnostics ran on 8,10,12,14 while replays ran. Affinity is not an exclusive
host reservation; single-run timing differences are not deterministic speedups.

### Deterministic solver comparison

The original 1,200-node fixture, unchanged executable and inputs, runs in
balanced order, three processes per revision and 30 iterations per process.
All seven pose-update, residual and consistency fields have **zero absolute
difference**. This comparison used the development binary with the same solver
implementation; the final Release additionally passes 90 fixture iterations
with zero difference in all seven fields (`release_numeric_qualification.json`).

| Mean per 30-iteration process | Old rc3 parallel | Previous local serial | Patched, bounded parallel |
| --- | --- | --- | --- |
| Wall time | 1.190 s | 1.463 s | 1.190 s |
| CPU time | 2.093 s | 1.460 s | 2.087 s |
| Peak RSS | 154.27 MiB | 135.58 MiB | 154.51 MiB |

The serial wall-time penalty is removed on this fixture. Parallel execution
uses more CPU time and memory than serial; it is not a total-work reduction.
`numeric_bound_qualification.json` retains every repetition.

## Save latency and full replay

New sparse events separate pending-loop drain, serialization, Atlas and PCD
work. They use the existing bounded asynchronous logger. Transaction ordering,
input parameters and loop-admission gates are unchanged. Pending verified
observations are still processed before a final save.
The large-map saves in this record run after input completion. Their timings
do not qualify a save requested while the robot continues supplying input.

The previous serial revision's complete normal-rate, two-floor replay saves
in **1.512 s**. Therefore the earlier **36.921 s** at 2× must not be reported
as the normal-rate disk-write cost. A development patched-parallel 2× replay
saves in **17.403 s**: drain **16.246 s**, serialization **0.153 s**, Atlas
**0.916 s**. This is a measured improvement, not a fixed upper latency bound.

Both 2× replays process all 13,781 input pairs and retain 1,661 keyframes.
Their final maps contain 1,538 and 1,537 loop constraints respectively.
The development run's 1,475 runtime-loop count excludes 62 constraints committed
during save; treating that count as the final map total would be incorrect.
Asynchronous snapshots change candidate and commit schedules, so pair sets need
not be identical even with unchanged acceptance gates.

The optional `graph_update_ms` frame field belongs to map extension and is not
populated by these mapping runs. Mapping graph work is represented by loop and
geometry timing, not by treating that absent field as zero work.

The final Release completes both full inputs without SIGSEGV. All processes exit
normally, all source stamps appear exactly once in the native trajectory, and
the old rc3 public reader loads both outputs.

| Final Release run | Input / backend / native rows | Keyframes / loop edges | Final save |
| --- | --- | --- | --- |
| Original failure input, 2× | 10,381 / 10,381 / 10,381 | 1,302 / 1,175 | 0.881 s |
| Completed two-floor route, 1× | 13,781 / 13,781 / 13,781 | 1,661 / 1,566 | 1.036 s |

Normal-rate final save has no outstanding loop wait: drain 0.017 ms,
serialization 148.25 ms and Atlas 881.58 ms. Four full ROS-node reloads cover
the two preceding maps and both final outputs. Dense pose differences are
below 5e-14 m, edge rotation differences below 1.3e-15 rad, and clouds and
non-pose edge fields survive exactly. The final node runs map_extension mode
to exercise graph/session/Atlas loading, then saves without new input.

| Two-floor 1× whole-node comparison | Previous serial `5e9bf8b` | Final parallel `111b1ac` |
| --- | --- | --- |
| Mean CPU cores | 0.720 | 0.886 |
| CPU seconds | 999.54 | 1,231.36 |
| Peak RSS high-water mark | 941.08 MiB | 1,123.52 MiB |
| Backend callback P99 | 7.07 ms | 6.86 ms |
| Final loop edges | 1,562 | 1,566 |
| Save service wall time | 1.512 s | 1.036 s |
| Historical online maximum position error | 17.146 cm | 17.146 cm |

The parallel revision uses about 23% more CPU time and 182 MiB more peak memory
in this normal-rate run. It restores solver latency; it is not a half-core or
memory optimization. Both runs suppress four initial awaiting-correction live
outputs and four invalid-input-time outputs. Every backend/native frame is
retained. Recorded cross-topic callback ordering is not a timestamp barrier.

The final normal-rate common-gauge online RMS/max is 7.891/18.396 cm; final
native RMS/max is 5.775/14.951 cm. The serial normal-rate final native result is
6.133/15.241 cm. Snapshot scheduling differs, so the small final-map change is
not claimed as an independently established accuracy improvement. The failed
input's historical online maximum remains 1.044 m and its gross-drift guard
remains FAIL. Neither run establishes 10 cm acceptance.

The [qualification index](large_mapping_followup_20261026.json) binds retained
evidence to the runtime hashes. `final_qualification.json` in the evidence root
contains complete timing, resource, reference-frame and reload records.

## Accuracy investigation: still open

The unchanged historical scorer aligns each output topic to its own first
pose. LIO and N3M begin at different timestamps. Their independently aligned
maximum errors must not be subtracted to attribute error to the backend.
`accuracy_audit.py` adds a diagnostic using one first-LIO-pose transform for
input, online output and final native trajectory. It does not replace the
acceptance scorer, fit the whole trajectory, or provide truth to the estimator.

| Frozen serial replay, common first-LIO gauge | Input LIO RMS / max | Online N3M RMS / max | Final native RMS / max |
| --- | --- | --- | --- |
| Original failed acquisition | 46.21 / 118.00 cm | 47.55 / 120.95 cm | 46.80 / 116.98 cm |
| Completed two-floor acquisition | 5.66 / 12.47 cm | 7.95 / 18.35 cm | 5.74 / 14.85 cm |

The historical first-N3M alignment still reports approximately 1.04 m and
17.10 cm online maxima. These are different references, not newly introduced
drift. Final optimized trajectories also differ from already published online
poses. Neither metric establishes general 10 cm acceptance.

Evidence identifies two separate issues:

1. The failed acquisition already has a **0.6905 m relative odometry error**
   between keyframes 362 and 363 (293.80–297.10 s). Its startup is also unsettled.
   Both frozen FAST-LIO logs report initialization over 20 samples with zero
   additional wait. Raw IMU and offline truth show the 0.2–0.3 s mean specific
   force direction 11.61 degrees from world vertical, versus 0.077 degrees in
   1.2–1.3 s. The second acquisition delays frontend startup by one second, but
   also changes motion limits; it is not a single-variable causal experiment.
   The single-variable raw replay below now isolates a substantial startup
   contribution. This does not establish a general frontend repair or prove
   every remaining source of drift. Acceleration is audited in recorded units,
   not mislabeled as m/s².
2. The backend also contributes error on the completed route. Some stored loop
   measurements have metre-scale disagreement with truth despite robust graph
   weighting. Their mere presence does not establish that they dominate the
   final trajectory. Geometry and odometry weights need separate investigation.

### Rejected accuracy ablations

All experiments use the production graph solver on frozen observations, with
truth only used afterward. Keyframe-only maxima below must not be confused
with dense online maxima.

| Completed-route frozen graph | Keyframe RMS / max |
| --- | --- |
| Odometry only | 5.72 / 12.29 cm |
| Odometry + loops | 6.60 / 12.98 cm |
| Odometry + planes | 6.02 / 16.31 cm |
| All existing observations | 5.83 / 14.75 cm |

Removing an observation family does not uniformly improve accuracy across both
routes. Applying robust noise to every mapping odometry edge changes the
completed-route maximum only from 14.746 to 14.735 cm, slightly worsening RMS
and rotation. It is not adopted.

Neighbor consensus alone is also unsuitable as a new blanket rejection gate:
its Reject group contains four measurements within 10 cm in the offline audit.
A visibility-bearing diagnostic identifies 29 poor stored loops without
rejecting a <=10 cm loop in this one route, but removing those 29 changes the
frozen-graph maximum by only about one micrometre. It neither explains the
remaining peak nor qualifies a cross-scene acceptance rule. No experimental
gate, ground-truth alignment, odometry reweighting or ablated solver enters the
delivered runtime.

The failed acquisition remains diagnostic and must not be promoted as a valid
navigation map. Next accuracy work must isolate startup/input defects from
backend geometric association and weighting on the same observations. Changes
to FAST-LIO or the simulator require their owning scope; this revision makes none.

## Raw startup isolation and online accuracy follow-up

The additional evidence is in `precision/` under the same evidence root.
Runtime remains **111b1ac**. No production algorithm, parameter profile,
acceptance gate, map format or external project is changed in this investigation.
Ground truth is used only after estimation, with the existing IMU/body extrinsic.

### A causal startup control

Replay the original acquisition's raw LiDAR, IMU and recorded clock into the
unchanged FAST-LIO executable and configuration. The only input change is
starting at storage time 1 s instead of 0 s. The route and sensor payloads are
unchanged. Private process mounts redirect legacy frontend log writes; the
frontend source, installed binary and configuration remain read-only.
The executable SHA256 remains
`86959088726e375e2c970fcef89d63525b91c39fb72ad3bc9eee25f0e7e1ce93`.

The first 60 s control reproduces the original frontend's 596 shared poses
with **zero component-wise difference**. Its own-first-pose maximum is
24.0445 cm. To exclude different alignment times as the explanation, both runs
are also aligned once at the same physical timestamp, **1.299989997 s**:

| Same 587 timestamps through 59.899695 s | Position RMS | Maximum |
| --- | --- | --- |
| Original raw start | 12.418 cm | 21.864 cm |
| Raw input starting at 1 s | 0.598 cm | 0.981 cm |

This is not a fitted whole-route alignment. The first-pose transform is fixed
for all later estimates. `startup_scores.json` and
`matched_startup_anchor.json` retain both scoring conventions.

The unchanged frontend reports initialization over **20 IMU samples** in both
runs. The original initialization log reports acceleration variance 0.100418;
the delayed run reports 0.000007, in the frontend's recorded units. The inspected
implementation checks motion only after more than 50 samples, while its
default zero-wait path already finishes after these 20 samples. This supports
an initialization-quality defect for this input; it does not prove that a
fixed one-second delay is safe on a different robot or scene.

An implementation remedy belongs at the frontend/acquisition initialization
boundary: validate a fresh stable IMU window before committing gravity and bias,
restart that window when it is disturbed, and expose readiness before mapping
or route motion begins. Simply discarding N3M outputs after FAST-LIO has already
initialized does not repair gravity, deskewed points or the frontend's local map.
This remedy is specified for the owning scope, not silently implemented here.

### Complete original route, with healthy startup

The normal-rate replay retains all **10,372** original frontend timestamps from
1.299989997 s through 1038.399989997 s, each exactly once in the backend and
native saved trajectory. At the same fixed 1.299989997 s alignment, original
frontend RMS/max is **35.583/86.813 cm**, versus **3.770/7.619 cm** after changing
the raw start. The previous 297 s large error no longer has its original
magnitude. This improvement is demonstrated over the complete recorded input,
not just the quiet opening minute.

All three output streams below share the first frontend pose's alignment in
each run. Online output is kept separate from the final corrected trajectory.

| Raw replay rate | Frontend RMS / max | Online N3M RMS / max | Final native RMS / max |
| --- | --- | --- | --- |
| 1× | 3.770 / 7.619 cm | 9.776 / 25.639 cm | 3.280 / 8.609 cm |
| 2× repeat | 3.784 / 7.635 cm | 8.248 / 19.681 cm | 3.763 / 11.058 cm |

The first run's online output exceeds 10 cm for about **258.5 s**. The graph's
existing current-pose log also reaches approximately 25.55 cm near 434.9 s;
its text precision is lower than the pose CSV. The issue is present in the
graph estimate, not merely an output-projection artifact. It is not a brief
isolated spike. The final map benefits from later observations that were
unavailable to online estimates.

The 2× repeat changes frontend poses by up to **5.58 mm** compared with 1×,
despite retaining exactly the same timestamps and raw source. Async graph
schedules also differ: 1× retains 1,289 keyframes/1,162 loops, while 2× retains
1,290/1,142. This is a pipeline repeat, not a deterministic backend-only A/B.
Neither the 1× final maximum nor the large startup gain establishes general
10 cm accuracy. The other completed two-floor acquisition retains its previous
unresolved accuracy result.

The normal-rate N3M process averages **0.649 cores**, with **891.32 MiB** peak
RSS high-water mark; frontend averages **0.200 cores**, with **373.85 MiB** peak.
Final save takes **0.817 s**. The 2× N3M process averages **1.233 cores**, peaks
at **910.25 MiB**, and saves in **0.835 s**. These are measured process-lifetime
costs, not a same-load performance improvement or a half-core qualification.
All estimators are confined to cores 0,2,4,6.

### Recording failure and recovery

The first run's independent pose observer and N3M complete, but its auxiliary
recorder aborts after **4,577** cloud/odometry pairs. Point's read-only query of
the actively written SQLite database creates a read lock; rosbag2 reports
`SQLite error (5): database is locked`. The recorder exits with SIGABRT.
The failure, partial database, missing metadata and first failed coverage check
are retained. It must not be described as a complete recorded frontend bag or
as a new N3M/GTSAM crash.

The repeat checks recorder liveness and does not inspect SQLite during writes.
It exits normally and produces a closed bag with **10,372 clouds and 10,372
odometry messages**, matching the observer, backend and native trajectory.
Use `precision/full_start1_repeat_2x/lio_source/bag` for future frozen-input
tests. The frontend wrapper's -2 status is the deliberately requested SIGINT;
the node logs show its shutdown signal. N3M, observer, recorder and replayer
return zero in the successful repeat.

### Solver controls rejected as a precision fix

Two frozen graphs compare approximate versus chain-rule-correct SE(3)
BetweenFactor derivatives, crossed with default versus tighter LM tolerances.
Measurements, information matrices, plane observations and gauge are fixed.
Finite differences confirm the exact residual Jacobians to about 1.3e-8,
whereas the dependency's default approximation has large errors on some
outlier constraints. That fact alone does not establish an accuracy benefit.

Default solves stop after one iteration. Tight tolerances require 15 iterations
on the failed graph and 17 on the completed graph. For the completed graph,
the approximate-derivative maximum changes only **14.7456 → 14.7415 cm**;
exact derivatives also leave its maximum near **14.744 cm**. The failed graph
remains above one metre, with small mixed changes. These controls do not explain
the online 25 cm error or justify a permanent increase in solver work.
No derivative variant or tighter tolerance is adopted.

### Causal prefix and correlation ablation

The closed partial recording contains all 4,577 cloud/odometry pairs through
458.899989997 s. A separately reindexed copy replays this exact prefix into the
unchanged backend. No later sensor observation is available. The 4,572 live
outputs reproduce the approximately 25.639 cm online maximum. The saved graph
contains 576 keyframes. Frozen-observation ablations retain the original
measurements, gauge and noise for each included factor family:

| Prefix graph, keyframe-only evaluation | RMS | Maximum | Last pose |
| --- | --- | --- | --- |
| Odometry only | 2.993 cm | 5.922 cm | 1.487 cm |
| Odometry + loops | 12.540 cm | 24.375 cm | 21.621 cm |
| Odometry + planes | 9.009 cm | 15.012 cm | 9.515 cm |
| All observations | 12.779 cm | 23.793 cm | 21.995 cm |

This localizes a substantial problem to loop constraints, with an additional
plane-association/weight contribution. Solving the prefix without future input
does not remove it. These diagnostic deletions are not an implemented policy:
removing all loops or planes would discard valid observations and their later
benefit. `prefix_ablation_scores.json` records the full comparison.

A second diagnostic retains every loop and scales each same-query group to
one total information budget. For a group of n measurements, each complete
robust cost is multiplied by 1/n; information and Cauchy transition are scaled
together, preserving the transition in measurement units. No truth is used to
choose groups. All variants use the same stricter convergence controls:

| Frozen graph maximum | Original group weight | Shared-query budget |
| --- | --- | --- |
| Causal prefix | 23.794 cm | 16.315 cm |
| Complete delayed-start route | 8.504 cm | 9.267 cm |
| Other completed two-floor route | 14.742 cm | 15.645 cm |

The prefix improves but still misses 10 cm, and both complete-route maxima
regress. A uniform same-query budget is therefore **not adopted**. Counts alone
do not identify which directions or observations share bias. The next backend
work should audit individual relative measurements and their shared scan/target
support, then calibrate directional uncertainty on held-out routes. Retain the
complete frozen frontend bag for those same-input comparisons.

All three new maps load through the real ROS node in map-extension mode and
save again. Dense position differences are below **4.5e-14 m**; cloud payloads
and non-pose edge fields survive exactly. Original source timestamps are
preserved. `reload_full_start1_qualification.json` indexes these checks.

Future trajectory corrections must not be used to certify past online accuracy.
Mature references considered are
[BALM](https://github.com/hku-mars/BALM) and
[HBA](https://github.com/hku-mars/HBA); this experiment reuses N3M's existing
solver and does not import a new framework.

## Integration boundary

Use the isolated `release/install/n3mapping` together with
`dependency/gtsam_install`. Preserve build/install directories and the manifest.
Place the selected GTSAM prefix before the original one in the launch
environment and confirm the `graph_solver_dependency` event. An old overlay
will safely retain serial containment and will not deliver the parallel timings.
No package version or existing rc1/rc2/rc3 tag is overwritten.

To reproduce the dependency: check out the GTSAM commit above, apply both
upstream PR #1158 commits, and build Release with TBB ON, system Eigen ON,
`GTSAM_BUILD_WITH_MARCH_NATIVE=OFF`, MKL OFF and the TBB allocator, into a new
prefix. Select that prefix through `GTSAM_DIR` when building N3M. A rebuilt
library has its own build ID; rerun qualification rather than copying an old
expected ID. Do not rebuild the navigation team's installed dependency in place.
