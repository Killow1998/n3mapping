# n3mapping Final Acceptance Protocol — 2026-08-14

Recorded in `America/Los_Angeles`. This protocol was frozen before FA-01 or
FA-02 final runs. It turns the remaining closeout work into bounded release
questions; a failed gate may reopen only the responsibility that produced the
failure.

The machine-readable authority for thresholds and frozen episode selectors is
[`evidence/final_acceptance_20260814/acceptance_contract.json`](evidence/final_acceptance_20260814/acceptance_contract.json).

## Preregistration snapshot

- Target tree: `deepseek/n3m-architecture-hardening-recovery`
- Preregistration HEAD: `13c3aff4558cc10af64548b17acd1b0d587e79c3`
- Stable baseline: `main` at
  `784d8c1ca25f92e69f680cc6a4ec411bf054e586`
- Realtime line: `dev/n3mapping-realtime-pipeline` at
  `7b814f6e8f0cb850d8381ab4bcc622b965dfb67a`
- Topology: recovery has 205 commits after common ancestor `c5bc745`; realtime
  contains main plus seven commits and has nine commits not in recovery.
- KITTI-360 evaluator root: `/home/user/DUALoc/KITTI360`
- M2DGR evaluator root: `/home/user/dataset/M2DGR`
- Humble derived-LIO replay:
  `/home/user/Desktop/n3mapping_cleanup_pending_delete_20260817/workspace_cache/files/n3mapping_v1_closeout/repro_0723_exact/lio_ros2`
- Frozen floor7 map: `map/n3map.pbstream`

The absolute local paths identify this workstation's source assets. Final
manifests must record content/inventory hashes so a path alone is never treated
as provenance.

## Scope boundary

FA-01 is a Humble/Jammy target-host throughput gate. It measures complete ROS
callbacks and the mapping loop timer; it is not a claim for every supported
computer. CPU and RSS are reported as host-specific capacity observations.

FA-02 is a deterministic backend loop-closure gate using dataset ground truth.
It intentionally uses `gt_pose_plus_lidar`, not a recorded LIO frontend. This
is sufficient for judging loop candidate/measurement/graph behavior, but it
must not be relabeled as an end-to-end SLAM or raw-sensor localization gate.
No owner RViz judgment is part of FA-02.

FA-03 is semantic consolidation, not a wholesale merge. The target tree stays
on the recovery architecture while every unique main/realtime behavior receives
an explicit keep, superseded, reject, or defer decision.

## FA-01 — tri-mode runtime performance

### Measurement contract

All three modes use the same Release build, product profile, ROS domain,
derived-LIO bag, source commit, sensor rate, sampler interval, and headless
publication surface. Each run gets a fresh output directory and records:

- exact executable/build identity, config, bag metadata and map identity;
- one frame record for every synchronized callback, with explicit mode;
- mapping loop-timer records, including lock wait and processing time;
- `/proc` CPU, RSS, high-water RSS and thread samples restricted to the
  callback runtime window;
- expected and processed frame counts, over-budget runs, and dropped/backlog
  evidence;
- mode-specific quality invariants from the frozen reference evidence.

Initial map load and initial relocalization are reported separately from the
steady-state window. The steady-state boundary is frozen as follows: mapping
uses every valid synchronized frame callback; localization and map extension
start with the first ordinary or strict tracking callback *after* the first
`FULL_6DOF_LOCKED` transition. There is no additional warm-up exclusion. A
mode cannot substitute another input or omit its slow path and remain
comparable.

### Throughput gate

The replay has a 100 ms sensor period. For each mode, inside the frozen
steady-state window:

- at least 500 steady-state frame records are required;
- at least 99 percent of expected synchronized frames must be processed;
- callback p95 must be at most 100 ms;
- at most 5 percent of callbacks may exceed their sensor period;
- no run may contain more than five consecutive over-budget callbacks;
- no queue-overflow, OOM, fatal error, non-finite pose, or unexpected tracking
  loss is allowed;
- mapping loop-cycle p95 must be at most 100 ms and its maximum at most 500 ms.

CPU cores, host-capacity percent and RSS remain mandatory report fields rather
than portable pass/fail limits. A future deployment hardware contract may add
stricter limits without changing these recorded results.

### Failure ownership

- Slow `target_prepare_ms` in ordinary localization permits a bounded,
  revision-aware prepared-target cache. It does not permit relaxed registration
  or lock gates.
- Slow loaded-map cache misses permit target transition/prefetch work. They do
  not permit changing map geometry or tracking acceptance.
- Mapping callback or loop-timer spikes permit work-queue/critical-section
  changes only after the dominant stage is measured.
- A quality mismatch vetoes the optimization even if the runtime target passes.

## FA-02 — automated dataset loop gate

### Frozen runs

The final gate contains four mapping-loop episodes selected independently of
the final algorithm output:

1. KITTI-360 `2013_05_28_drive_0005_sync`, first 900 stride-5 aligned frames,
   official calibration, 0.5 m symmetric input voxel. This is the outdoor
   positive/revisit episode.
2. KITTI-360 `2013_05_28_drive_0003_sync`, first 900 aligned frames, official
   calibration, 0.5 m symmetric input voxel. This is the low-overlap outdoor
   control.
3. M2DGR `gate_02`, first 600 stride-5 aligned frames, 0.05 s maximum alignment
   error, normalized GT origin and 0.5 m symmetric input voxel. Its measured
   quaternion permits translation and rotation grading.
4. M2DGR `hall_05`, first 500 stride-5 aligned frames, 0.05 s maximum alignment
   error, normalized GT origin and 0.5 m symmetric input voxel. Its
   trajectory-derived yaw permits place/translation grading but not an
   authoritative full-attitude verdict.

The freeze step must write exact frame tokens and full SHA-256 payload hashes,
plus GT and calibration hashes. Missing payload, calibration, timestamp match,
or hash produces `INVALID_EVIDENCE`, never a partial PASS.

### Ground-truth semantics

- Minimum keyframe-ID separation: 20.
- Place opportunity: GT translation distance at most 5 m.
- Same-heading diagnostic: absolute GT yaw difference at most 45 degrees.
- Correct accepted measurement: place opportunity plus translation error at
  most 1 m and, where attitude is authoritative, rotation error at most
  10 degrees.
- Catastrophic false loop: GT place distance greater than 10 m, accepted
  measurement translation error greater than 2 m, or authoritative rotation
  error greater than 30 degrees.
- Revisit segments are maximal clusters of opportunity query IDs separated by
  no more than five keyframe IDs. One correct accepted loop covers a segment.

Pair-count coverage is diagnostic only because a single physical revisit can
create many redundant GT pairs.

### Loop and trajectory gate

Across every episode and in the pooled summary:

- catastrophic false-loop count must be zero;
- accepted place/measurement precision must be at least 0.95;
- every positive episode must accept at least one correct loop;
- revisit-segment recall must be at least 0.80;
- loop-induced graph output must contain no non-finite pose or optimizer error;
- ATE translation RMSE must be at most 0.50 m and p95 at most 1.00 m;
- ATE rotation RMSE must be at most 5 degrees where attitude is authoritative;
- consecutive-frame RPE translation RMSE must be at most 0.20 m;
- consecutive-frame RPE rotation RMSE must be at most 2 degrees where attitude
  is authoritative.

The final report must keep strict same-heading precision, place precision,
measurement correctness and recall separate. Relaxing a yaw label after seeing
results is prohibited; any label defect requires a new contract version and a
complete rerun.

## FA-03 — single canonical development line

The recovery tree is the integration base. Current preliminary decisions are:

| Unique lineage behavior | Preregistered treatment |
| --- | --- |
| `main` stable-runtime revert | Reject as target-tree content; recovery intentionally supersedes it. |
| Noetic transformed odometry twist | Keep pending focused parity/contract tests. |
| Realtime prepared-target matcher API | Superseded by recovery's current prepared source/target API. |
| Realtime ordinary-localization background target cache | Defer; eligible only if FA-01 attributes a failure to ordinary target preparation. |
| Stop localization odometry before lock | Superseded by current relocalization output authority; verify by tests. |
| Realtime odometry shadow topic removal | Superseded; the recovery tree does not publish that shadow topic. |
| Accept high-quality but unconverged GICP tracking | Reject; it weakens the current fail-closed registration contract. |
| Periodic Noetic JSON status | Keep only after consumer and typed-status compatibility audit; do not replace current authoritative relocalization status. |

After FA-01 and FA-02 pass, the retained semantics are implemented on a
temporary integration branch, all three histories are made reachable from the
result, and the result is promoted to canonical `main`. Remote branch deletion
is allowed only after exact ref targets, ancestry reachability, CI and recovery
commands are recorded.

## Terminal classification

- `PASS`: every required artifact is complete and every frozen gate passes.
- `FAIL_<RESPONSIBILITY>`: evidence is valid and one frozen criterion fails;
  reopen only that responsibility.
- `INVALID_EVIDENCE`: provenance, schema, frame count or hashes are incomplete;
  repair the evidence path without tuning the algorithm.
- `BLOCKED_INPUT`: a required local dataset/input is absent. This status is not
  currently applicable; all required roots were found during preregistration.
