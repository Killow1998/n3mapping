# n3mapping Final Industrial Closeout Review — 2026-08-15

Recorded in Pacific daylight time (PDT, UTC-07:00). This supersedes the
interim 2026-08-14 verdict in the previous revision of this file.

## Executive verdict

- **Final product closeout and single development line: PASS.**
- **Maintenance freeze for the qualified floor7 workflow: GO.**
- **General industrial release: NO-GO until the operational release gates
  below are closed.**
- **Further open-ended algorithm development is not recommended.**

The distinction is deliberate. The architecture roadmap, performance gate,
dataset-backed loop gate, build matrix, and branch integration are complete.
That is enough to stop feature development and maintain one canonical product
line. It is not evidence that every deployment computer, map scale, sensor
frontend, failure mode, or environment has been certified.

## What is now accepted

| Acceptance surface | Result | Scope |
| --- | --- | --- |
| Architecture-hardening roadmap | COMPLETE | Every work package has an evidence-backed PASS or an explicit NO-GO/default-off terminal decision. |
| FA-01 tri-mode performance | PASS | Same commit, bag, configuration, replay rate, and target host for mapping, localization, and map extension. |
| FA-02 dataset-GT loop closure | PASS | Automated backend loop-closure gate over KITTI-360 and M2DGR; no RViz or manual truth labeling. |
| Product build | PASS | Release, research tools OFF, verified identity for 8f11ea22e4df506519dbeca6d833a29870f0c8b9; 55/55 CTest targets and 494 assertions, no error/failure/skip. |
| Research build | PASS | Fresh isolated Release build, research tools ON; 60/60 CTest targets and 490 assertions, no error/failure/skip. It correctly identifies itself as non-product. |
| Remote pre-release CI | PASS | [Run 31878531142](https://github.com/Killow1998/n3mapping/actions/runs/31878531142): Humble/Jammy and Noetic/Focal build/test passed. |
| Canonical-main CI | PASS | [Run 31878902744](https://github.com/Killow1998/n3mapping/actions/runs/31878902744): Humble/Jammy and Noetic/Focal build/test passed after the fast-forward to unified main. |
| Branch semantics | PASS | Every main/realtime-only commit was reviewed; rejected and superseded code was not imported. Both histories are reachable from canonical main. |

The Product and Research builds intentionally have different identity
semantics. Research tools are required to run dataset evaluation but are not
eligible for a production bundle.

## FA-01: where the runtime cost actually is

The table is the n3mapping process only. FAST_LIO is an external frontend and
is not included in these CPU/RSS figures. Resource metrics remain report-only
until a deployment-hardware contract is declared.

| Mode | Callback p95 | Over 100 ms | Max consecutive over budget | CPU p95 | RSS p95 | Processed input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mapping | 0.337 ms | 0.000% | 0 | 0.020 cores | 82.6 MiB | 100% |
| Localization | 59.908 ms | 0.812% | 2 | 1.533 cores | 300.1 MiB | 100% |
| Map extension / resuming | 44.669 ms | 4.708% | 2 | 8.251 cores | 1305.4 MiB | 100% |

Confirmed conclusions:

- Map extension is the highest-cost mode by a large margin in CPU and memory.
- Localization is materially lighter than map extension on the same replay.
- All three modes pass the declared 100 ms callback p95, backlog/drop, fatal,
  OOM, non-finite pose, and quality gates.
- This is a same-host non-regression result, not a universal hardware sizing
  claim. In particular, map extension reached 1396.8 MiB maximum RSS and needs
  an explicit target-computer budget before a general release.

The durable result is
[FA-01 result](evidence/final_acceptance_20260814/fa01_verdict_2d9b359.json).
The formal runtime directories are:

- /tmp/n3mapping_fa01_mapping_2d9b359_20260815_011612
- /tmp/n3mapping_fa01_localization_2d9b359_20260815_011847
- /tmp/n3mapping_fa01_map_extension_2d9b359_20260815_011214

## FA-02: what KITTI-360 and M2DGR prove

FA-02 supplies ground-truth poses plus LiDAR clouds to the backend evaluator.
Ground truth is used only by the offline oracle; it is not consumed by product
runtime behavior.

Aggregate result:

- 118 accepted loop closures;
- 21/21 authoritative SE(3) measurements correct, precision 1.000;
- 94/97 position-only M2DGR hall loops place-consistent, diagnostic precision
  0.969;
- 0 catastrophic false loops;
- 3/3 positive revisit segments hit, recall 1.000;
- KITTI-360 drive 0003 and M2DGR gate_02 low-overlap controls accepted no
  loops;
- no optimizer error, non-finite trajectory, or graph-structure regression.

For KITTI-360 drive 0005, final translation ATE RMSE is 0.331 m and p95 is
0.689 m; rotation RMSE is 0.067 degrees.

This closes the automated **backend loop-closure** gate. It does not certify:

- LiDAR/IMU frontend quality or time synchronization;
- end-to-end public-dataset mapping/localization when runtime odometry comes
  from a real LIO;
- cross-session relocalization of arbitrary maps;
- the correctness of an autonomous multi-session map federation service.

Older KITTI-360/M2DGR relocalization diagnostics that had low lock recall asked
a different question and remain useful fail-closed evidence. They do not
contradict the now-passing backend loop-closure gate.

The durable result is
[FA-02 result](evidence/final_acceptance_20260814/fa02_result_ac54ed2.json).
The checksummed formal directory is
/tmp/n3mapping_fa02_ac54ed2_20260815_023901.

## FA-03: one development line without importing obsolete code

The recovery code tree was selected as the product tree. Main and realtime
histories were connected with audited ours merges, preserving ancestry
without changing the selected tree.

Key semantic decisions:

- reject the old main runtime-baseline revert;
- reject transformed odometry twist because the upstream FAST_LIO message
  mixes world-frame linear velocity and body-frame angular velocity under one
  child frame, so one rigid rotation cannot make both fields correct;
- keep the evolved bounded PreparedTarget/cache/prefetch implementation already
  present in recovery; realtime cache commits are superseded;
- keep the typed Humble/Noetic RelocalizationStatus and
  RelocalizationOutputAuthority; reject the unconsumed Noetic-only JSON
  heartbeat;
- reject acceptance of non-converged GICP tracking matches and preserve
  fail-closed behavior.

The complete decision ledger is
[FA-03 semantic audit](evidence/final_acceptance_20260814/fa03_semantic_integration.json).

## Industrial acceptance boundary

| Product question | Final classification |
| --- | --- |
| Can development stop on one maintained branch? | **YES** |
| Is the qualified floor7 mapping/localization/map-extension workflow suitable for controlled use? | **YES, within the frozen profile and evidence boundary** |
| Is backend loop closure dataset-tested without manual RViz review? | **YES** |
| Is arbitrary cross-session automatic merge/federation certified? | **NO; out of the frozen product scope** |
| Are CPU/RSS budgets certified on every deployment computer? | **NO** |
| Are large maps beyond the current 1 GiB prepared-Atlas/load contract solved? | **NO** |
| Are restart, disk-full, power-loss, long-duration, and resource-exhaustion cases certified? | **NO** |
| Is there a protected release branch, versioned release artifact, and rollback bundle? | **NO** |

The remaining NO answers are release/operations projects, not a reason to
restart the architecture-hardening algorithm roadmap.

## Maintenance-freeze policy

After ref cleanup, retain:

- one development branch: main;
- two permanent historical tags: archive/humble and archive/noetic.

Allow only:

- reproducible correctness, security, compatibility, and build fixes;
- release packaging and operator documentation;
- a bounded acceptance task with predeclared falsifiers.

Do not reopen by default:

- C1/C2/C3 threshold promotion or persistence authority;
- non-converged tracking acceptance;
- builtin-LIO duplication;
- shadow graph writeback;
- arbitrary automatic cross-session map federation;
- another unbounded performance or relocalization roadmap.

## If a general industrial release is later required

Open a separate release project containing only these gates:

1. **Deployment hardware contract:** declare supported computers and enforce
   per-mode CPU, RSS, latency, backlog, and thermal budgets.
2. **Automated end-to-end datasets:** run LiDAR/IMU through the selected frozen
   frontend and n3mapping; use KITTI-360/M2DGR ground truth only in the
   evaluator, never as runtime odometry. No routine manual RViz gate is needed.
3. **Operational fault matrix:** long-duration restart, power-loss, disk-full,
   corrupt-map, old-pbstream compatibility, and resource-exhaustion tests.
4. **Release governance:** protect main or a release branch, require the
   Humble/Noetic checks, update the package version, and publish a reproducible
   binary/config/rollback bundle with commit and profile hashes.

Until those four gates are requested and funded, the honest endpoint is a
maintenance freeze, not more feature development.
