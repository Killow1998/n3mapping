# Changelog

## Unreleased — Humble startup input and branch integration

- Merge `dev/noetic-core-modularity-20261019` into the current shared pipeline,
  retaining the newer asynchronous MapArtifacts and both thin adapters. Share
  dependency/protobuf setup and remove inherited wrapper includes from the core.
  Keep native colcon/catkin builds and immutable rc1/rc2 tags.
- Replace Humble's reliable depth-1 live history with the existing bounded sync
  queue, and coalesce pending input before projection. Preserve latest-input,
  timestamp, invalid-input and backend-authority behavior. Count coalesced samples;
  include drain time in the existing live processing timing.
- Strengthen the real-node logging E2E: input now runs during initial startup and
  process restart, without waiting for discovery or health before publication.
  Keep the existing delivery, latency, invalid-input/recovery, save and quota checks.

See [startup and merge evidence](docs/evidence/runtime_cpu_20261017/README.md#humble-startup-input-and-branch-integration--2026-10-23).

## Unreleased — shared bounded asynchronous logging

- Route both ROS adapters, glog and existing diagnostic streams through one
  ROS-free worker. Bound queued records by allocated capacity, batch file writes,
  aggregate frequent observations, and reserve queue space for critical events.
- Enforce a shared 100,000,000-byte ceiling over registered current logs, archives
  and restarts. Persist destination ownership, account for configured external
  streams, repair torn tails, and reject a second owner of the same instance.
- Add shared lifecycle/state/lock/loop/save events and memory-resident 1 Hz health
  reporting. Count queue loss, oversized records and write failures; preserve
  failure evidence across restart. Align Noetic frame/loop/live timing with Humble.
- Disable duplicate native node logs. Extend support bundles to collect managed
  health, events and rotated logs without following external catalog paths.
- Fix Noetic logging teardown after ROS has already removed its appender.
  Four-core development validation passes Humble 66/66 and Noetic 48/48, including
  actual-node write failure, invalid-input recovery, map save and process restart.

The original logging qualification waited for a ready executor. The following
startup correction removes that test precondition; see the newer startup
evidence. No new whole-route, target-hardware or Release qualification
is claimed. Existing rc1
and rc2 are unchanged. See [logging evidence](docs/evidence/runtime_cpu_20261017/README.md#shared-asynchronous-logging--2026-10-23).

## Unreleased — shared artifacts and ROS adapter alignment

- Share map snapshots, save orchestration, global-cloud caching and debug PCDs
  between Humble and Noetic through the ROS-free `MapArtifacts`. Preserve
  post-async-commit save ordering and existing map formats.
- Narrow session/loop/registration header dependencies. Three uncached compile
  samples per side show representative Humble backend/map compilation medians
  falling 10.354→4.293 s and 11.981→5.791 s; this is not a runtime-speed claim.
- Split the Noetic monolith into files matching Humble lifecycle, live output,
  backend, visualization and map responsibilities. Keep state/locks in one
  owner and algorithms in the shared core. Extend the existing boundary check
  to all split files without weakening live-queue/projection guards.
- Fix dependency preflight falsely matching GPU/research tokens in parent
  directory names; still check actual SONAMEs and resolved library filenames.
- Four-core RelWithDebInfo: Humble 64/64 and Noetic 46/46 tests pass, including
  real-node E2E. These are development builds, not a new Release candidate.
- Audit and record outstanding synchronous log I/O, incomplete cross-ROS event
  coverage and missing aggregate 100 MB retention. No logging-policy compliance
  or new full-route/runtime-performance qualification is claimed.

See the [stage evidence](docs/evidence/runtime_cpu_20261017/README.md#shared-map-artifacts-and-ros-alignment--2026-10-23).
Existing rc1/rc2 tags remain unchanged.

## Unreleased — Noetic validation follow-up

- Make the existing loop-detector fixture compatible with Focal's PCL 1.10.
- Align the existing Noetic node smoke test with the shared output contract:
  unavailable estimates have zero pose time, input association retains source
  time, and invalidated projections need a fresh trusted backend correction.
  Register this real-node test in CTest and retain invalid-input/recovery checks.
- Validate clean commit `19f099d` in Noetic/Focal Docker: full Release build and
  46/46 CTest targets pass on four physical cores. Audit all four configurations'
  compiler commands and run nine realtime cases in each non-Release profile.
  Humble's two affected fixture/contract targets also pass.
- Correct the earlier environment assessment: the host lacked a native Noetic
  underlay, but an existing Docker image was available. Install the declared
  Python 3.8 dependencies and use Focal-compatible CTest invocation.

Runtime algorithms and configuration are unchanged from rc2; `v1.1.0-rc1` and
`v1.1.0-rc2` remain immutable. See the
[Noetic evidence](docs/evidence/runtime_cpu_20261017/README.md#noetic-docker-validation--2026-10-23).

## [1.1.0-rc2] - 2026-10-23

Build-configuration candidate; ROS package versions remain `1.1.0` and the
`v1.1.0-rc1` tag is retained.

- Remove unconditional `-O3` from both wrappers. CMake's selected build type
  now controls optimization; Release retains `-O3 -DNDEBUG` on GCC.
- Document and validate ccache compiler launchers, retained caches, four-job
  builds and separate development/release build, install and log directories.
  Keep native colcon/catkin entry points and the clean-commit Product identity.
- Update the hardware instructions to the versioned rc2 install. No runtime
  algorithm, acceptance threshold, map format or default parameter changes.

See the [build-profile evidence](docs/evidence/runtime_cpu_20261017/README.md#build-profiles-and-rc2--2026-10-23)
for configuration, cache, regression and Release comparison results. The
subsequent Noetic build/node-smoke qualification is recorded above; full Noetic
route timing and target-hardware acceptance remain unverified.

## [1.1.0-rc1] - 2026-10-23

Hardware-test candidate; both ROS manifests use numeric version `1.1.0`.

- Consolidate the existing online asynchronous geometry/loop pipeline,
  incremental relocalization reuse, acquisition deduplication, bounded memory
  and real-data input/timing diagnostics. No new algorithm or acceptance rule
  is introduced by this release closeout.
- Split the Humble node by lifecycle, live output, backend processing,
  visualization, map IO and authority probing; keep one owner of state and locks.
  Link the real node and E2E test to one compiled implementation.
- Split extension lifecycle, keyframe transactions, cross-session loops and
  submap diagnostics. Fold redundant state/getter tests into the serialized-map
  lifecycle and remove an ID test that bypassed the extension API; retain real
  anchor/odometry commits, repeated-acquisition and failure-rollback coverage.
- Keep rejected geometry-cache and uncertainty prototypes out of production.
  Preserve their ablation evidence and the valid loop observations. Package
  version does not change the `pbstream` format or acceptance thresholds.
- Freeze an explicit real-hardware test boundary: remaining live errors above
  10 cm, difficult-case CPU cost, unqualified M2DGR reference/time calibration
  and correlated uncertainty are not declared solved.

Validation: 70/70 CTest, paired original14 and both motion routes, complete
fresh online extension and map/Atlas reload checks. See the
[closeout evidence](docs/evidence/runtime_cpu_20261017/README.md#hardware-test-release-closeout--2026-10-23)
and [field testing](docs/field_testing_zh.md) for isolated installation and
build-identity checks. The following development evidence records the changes
incorporated since 1.0.1, including their workload-specific limits.

### Incorporated development evidence

- Reuse exact acquisitions before plane fitting and extension graph insertion;
  preserve sampling, fresh continuation and all tracking/dense outputs. Four-core
  repeated-route CPU 0.857→0.408 cores, runtime RSS
  2429→1465 MiB; this applies to identical replay.
- Log actual selected tracking curvature separately from fixed edge noise.
  Reject covariance/factor-removal ablations that regress precision. Final original14,
  both routes, repeated/fresh online extensions, real checkpoints and 69 CTest
  targets pass; common deskew calibration remains open. See [evidence](docs/evidence/runtime_cpu_20261017/README.md#acquisition-identity-and-source-matched-uncertainty--2026-10-21).

- Expose geometric-loop information mode and fixed loaded-map tracking noise
  in the configuration summary, including the effective vertical-noise fallback.
  Correct uncertainty evidence to distinguish those constraint sources and
  exclude exact acquisition self-matches from independent calibration. Saved
  weights and registration behavior are unchanged; see the [repeat/source audit](docs/evidence/runtime_cpu_20261017/README.md#repeated-input-determinism-and-constraint-source-correction--2026-10-21).

- Prune exact correspondence search using occupied subtree bounds and retain
  Atlas visibility crops under the existing byte budget. Four-core, three-repeat
  03 medians: 0.569→0.552 cores; startup CPU 10.359→9.199 s. Original14 and both
  routes retain every decision and their error maxima; 69 CTest targets, 1,712
  paired registration solves and a real checkpoint pass. Half-core remains open;
  see [timing and limits](docs/evidence/runtime_cpu_20261017/README.md#occupied-search-bounds-and-scan-level-loop-bias--2026-10-21).
- Trace the worst audited mapping-loop bias to both endpoint scans' deskew:
  16.757→0.495 mm in an observer-only cross-ablation. Record the shared scan-bias
  ambiguity before further uncertainty calibration; production weights remain
  unchanged. Remove unsuccessful arithmetic/traversal variants from production.

- Resume fresh relocalization confirmation at its accepted geometric scale,
  retrying the full cascade for failed or large local corrections. Original14
  and both independent routes preserve all decisions and outputs; all 69 CTest
  targets pass. Keep 03 half-core open and retain explicit precision/timing
  limits in the [paired evidence](docs/evidence/runtime_cpu_20261017/README.md#confirmation-scale-reuse-and-session-separated-uncertainty--2026-10-21).
- Audit spatially correlated loop residuals and voxel-phase sensitivity by
  session; remove five unsuccessful search prototypes. Production loop weights
  remain unchanged because the proposed floors do not explain common bias.

- Share loaded Atlas tracking targets and confirm one fresh prior-pose recovery
  candidate without suppressing descriptor regions or inheriting old votes.
  New-route recovery advances 6.4 input seconds; paired CPU
  0.820→0.363 cores. Full original14 and both routes retain outputs
  below 10 cm; all 69 CTest targets pass. Small precision tradeoffs and remaining
  03/14 cost are explicit in the [evidence](docs/evidence/runtime_cpu_20261017/README.md#shared-atlas-tracking-and-fresh-recovery-confirmation--2026-10-21).
- Diagnose opposite-wall reference coverage and fitted-residual overconfidence;
  reject the slower thread split and unqualified global robust-weight changes.
  Keep production loop weights pending directional/correlation calibration.

- Bound local tracking recovery at the existing voxel scale, retaining broad
  geometric admission, full-query coverage and first-return checks. Fresh-route
  maximum 6.436→3.913 cm and paired CPU 0.849→0.819 cores; recovery lock remains.
- Pack temporary map assembly without changing raw clouds or voxel values.
  Paired complete-extension runtime RSS 2.088→1.816 GiB; CPU remains comparable.
- Complete original14 before/after repeated extension, both motion routes, three
  extensions, real checkpoints and final-source 69-target regression/equivalence
  checks. Remove the no-benefit seed-refresh prototype and reject unproven uniform
  weight changes. Half-core and broad uncertainty calibration remain open; see
  the [recovery/resource evidence](docs/evidence/runtime_cpu_20261017/README.md#recovery-geometry-compact-map-assembly-and-uncertainty-ablation--2026-10-21).

- Correct newly appended mapping keyframes using committed plane support before
  background refinement; transactionally preserve observations and re-associate
  worker tails when plane indices change.
- Divide geometry information for exactly repeated raw acquisitions, preserve
  robust residual scales, and persist weights in map format 2.10.0.
- Complete four-core mapping, original14 before/after extension, original and newly
  recorded motion, repeated/fresh extensions, real checkpoints and 69 CTest targets.
  Mapping maximum 11.593→9.289 cm,
  with zero outputs over 10 cm and 0.413 average cores. Record small
  accuracy tradeoffs, the fresh route's existing recovery lock and remaining
  extension/covariance limits in the [timely geometry evidence](docs/evidence/runtime_cpu_20261017/README.md#timely-keyframe-geometry-and-repeat-input-information--2026-10-21).

- Qualify two saved/reloaded extensions on repeated input, original14 after both,
  and a three-session map/Atlas/trajectory checkpoint on unchanged `a16a35d` code.
  CPU 0.871/0.882 cores; live maximum
  4.280/7.237 cm. Record small map/static regressions and correlated-input limits.
- Diagnose remaining mapping peaks using causal input prefixes and exact adjacent
  registration replay. Existing geometry can correct the tested anchors, but
  timely online delivery remains open; no gate is relaxed. See the
  [multisession and causal evidence](docs/evidence/runtime_cpu_20261017/README.md#repeated-extension-sessions-and-causal-peak-diagnosis--2026-10-21).

- Refresh existing shared geometry even when the current loop-query job admits
  no loop. Preserve independent loop gates and the async commit boundary.
- Complete four-core mapping, original14, independent route, real checkpoint
  and 69 CTest targets. Mapping online RMSE 2.199→1.702 cm;
  CPU 0.377→0.410 cores. Eight outputs still exceed 10 cm;
  maximum 11.593 cm. Small metric regressions and two omitted
  valid loop pairs are recorded in the [geometry refresh evidence](docs/evidence/runtime_cpu_20261017/README.md#geometry-refresh-independent-of-loop-admission--2026-10-21).

- Reuse exactly unchanged cross-level neighborhoods and reconstruct fixed-spectrum
  target covariance from normals. Avoid temporary covariance arrays at Atlas load;
  retain arbitrary serialized matrices and the existing map/Atlas formats.
- Final four-core Atlas ABBA reduces CPU 7.953→6.261 s and removes
  264.661 MiB of covariance storage on the tested map. Full mapping, three
  original14 suites, independent routes, extension and real checkpoints retain
  the reliability baseline; all 69 CTest targets pass. Whole-extension half-core
  and online mapping peaks remain open. See the [Atlas evidence](docs/evidence/runtime_cpu_20261017/README.md#exact-atlas-feature-reuse-and-compact-target-covariance--2026-10-21).

- Fix tracking recovery's missing global-fitness check, make pose smoothing
  invariant to odometry origin, and preserve Atlas-verified extension output
  across conditioned graph append solves.
- Partition incoherent plane groups without relaxing geometry gates; budget and
  coalesce extension refinement while retaining all constraints. Reduce point
  preparation allocations and retain only required features at each Atlas level.
- Complete four-core full mapping, two original14 suites, independent motion,
  extension, real checkpoint/reload and budget-off ablation. Moving maximum
  16.512→7.744 cm on the same map; extension maximum
  21.505→4.278 cm, CPU 2.036→0.887 cores,
  runtime RSS 4.186→2.244 GiB. Maximum pending correction age
  32.73 s; whole-extension half-core and all-frame
  mapping below 10 cm remain open. Small map/static regressions are recorded.
  All 69 CTest targets pass. See the [refinement evidence](docs/evidence/runtime_cpu_20261017/README.md#tracking-authority-plane-partition-and-extension-budget--2026-10-21).

- Integrate shared-plane joint optimization into asynchronous mapping and strict
  extension. Catch up appended frames without repeated registration/global solves;
  install optimized map/Atlas/tracking references together. Format 2.9.0 retains
  plane moments, gauge and dense correction provenance across save/reload.
- Normalize rotation matrices at graph boundaries to prevent repeated async
  transport from amplifying numerical scale error. Isolate dense correction by
  session, including overlapping timestamps. Share Atlas preparation at load.
- Complete four-core paced mapping, original fixed14, independent motion, strict
  extension, reload continuation and another full fixed14 on the extended map.
  That integration baseline averaged 0.412 mapping cores and 2.036 extension
  cores, with 4.186 GiB extension runtime RSS. Both fixed14 runs retained every
  output. At that stage, complete motion below 10 cm and improvement in every
  metric remained unqualified. Real checkpoint
  Atlas points match exactly; all 69 CTest targets pass across final runs.
  See the [online integration evidence](docs/evidence/runtime_cpu_20261017/README.md#asynchronous-joint-geometry-mapping-extension-and-checkpoints--2026-10-21).

- Add an opt-in shared-plane geometry evaluation tool with oriented association,
  per-scan moments, connected-surface aggregation and the existing consistent
  robust-noise implementation. Frozen four-core full14 and independent-route
  comparisons demonstrate lower map and localization error, while retaining
  all lock/output counts. Directional null-space and robust-gradient controls
  pass; causal mapping snapshots also improve without future observations.
  That earlier research stage left individual angular regressions, the 15.28 cm
  moving peak, statistical uncertainty calibration and online integration open;
  it changed no runtime defaults. The integration above supplies the subsequent
  runtime and checkpoint evidence. See the
  [geometry association evidence](docs/evidence/runtime_cpu_20261017/README.md#oriented-shared-plane-geometry-and-directional-information--2026-10-20).

- Log raw tracking registration and visibility-selected endpoints before
  correction smoothing, with explicit validity flags. Frozen four-core
  map-only comparisons reject the tested joint-geometry and propagation
  candidates; the accepted localization algorithm and map remain unchanged.
  The 17.28 cm moving-route peak is reproduced by a low-overlap original-scan
  fallback, while the same scan and actual prediction admit a 6.42 cm
  full-map diagnostic fit. This diagnoses the peak without claiming a
  qualified repair or a new online geometry backend. See the
  [complete comparison](docs/evidence/runtime_cpu_20261017/README.md#spatial-geometry-ablation-and-motion-peak-attribution--2026-10-20).

- Calibrate mapping constraints using original scan-pair refinement and neighbor
  disagreement in the relative-pose covariance. Exclude query self-evidence and
  restore distinct short-revisit neighbor measurements. Main mapping YAML uses
  the tested ICP-information mode; numerical gates and the frozen localization
  profile are retained.
- Correct robust nonlinear cost versus IRLS linearization and log the actual
  post-calibration edge information. Share fine constraint refinement and split
  adjacent-constraint construction out of the main core.
- Admit bounded/original-scan tracking recovery only after complete fine,
  motion, global-coverage and published-pose ray checks. A rejected candidate
  cannot suppress the original retry. Complete partial-map replay adds no wrong output or geometric
  confirmation; the existing ambiguous global lock remains unresolved.
- All original 14 stationary recordings now stay below 10 cm on the causal
  rebuilt map: 4.40 cm pooled RMSE / 9.04 cm maximum, with every position and
  angular RMSE/maximum improved. Complete independent motion has one
  initialization, five geometry failures, 5.06 cm RMSE and 17.28 cm maximum.
  Its remaining 192 outputs above 10 cm keep full-route acceptance open.
  Full CTest passes 61/61; default/Atlas/kill each pass 15/15.
  See the [complete qualification](docs/evidence/runtime_cpu_20261017/README.md#independent-constraint-calibration-and-tracking-recovery--2026-10-20).

- Correct loop information coordinates when a pretransformed source estimates a
  correction and when a full-six-DoF edge is reversed. Two physical residual
  regressions fail on the old core and pass on the repair. Original fixed14,
  independent motion, strict extension and default causal mapping retain their
  behavior; full CTest passes 61/61.
- Earlier diagnostic stages attribute stationary 05's accuracy regression to local map motion,
  with only 0.162 mm residual after diagnostic transport. Complete map/query
  ablations reach 10.86 cm pooled fixed14 RMSE but still regress individual
  position/angular metrics; those candidates were withdrawn. The subsequent
  calibration above supersedes that candidate-only result. Correct information
  coordinates do not certify covariance calibration or 10 cm accuracy.

- Fix ordinary tracking accepting a distant coarse registration without fine
  geometric confirmation. Enforce existing motion limits and recheck suspicious,
  otherwise admissible proposals with prepared fine GICP from the prediction.
  Reject failed rechecks; new recovery authority additionally requires full-map coverage.
  Complete original fixed14 behavior and strict-extension outputs are preserved.
  The full-map motion recording has no remaining >1 m / >5° output; the separate
  aliased partial-map global lock remains unresolved.
- Use canonical physical roll/pitch/yaw for loop admission, segment consensus
  and diagnostics. Small roll corrections no longer become false half-turns.
  Complete causal mapping increases accepted loops 195→398 and improves revisit
  translation RMSE 3.45→2.61 cm. Rebuilt-map accuracy still has local regressions;
  the subsequent calibration above addresses those local regressions.
- Add research-only map × deskew ablation tools and complete four-core accuracy
  evidence. Independent authored geometry and a mapping-only known-trajectory
  control isolate map bias from within-scan deskew error on this fixture.
  Normal reference-map localization passes all 14 stationary recordings below
  10 cm; the subsequent tracking repair brings all 2,232 independent motion
  outputs below 10 cm on that reference too. Production accuracy is not
  qualified by truth-assisted reference construction.
- Evaluate loop covariance calibration through complete causal online mapping,
  all 14 original recordings, independent motion and strict extension.
  Reject promotion despite lower position RMSE: stationary 08 and moving-route
  rotation regress. Preserve numerical defaults, quality gates and the frozen
  product profile; clarify constraint units and dataset-specific assumptions.
- Fix strict extension becoming permanently LOST and a valid ICP proposal being
  overwritten by a prediction that fails the original quality gate. Fresh
  geometry is still required before publishing or adding a keyframe.
- Keep loaded-reference crop centers consistent with frozen map geometry.
  Share one prepared registration target per loaded map, retain local raw
  visibility crops, and remove speculative preparation workers. Complete
  independent motion retains all 500 old successful frames and now delivers
  all 2,232 post-initialization outputs. Against the reliable per-anchor
  control, CPU falls 896.7→186.5 s and peak RSS 2.99→1.86 GiB.
- Run online mapping-loop geometry and graph solving on an isolated worker.
  Keep retrieval with the owner, preserve graph updates arriving during a
  solve, reject obsolete map generations and drain pending work before saving.
  Humble and Noetic use the common implementation; strict extension tracking
  constraints remain inline. Add queue/worker/snapshot/commit diagnostics.
- Add complete loop-on/off consistency evidence: repeated visits improve, but
  final map absolute error remains above 10 cm. The complete 10 Hz asynchronous
  replay preserves 195 loops and 13,723 outputs without a 100 ms owner overrun.
- Batch online loop-neighborhood verification by target geometry, preparing one
  target at a time while retaining each candidate's independent solves and
  consensus. Split registration and graph admission into separate implementation
  files shared by mapping and map extension.
- Reuse prepared source/target geometry across cross-session yaw hypotheses.
  Keep candidate graph trials; run the alternative consensus-measurement graph
  trial only when its mapping debug consumer is enabled.
- Treat zero-correspondence registration as absent neighborhood evidence, even
  if the optimizer reports convergence. It supplies neither confirmation nor
  contradiction.
- Four-core causal replay preserves 195 loops and 13,723 mapping outputs from
  13,791 inputs. Mapping-loop CPU falls 20.0%, with unchanged numerical gates and
  pose differences below 1e-8 m/rad. Full CTest: 61/61.
- Reject pose interpolation after full-route ablation: it adds CPU and worsens
  position error despite matching the short-prefix results. Retain behavioral
  recovery, stale-map, graph-tail and save/drain coverage.
- The 10 cm absolute-accuracy goal, extension half-core budget, cold startup
  latency and complete ROS-node throughput remain open. Common-frame extension
  position error improves, while rotation RMSE changes 0.133→0.138°.

See the [online-loop evidence](docs/evidence/runtime_cpu_20261017/README.md#online-loop-verification-and-accuracy-follow-up--2026-10-20)
and [recovery/asynchronous-loop qualification](docs/evidence/runtime_cpu_20261017/README.md#extension-recovery-and-asynchronous-online-loops--2026-10-20).
The subsequent [reference-map, deskew and loop-calibration ablation](docs/evidence/runtime_cpu_20261017/README.md#reference-map-deskew-and-loop-calibration-ablation--2026-10-20)
separates stationary map bias from the moving tracking failure; the
[tracking and loop-angle follow-up](docs/evidence/runtime_cpu_20261017/README.md#tracking-admission-loop-angle-branches-and-map-deformation--2026-10-20)
records its repair, rejected recovery variants and rebuilt-map accuracy limits.

## [1.0.1] - 2026-10-20

### Maintenance

- Consolidate the accepted localization performance and reliability work: incremental candidate admission, bounded seed-proposal reuse, fresh continuous confirmation, prepared registration/visibility data, exact free-space queries and full-circle relocalization yaw planning.
- Keep solver stages and numerical acceptance gates. The original 14-position paired benchmark reduces process CPU by 15.3% (0.613225 to 0.519290 cores) and first-lock median wall time from 4.799 to 3.570 s on four physical P cores. Loading, frontend, simulator and DDS are excluded; difficult half-core operation remains unqualified.
- Keep seed planning in its own implementation file, remove an overwritten tracking-confidence calculation, and remove a settings-readback test that did not exercise registration behavior. Preserve behavioral regression and E2E coverage.
- Withdraw source-layout, direct fine-tracking, neighbor-certificate and registration-kernel experiments that lack consistent overall benefit or regress accuracy/latency. Retain their evidence, without adding runtime switches or abstractions.
- Pause further performance research. The next priorities are relocalization accuracy and the existing online loop path during mapping/map extension. The 10 cm requirement and loop-consistency improvement remain unqualified; offline replay will validate online behavior.

Validation and artifact identities are recorded in the [maintenance closeout](docs/evidence/runtime_cpu_20261017/README.md#maintenance-release-closeout--2026-10-20).

## Development notes - 2026-05-19

### Architecture

- Split the backend into a ROS-free `n3mapping_core` library and a thin Humble wrapper target.
- Move Humble node implementation into `humble/src/` and keep backend ownership inside `N3MappingCore`.
- Add ROS-independent core frame/output types for external-LIO integration without depending on ROS message headers.
- Add Humble conversion/config helper layers so parameter loading and message conversion stay outside core code.
- Make `noetic/` and `humble/` sibling wrapper packages that both keep the ROS package name `n3mapping`, share the root core and config, and keep wrapper-local launch resources.
- Add a Noetic wrapper under `noetic/` for Noetic integration without changing the shared ROS-free core.
- Move Humble wrapper sources into `humble/src` and `humble/include` so the root source/include trees contain only ROS-free core code.
- Add CMake modules for core, Humble wrapper, tests, and boundary checks.

### ROS Wrapper and Runtime

- Build the `noetic/` package under ROS 1 Noetic with catkin while preserving the same ROS-free core used by Humble.
- Promote the Noetic wrapper from skeleton to a runnable thin adapter that subscribes to external LIO cloud/odometry, calls `N3MappingCore`, and publishes odometry, path, body/world clouds, loop markers, global map, relocalization lock, TF, and `/n3mapping/save_map`.
- Preserve Noetic launch entry points for `mapping`, `localization`, and `map_extension`; runtime topic overrides should be supplied through an external `config_file` instead of changing shared config resources.
- Use one Noetic RViz config for all Noetic launch files and one Humble RViz config for all Humble launch files.
- Add Noetic synthetic relocalization eval and RViz visualization tools so saved maps can be tested under Noetic as well as Humble.
- Require an explicit `map:=...` for the Noetic synthetic visualization launch instead of defaulting to a package-relative map path that does not exist after the wrapper split.
- Disable Noetic synthetic RViz save prompts to avoid committing local window-state churn after interactive tests.
- Add `rviz:=false` support to Humble mapping, localization, map extension, and synthetic relocalization launch files while keeping RViz enabled by default.
- Disable Humble RViz save prompts in the shared Humble RViz configs to avoid local interactive-test churn.
- Document the shared Humble/Noetic `rviz:=false` headless workflow and keep synthetic visualization launches explicit about `map:=...`.
- Add Humble wrapper parity fixes for `/n3mapping/save_map`, global map timer/fallback publishing, TF timestamp guarding, loop timer core mutex protection, and map-extension relocalization-lock frame publication.
- Align Humble invalid-frame return logic with Noetic so mapping/map-extension failed non-keyframe frames are dropped while localization fallback poses still publish.
- Share Humble/Noetic `optimization.log` semantics for optimization diagnostics and reduce terminal optimization spam.
- Document the ROS 2 `std_srvs` dependency in the README.
- Update wrapper package maintainer metadata to `killow <killow1998@gmail.com>`.
- Add the repository BSD-3-Clause `LICENSE` file.
- Split wrapper launch resources so Humble installs only ROS 2 `.launch.py`/RViz files and Noetic installs only ROS 1 `.launch`/RViz files, while both continue to use the shared config directory.
- Confirm the split launch resources under Noetic: catkin build, local ctest, headless `mapping`/`localization`/`map_extension` launch smoke checks, `/n3mapping/save_map`, `optimization.log`, and package-share launch-resource checks all pass without additional code changes.
- Document the wrapper-local launch naming rule in the README so Humble users launch `*.launch.py` files and Noetic users launch `*.launch` files after selecting the matching wrapper.
- Add a README heading before the Humble/Noetic command blocks so build, run, and test commands are visually separated from map/log documentation.
- Add `scripts/select_distro_wrapper.sh` as a local wrapper-profile switch with `auto`, `status`, `noetic`, `humble`, and `clear` modes, avoiding committed mutually exclusive ignore files.
- Move run-mode parsing and frame dispatch into the shared core API (`CoreRunMode`, `processFrame`) so Noetic/Humble wrappers do not duplicate backend mode selection.
- Move map snapshot save semantics into `N3MappingCore::saveMapSnapshot()` so ROS wrappers do not duplicate backend save logic.
- Add shared runtime config fields for `global_map_publish_hz` and `save_global_map_voxel_size` so published-map and saved-map resolution can differ across wrappers without changing the shared config file defaults.
- Preserve the original `n3mapping_node` executable while routing mapping, localization, map extension, pending loop processing, map save/load, and global map save through `N3MappingCore`.
- Publish the global map during mapping mode so RViz can inspect the current map without switching modes.
- Keep loop closure marker history instead of replacing old loop markers with the newest loop only.
- Add loop optimization impact diagnostics and write optimization logs to local log files instead of relying on terminal output.
- Update package metadata and README to describe the RHPD-primary backend, ROS-free core, Humble wrapper workflow, dependencies, launch files, tests, and synthetic relocalization tools.

### Synthetic Relocalization Validation

- Add a synthetic relocalization core smoke test and descriptor-driven relocalization evaluation tool.
- Add a matrix runner for dropout, noise, yaw, and query-source sweeps.
- Add a Humble publisher and RViz visualization launch for before/after/ground-truth relocalization clouds.
- Add random periodic visualization mode with configurable test count, interval, random seed, dropout, noise, and fake odometry yaw.
- Report matched keyframe ID, translation/yaw accuracy, and pass/fail markers in synthetic relocalization outputs.
- Support `same_keyframe`, `local_submap`, and diagnostic `global_map` synthetic query sources.

### Relocalization

- Add a relocalization-only frame-level RHPD index to improve recall for queries that match individual historical frames better than submap descriptors.
- Add `RhpdFrame` candidate source tracking while preserving RHPD-primary and ScanContext-auxiliary semantics.
- Improve ScanContext yaw verification by testing adjacent sector yaw hypotheses instead of a forced 180-degree alternative.
- Select relocalization basins using ICP fitness plus fused descriptor score.
- Define temporal winner continuity by the physical LiDAR pose at the current odometry state instead of the nearest keyframe ID or origin-dependent raw transform translation.
- Require moving multi-view windows to provide non-negative ray-consistency evidence while keeping stationary descriptor-led relocalization unchanged.
- Add an offline oracle/descriptor registration-seed probe to isolate retrieval, initialization, registration, and lock-decision failures without exposing ground truth to runtime localization.
- Correct KITTI-360 evaluation to compose pose/IMU, camera, and Velodyne frames using the official calibration contract; default `auto` to this contract and fail closed when required calibration is missing or malformed.
- Freeze and verify KITTI-360 official calibration hashes in episode manifests, verify previously recorded GT hashes before execution, and require an explicit escape hatch for legacy unhashed calibration evidence.
- Finalize episode benchmark outputs with payload checksums and an atomic `COMPLETE` sentinel; require finalized evidence in the diagnostic tool unless legacy input is explicitly allowed.
- Write oracle candidate diagnostics to a separate finalized output directory bound to the input manifest and benchmark hashes, leaving benchmark evidence immutable.
- Extend the offline registration probe with the exact production fitness/inlier/confidence sub-gates, initial-vs-refined visibility endpoint decision, and per-projection resolution/coverage/support/conflict counts, sharing one behavior-preserving quality calculation with runtime basin initialization.
- Add a point-budgeted `registration_probe_motion` evaluator mode for KITTI-360 and M2DGR, recording query frame/point counts so motion-footprint evidence can be compared with single-scan registration without changing production behavior.

### Tests and Checks

- Add core type, config, Humble conversion, `N3MappingCore`, synthetic relocalization, no-ROS-core, Humble wrapper-boundary, and Noetic wrapper checks.
- Keep existing RHPD, loop detector, map serializer, loop closure, and relocalization regression tests in the refactored target layout.
- Validate the current test suite at 272 tests passing after the wrapper split and synthetic relocalization additions.
- Add global-coordinate moving-winner coverage, registration-probe precondition checks, official KITTI-360 calibration direction/fail-closed coverage, and debug evidence fields.
- Revalidate Noetic after Humble wrapper parity updates: `catkin build n3mapping --no-status -j2 --catkin-make-args run_tests`, local `ctest --output-on-failure`, headless `mapping`/`localization`/`map_extension` launch initialization, and `/n3mapping/save_map` service discovery all pass.

### Retrieval Pipeline

- Make RHPD the primary descriptor for mapping loop candidate retrieval and global relocalization candidate recall.
- Keep ScanContext as auxiliary only: yaw hint, weak rerank, optional loose veto, and fallback when RHPD is disabled or unavailable.
- Split loop candidate descriptor fields into explicit `rhpd_distance`, `sc_distance`, `yaw_diff_rad`, source flags, and fused score to avoid mixing RHPD distance into ScanContext semantics.

### RHPD

- Add lightweight negative-space, vertical-token, and PCA-anisotropy-confidence augmentation without training or learned regression.
- Fix 180-degree yaw flip handling across XY, XZ, YZ planes and negative-space sectors.
- Upgrade map metadata version to `2.2.0` so old maps rebuild RHPD under the new descriptor semantics.

### Bug Fixes

- Fix loop edge measurement direction: `MatchToQuery` uses `T_match_query`; `QueryToMatch` uses `T_match_query.inverse()`.
- Fix mapping-loop ICP measurement composition by applying residual correction to the optimized pose estimate.
- Fix yaw initialization in loop verification and relocalization to avoid rotating translation around the map origin.
- Harden map deserialization against malformed point cloud, ScanContext, RHPD, and information-matrix fields.
- Preserve serialized edge information matrices when rebuilding graph optimizer factors from loaded maps.

### Tests

- Add RHPD dimension, sparse-cloud, distance symmetry, 180-degree yaw, and structure-separation coverage.
- Add descriptor configuration injection coverage for ScanContext dimensions/sector angle and RHPD range/height bounds.
- Update loop closure direction tests and relocalization tests for the RHPD-primary path.
- Add malformed RHPD serialization coverage.

## [1.0.0] - 2026-04-08

### Release

- Align changelog versioning with `package.xml` (`1.0.0`) as the first stable package version.
- Keep runtime interfaces unchanged (`mapping` / `localization` / `map_extension`) and preserve existing Humble launch/build workflow.

### Documentation and Naming Consistency

- Rename test files/targets to current module semantics (`WorldLocalizing`, `MappingResuming`) to remove legacy module naming residue in the test surface.
- Clarify that `loop_closest_id_th`, `loop_min_id_interval`, and `loop_max_range` are compatibility/legacy parameters and are not used in the active mapping loop-candidate retrieval/verification path.

### Migration Continuity

- Preserve the migration continuity established in `0.3.0` without introducing new runtime features in this release.

## [0.3.0] - 2026-03-25

### Migration

- **Noetic -> Humble behavior alignment**: Port noetic-validated behavior to Humble without Noetic fallback path.
- **Hybrid ScanContext integration**: Add and wire Hybrid ScanContext pipeline into loop/relocalization flow.
- **RHPD integration**: Add RHPD descriptor compute/store/load chain and integrate with keyframe + map serialization pipeline.
- **Relocalization temporal logic alignment**: Migrate temporal-hypothesis/retry behavior and unstable-submap strategy from noetic flow.
- **Loop pipeline alignment**: Use noetic-style distance-candidate selection + ICP verification in mapping loop handling.

### Map Versioning

- **Map metadata version**: `MAP_VERSION` upgraded to `2.0.0`.
- **Load policy enforced**:
  - `file_version < 2.0.0`: allow load and rebuild RHPD.
  - `file_version > 2.0.0`: reject load.
  - `file_version == 2.0.0` with missing/invalid RHPD: rebuild RHPD.
- **Save policy enforced**: Always save map with version `2.0.0` and complete RHPD fields.

### Stability Fixes

- **Global map save crash fix**: Replace unstable voxel-grid path in `saveGlobalMap` with deterministic manual voxel aggregation to eliminate test/runtime segfault in map serializer path.
- **Humble build compatibility fixes**: Resolve pointer/allocation and parameter type issues encountered during migration integration.

### Configuration and Docs

- **Humble config synchronization**: Sync noetic-validated parameter set and defaults into Humble config path, including loop distance-candidate, relocalization temporal/retry, and RHPD parameters.
- **README update**:
  - Keep English Humble workflow.
  - Add explicit `gtsam` build step with noetic-aligned CMake args:
    - `-DGTSAM_USE_SYSTEM_EIGEN=ON`
    - `-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF`
- **Workspace layout consistency**: Keep `map/` directory in git using `.gitkeep` + `.gitignore` exception to align branch directory layout.

### Validation

- Build/test validated under memory-safe parallel limits:
  - `--parallel-workers 2`
  - `CMAKE_BUILD_PARALLEL_LEVEL=2`
  - `MAKEFLAGS=-j2`
- Test result: **15/15 passing**.
- Startup verification: `mapping`, `localization`, `map_extension` all initialize `n3mapping_node` successfully in short-run launch checks.

## [0.2.0] - 2026-02-27

### Refactor

- **Rename module**: `RelocalizationModule` → `WorldLocalizing` to better reflect its responsibility (global relocalization + tracking localization)
- **Rename module**: `MapExtensionModule` → `MappingResuming` for clearer semantics around map resumption
- **Extract LoopClosureManager**: Decouple loop verification, filtering, edge construction, and optimization triggering from n3mapping_node into a dedicated class
- **Extract ModeHandlers**: Move mapping/localization/resuming processing logic into standalone handler classes (`MappingModeHandler`, `LocalizationModeHandler`, `MapResumingModeHandler`) to reduce node complexity
- **Code formatting**: Apply clang-format uniformly across all source files

### Features

- **Global map visualization**: Automatically load and publish `global_map.pcd` on `/n3mapping/global_map` at startup in localization mode, using `transient_local` QoS to ensure late-joining subscribers receive the map
- **Localization RViz config**: Add `launch/n3_loc.rviz` with pre-configured displays for point clouds, odometry, path, TF, and global map
- **Loop closure consistency gate**: Validate ICP measurements against the existing pose graph before adding loop edges; reject outlier loops based on pre-error checks
  - New parameter `loop_max_pre_translation_error` (default: 2.0m)
  - New parameter `loop_max_pre_rotation_error` (default: 0.2rad)
  - New parameter `loop_use_icp_information` (default: false)
- **Loop closure diagnostics**: Switch optimization log to overwrite mode; add detailed fields: loop_edge, loop_pose, loop_meas, loop_info, loop_result, loop_post_pose
- **Color-coded trajectory markers**: `publishPathMarkers()` publishes MarkerArray on `loop_closure_markers` topic with blue LINE_STRIP (loaded map trajectory) and green LINE_STRIP (new trajectory)
- **Localization path separation**: `publishPath()` uses real-time `localization_path_` in localization mode; iterates keyframes in mapping/resuming modes

### Bug Fixes

- **Information matrix ordering** (`point_cloud_matcher.cpp`): small_gicp outputs the Hessian in `[rx,ry,rz,tx,ty,tz]` order, but GTSAM expects `[tx,ty,tz,rx,ry,rz]`. Fixed 6×6 block swapping in `align()`, `alignCloud()` coarse registration, and `alignCloud()` fine registration (3 locations)
- **Z-axis jitter in corridor environments** (`world_localizing.cpp`): In corridors, floor/ceiling horizontal planes provide weak Z constraints in GICP, yet overall fitness score remains high from wall matching, causing Z oscillation to propagate via high fusion weight. Fix: compute Z-axis fusion weight independently (`w_icp_z = confidence × 0.3`, capped at 0.5), while XY retains the full weight

### Configuration

- `loop_noise_position`: 0.01 → 0.3
- `loop_noise_rotation`: 0.01 → 0.1
- `loop_min_inlier_ratio`: 0.6 → 0.7
- `robust_kernel_delta`: 100.0 → 1.0
- `loop_max_pre_translation_error`: 5.0 → 2.0
- `loop_max_pre_rotation_error`: 1.0 → 0.2

### Tests

- All test cases updated to use renamed classes (`WorldLocalizing`, `MappingResuming`, `MappingResumingState`)
- Add `test_loop_closure_manager.cpp`: covers loop filtering, best-per-query selection, edge direction, and optimizer invocation
- 15/15 tests passing

### File Change Summary

- Added (9): `world_localizing.h/.cpp`, `mapping_resuming.h/.cpp`, `loop_closure_manager.h/.cpp`, `mode_handlers.h/.cpp`, `n3_loc.rviz`
- Deleted (4): `relocalization_module.h/.cpp`, `map_extension_module.h/.cpp`
- Modified (16): `CMakeLists.txt`, `config.h`, `config.cpp`, `graph_optimizer.h`, `loop_detector.h/.cpp`, `n3mapping_node.cpp`, `point_cloud_matcher.cpp`, `n3mapping.yaml`, `localization.launch.py`, `package.xml`, 5 test files
- New test (1): `test_loop_closure_manager.cpp`
