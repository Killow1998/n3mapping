# Development and evaluation

[Project overview](../README.md) · [Build profiles](getting_started.md) · [ROS interface](ros_interface.md)

## Source layout

```text
include/ and src/    C++ backend core
humble/              ROS 2 adapter, launch files and RViz configuration
noetic/              ROS 1 adapter, launch files and RViz configuration
config/              Shared runtime configuration
cmake/               Shared dependencies, targets and core tests
test/                Core and native ROS regression tests
tools/               Map utilities and offline evaluation
```

## Architecture and ownership

`N3MappingCore` accepts frontend `LioFrame` values and returns `BackendOutput`.
Mapping, localization, loop closure, graph optimization and serialization stay in
the ROS-free core. Noetic and Humble adapt parameters, transport, messages, TF
and services. Do not commit wrapper-selection markers, generated maps, local
runtime settings or build products.

The private `src/core/n3mapping_session.h` owns backend components as one lifetime
unit. It wires borrowed references once and is replaced transactionally when a
map loads. Components are direct members; the session has no duplicate `Config`
and no forwarding accessors. Graph constraints and optimized-pose projection live
in `n3mapping_constraints.cpp`; dense trajectory recording and correction live in
`n3mapping_trajectory.cpp`.

`core/loop_closure_result.h` is the result contract shared by the facade and loop
processing. Loop processing does not include the facade. `loop_types.h` carries
verification data; only detector callers include `loop_detector.h`.
`mapping_resuming.h` forward-declares borrowed services. Implementations include
the owners they actually call, including the asynchronous loop implementation.

`Config` declares types in `config.h`; `src/config_defaults.cpp` owns defaults.
`config_fields.h` owns external names and canonical ordering for both adapters
and provenance. A default-only change rebuilds its implementation; a field/type
change changes the shared contract. Noetic retains its three parameter namespace
forms and their existing precedence.

Mapping and map continuation use `loop_min_path_length_m` for same-session
loop candidates from RHPD, heading fallback and spatial search. The distance is
the odometry arc along the intervening keyframes, including frames without
descriptors. Descriptor and spatial candidates within a session must be within
2.0 m vertically, matching the loaded-map proposal limit. The loop referee uses
the arc length for its temporal-gap feature. Cross-session candidates skip the
arc-length cutoff and keep the existing cross-session verification. Use the `heading_*` settings for the
heading descriptor and `rhpd_use_heading_yaw` to use it in RHPD matching.
The saved descriptor signature and protobuf field numbers remain compatible
with existing maps.

Relocalization uses the prepared global Atlas when available and a bounded
local target cache otherwise. The optional `reloc_initial_prior_*` parameters
apply until the first localization lock. XY and Z radii filter retrieval seeds
and the final registered sensor position in map coordinates. Optional
`reloc_initial_prior_yaw` and `reloc_initial_prior_yaw_tolerance` (radians) also
limit the final registered heading, with wraparound at ±π. A zero radius or
tolerance leaves that dimension unrestricted. Enabling the prior requires at
least one positive bound. It is disabled by default. A subsequent recovery
after tracking loss searches without the initial prior.

The Atlas uses `global_map_voxel_size` with a
1 mm minimum. PCD exports use `save_global_map_voxel_size`. A joint save
reuses the Atlas point cloud when both effective voxel sizes match.

An Atlas can share the finest plane geometry with GICP when the map grid
already limits target resolution. The sidecar records the GICP resolution
and stores the shared points, normals and search index once. V5 also supports
a separate refinement level when the configured resolution needs one.

Localization with an Atlas releases the loaded keyframe point clouds after
descriptor preparation. Each keyframe keeps indices for Atlas voxels it
observed. Directional verification can still check independent observations
and the global map publisher reuses the Atlas. Switching to map extension
restores the original point clouds from the loaded pbstream after checking
its identity.

With an Atlas and full translation support, steady localization tracks with
point-to-plane registration. It prepares coarse source points and GICP
covariances on demand for the existing fallback. Initial relocalization and
strict map-continuation tracking keep the configured GICP refinement.
Map continuation uses both visibility viewpoints by default. The optional
endpoint-only path reduced check time in the SpRobo comparison and raised the
maximum position error from 4.9 cm to 13.1 cm on the same source map.

Relocalization ranks places with the RHPD coarse key and its 180-degree flip
before computing full descriptor distances. It screens yaw seeds with coarse
point-to-plane geometry before refinement. ROS development and product profiles
retain live hypotheses across confirmation windows with decaying evidence.
Descriptors introduce new places. Current geometry, visibility and odometry
confirm those already under consideration.

Strong static matches can lock on the third observation after the usual
geometric and free-space checks. New closer returns withhold directional
support, while returns through expected surfaces count as contradictions.
Sufficient known free-space conflict vetoes even a single remaining pose.

`graph_types.h` exposes graph-edge data and the optimizer interface without GTSAM
headers. `GraphOptimizer` keeps solver objects in private `graph_optimizer_state.h`.
Its implementation files own GTSAM dependencies; unrelated consumers do not.

External cloud, configuration and persisted-map inputs are validated at their
entry points. Internal preprocessing and diagnostic output parameters use
references when every caller supplies an object. `MapSerializer::saveMap`
validates clouds before encoding; its private encoder trusts those validated
points. Atlas identity/cache orchestration belongs to `localization_atlas.cpp`,
and prepared-level wire conversion to `localization_atlas_codec.cpp`.

Atlas V5 keeps the `.localization_atlas.pb` suffix and binds the cache to its
source pbstream and configuration by SHA-256. Its little-endian binary
container starts with `N3ATLAS5`, a 64-bit section count, and entries containing
a 32-bit tag, 32-bit flags, 64-bit offset and 64-bit size. Sections start on
8-byte boundaries. Tags 1–5 identify metadata protobuf, global XYZI float
points, plane levels (flags 0 and 1), optional refine level and optional
free-space blocks. Each prepared level stores contiguous float XYZ points and
float XYZ normals at 12 bytes per point per array. The two arrays have
8-byte-aligned starts. Covariance derived from the normal needs no array.
Explicit covariance uses nine row-major doubles per point. Search splits use
the same rounded coordinates as the persisted points.

The writer streams each section to a temporary file, maps the prepared levels
to check the result and atomically publishes it. The reader addresses the large
sections with 64-bit offsets and maps their payloads read-only. The global
cloud is converted once to PCL's public XYZI type for existing callers. The
reader restores the search index once. Free-space scoring reads only populated
8×8×8 blocks through an in-memory directory. Its interpolation and occupancy
rules are unchanged. The existing background worker warms mapped search
geometry for observation bounds. Page advice controls residency and a missed
prefetch can still cause a page fault during registration.

`SpatialPointBlocks` rejects raw visibility blocks while preserving the existing
point order and spherical crop. Native and mapped targets share the registration
algorithm through private storage adapters. The mapping owns its background
worker and joins it before releasing the file; prepared targets retain that owner
across Atlas replacement. Boost mapping headers stay private. Noetic and Humble
use the same implementation. Compile C++ consumers with matching headers and
library: the prepared-target layout has changed; ROS interfaces are unchanged.

`registration_search.cpp` owns the native and mapped nearest-search and traversal
instantiations. The search selects occupied-bound checking once at entry; its
traversal, rejection distance and tie order stay shared. Native registration and
mapped registration compile separately around `registration_solver.h`, which
owns their common factor/rejector implementation. This keeps the hot traversal
and factor code from competing within one large template compilation unit.
Changing storage adapters must be checked against generated hot code and CPU;
unchanged geometry and call counts alone do not establish unchanged compute.
The [online CPU audit](evidence/online_cpu_20261114.json) records the earlier
controlled ablations and four-core replay for the previous Atlas format.

The default cache path rebuilds pre-V5 sidecars from the source pbstream
when needed. An explicit `reloc_atlas_path` requires a matching V5 sidecar.
The pbstream format and the TGW navigation resource reader are unaffected.

`RelocalizationDebugEmitter` owns sequencing, hypothesis snapshots and persistence.
Its header forward-declares event payloads. Localization implementations include
the payload schema where they populate events; schema details do not flow through
the public localizer header. Diagnostic code does not decide whether to lock.

Offline evaluation separates three responsibilities:

- `synthetic_relocalization_query.h`: scan synthesis, visibility and pose jitter.
- `query_pose_manifest.h`: external CSV validation and typed pose entries.
- `synthetic_episode_state.h`: evaluation-only episode transitions and labels.

CSV numeric conversion uses the standard C conversion functions underlying
`std::stoll`/`std::stod`, preserving full-token, range and finite-value checks
without exception control flow. Runtime localization remains the authority for
lock and tracking results.

Free-space scoring interpolates occupied/free layers at voxel centers. Both veto
and elimination use the existing `reloc_free_space_kill_sigmas` separation rule.
An insignificant score difference cannot overturn the geometric winner; a
significantly better distinct candidate still vetoes it. This is a sampling
variance approximation, not calibrated cross-scene probability. Registration,
confirmation, visibility and ambiguity gates remain separate decisions.

Validation records: [compile dependencies](evidence/compile_dependencies_20261029.json),
[serialization cleanup](evidence/cleanup_20261029.json),
[reload/relock repair](evidence/reload_relock_20261029.json), and
[ownership and diagnostic cleanup](evidence/cleanup_20261107.json).
Earlier compile-time measurements apply to their recorded build configurations,
not SLAM runtime speed. Free-space interpolation follows the approach in
[Cartographer's interpolated grid](https://github.com/cartographer-project/cartographer/blob/master/cartographer/mapping/internal/3d/scan_matching/interpolated_grid.h)
with trilinear sampling, without an optimizer dependency.

### Directional support before locking

Before the directional witness, the selected current registration must constrain
horizontal translation and yaw in the matched map. Its information matrix is
projected onto those three source-frame directions and checked for numerical rank.
This is a necessary observability check, not a confidence calibration. Candidate
generation and reuse may retain partial constraints; only publishing a lock
requires all three planar directions. A bare floor cannot establish a place even
when the query contains nonplanar structure and the residual gates pass.

The lock gate evaluates the already prepared plane geometry once at the selected
pose, without another optimizer. It uses the scalar normal projection and its
Jacobian outer product. Neither GICP's regularized covariance nor componentwise
normal weighting establishes this geometric rank: both can add tangential
information to a tilted plane. Solver residuals and their quality thresholds
remain unchanged. This evaluation is counted in
`fixed_pose_ms` and included in `directional_check_ms`; those counters overlap.
The initialization, confirmation and tracking-recovery paths retain the matrix
that belongs to the selected pose. An unobservable match reports
`matched_planar_geometry_unobservable` and follows the existing rejected-place
seed invalidation policy.

The final witness uses point-to-plane Jacobians for yaw and horizontal translation.
`planar_min_support = min_v (vᵀ H_explained v) / (vᵀ H_all v)` measures the
least explained planar direction. Whitening makes `H_all` the identity. The fixed
0.5 gate requires an information majority in every direction; it is an engineering
policy, not a calibrated probability or a ratio between the two best candidates.
It assumes largely covered, approximately static geometry. Limited coverage or
changed occlusion can leave a correct candidate searching; 0.5 is not a theorem
that separates every correct and incorrect pose.

The witness first checks the fused map. Fusion combines different visits, so an
old foreground can hide a return supported by another complete observation. If
the fused map has finite but insufficient support, the final gate independently
checks the candidate's seed and support keyframe clouds in their optimized
reference frames. Duplicate references are checked once. An observation passes directly when support exceeds 0.5 in every planar direction.
For partial coverage, the same historical observation must instead pass both
coverage-conditioned forward support and unconditional reverse support, each
above 0.5 in every planar direction. The reverse check prepares normals in the
historical sensor frame and projects the current query into that frame. It uses
the existing nearest-return model and performs no additional registration.
Directional evidence from different observations is never pooled. Unobservable
query or matched-map geometry still fails.

Within each tested point cloud, a bearing supports its query surface when its
nearest return agrees in range or point-to-plane distance, using the existing
tolerance. This handles grazing-plane sampling without selecting a farther
matching background through that observation's foreground. The query normal fit
is reused; the recheck performs no registration. Its time is included in
`directional_check_ms`; an accepted reference logs its ID and both support scores.
Candidate ranking, temporal confirmation, ambiguity checks and free-space vetoes
retain their existing definitions; floor matches cannot compensate for unsupported
walls.

Low support alone does not prove geometric contradiction. In the whitened query
basis, `H_support + H_conflict + H_unknown = I`. Known map bearings contribute to
`H_known`; a bearing contributes to `H_conflict = H_known - H_support` only when
neither range nor the observed surface agrees. These are geometric information
fractions, not independent measurement counts or calibrated probabilities.

Failed witnesses invalidate cached seeds only when
`lambda_max(H_conflict) >= 0.5` in every checked observation: even perfect
agreement in all unknown bearings could not pass any existing support gate.
A conflicting fused map cannot invalidate a seed when a stored observation
only lacks coverage. This reports `directional_visibility_disagreement`. Otherwise it
reports `directional_visibility_insufficient_coverage`, withholds localization,
and preserves existing seeds for the next observation. Query or matched-map
degeneracy still follows its separate rejection path. Repeating a stationary
observation does not convert missing coverage into evidence.

`planar_min_known` measures the least-covered direction;
`planar_min_support_given_known` measures agreement conditional on coverage;
`planar_max_conflict` measures the strongest contradiction. The coverage and
reference debug events expose all three. Conditional agreement alone cannot accept a pose: discarding unknown bearings
can make a small, ambiguous overlap look good. The reciprocal alternative also
requires the historical observation to be supported by the query in all planar
directions. `directional_min_support` retains its raw forward definition and may
therefore be below 0.5 on an accepted reciprocal witness.
`Reloc/DirectionalReciprocal` records the reference, conditional forward score
and unconditional reverse score; both checks are timed in `directional_check_ms`.
Known/unknown radial counts and nearer-map-return diagnostics remain separate.

The [reciprocal witness audit](evidence/directional_reciprocal_20261111.json)
records the preceding directional-gate qualification. Initial retrieval now accepts
an explicit rank offset without owning search state. When no valid initial geometry
is found, or stationary initial geometry fails likelihood or contradicts the map, localization requests
the next bounded batch of places.
A margin/ambiguity failure alone does not advance past a viable place. Existing
seed caches and acceptance thresholds remain unchanged. An exhausted batch,
a viewpoint change beyond the existing stationary aggregation limits, a successful
lock, reset or map replacement returns retrieval to the first batch. Recovery after
an established map pose and motion-gated confirmation failures retain their original
retrieval policy. `Reloc/Retrieval` reports the offset through the bounded logger.

Relocalization groups and reuses candidates by three-dimensional map distance.
The same spatial radius applies when checking descriptor support and invalidating
failed seed poses. XY proximity alone must not merge vertically separated regions.
This is a spatial grouping rule, not a semantic floor classifier. The
[spatial-basin audit](evidence/spatial_basins_20261111.json) records the repeated-floor
E2E and the four-core cost comparison; the difficult-search half-core target remains
unmet.

A descriptor proposes an overlapping observation whose reference position can
differ from the registered sensor position. Admission and confirmation therefore
recognize descriptor support around either the original seed reference or the
nearest reference at the solved pose, using the same three-dimensional radius.
Confirmation still evaluates each new observation and retains the full final
registration and existing ambiguity, visibility and observability gates. The
[confirmation-support audit](evidence/confirmation_support_20261114.json) records
the four-core regression and the distinction between search CPU and whole-clip CPU.

Established tracking without a recent rejection first checks fresh plane coverage
at the odometry prediction, then tries the existing fine GICP directly. A proposal
that fails geometric quality or exceeds the existing capture limits uses the
original coarse-to-fine solver. Existing tracking failures use that solver from
the start. Selected-pose geometry, motion, visibility, free-space and observability
gates remain downstream. The [tracking-geometry audit](evidence/tracking_geometry_20261114.json)
records the shared Humble/Noetic implementation and four-core regression.

The [retrieval audit](evidence/retrieval_20261111.json) separates descriptor recall,
rank paging and motion/recovery regressions. Paging does not make descriptor scores
probabilities, guarantee exhaustive coverage under changing ranks, or solve the
remaining motion-visibility calibration.

The [cross-scene audit](evidence/directional_cross_scene_20261110.json) records
sequence-held-out analysis, controlled coverage ablations and end-to-end checks.
No universal cross-scene calibration of 0.5 is claimed. Earlier evidence covers
[grazing planes](evidence/directional_visibility_20261103.json),
[historical observations](evidence/directional_reference_20261104.json) and
[matched-map rank](evidence/n3m_issues_20261109.json).

The [Atlas cache audit](evidence/atlas_cache_20261029.json) distinguishes that
frozen regression fixture from normal deployment. Its explicit legacy Atlas
omits descriptor precomputation; the existing default path checks cache currency
and rebuilds outdated artifacts. With the same binary and 1,271-keyframe map,
four-core fixed-14 mean map-ready time is 5.058 s with the legacy artifact versus
1.479 s with a freshly compiled cache. The latter excludes its one-time compile
cost. Peak RSS across those fixed positions is 1,225.4 versus 1,097.0 MiB; CPU
seconds per input second are 0.379 versus 0.361, including both search and tracking.
This measures an existing cache capability, not a newly accelerated algorithm.

All 14 positions retain frame-4 locks. Both complete localization routes retain
their output/lock decisions; maximum live-pose matrix differences are below
6e-14. The independent route did not show a consistent CPU saving, including
a reversed-order repeat. Search 03 still uses 7.17 CPU seconds before locking,
41 registration calls and 109 visibility evaluations; descriptor work uses only
0.097 CPU seconds. Subsequent compute reduction should target measured registration
and verification work. Do not report a mixed 30-second average as active-search
CPU usage, or a frozen old-cache load as the current default startup cost.

The subsequent [03 geometry-call audit](evidence/geometry_audit_20261029.json)
fingerprinted the first five input frames without changing production code.
All 41 registration calls had distinct fingerprints covering prepared geometry,
tree identity, pose and settings. All 110 visibility calls, including the final
planar-support witness, also had distinct complete-input fingerprints. Instrumented
replays retained the frame-4 lock and its pose to floating precision. Hashing and
instrumentation perturb timing; these runs are not performance comparisons.
The result does not establish that every call is necessary, or cover other routes.
It does not support adding an identical-input result cache for this segment.
Further optimization needs a specific hypothesis, unchanged acceptance behavior
on the frozen regressions, and a repeatable CPU benefit beyond baseline variation.

The bounded correspondence search uses a previous correspondence only as a
search bound. On an ambiguous floating-point miss, it consults the original
small_gicp search and reapplies the unchanged distance gate. This covers the
ARM64 hint/leaf rounding failure reported in [#8](https://github.com/Killow1998/n3mapping/issues/8).
[PR #9](https://github.com/Killow1998/n3mapping/pull/9) also requests TCP_NODELAY
on all four Noetic odometry subscriptions, preserving synchronization and
output-authority rules. Four-core integration passed Humble 69/69 and Noetic
67/67. The RK3588/GCC11 registration suite passed 10/10 before and after;
it did not reproduce the original Jetson/GCC9 failure. The interleaved
optimizer-process benchmark used approximately 3.7% more CPU time with unchanged
median peak RSS. These are microbenchmark results, not whole-SLAM resource
measurements. See the [integration evidence](evidence/pr9_arm64_20261029.json).

Localization changes should follow the existing ownership boundaries:

| Responsibility | Owner |
| --- | --- |
| Input-source lifecycle and backend orchestration | `N3MappingCore` localization facade |
| Session state, reset and map replacement | `WorldLocalizing` |
| Query history and place retrieval | `RelocalizationQueryBuilder`, `RelocalizationPlaceIndex` |
| Candidate registration and temporal acceptance | Candidate evaluator, hypothesis manager and decision policy |
| Tracking targets, bounded local caches and the immutable loaded-map target | `TrackingTargetCache` |
| Online loop retrieval, isolated geometry/graph work and revision-checked commit | `N3MappingCore` loop orchestration and shared loop-processing implementation |
| Trusted corrections and live pose/cloud projection | `core::RealtimeOutput` |
| Post-commit map snapshots, global-cloud cache and debug PCDs | `MapArtifacts` |

Both ROS adapters keep one state/lock owner in their private
`src/n3mapping_node.h` and use corresponding files for the same responsibilities:

| File in `humble/src/` and `noetic/src/` | Adapter responsibility |
| --- | --- |
| `n3mapping_node.cpp` | Process startup, execution and shutdown |
| `n3mapping_node_lifecycle.cpp` | Parameters, ROS resources and lifecycle |
| `n3mapping_node_backend.cpp` | Input conversion, serial core calls, forwarding results |
| `n3mapping_node_live.cpp` | Independent live queues, odometry/cloud/TF/status publication |
| `n3mapping_node_map.cpp` | Map service calls and map publication |
| `n3mapping_node_visualization.cpp` | Paths, markers and optimization-log formatting |

`N3MappingCore` owns SLAM decisions; `core::RealtimeOutput` owns output authority.
The ROS-free `MapArtifacts` owns map snapshots, save orchestration, global-cloud
caching and debug PCDs. Saving captures poses after draining pending loop work;
immutable clouds are shared, and cache visitors run under the cache lock.
Map artifact operations take references to their required core, result and error
objects. Keyframe snapshots come from the core's non-null owned keyframe list.
The global cloud is borrowed only for the duration of the cache visitor.
`MapArtifacts` retains the save path, save policy and run mode instead of a
second complete configuration. Adapter map callbacks and services run on the
serial backend owner; live input workers do not access map artifacts.
Adapters track map revisions and publish on demand; middleware retains the
latched message without a second adapter copy. Core headers expose value contracts
through `loop_types.h` and `registration_probe.h` without pulling session and
registration implementation headers into every adapter source.
`keyframe_map_revision.h` is the shared value contract for map generations,
structure changes and pose changes. The core facade and display cache include
that contract without importing the keyframe manager implementation.

Both adapters compile their node methods into an internal static implementation
target. Humble's real node and live-output E2E link the same objects; Noetic's
E2E launches its real executable. The existing Humble-only authority probe
remains in `n3mapping_node_probe.cpp`. Transport APIs and diagnostic coverage
still differ; file alignment does not establish complete feature parity.
See the [alignment and compile measurements](evidence/runtime_cpu_20261017/README.md#shared-map-artifacts-and-ros-alignment--2026-10-23).
The core's `MappingResuming` similarly separates session lifecycle, keyframe
transactions, acquisition identity, cross-session loops and submap diagnostics.
These are implementation boundaries, without new managers or state copies.
`N3MappingCore` keeps mapping, extension, load and save implementations in
corresponding files. `MapSerializer` separates pbstream read/write from PCD export;
its public header does not import graph, loop-detector or generated protobuf
implementation headers. `n3map_proto_reader.cpp` owns file admission, protobuf
parsing and resource limits. `n3map_proto_utils.cpp` owns geometry and metadata
decoding. Both the SLAM core and standalone navigation resource reader build the
same reader and decoder sources, without a dependency on one another.
Invalid maps, resource-limit failures and transactional rollback remain boundary
responsibilities. Internal operations consume the validated values directly.
Both distributions register core tests from
`cmake/n3mapping_core_tests.cmake` through native ament/catkin macros. These tests
use GoogleTest directly, without starting a ROS context. Adapter tests retain
their native middleware setup.
Use the relevant behavioral tests during a module change and the integrated
mapping/localization/extension checks at release boundaries.
The [2026-11-15 cleanup evidence](evidence/code_cleanup_20261115.json) records
the ownership and dependency reductions, non-comment token comparison,
version-parser ablation, both native ROS suites, and the unchanged 14-point
and two-route localization results.

Localization callback admission in Humble stays on its serial backend owner.
`N3MappingNode::spinOnce` uses bounded repeated collection before consuming the
latest complete pair; the existing wrapper lint rejects a regression to
one-pass admission. Noetic retains its callback-queue draining loop.
Both adapters report raw cloud receipt to `core::RealtimeOutput` and run its
input-health check on the existing live-odometry callback lane. Cloud arrival,
processed observation and accepted map correction remain separate events.
Backend observation times use `core::TimeStamp` throughout this boundary, so
floating-point seconds cannot enter the outage-recovery comparison. JSON is
the conversion boundary for seconds.

C++ checks use `.clang-tidy` and `.clang-format`; adapter formatting retains four
spaces. Enable `CMAKE_EXPORT_COMPILE_COMMANDS` for the native build and run
`clang-tidy -p <build-directory> <source.cpp>`. The standalone
`test/realtime_contract` CMake entry also accepts
`-DCMAKE_CXX_CLANG_TIDY=clang-tidy` to analyze production output sources during
the build. Install the matching Clang/OpenMP development headers when using a
standalone LLVM installation. Use `clang-format-diff -p1` on changed C++ hunks
to avoid reformatting unrelated code. Python checks use `ruff.toml` with
`ruff check test/map_extension_ros_e2e.py`. The shared native ROS E2E covers
both cloud-only and complete input outages, recovery and extension admission.

Search, tracking, map evidence and offline probes have separate implementation
files. Session reset preserves map-derived tracking targets; map replacement
invalidates their generation and queued work. Loaded-map tracking prepares one
registration target per frozen map generation, with a four-entry cache of exact
raw visibility crops. Crop centers belong to that same frozen geometry.
Speculative target-preparation threads have been removed.

Tracking translation support comes from admitted correspondences, not all
surfaces in the query. The private `tracking_geometry.h` helper removes GICP's
isotropic tangential regularization from the translation information block.
Directions supported by fewer than `gicp_num_neighbors` effective plane
observations retain the odometry prediction; rotation is unchanged. This is
a support test, not a calibrated posterior covariance. A changed pose is
evaluated using the selected registration target and metric before the existing
quality and motion gates. Recovery keeps that target alive through evaluation.
Initial global relocalization does not use this projection. Already deformed
map artifacts require reconstruction from their source recordings.

Tracking recovery separates correspondence coverage from first-return agreement.
The existing narrow GICP coverage and pose-change bounds still apply. When a
changed foreground leaves first-return agreement below the correspondence gate,
positive visibility evidence additionally requires stable horizontal and yaw
support from tight point-to-plane correspondences. Each planar direction needs
at least `gicp_num_neighbors` effective observations; a floor alone cannot pass.
The directional assessment uses the normal covariance variance from the
supported small_gicp covariance model (0.001). Isotropic fallback covariances
can occur at any point when there are too few neighbors.
If narrow optimization fails, the predicted pose may instead undergo the same
fresh geometric verification with the original GICP metric. This does not carry
an old successful status through missing or contradictory scans. Ordinary
tracking and strict map-extension gates retain their existing behavior.
The [changed-map tracking audit](evidence/tracking_recovery_20261116.json) records
the field replay, portable geometry checks and four-core qualification.

Mapping-loop retrieval stays with the serialized owner. Geometry verification
and graph solving use an isolated snapshot; the owner commits only a matching
map generation and revision. Appended keyframes catch up through a conditioned
tail solve. A conflicting graph revision requires readmission and re-solving,
while retaining measured registration evidence. Saving drains pending loop
work. Map continuation conditions every graph solve on the exact loaded poses.
The graph retains its persistent factors. Shared planes can optimize against
both loaded and new observations, and the loaded Atlas remains available for
tracking and loop visibility checks.

Continuation uses the same asynchronous loop worker as mapping. A keyframe
already constrained by loaded-map tracking can still close a loop within the
new session. When tracking cannot establish that constraint, descriptor
retrieval can recover a loaded-map match outside the spatial search radius.
Cross-session verification requires Atlas visibility and neighborhood consensus.
A missing Atlas rejects cross-session candidates at the visibility gate.
The trial and commit interpolate large corrections from the preceding
loaded-map anchor to the return keyframe, then carry the full correction
through any unanchored tail up to the next loaded-map anchor. The temporary
solve excludes shared-plane observations on moved poses. The worker
reassociates planes after correcting the trajectory. After commit, the newest
continuation keyframe defines the map-to-odometry transform.

Causal descriptor neighborhoods stay within the keyframe's session in both pose
domains. Automatic Atlas preparation rebuilds older place caches once. Explicit
historical Atlas files retain their geometry and refresh descriptor inputs
at runtime.

Continuation map saves use the final optimized poses and require at least three
new keyframes by default. An old point is removed when three rays from distinct
new keyframes provide a return behind it by more than 0.35 m and the surrounding
angular bins contain no nearer return. A return near the old point's depth in
another view keeps the point. The default range is 20 m and the angular
resolution is 1 degree. Set `continuation_penetration_cleaning_enable`
to `false` to save without this step. The other `continuation_penetration_*`
fields control the distance, vote count, angular resolution, and range.

Keyframe IDs and poses are preserved. A keyframe that would become empty keeps
its original cloud. The saved map, Atlas, and PCD use the same surviving
points. Successful map publication also updates the live keyframe clouds,
affected descriptors, and reference geometry. The `[ContinuationCleaning]`
log reports coverage, points removed, protected empty-frame cases, and
cleaning computation time.

`loop_geometry_cpu_budget` defaults to **0.5** CPU seconds per wall second for
background extension geometry; **0** disables pacing. It leaves ordinary mapping
loop processing unchanged. Successive extension requests coalesce while every
measured constraint remains in the graph. The interval uses process CPU during
the job, including parallel feature work and concurrent foreground work, as a
conservative upper bound. It is not a cap on total node CPU. Pending age retains
the oldest request time; saving bypasses pacing to finish outstanding work.

The deterministic `test_localization_episode` passes actual saved/loaded maps
through the production core and live-output owner. It covers initialization,
moving outside the map, reentry, invalid odometry, source replacement, gradual
yaw-drift recovery, fresh map lifecycles,
ambiguous places, no candidates and strict map-extension tracking. Together
with ROS publication tests, it is the integration check for changes within
these boundaries:

```bash
ctest --test-dir build/n3mapping --output-on-failure \
  -R 'test_(localization_episode|world_localizing|humble_live_output)$'
```

The episode test also accepts `N3MAPPING_EPISODE_ATLAS=1` and
`N3MAPPING_EPISODE_FREE_SPACE_MODE=kill` for the existing algorithm variants.
`N3MAPPING_EPISODE_TRACE=/tmp/episode.csv` records per-frame decisions, poses,
timestamps, work counts and timings for ablations. Synthetic episodes measure
backend behavior; real-sensor drift and robot acceptance require recorded or
live sensor input.

## Humble tests

```bash
# After the selected profile has built successfully:
source "install/$N3M_PROFILE/setup.bash"
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1
ROS_LOG_DIR="$PWD/log/$N3M_PROFILE/ros_test" \
  ctest --test-dir "build/$N3M_PROFILE/n3mapping" --output-on-failure -j1
```

For a local change, append `-R '<affected test expression>'` to run the relevant
existing tests; run the full suite for a release candidate. Keep research tools
in a separate development profile if enabling `N3MAPPING_BUILD_RESEARCH_TOOLS`.

## Noetic tests

Install `requirements-product-tools.txt` before running the complete suite;
Python 3.8 needs its declared `backports.zoneinfo` dependency. If GTSAM or
small_gicp was installed to a custom prefix without a ROS environment hook,
also make that prefix's `lib` directory available to the runtime linker.

```bash
catkin build --profile "$N3M_PROFILE" n3mapping --no-status -j4 -p1 \
  --catkin-make-args run_tests
source "devel/$N3M_PROFILE/setup.bash"
(
  cd "build/$N3M_PROFILE/n3mapping"
  ctest --output-on-failure -j1
)
```

## Offline evaluation

Recorded cloud/odometry manifests use the explicit target
`n3mapping_relocalization_manifest_eval`; it can be built without enabling all
research targets. Supply `--map`, `--atlas`, `--manifest` and `--output`.
For a deployment-parameter comparison, `--lock-min-margin VALUE` overrides only
the lock margin. The report identifies the override in `config_source` and
records the actual value in `relocalization_contract.lock_min_margin`. Such a run
is not evidence for the unchanged product defaults. `frame_status.csv` includes
`directional_checked`, `directional_min_support` and `directional_check_ms`;
an unchecked witness is not a zero-support observation.

Enable `N3MAPPING_BUILD_RESEARCH_TOOLS=ON` in a separate development build to use
the research tools below. These commands use Humble; under Noetic use `rosrun`
in place of `ros2 run`, and `.launch` in place of `.launch.py` for `roslaunch`.

RViz visualization:

```bash
# First set the map and tool options in config/launch_humble.yaml.
# Noetic uses config/launch_noetic.yaml; neither launch accepts arguments.
ros2 launch n3mapping synthetic_relocalization_visualization.launch.py
```

Headless batch evaluation:

```bash
ros2 run n3mapping n3mapping_synthetic_relocalization_eval \
  --map /path/to/n3map.pbstream \
  --max_queries 100 \
  --query_source local_submap \
  --strict
```

KITTI360 lidar/pose alignment smoke:

```bash
ros2 run n3mapping n3mapping_kitti360_reader \
  --kitti_root /path/to/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --output /tmp/n3mapping_kitti360_reader_test \
  --max_frames 50 \
  --dump_sample_pcd
```

This is an offline evaluation input reader only; it does not run mapping or
relocalization and does not replace real robot bags. See the
[dataset evidence contract](industrial_closeout_review_20260814.md#fa-02-what-kitti-360-and-m2dgr-prove).

KITTI360 offline mapping-loop smoke:

```bash
ros2 run n3mapping n3mapping_kitti360_eval \
  --kitti_root /path/to/KITTI360 \
  --sequence 2013_05_28_drive_0003_sync \
  --mode mapping_loop \
  --calib_mode auto \
  --max_frames 200 \
  --stride 1 \
  --output /tmp/n3mapping_kitti360_mapping_loop
```

Label KITTI360 loop candidates with keyframe ground truth:

```bash
ros2 run n3mapping n3mapping_loop_debug_analyze.py \
  --loop_debug /tmp/n3mapping_kitti360_mapping_loop/loop_debug.jsonl \
  --keyframes_gt /tmp/n3mapping_kitti360_mapping_loop/keyframes_gt.csv \
  --accepted_loops /tmp/n3mapping_kitti360_mapping_loop/accepted_loops.csv \
  --output /tmp/n3mapping_kitti360_mapping_loop/loop_gt_analysis
```

Summarize one or more eval runs into a metric matrix:

```bash
ros2 run n3mapping n3mapping_eval_matrix.py \
  --run kitti360_drive0003=/tmp/n3mapping_kitti360_mapping_loop \
  --output /tmp/n3mapping_eval_matrix
```

Benchmark interpretation and current acceptance limits are recorded in the
[industrial closeout review](industrial_closeout_review_20260814.md).

M2DGR extracted-cloud offline eval:

```bash
ros2 run n3mapping n3mapping_m2dgr_eval \
  --m2dgr_root /path/to/M2DGR \
  --sequence hall_03 \
  --lidar_dir /path/to/M2DGR/hall_03/velodyne_points \
  --gt /path/to/M2DGR/hall_03/groundtruth.txt \
  --mode mapping_loop \
  --output /tmp/n3mapping_m2dgr_hall03_mapping
```

Details and evidence limitations are in the
[dataset evidence contract](industrial_closeout_review_20260814.md#fa-02-what-kitti-360-and-m2dgr-prove).

For raw M2DGR LiDAR/IMU replay, first convert the selected ROS1 sensor topics to
ROS2. Its FLOAT32 `time` field contains seconds relative to a cloud header near
the scan end. Normalize that representation before FAST_LIO:

```bash
python3 tools/n3mapping_normalize_point_time.py \
  --bag /path/to/raw_ros2 --output /path/to/normalized_ros2
```

This offline tool requires `numpy` and `rosbags`. It writes a new bag with a
scan-start header and nonnegative offsets, retaining absolute point sampling
times, other point fields, record times and other message streams. It does not
perform motion compensation. Configure FAST_LIO's `preprocess.timestamp_unit`
as `0` (seconds), select `/handsfree/imu`, and use the matching LiDAR-to-Handsfree
extrinsic; the Xsens device supplies outdoor truth and is a different IMU.
Check the adapter's `point_time_audit.json` and complete frontend frame
acknowledgements before interpreting an accuracy score.
When replaying into nodes configured with `use_sim_time`, give clock ownership
to the replay process:

```bash
python3 tools/n3mapping_deterministic_ros2_replay.py \
  --bag /path/to/normalized_ros2 \
  --topic-depth /velodyne_points=100 --topic-depth /handsfree/imu=4000 \
  --clock --flow-control-lidar-topic /velodyne_points \
  --lidar-ack-topic /n3m_replay_ack
```

The frontend must publish the configured frame acknowledgement. `--clock`
publishes monotonic time before each sensor message without changing its payload;
do not also publish `/clock` from a downstream observer. With runtime debug
logging enabled, `live_odometry` records report exact input/correction/current
nanoseconds and the publication decision, independently of backend frame counts.
The [2026-10-21 real-sensor baseline](evidence/runtime_cpu_20261017/README.md#real-sensor-baseline-and-input-contract--2026-10-21)
records M2DGR and MID360 results, reference limitations and complete ROS replay.

## Results and changes

Use the [changelog](../CHANGELOG.md) for release history and the
[field guide](field_testing_zh.md) for hardware testing. Measured results,
ablations and dataset limitations belong in the existing `docs/evidence/`
records. The [runtime measurements](evidence/runtime_cpu_20261017/README.md)
link the corresponding full-route experiments.
