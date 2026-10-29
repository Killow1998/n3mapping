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

- Keep backend mapping, loop closure, relocalization, graph optimization, and map serialization logic in the ROS-free core.
- Keep ROS wrappers as thin adapters for parameters, messages, topics, services, TF, and launch defaults.
- Keep Humble and Noetic runtime behavior consistent.
- Do not commit local wrapper-selection marker files.
- Do not commit generated maps, logs, build artifacts, or local test configs.

`Config` declares parameter types in `config.h`; its out-of-line constructor in
`src/config_defaults.cpp` owns the defaults. Changing a default recompiles one
implementation file; changing a field or its type still changes the shared
contract. `forEachConfigField` in `config_fields.h` owns the external names and
canonical ordering used by both
ROS adapters and runtime provenance. Add a field there once; transport-specific
parameter conversion stays in the adapter. Noetic retains its three supported
namespace forms and their existing precedence.

`graph_types.h` carries graph-edge data and the loop-optimizer interface without
GTSAM headers. `GraphOptimizer` owns its solver objects through the private
`src/graph_optimizer_state.h`; only its seven implementation files include
`ISAM2.h`. Consumers include the full optimizer declaration only where they call
its methods. A solver implementation edit does not require recompiling its
unrelated consumers. Rebuild C++ consumers when adopting this object-layout and
`Config` constructor change; ROS interfaces and parameter values are unchanged.

The [compile-dependency comparison](evidence/compile_dependencies_20261029.json)
uses Release flags, disabled ccache and four-core affinity on Jammy and RK3588.
Three representative consumers reduce compile CPU time by approximately 54–63%
on both machines; the solver file itself shows no consistent improvement.
A default-only edit rebuilt one file and completed in 7.44 s on Jammy, including
linking. These are translation-unit and incremental-build measurements, not
full clean-build or SLAM runtime speedups. The qualified GTSAM dependency remains
unchanged. The final Humble/Noetic suites passed 69/69 and 67/67; the same reload
relock test also failed intermittently on both the candidate and unmodified
baseline. That existing reliability issue remains open in the evidence record.

Registration entry points validate external clouds. Private preprocessing and
result-copy helpers accept references; they do not repeat nullable-pointer
checks. Multi-scale registration keeps its stage result as the source of truth
for convergence and quality statistics.

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
Adapters track map revisions and publish on demand; middleware retains the
latched message without a second adapter copy. Core headers expose value contracts
through `loop_types.h` and `registration_probe.h` without pulling session and
registration implementation headers into every adapter source.

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
implementation headers. Both distributions register core tests from
`cmake/n3mapping_core_tests.cmake` through native ament/catkin macros. These tests
use GoogleTest directly, without starting a ROS context. Adapter tests retain
their native middleware setup.
Use the relevant behavioral tests during a module change and the integrated
mapping/localization/extension checks at release boundaries.

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

Mapping-loop retrieval stays with the serialized owner. Geometry verification
and graph solving use an isolated snapshot; the owner commits only a matching
map generation and revision. Appended keyframes catch up through a conditioned
tail solve. A conflicting graph revision requires readmission and re-solving,
while retaining measured registration evidence. Saving drains pending loop
work. Strict map extension verifies loaded-map geometry and commits its tracking
constraint inline. This append solve preserves the Atlas-verified live pose;
the worker later installs the optimized reference and its coordinate correction
together.

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

Enable `N3MAPPING_BUILD_RESEARCH_TOOLS=ON` in a separate development build to use
the research tools below. These commands use Humble; under Noetic use `rosrun`
in place of `ros2 run`, and `.launch` in place of `.launch.py` for `roslaunch`.

RViz visualization:

```bash
ros2 launch n3mapping synthetic_relocalization_visualization.launch.py \
  map:=/path/to/n3map.pbstream
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
