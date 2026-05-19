# Changelog

## [Unreleased] - 2026-05-19

### Architecture

- Split the backend into a ROS-free `n3mapping_core` library and a thin Humble wrapper target.
- Move Humble node implementation into `humble/src/` and keep backend ownership inside `N3MappingCore`.
- Add ROS-independent core frame/output types for external-LIO integration without depending on ROS message headers.
- Add Humble conversion/config helper layers so parameter loading and message conversion stay outside core code.
- Make `noetic/` and `humble/` sibling wrapper packages that both keep the ROS package name `n3mapping` while sharing the root core, config, and launch resources.
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

### Tests and Checks

- Add core type, config, Humble conversion, `N3MappingCore`, synthetic relocalization, no-ROS-core, Humble wrapper-boundary, and Noetic wrapper checks.
- Keep existing RHPD, loop detector, map serializer, loop closure, and relocalization regression tests in the refactored target layout.
- Validate the current test suite at 272 tests passing after the wrapper split and synthetic relocalization additions.
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
