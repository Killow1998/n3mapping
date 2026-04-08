# Changelog

## [1.0.0] - 2026-04-08

### Release

- Align changelog versioning with `package.xml` (`1.0.0`) as the first stable package version.
- Keep runtime interfaces unchanged (`mapping` / `localization` / `map_extension`) and preserve existing ROS2 launch/build workflow.

### Documentation and Naming Consistency

- Rename test files/targets to current module semantics (`WorldLocalizing`, `MappingResuming`) to remove legacy module naming residue in the test surface.
- Clarify that `loop_closest_id_th`, `loop_min_id_interval`, and `loop_max_range` are compatibility/legacy parameters and are not used in the active mapping loop-candidate retrieval/verification path.

### Migration Continuity

- Preserve the migration continuity established in `0.3.0` without introducing new runtime features in this release.

## [0.3.0] - 2026-03-25

### Migration

- **Noetic -> Humble behavior alignment**: Port noetic-validated behavior to ROS2 `main` without ROS1 fallback path.
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
- **ROS2 build compatibility fixes**: Resolve pointer/allocation and parameter type issues encountered during migration integration.

### Configuration and Docs

- **ROS2 config synchronization**: Sync noetic-validated parameter set and defaults into humble config path, including loop distance-candidate, relocalization temporal/retry, and RHPD parameters.
- **README update**:
  - Keep English ROS2 workflow.
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
