# Large mapping crash qualification — 2026-10-25

## Incident and scope

The original rc3 crashed with SIGSEGV during acquisition and same-input replay.
The replay stack entered GTSAM/iSAM2's TBB parallel elimination and
`concurrent_unordered_map::init_bucket`. The exact race or lifetime defect is
not established. Original evidence is preserved under:

`/home/user/ros_ws/etc_host_20261024/etc_ws/docs/validation/humble_navigation_performance_2026-10-25/large_scene_route_scope/acquisition01_failure/n3`.

The navigation team's frozen candidate adds process-wide TBB guards at three
additional solve entries. It completed the failed input, save/reload and a later
356 m / 50-waypoint two-floor acquisition. That candidate and all frozen
navigation libraries remain unchanged. Those results qualify the candidate's
containment, not a proven dependency repair or general accuracy claim.

## Revised implementation

The N3M revision replaces both existing and candidate global guards with a
single private local-arena boundary across all five solve entries. Interfaces,
map schema, solver parameters, acceptance gates and dependencies are unchanged.
See [ownership and execution design](../graph_solver_execution.md).

The source audit found separate mutable solver state in graph snapshots,
immutable shared factors, serialized live commits and join-before-release
worker lifetime. It did not prove that every dependency operation is race-free.
The installed GTSAM elimination traversal also deserves further dependency
investigation; a small ASan traversal probe did not reproduce the recorded crash.
ThreadSanitizer could not initialize in the initial diagnostic environment
(`unexpected memory mapping`), so no clean TSan result is claimed.

## Qualification status at implementation commit

Isolated Release build, four physical CPUs (0,2,4,6), OpenMP 4, passive waiting:
11 focused CTest targets pass, including three new concurrency/numerical tests.
Two logging/persistence targets required the authorized host environment after
sandbox log-output failures; both pass there. The deterministic concurrent
snapshot comparison retains a 1e-10 matrix tolerance.

Complete replay, resource comparison and verified build identity are pending
at this implementation commit. This record will be completed from their output,
not inferred from the frozen candidate's prior results. Current evidence root:
`build/n3mapping/large_mapping_fix_20261025` in the parent ROS workspace.

No changes to TGW, ETC, FAST-LIO, SpRobo or installed GTSAM are part of this fix.
No existing release tag or navigation installation is replaced.
