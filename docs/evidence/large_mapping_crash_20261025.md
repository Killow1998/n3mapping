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

## Ownership and root-cause assessment

The audit found separate mutable ISAM2/Bayes-tree state in graph snapshots,
immutable shared factors, serialized live commits and join-before-release
worker lifetime. Existing regressions cover replaced-map jobs, save-time drain,
failed-load rollback and concurrent snapshots. No N3M lifetime defect was proven.

The installed GTSAM elimination code grows a parent's `childFactors` while
parallel child visitors can publish results into it. This is a candidate race
path for further investigation, not a demonstrated explanation of the recorded
`init_bucket` SIGSEGV. The [current upstream implementation](https://gtsam.org/doxygen/a00887_source.html)
contains a write mutex around these accesses; this does not establish that a
particular upstream change fixes this incident.

An ASan probe instantiating the installed traversal templates completed 20 runs
with both one and four threads. GTSAM and its TBB allocator were not fully
instrumented, so absence of an ASan finding is not proof of memory safety.
TSan initially could not initialize (`unexpected memory mapping`). On the host,
with ASLR disabled only for the diagnostic child, it reported task-publication
races. A GTSAM-free control using immutable task captures and an atomic sum
reported the same class of race with the uninstrumented system TBB. The serial
GTSAM control passed. These reports are insufficient for root-cause attribution;
a fully instrumented dependency build is still needed for that investigation.
No sanitizer suppression, dependency patch or global kernel setting was applied.

**Classification:** a qualified N3M execution-containment fix. It removes the
process-wide throttling workaround and contains the observed parallel crash
path. It is not a proven repair of GTSAM internals. Restoring graph-internal
parallelism requires a separate dependency revision and equivalent regression.

## Qualified build and tests — completed 2026-10-26

Runtime revision: **`0b4f4a928ab3c8709680c21f7452e364a01e4a92`** on local main.
The tested build stays fixed at this implementation commit; later documentation
commits do not replace its binaries. ROS package version remains 1.1.0.

- Humble Release, clean-source identity `verified=true`, research tools OFF.
- CMake-selected `-O3 -DNDEBUG`, ccache, isolated build/install/log directories.
- Four physical CPU affinity entries: 0,2,4,6; OpenMP 4, passive wait,
  OpenBLAS 1. This is affinity, not exclusive CPU reservation.
- **67/67 CTest targets passed**, including the three new execution tests and
  real ROS-node E2Es. Two initial ROS failures were missing type-support libraries
  because sandbox colcon had not finalized its environment. Host install
  completion and both retests passed; original failure output is retained.
- New snapshot tests compare independent concurrent graph owners with serial
  results at 1e-10 tolerance and require unrelated TBB work to remain concurrent.

All evidence below is under the parent ROS workspace's
`build/n3mapping/large_mapping_fix_20261025/`.
The tested install prefix is `release/install/n3mapping`; it is a symlink install,
so retain `release/build` alongside it.

| Artifact | SHA256 |
| --- | --- |
| `release/install/n3mapping/lib/n3mapping/n3mapping_node` | `157450d66b6680587a8901b0f978b650883382c10ff0a5e7efe509dfb732e13a` |
| `release/install/n3mapping/lib/libn3mapping_core.so` | `d9d87a83a11c4018c455f4b5dd03c09b7814880efc63190122cd215815c701f7` |
| installed GTSAM 4.1.1 | `546f8b37a0ff0edc5fafe85f1a8830d114e77a9d8f08e23091b10e08e0a8e41a` |

[The retained qualification summary](large_mapping_crash_20261025.json) preserves
the measurements and evidence checksums. `build_manifest.json` binds
executable/core/configuration hashes, actual loaded
solver dependencies, original tags and compiler settings. `qualification.json`
indexes the detailed replay, reload, numerical and test evidence.

## Complete replay and persistence

Only original `/Odometry`, `/cloud_registered_body` and recorded 500 Hz `/clock`
were replayed. No simulator, FAST-LIO or truth topic was started. Both A/B runs
used identical optional timing diagnostics and transport; truth was used only
by the unchanged offline quality scorer. All processes used private domain 231
and exited normally. The original fault is already reproduced in the preserved
rc3 evidence; it was not repeatedly triggered for this qualification.

| Run | Rate | Backend / native trajectory rows | Keyframes / edges | Save service wall time |
| --- | --- | --- | --- | --- |
| Original failed input, frozen global guard | 1× | 10,381 / 10,381 | 1,302 / 2,492 | 0.88 s |
| Same input, local arena revision | 1× | 10,381 / 10,381 | 1,302 / 2,493 | 0.91 s |
| Completed 356 m, 50-waypoint two-floor input, local arena | 2× | 13,781 / 13,781 | 1,661 / 3,198 | 36.92 s |

All runs completed without SIGSEGV. Native trajectory stamps exactly match the
source LIO sets; sequence numbers are continuous, values finite, source=native,
degraded=false. The original rc3 navigation reader loads all three outputs.
Each map was then loaded by the **complete revised ROS node in map_extension
mode**, saved without new input and compared: keyframe poses unchanged exactly,
dense translation difference below 4.4e-14 m, constraint rotation difference
below 1.4e-15 rad, clouds and non-pose constraint fields unchanged exactly.
This exercises full session/graph/Atlas loading, beyond the navigation reader.

The 2× save duration includes waiting for outstanding asynchronous loop work;
it is not a pure disk-write benchmark. It needs to be allowed for by callers.
No same-rate frozen-candidate A/B on that second route is claimed.

The failed-input live quality guard remains FAIL, just as before: maximum N3M
position error 1.043989759 m under the unchanged 1 m / 5 degree gross-drift guard.
The completed-route replay passes that guard (N3M maximum 0.171017 m; original
live acquisition 0.171460 m). These are simulation results, not 10 cm or hardware
acceptance. The failed-input map remains diagnostic and must not be promoted
into a successful navigation acquisition.

## Numerical behavior and cost

The original 1,200-node cross-linked graph fixture is unchanged. Balanced order,
three processes per revision, 30 iterations per process: all seven residual,
pose-update and consistency fields have **zero absolute difference** between
old rc3, the frozen global guard and the local arena revision.

| Fixture, mean per 30-iteration process | Old rc3 parallel path | Frozen global guard | Local arena |
| --- | --- | --- | --- |
| Wall time | 1.193 s | 1.490 s | 1.470 s |
| CPU time | 2.097 s | 1.500 s | 1.463 s |
| Peak RSS | 151.42 MiB | 135.73 MiB | 135.56 MiB |

Thus graph-internal containment has a measurable cost on this safe fixture:
about 23% more wall time and 30% less CPU time than old rc3. It does not justify
claiming that serial solves are universally faster. Local isolation avoids the
additional unrelated-work restriction of the frozen global guard.

| Failed-input 1× whole-node A/B | Frozen global guard | Local arena |
| --- | --- | --- |
| Mean CPU cores, including startup/drain/save | 0.711 | 0.706 |
| CPU seconds | 745.67 | 741.87 |
| Peak RSS high-water mark | 889.93 MiB | 870.28 MiB |
| Backend callback P95 / P99 | 4.45 / 6.19 ms | 4.35 / 5.86 ms |
| Async loop worker mean / P95 | 2.183 / 4.926 s | 2.188 / 5.237 s |

Whole-node average CPU/RAM and callback latency are close. Async worker P95 is
6.3% higher in this single A/B, with different batch/commit schedules and one
additional accepted loop; this cannot isolate an arena-only latency effect.
The final native trajectories differ by at most 6.74 mm in translation; live
output differences peak at 68.99 mm. No bitwise whole-route equivalence is
claimed. The 2× route averages 1.138 cores, peaks at 946.61 MiB RSS, and has a
4.65 ms backend callback P95.

Timing aggregation includes **all rotated files**, with every backend frame
accounted for. `graph_update_ms` is absent in the existing mapping instrumentation
and is reported unavailable, not zero. Callback, async worker and geometry-solve
wall times are measured; the controlled graph fixture supplies the solver-cost
comparison. `/clock` and sensor callbacks have no cross-topic ordering barrier:
recorded live `invalid_input_time` projection counts are 1 in each 1× run and
4 in the 2× run. Status publication can repeat a rejection; its counts are
retained separately in the observer evidence.
The matching native/backend counts show these projection rejections did not
remove mapping input. This fix does not alter time-validity gates.

## Navigation integration and remaining limits

Select the complete revised prefix explicitly; do not combine its core with an
old node and call the mixed process verified. From the parent ROS workspace:

```bash
source /opt/ros/humble/setup.bash
source build/n3mapping/large_mapping_fix_20261025/release/install/setup.bash
export LD_LIBRARY_PATH="$PWD/install/gtsam/lib:$PWD/install/small_gicp/lib:$LD_LIBRARY_PATH"
build/n3mapping/large_mapping_fix_20261025/release/install/n3mapping/lib/n3mapping/n3mapping_node --build-identity-json
```

The command must report the exact implementation commit above, Release,
research_tools=OFF and verified=true. Reuse the navigation thread's existing
N3M interfaces and parameters, selecting this executable/install prefix in its
own integration change. The new and frozen libraries expose the same 928
strong `n3mapping` definitions; this supplements behavioral tests and is not a
complete C++ ABI proof.

The precise GTSAM crash cause remains open. Future parallel dependency work
needs a trustworthy instrumented build and the same recorded large graphs.
This stage qualifies full failure-prefix and two-floor-route replays, not a
multi-hour soak or a new Noetic binary. Short cross-topic projection rejections,
the original failed-input drift and asynchronous save latency remain visible.
No TGW, ETC, FAST-LIO, SpRobo, installed GTSAM, frozen navigation library or old
release tag was changed. No deployment or remote push was performed.
