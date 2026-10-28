# Localized-map extension handoff — 2026-10-27–28

This development change builds on `d2c4233`. A node already localized in its
loaded map can start extension without another global search or restarting the
odometry source. It is not a new hardware release. rc4, the installed programs,
and navigation's frozen libraries remain unchanged.

## Contract and ownership

Both adapters expose `/n3mapping/start_map_extension` (`std_srvs/Trigger`). The
backend owner's existing mutex serializes the request with frame processing,
loop commits and map saving. `N3MappingCore::startMapExtension()` requires a
loaded map and the latest processed observation to be `FULL_6DOF_LOCKED` with
`GEOMETRICALLY_CORRECTED` pose source. Searching, invalid input and coasting
cannot authorize this transition. Repeated successful requests are idempotent.

`MappingResuming` shares its existing anchor/session initialization between
cold startup and this handoff. The handoff adopts the current map-to-odometry
transform and matched map anchor; it does not reset the localizer or repeat
candidate registration. Existing loaded-target preparation and acquisition-time
index construction still run if needed. A large-map switch is therefore not
guaranteed to take the small-fixture service time below.

The last accepted input is retained through a shared cloud reference, at most
one scan, until the next localization result, map replacement or handoff. It
seeds the existing odometry sanity check so a jump on the first extension frame
cannot bypass the check. No new worker, graph copy, global lock or acceptance
threshold is introduced. The adapters discard a pending localization pair before
changing mode, so it cannot run after a newer extension pair.

`RealtimeOutput` changes mode under its existing snapshot lock. It preserves
the correction, projection session/generation and lock epoch. `MapArtifacts`
retains its map cache and enables the existing save path. The save service is
advertised in localization but still rejects writes there. Configure the
extension destination with `map_save_path` at startup; the runtime status
`mode` reports the active role. Changing maps or restarting the LIO source still
requires the normal localization lifecycle. No downstream task coordinator was
modified or deployed.

## Four-core validation

All builds and tests use CPUs `0,2,4,6`, Release, ccache and `num_threads=4`.
Humble uses the isolated qualified GTSAM dependency. Noetic uses the existing
Focal Docker image and its existing serial-containment dependency path.

| Check | Humble | Noetic |
| --- | ---: | ---: |
| Full CTest | 69/69, 49.89 s | 67/67, 61.36 s |
| Native ROS service handoff, final run | 11.84 ms | 6.46 ms |
| Native extension maximum position error | 0.311 mm | 0.311 mm |
| Native reload maximum position error | 0.264 mm | 0.131 mm |
| Additional lock events at handoff | 0 | 0 |

Service durations are single integration-test observations, not isolated
performance benchmarks.

The native test creates a partial map through the real ROS node, restarts in
localization with a translated/rotated odometry frame, checks refusal before
lock and refusal to save, then switches the running node. It drives through the
doorway and returns, checks the same projection session and absence of new
search timings, saves, reloads and localizes again. The original pbstream hash
is unchanged. It supplies synthetic Odometry and PointCloud2; it does not run
FAST-LIO or establish real-scene submillimeter accuracy.

The existing core doorway-route E2E now covers cold extension and this handoff.
The handoff case uses a 0.7 rad rotation and `(12,-7,0.3)` translation between
odometry and map frames. It covers tracking loss beyond the partial map,
recovery, asynchronous loop processing, one session anchor, invalid source
rejection and map/trajectory persistence. Its maximum position error is
1.367 mm. A supplementary regression, passed in both builds, rejects an
impossible jump immediately after the handoff without inserting a keyframe.
The live-output unit check preserves correction/session/generation/lock epoch.

Two validation setup issues were resolved without changing production policy:

- The first Humble suite loaded the old installed GTSAM via `LD_LIBRARY_PATH`.
  The runtime identity check correctly selected serial containment, while the
  test expected the qualified build's parallel arena. Prepending the matching
  isolated dependency after sourcing the workspace fixes the test environment.
- The native Noetic fixture initially published its first reload scan before
  the new node was ready. Waiting for output from a new projection session,
  rather than only publisher connection counts, fixes the fixture startup.

The complete suites passed after these corrections. Only the affected route
and native E2Es were repeated after test cleanup. No algorithm thresholds were
relaxed and no tests were skipped.

## Read-only Jetson rc4 audit

The existing measurement used four online Orin NX cores and rc4 `74b6961`.
The localization input map contains **250 keyframes**; the 49-keyframe map
belongs to a different mapping measurement. Atlas was explicitly disabled.
The scans were derived from saved map observations, not independently recorded
raw sensor data. These measurements establish cost, not localization accuracy.

Repeated 200 Hz status messages were deduplicated by `observation_stamp`:

| Completed search observations visible in status | Static startup | Moving startup |
| --- | ---: | ---: |
| Observations | 5 | 12 |
| Search wall time sum | 21.415 s | 53.083 s |
| Process CPU time sum | 44.475 s | 102.172 s |
| Target preparation wall time sum | 16.236 s | 40.550 s |
| Alignment call wall time sum | 3.036 s | 8.089 s |
| Descriptor wall time sum | 0.102 s | 0.187 s |
| Target preparations / cache hits | 36 / 0 | 64 / 0 |

Target preparation accounts for about 76% of these search wall-time sums. The
effective legacy target-cache budget is 128 MiB; observed retained occupancy is
only one or two targets. Zero hits and repeated preparation warrant a controlled
cache/target-reuse experiment. This does not yet prove which eviction or cache
key change causes each miss. Descriptor reduction is not the first target for
this measured workload. Enabling Atlas needs a CPU/RAM comparison, not a blanket
recommendation.

The runtime log contains two moving-startup rejections with
`directional_visibility_disagreement`, with directional support approximately
0.476 and 0.398. The likelihood, margin and temporal guards passed for these
attempts. The status sums above exclude later search completions that were not
observed before the subscriber stopped. Derived scans retain an earlier
acquisition's visibility and do not recompute rays at the synthesized viewpoint;
this audit does not justify weakening the directional guard.

The original resource measurement reports static search averaging 2.08 cores,
343.8 MiB peak RSS and a sampled 3.57-core peak; locked static tracking averages
0.61 cores. The handoff change removes a redundant startup search when entering
extension; it does not fix cold-search cost or demonstrate the half-core goal.
No new SLAM process was started on the Jetson during this audit.

## Artifacts and remaining acceptance

[Machine-readable results](localization_to_extension_20261027.json) record
production/test source hashes, four candidate binary hashes and selected audit
counters. Local builds, raw logs and native E2E maps are retained under
`build/n3mapping/localization_to_extension_20261027` in the parent workspace.
The candidates retain `UNVERIFIED` development build identity. Twelve installed,
previous-candidate and frozen binaries match their pre-task hashes. Concurrent
FAST-LIVO2 documentation edits are excluded from this change.

The [RK3588 follow-up](rk3588_resources_20261028.md) measures this handoff on the
same 250-keyframe input, including service latency, CPU/RAM and map/Atlas reload.
It also compares the existing precomputed Atlas against repeated legacy target
preparation. Jetson remains a read-only audit; independent real-sensor reliability
and long-duration moving-extension resource bounds still need hardware tests.
