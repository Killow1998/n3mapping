# Logging and local DDS setup

[Project overview](../README.md) · [ROS interface](ros_interface.md)

## Managed logs

Mapping and map extension save:

```text
<map_save_path>/n3map.pbstream
<map_save_path>/global_map.pcd
<map_save_path>/optimization.log
```

Humble and Noetic share the ROS-free asynchronous logger. `optimization.log`,
loop/relocalization/trial/performance JSONL and node-owned ROS/glog messages use
one bounded queue and one disk worker. Producers perform no log filesystem I/O.
The queue holds at most 2 MiB, reserving its last quarter for warnings and critical
events; individual records are limited to 64 KiB. The worker writes complete
records in batches, normally every 200 ms. Critical bursts wake it after a short
20 ms coalescing window. Frame, live-output and decision counters are aggregated;
detailed per-frame/per-candidate diagnostics remain opt-in in both ROS versions.
No matching, locking or geometry thresholds change.

**The shared limit is 100 MB (100,000,000 bytes) per instance, across all managed
current files, archives and restarts.** It is not a separate budget per stream.
`N3MAPPING_LOG_DIR` identifies the instance and defaults to
`${ROS_HOME:-$HOME/.ros}/n3mapping/logs`. Keep this directory stable across restarts;
concurrent instances must use distinct directories. `N3MAPPING_LOG_TOTAL_BYTES`
can lower the limit (minimum 1 MiB), but cannot raise it above 100 MB. The catalog
tracks existing configured destinations even outside this directory. The worker
expires the oldest managed files before exceeding the total budget; files no
longer selected by a new configuration remain accounted for. Startup preserves
complete records and removes an incomplete trailing record. Optimization logs
now append across restarts instead of being truncated.

The instance directory contains `runtime.log`, `events.jsonl`, `catalog.json`
and `logger_health.json`. Recognizable historical `n3mapping_node.*.log.*` files
in `GLOG_log_dir` also join retention; without an explicit instance directory,
the previous source-tree `logs/` location is checked as a fallback. Unrelated
files are never deleted. Previously configured external debug destinations are
managed once registered in the catalog; unknown historical paths cannot be
discovered automatically. Map files, bags and logs owned by roscore, launchers,
frontends or terminal capture are separate artifacts/services.

Default events cover process lifecycle, map load/save outcomes, backend and
live-output state changes, relocalization locks, loop commits/rebases/stale jobs,
and logging health. Existing detailed diagnostics retain their formats.
`/n3mapping/logging_status` publishes memory-resident counters once per wall-clock
second on the independent live queue. Queue overflow, critical loss, oversized
records, write failures, lost records, last failed destination and repaired tail
bytes are observable even if the log disk is unavailable. Health snapshots and
failure summaries are also persisted when storage permits. `retained_bytes`
includes a conservative 576 KiB metadata reservation; it is not exact disk usage.

Queue admission is not a persistence acknowledgement. A full queue drops and
counts records instead of waiting for disk; critical traffic has reserved space
but is not guaranteed lossless. Normal shutdown drains the worker. Fatal logging
has a bounded best-effort drain; power-loss durability is not promised. An invalid
catalog or inaccessible instance directory disables file writing and reports
`storage_ready=false`; repair that condition and restart. Per-destination write
failures recover on subsequent batches without stopping SLAM. Native ROS/glog
file and console duplication is disabled for node diagnostics; use the status
topic and managed logs to inspect normal runtime messages.

For real-device update/build commands, per-run log capture, and local support ZIP
export, see [实机测试与问题反馈](field_testing_zh.md). The support tool
does not launch frontends, upload logs, or include maps/bags. Detailed JSONL
logging is opt-in; default matching and locking behavior is unchanged.

## Humble and Fast DDS shared memory

For other Humble applications as well, check **DDS shared-memory segment capacity
against serialized message size and simultaneous traffic** when local point-cloud
delivery stalls or odometry callbacks disappear. Available system RAM and DDS
segment capacity are different limits. Fast DDS 2.6 documents a default segment
of 512 KiB and warns that a segment no larger than a message risks data loss;
see its [SHM transport documentation](https://fast-dds.docs.eprosima.com/en/2.6.x/fastdds/transport/shared_memory/shared_memory.html).
This applies to the selected Fast DDS transport/version, not every ROS 2 RMW.

In our Humble replay (`rmw_fastrtps_cpp` 6.2.10, Fast DDS 2.6.11), a M2DGR
frontend cloud serialized to 1,091,869 bytes while the observed default SHM
mapping was about 0.5 MiB.
In that 2026-10-21 replay, with the original depth-1 live subscription, several runs lost live callbacks
despite complete backend input. A 64 MiB SHM profile restored all 3,265 callbacks
in the complete `gate_02` replay, with 3,250 outputs after 15 startup waits and a
maximum input timestamp gap of 106.8 ms. Increasing the subscription depth or
changing executor spinning did not resolve the delivery problem. The profile
is an opt-in, validated **single-host** configuration, not a universal memory
budget or a replacement for measuring the actual workload.
The separate small-message startup/history correction is documented in the
[startup evidence](evidence/runtime_cpu_20261017/README.md#humble-startup-input-and-branch-integration--2026-10-23).
Changing only the segment back to 512 KiB reproduces the failure: 733 callbacks
from 3,265 backend inputs, with a maximum 9.819 s input gap. All raw sensor inputs
remain present; the failed delivery check is retained.

Before starting **all participating processes**, including sensor/replay
publishers, FAST_LIO, n3m and recording tools, set:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/absolute/path/to/n3mapping/config/fastdds_local.xml
```

The [profile](../config/fastdds_local.xml) enables a 64 MiB SHM segment per participant
and loopback UDP. Its UDP interface whitelist is deliberately local-only; adapt
the interface configuration before using multiple computers. Existing processes
must be restarted to pick up their environment. The
[Humble RMW documentation](https://github.com/ros2/rmw_fastrtps/blob/humble/README.md)
describes loading the XML. SHM transport is distinct from Fast DDS Data Sharing;
this profile does not enable zero-copy Data Sharing or change subscription QoS.

For future incidents, retain the RMW/Fast DDS versions, loaded profile, largest
serialized frame and actual `fastrtps` mappings in `/proc/<pid>/maps`. Compare
publisher, subscriber callback and output counts, timestamp gaps and callback
phase timings with the same input before and after changing the segment. Size
for outstanding traffic as well as one frame, and measure process RSS separately.
The [complete six-sequence record](evidence/runtime_cpu_20261017/README.md#real-data-clocks-references-and-shared-loop-solving--2026-10-21)
contains the paired controls, timing/reference audit, CPU/RAM measurements and
remaining accuracy limits. Raw replay evidence stays under
`build/n3mapping/real_robustness_20261021`, including the failed controls and all
six successful `full_ros_shm64` delivery/checkpoint validations.
