# ROS interface

[Project overview](../README.md) · [Setup](getting_started.md) · [Logging](logging.md)

## Topics and services

| Name | Kind | Type | Meaning |
| --- | --- | --- | --- |
| `/n3mapping/odometry` | topic | `nav_msgs/Odometry` | Optimized/relocalized pose and measured body-frame twist |
| `/localization/status` | topic | `std_msgs/String` | Provider-neutral JSON status, configurable with `output_status_topic` |
| `/n3mapping/path` | topic | `nav_msgs/Path` | Output trajectory |
| `/n3mapping/cloud_body` | topic | `sensor_msgs/PointCloud2` | Current cloud in body frame |
| `/n3mapping/cloud_world` | topic | `sensor_msgs/PointCloud2` | Current cloud transformed to world/map frame |
| `/n3mapping/global_map` | topic | `sensor_msgs/PointCloud2` | Published global map |
| `/n3mapping/loop_closure_markers` | topic | `visualization_msgs/MarkerArray` | Loop closure and trajectory markers |
| `/n3mapping/relocalization_lock` | topic | `std_msgs/UInt32` | Relocalization lock event counter |
| `/n3mapping/save_map` | service | `std_srvs/Trigger` / `std_srvs/srv/Trigger` | Save map files |
| `/n3mapping/start_map_extension` | service | `std_srvs/Trigger` / `std_srvs/srv/Trigger` | Continue the loaded map from the current verified localization |

Global map visualization is built on demand when a subscriber appears, then
latched/transient-local for later subscribers. Both adapters use the loaded
keyframes and their map revision; an adjacent `global_map.pcd` is an export,
not an unchecked replacement for the selected map. Extension and loop corrections
invalidate the cache. Headless localization does not construct this display cloud.

## Switch from localization to map extension

An already localized node can switch to map extension without restarting the
front end or running global relocalization again. This one-way service requires
the latest processed observation to have a full geometric lock; searching or
odometry-only coasting returns `success: false`. Repeated successful calls keep
the same extension session. The live correction, projection session and lock
counter are retained. Use `map_save_path` to select the extension destination
when starting the node. The save service is available during localization but
rejects writes until extension starts. A different map or restarted LIO source
still needs the normal localization lifecycle. See the
[handoff validation and Jetson audit](evidence/localization_to_extension_20261027.md).

## Status and freshness

External consumers do not need N3Mapping's private status message. The generic
status contains `mode`, `state`, numeric `stamp` (current estimate source time),
`observation_stamp` (last processed global scan source time), `correction_stamp`
(last accepted map correction source time), `frame_id`, and `reason`. Times are
seconds; zero observation/correction time means none is available. Forwarding a
current estimate never refreshes an old observation or correction time. In
localization mode, `tracking` requires an authoritative full localization pose.
Other states are `localizing`, `degraded`, `lost`, and `error`.
Mapping reports `initializing` with `waiting_for_static_start_guard` while the
core's startup protection is holding output, then `tracking` when output is
available. Invalid mapping input and diverged odometry report `error`; a started
process alone is not evidence of mapping output. These observations do not grant
navigation authority or identify the consumer's selected map.
Mapping startup accepts a continuous three-second stable scan/odometry window
(pose stays within 2 cm and 0.01 rad of its window anchor, input gaps at most
0.5 s). This allows a settled stationary front end to start a new mapping session
without waiting for physical movement. Unsettled input retains scan-based motion
detection and the existing 120-second fallback. The window is a bounded startup
consistency check, not proof against future estimator drift. No ROS-specific
readiness flag or upstream process-age assumption is used.
The topic is not latched; consumers check current estimate source and receive
freshness instead of treating a historical lock as current readiness. Correction
age alone is not proof of estimator failure or accuracy. For example:

```json
{"mode":"localization","state":"tracking","stamp":123.45,"observation_stamp":123.1,"correction_stamp":123.0,"frame_id":"map","reason":""}
```

## Frames and velocities

`input_linear_velocity_frame` defaults to `child`, as required by standard
Odometry. The shared-core FAST-LIO output-contract repair of 2026-10-29 uses
`child`; its revisions through `783d62f` used parent-frame linear velocity.
Select `parent` explicitly only for a frontend that reports that convention. N3 rotates
that linear velocity and its covariance using the input odometry orientation;
angular velocity stays in the child frame. Global localization corrections do
not create velocity. The input child frame must match `body_frame`. Invalid
twist/covariance or mismatched child frames suppress output odometry/TF rather
than publishing a fabricated zero measurement. These conversions are shared by
the thin Noetic/Humble adapters, not the backend core.

The repaired FAST-LIO velocities refer to the IMU origin. Pose covariance is
current and expressed in parent axes, but twist covariance is not yet supplied;
its zero-filled default is not calibrated velocity uncertainty. N3 forwards the
input twist and does not estimate missing uncertainty or move its reference point.

## Pose and cloud association

Both wrappers publish live odometry on an independent input callback in
all three run modes. The ROS-free `core::RealtimeOutput` owns authorization and
combines the latest raw pose with the last trusted map-to-odom correction for
the existing global output. There are no public local-odometry or map-correction topics. Exact
applied correction and paired-pose metadata are exposed by the existing status
JSON contract below; their authority remains N3-owned. Each adapter
forwards body clouds and transforms world clouds using the
scan's paired odometry, never the newest callback pose. Cloud conversion and
publication use a different single-worker queue from raw odometry; neither runs
under the projection lock or waits for registration.

The first valid odometry or paired scan freezes the map pose and applied
correction stamp for that source odometry timestamp. A later correction cannot
change that observation's geometry. The cache retains at most `sync_queue_size`
poses; evicted observations are rejected, not reconstructed using a newer
correction. An already-reserved in-flight cloud keeps its copied immutable basis
after eviction unless its generation is invalidated. Approximate synchronization
preserves the cloud's original stamp and associates the basis by its paired
odometry stamp. Invalid input cancels
retained bases and in-flight cloud work at its final publication check; ordinary
geometric observation rejection still retains the trusted anchor.
The check admits completed work; it is not an atomic transaction with ROS publish.

`/localization/status` in both wrappers may include a `projection` object (schema version
1). `kind=odom` accompanies the existing scalar estimate; `kind=cloud` is an
association-only event and must not update scalar state or freshness. Cloud
events retain the latest scalar snapshot without changing its timestamps. Every
available record carries `session_id`, `generation`, `basis_revision`,
`world_frame`, `source_frame`, `body_frame`, integer `odom_stamp_nsec`, and the
finite rigid `map_from_source` and `source_from_body` transforms. Each transform
contains `translation:[x,y,z]` and `quaternion_xyzw:[x,y,z,w]`. Cloud records also
carry integer `cloud_stamp_nsec` and `cloud_frame` (the body-cloud frame);
`world_frame` identifies the corresponding world-cloud output. Both clouds keep
their original source stamp, even when approximate synchronization paired them
with a different odometry stamp. The full paired pose is present even if the
independent raw-odometry subscription dropped that observation.

`available=false` is an odometry invalidation record carrying session,
generation, frames and input stamp, but no invented transform. It cancels old
associations; a newer timestamp alone cannot restore availability. Recovery
requires a genuinely available full-pose record backed by an authorized
correction. Ordinary rejected backend observations do not revoke the anchor.
Consumers must join by the exact integer stamp and declared frames, wait or drop
unpaired inputs within their existing retention/freshness bounds, and never
infer a correction from timestamp differences. A late cloud's frozen correction
only converts that observation into continuous source coordinates; it must not
roll back the current view correction selected by newer valid odometry.
Body-only subscriptions still receive the association; without world-cloud
subscribers the adapter skips full cloud conversion and transformation.

N3 publishes the direct world-to-body TF with the same source pose as its odometry.
A rejected observation leaves the last trusted correction unchanged. Correction
age alone does not revoke output or imply known drift; invalid timestamps, poses
or source frames still invalidate the estimate. Live output keeps actual input
timestamps. Existing localization JSON separates `stamp` (current estimate),
`observation_stamp` (last processed scan) and `correction_stamp` (the anchor
actually applied to this estimate), in seconds. A cloud-first frozen estimate
can use an older anchor than the newest backend observation. Quality reflects
the backend observation, not a fabricated covariance guarantee. A `LOST` backend
observation also leaves an existing trusted anchor intact while raw input stays
valid; it grants no new initialization authority or lock epoch.

## Callback scheduling

Humble uses separate callback groups and independently spun serial executors for
live odometry and paired clouds. The backend, dense trajectory recorder, loop
processing and save service keep one serial executor. Raw odometry uses the
configured synchronization queue (default 100, reliable, volatile) for transport
recovery, then drains at most that many samples and projects only the newest
pending input. Coalesced inputs are counted in the asynchronous event aggregates.
This avoids the reproduced Fast DDS depth-1 startup stall while preserving the
latest-input behavior; timestamp and authority guards still apply. Other
subscriptions retain their configured queues and existing QoS. Product V1 uses exact
synchronization; other profiles keep `sync_time_tolerance`. Localization drains
received backend pairs into the existing `LatestFrameSlot`, replacing pending
work with the newest complete pair before processing it. Mapping and extension
retain their serial paired backend processing. Shutdown stops and joins both
live workers before saving maps. Path and authority events keep backend cadence;
TF keeps odometry source time and suppresses zero/non-increasing timestamps.
