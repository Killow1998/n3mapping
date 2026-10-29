# N3Mapping

**Loop closure, relocalization, and map reuse for LiDAR–inertial odometry.**

N3Mapping is a C++ SLAM backend for external LIO frontends such as FAST-LIO.
It takes deskewed point clouds and odometry, optimizes the map and trajectory,
and localizes the robot against a saved point-cloud map.

ROS 2 Humble and ROS 1 Noetic share the same backend through thin ROS adapters.

## What it does

- **Mapping:** build a 3D point-cloud map with online loop closure and asynchronous graph and geometry optimization.
- **Localization:** find a place in a saved map with RHPD descriptors, verify it geometrically, and track the robot's pose.
- **Map extension:** continue an existing map, including switching directly from an established localization.
- **Map reuse:** save and reload maps, export PCD point clouds, and precompute an Atlas for repeated localization.

```text
LIO frontend                     N3Mapping                      Outputs
────────────                     ─────────                      ───────
Deskewed body-frame clouds ──┐    Place recognition          ┌── Corrected poses
                            ├──▶ Geometric registration ───┼── Trajectories
Odometry ───────────────────┘    Loop closure & optimization└── Reusable maps
```

The LIO frontend handles sensor fusion and deskewing. N3Mapping handles global
map alignment, loop closure, and map persistence.

## Recommended frontends

These maintained forks have been tested with N3Mapping. Both provide native
Noetic and Humble adapters around a shared C++ core on their `main` branch.
Choose the frontend for your sensor inputs:

| Repository | Sensor inputs | N3M `cloud_topic` | N3M `odom_topic` |
| --- | --- | --- | --- |
| [Killow1998/FAST_LIO](https://github.com/Killow1998/FAST_LIO) | LiDAR + IMU | `/cloud_registered_body` | `/Odometry` |
| [Killow1998/FAST-LIVO2](https://github.com/Killow1998/FAST-LIVO2) | LiDAR + IMU + calibrated RGB camera | `/n3m_input/cloud_body` | `/n3m_input/odometry` |

For FAST-LIO, keep `publish.scan_publish_en` and
`publish.scan_bodyframe_pub_en` enabled. For FAST-LIVO2, use the full body-cloud
output paired with the final post-visual pose; `/cloud_registered` may contain
only camera-visible points in the world frame. In both cases, configure the
sensor extrinsics and timestamps, and keep the cloud's body frame consistent
with odometry. The IMU body origin may differ from the robot base.

See the [FAST-LIO shared-core validation](https://github.com/Killow1998/FAST_LIO/blob/main/docs/core_wrappers_20261027.md)
and [FAST-LIVO2 integration validation](docs/evidence/fast_livo2_frontend_20261028.md)
for tested revisions and scope. The FAST-LIVO2 Avia/MID360 continuation and
cross-ROS reload results use simulation; general hardware accuracy remains
unqualified. Corrupted maps from before the tracking repair require regeneration.

## Getting started

Build with the [Humble instructions](docs/getting_started.md#ros-2-humble) or
[Noetic instructions](docs/getting_started.md#ros-1-noetic), then source the workspace.

Copy [the default configuration](config/n3mapping.yaml) and set your frontend's
topics, coordinate frames, and map paths. This example uses the FAST-LIO topic
pair; select the FAST-LIVO2 pair from the table above when using that frontend:

```yaml
n3mapping_node:
  ros__parameters:
    cloud_topic: "/cloud_registered_body"
    odom_topic: "/Odometry"
    body_frame: "body"
    world_frame: "map"
    map_path: "/path/to/n3map.pbstream"
    map_save_path: "/path/to/output_map"
```

### ROS 2 Humble

```bash
# Choose a startup mode.
ros2 launch n3mapping mapping.launch.py config_file:=/path/to/n3mapping.yaml
ros2 launch n3mapping localization.launch.py config_file:=/path/to/n3mapping.yaml
ros2 launch n3mapping map_extension.launch.py config_file:=/path/to/n3mapping.yaml
```

### ROS 1 Noetic

```bash
# Choose a startup mode.
roslaunch n3mapping mapping.launch config_file:=/path/to/n3mapping.yaml
roslaunch n3mapping localization.launch config_file:=/path/to/n3mapping.yaml
roslaunch n3mapping map_extension.launch config_file:=/path/to/n3mapping.yaml
```

Add `rviz:=false` to run without RViz. Atlas is enabled by default and prepared
automatically when needed. See [Atlas setup](docs/getting_started.md#prepare-an-atlas-for-localization)
to prepare it ahead of startup or manage a fixed deployment.

## Use and reuse a map

The main outputs are `/n3mapping/odometry`, `/n3mapping/path`, and
`/n3mapping/global_map`. `/localization/status` reports the current mode and
localization state. See the [ROS interface](docs/ros_interface.md) for message
fields, frames, and integration details.

Save a mapping or extension session:

```bash
# Humble
ros2 service call /n3mapping/save_map std_srvs/srv/Trigger '{}'

# Noetic
rosservice call /n3mapping/save_map '{}'
```

An already localized node can start extending the same map without another
global search. Configure `map_save_path` before startup, then call:

```bash
# Humble
ros2 service call /n3mapping/start_map_extension std_srvs/srv/Trigger '{}'

# Noetic
rosservice call /n3mapping/start_map_extension '{}'
```

This switch requires a current geometric lock. It preserves the live coordinate
transform and localization session.

Saved maps contain `n3map.pbstream` for continued mapping and localization,
its matching `.localization_atlas.pb` cache, and `global_map.pcd` for point-cloud
tools. Keep the map and Atlas together when deploying to another machine.

## Documentation

| Guide | Contents |
| --- | --- |
| [Build and setup](docs/getting_started.md) | Dependencies, ROS builds, configuration, Atlas preparation |
| [ROS interface](docs/ros_interface.md) | Topics, services, coordinate frames, status and timing |
| [Logging and DDS](docs/logging.md) | Log storage, health counters, local point-cloud transport |
| [Development](docs/development.md) | Core architecture, tests, offline evaluation |
| [Field testing](docs/field_testing_zh.md) | Release installation and robot testing, in Chinese |
| [Changelog](CHANGELOG.md) | Release history |

## License

[BSD 3-Clause](LICENSE).
