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

## Getting started

Build with the [Humble instructions](docs/getting_started.md#ros-2-humble) or
[Noetic instructions](docs/getting_started.md#ros-1-noetic), then source the workspace.

Copy [the default configuration](config/n3mapping.yaml) and set your frontend's
topics, coordinate frames, and map paths:

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
