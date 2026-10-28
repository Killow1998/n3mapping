# Build and start N3Mapping

[Project overview](../README.md) · [ROS interface](ros_interface.md) · [Development](development.md)

## Prerequisites

Use ROS 2 Humble on Ubuntu 22.04 or ROS 1 Noetic on Ubuntu 20.04.
N3Mapping consumes deskewed body-frame point clouds and odometry from a running
LIO frontend. The frontend can run in the same workspace or publish from another one.

The C++ dependencies are PCL, Eigen3, OpenCV, Protobuf, OpenSSL, glog, OpenMP,
TBB, GTSAM and small_gicp. Install the Ubuntu development packages and build tools:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ccache \
  libprotobuf-dev protobuf-compiler libgoogle-glog-dev libpcl-dev \
  libeigen3-dev libopencv-dev libboost-all-dev libtbb-dev libssl-dev
```

Provide [GTSAM](https://github.com/borglab/gtsam/tree/4.1.1) and
[small_gicp](https://github.com/koide3/small_gicp) through your dependency workspace
or CMake prefix.
GTSAM 4.1.1 is used by the existing build profiles; build it with
`GTSAM_USE_SYSTEM_EIGEN=ON` so it uses the same Eigen as PCL. For dependencies
without a ROS environment hook, set `CMAKE_PREFIX_PATH` and `LD_LIBRARY_PATH`
to their install prefix and its `lib` directory. The
[graph solver guide](graph_solver_execution.md) describes the GTSAM repair
required for qualified parallel graph solves.

## Workspace

```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
git clone https://github.com/Killow1998/n3mapping.git
```

Select the adapter before running `rosdep`, so only the intended package is
visible. The selection script creates local ignore markers.

## ROS 2 Humble

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh humble
rosdep install --from-paths src --ignore-src -r -y
```

Keep the cache and build directories between edits. Each build profile has its
own build, install and log paths.

```bash
source /opt/ros/humble/setup.bash
cd ~/ros_ws
# Source any dependency workspace here, before the selected profile below.
src/n3mapping/scripts/select_distro_wrapper.sh humble

export CCACHE_DIR="$PWD/build/ccache" CCACHE_TEMPDIR="$PWD/build/ccache/tmp"
mkdir -p "$CCACHE_TEMPDIR"
ccache --max-size 8G
export CMAKE_BUILD_PARALLEL_LEVEL=4 MAKEFLAGS=-j4
N3M_BUILD_TYPE=RelWithDebInfo
N3M_PROFILE="n3mapping-$N3M_BUILD_TYPE"

colcon --log-base "log/$N3M_PROFILE" build \
  --symlink-install --executor sequential --packages-up-to n3mapping \
  --build-base "build/$N3M_PROFILE" --install-base "install/$N3M_PROFILE" \
  --cmake-args -DCMAKE_BUILD_TYPE="$N3M_BUILD_TYPE" -DBUILD_TESTING=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DN3MAPPING_BUILD_RESEARCH_TOOLS=OFF -DN3MAPPING_PRODUCT_COMMIT=

source "install/$N3M_PROFILE/setup.bash"
ros2 pkg prefix n3mapping
```

Once dependencies are built and sourced, use `--packages-select n3mapping` for
package-only iteration. In a new terminal, restore the cache environment and
source ROS/dependencies first, then the chosen profile last. Use `Debug` for
unoptimized debugging, `RelWithDebInfo` for daily development, and `Release` for
runtime comparisons; each value selects its own directories. With GCC they use
`-g`, `-O2 -g -DNDEBUG`, and `-O3 -DNDEBUG`, respectively; `MinSizeRel`
uses `-Os -DNDEBUG`. The wrappers do not override CMake's optimization flags.
Inspect `build/$N3M_PROFILE/n3mapping/compile_commands.json` and `ccache --show-stats`
to verify the effective flags and cache activity. An unchanged build has no
compilations to cache. Cache hits accelerate compilation, not runtime execution.

## ROS 1 Noetic

```bash
source /opt/ros/noetic/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh noetic
rosdep install --from-paths src --ignore-src -r -y
```

```bash
source /opt/ros/noetic/setup.bash
cd ~/ros_ws
src/n3mapping/scripts/select_distro_wrapper.sh noetic

export CCACHE_DIR="$PWD/build/ccache" CCACHE_TEMPDIR="$PWD/build/ccache/tmp"
mkdir -p "$CCACHE_TEMPDIR"
N3M_BUILD_TYPE=RelWithDebInfo
N3M_PROFILE="n3mapping-noetic-$N3M_BUILD_TYPE"
catkin config --init --profile "$N3M_PROFILE" \
  --build-space "build/$N3M_PROFILE" --devel-space "devel/$N3M_PROFILE" \
  --install-space "install/$N3M_PROFILE" --log-space "log/$N3M_PROFILE" \
  --cmake-args -DCMAKE_BUILD_TYPE="$N3M_BUILD_TYPE" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DN3MAPPING_BUILD_RESEARCH_TOOLS=OFF -DN3MAPPING_PRODUCT_COMMIT=
catkin build --profile "$N3M_PROFILE" n3mapping --no-status -j4 -p1

source "devel/$N3M_PROFILE/setup.bash"
```

Use a separate `Release` profile for
performance testing and `Debug` for unoptimized debugging. Native
[catkin profiles](https://catkin-tools.readthedocs.io/en/latest/verbs/catkin_profile.html)
keep build/devel/install/log paths separate; retain existing profiles and caches.
After dependencies are ready, `catkin build ... --no-deps` limits iteration to
n3mapping.

## Configure and launch

Copy [config/n3mapping.yaml](../config/n3mapping.yaml) for your robot. Set the
input topics, `body_frame`, `world_frame`, `map_path` and `map_save_path`.
Use a separate save directory when extending a map you want to retain.
The [ROS interface](ros_interface.md) defines frame and velocity conventions.

Pass your configuration to the selected launch file:

```bash
# Humble
ros2 launch n3mapping mapping.launch.py config_file:=/path/to/n3mapping.yaml

# Noetic
roslaunch n3mapping mapping.launch config_file:=/path/to/n3mapping.yaml
```

Replace `mapping` with `localization` or `map_extension` to change the startup
mode. Add `rviz:=false` for headless operation. The launch files default to
`OMP_WAIT_POLICY=PASSIVE` unless it is already set in the environment.

Empty map paths fall back to the source-tree `map/` directory. Set explicit
paths for installed deployments. See the [field guide](field_testing_zh.md)
for versioned release installation and test capture.

## Prepare an Atlas for localization

An Atlas stores map-derived registration geometry, frame/submap descriptors,
heading features and visibility data so localization can reuse them. Compile it after building and
sourcing the workspace:

```bash
# Humble
ros2 run n3mapping n3mapping_localization_atlas_compile --map /path/to/n3map.pbstream

# Noetic
rosrun n3mapping n3mapping_localization_atlas_compile --map /path/to/n3map.pbstream
```

The output is `<map_path>.localization_atlas.pb`. Set `reloc_atlas_enable: true`
and use `reloc_atlas_path` for a custom location. The Product localization
profile requires a matching Atlas. The loader verifies map and configuration
fingerprints. Rebuild the cache after changing the map or relevant parameters;
`--output` selects a new file and `--force` replaces an existing one.

Newly compiled Atlases use compact v3 storage. The loader also reads v2 caches;
missing or incompatible descriptor caches rebuild from the saved point clouds.
Recompile an older Atlas to obtain the startup savings. Older N3M binaries need
their original v2 Atlas, so use `--output` when comparing versions.

Humble launch files use a 64 MiB Fast DDS SHM profile with normal UDP interfaces
when `FASTRTPS_DEFAULT_PROFILES_FILE` is unset. Existing profiles are preserved;
`dds_profile:=/path/to/profile.xml` overrides the path. See [DDS and container setup](logging.md#humble-and-fast-dds-shared-memory)
for the supplied single-host profile and container capacity.

See [development and tests](development.md) for build profiles, E2E checks and
offline evaluation tools, and [logging and DDS setup](logging.md) for diagnostics.
