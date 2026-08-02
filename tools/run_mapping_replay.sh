#!/usr/bin/env bash
# Re-runs mapping on the same LIO bag with the loop-closure debug stream on.
#
# The saved map holds 258 edges: 256 odometry, 2 loop. Within 2 m and more than
# 20 keyframes apart there are 162 geometric opportunities, so 160 of them died
# somewhere between detection and the graph, and the optimisation log only ever
# recorded the two that survived. loop_debug_enable writes one JSONL record per
# candidate with the gate that rejected it.
set -eo pipefail

export ROS_DOMAIN_ID="${1:-77}"
lio_bag="/home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_product_v1/20260723/0723_gravity_v2/full/lio_ros2"
map_output="${2:-/home/user/ros_ws/n3mapping_v1_closeout/s5b_keepall_map}"
log_dir="${3:-/home/user/ros_ws/n3mapping_v1_closeout/s5b_keepall_logs}"

root="/home/user/ros_ws/to_migrate_ws"
build="/home/user/ros_ws/n3mapping_fs_build/install/n3mapping"
node="$build/lib/n3mapping/n3mapping_node"
params="${PARAMS:?set PARAMS to a per-stage params file}"
replay="$build/lib/n3mapping/n3mapping_deterministic_ros2_replay.py"

source /opt/ros/humble/setup.bash
source "$root/install/setup.bash"
export LD_LIBRARY_PATH="$build/lib:$root/install/gtsam/lib:$root/install/small_gicp/lib:${LD_LIBRARY_PATH:-}"
set -u

# A run that was interrupted leaves its node alive, and a second run started
# alongside it competes for the same topics and silently mixes two mappings into
# one output. Refuse rather than rely on remembering to clean up.
if pgrep -x n3mapping_node >/dev/null; then
  echo "a n3mapping_node is already running; kill it before starting" >&2
  exit 3
fi

rm -rf "$map_output" "$log_dir"
mkdir -p "$map_output" "$log_dir"

cp "$params" "$log_dir/params.effective.yaml"
{
  echo "bag=$lio_bag"
  echo "rate=${REPLAY_RATE:-1.0}"
  echo "params=$params"
  grep -E "odom_noise_rotation|floor_attitude|loop_max_range|loop_keep_all_verified|loop_min_path_length_m|robust_kernel|loop_noise|loop_axis|mapping_static|icp_refine_max" "$params"
} | tee "$log_dir/provenance.txt"

node_pid=""
cleanup() {
  if [[ -n "$node_pid" ]] && kill -0 "$node_pid" 2>/dev/null; then
    kill -INT "$node_pid"
    wait "$node_pid" || true
  fi
}
trap cleanup EXIT INT TERM

"$node" \
  --ros-args \
  --params-file "$params" \
  -p mode:=mapping \
  -p cloud_topic:=/cloud_registered_body \
  -p odom_topic:=/Odometry \
  -p map_save_path:="$map_output" \
  -p save_global_map_on_shutdown:=true \
  >"$log_dir/n3mapping.log" 2>&1 &
node_pid="$!"

sleep 5
# The LIO bag was recorded during a 0.5x raw replay, so its stored timeline runs
# at twice real duration; 2.0 restores the original ~10 Hz cadence.
python3 -B "$replay" \
  --bag "$lio_bag" \
  --topic-depth /cloud_registered_body=20 \
  --topic-depth /Odometry=50 \
  --rate 2.0 \
  --min-subscriptions 1 \
  --match-timeout-s 180 \
  --ack-timeout-s 180 \
  >"$log_dir/replay.jsonl" 2>"$log_dir/replay.stderr"

sleep 20
cleanup
trap - EXIT INT TERM
echo "MAPPING DONE"
ls -la "$map_output"
