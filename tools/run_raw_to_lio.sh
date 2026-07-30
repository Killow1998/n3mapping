#!/usr/bin/env bash
# Replays the converted raw bag through FAST_LIO and records a fresh LIO bag.
#
# Reproduces the chain the 0723_gravity_v2 evidence documents -- converted ROS2
# raw bag in, /Odometry + /cloud_registered_body + /lio_gravity out -- at the
# same replay rate of 0.5 with the same acknowledgement flow control, so the
# only thing that differs between runs is the estimator itself.
set -eo pipefail

TAG="${1:?usage: run_lio.sh <tag> [ros_domain_id]}"
export ROS_DOMAIN_ID="${2:-90}"

root="/home/user/ros_ws/to_migrate_ws"
raw="$root/artifacts/n3mapping_product_v1/20260723/0723_derived/full_ros2"
out="/home/user/ros_ws/lio_runs/$TAG"
params="$root/src/FAST_LIO/config/mid360.yaml"
replay="$root/src/n3mapping/tools/n3mapping_deterministic_ros2_replay.py"

source /opt/ros/humble/setup.bash
source "$root/install/setup.bash"
set -u

if [[ -e "$out" ]]; then
  echo "output exists, refusing to overwrite: $out" >&2
  exit 2
fi
mkdir -p "$out/logs"

lio_pid=""; rec_pid=""
cleanup() {
  for pid in "$rec_pid" "$lio_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -INT "$pid"; wait "$pid" || true
    fi
  done
}
trap cleanup EXIT INT TERM

ros2 run fast_lio fastlio_mapping \
  --ros-args --params-file "$params" \
  -p common.lid_topic:=/go2w/livox/lidar \
  -p common.imu_topic:=/go2w/livox/imu \
  -p common.gravity_topic:=/lio_gravity \
  -p pcd_save.pcd_save_en:=false \
  >"$out/logs/fast_lio.stdout.log" 2>"$out/logs/fast_lio.stderr.log" &
lio_pid="$!"
sleep 8

ros2 bag record -o "$out/lio_ros2" \
  /go2w/livox/lidar /go2w/livox/imu /Odometry /cloud_registered_body /lio_gravity \
  >"$out/logs/recorder.stdout.log" 2>"$out/logs/recorder.stderr.log" &
rec_pid="$!"
sleep 5

python3 -B "$replay" \
  --bag "$raw" \
  --topic-depth /go2w/livox/lidar=20 \
  --topic-depth /go2w/livox/imu=200 \
  --rate 0.5 \
  --min-subscriptions 1 \
  --match-timeout-s 180 \
  --ack-timeout-s 180 \
  >"$out/logs/replay.jsonl" 2>"$out/logs/replay.stderr"

sleep 15
cleanup
trap - EXIT INT TERM
echo "LIO RUN DONE: $out"
du -sh "$out/lio_ros2"
