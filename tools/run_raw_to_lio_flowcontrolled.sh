#!/usr/bin/env bash
# Reproduces the 0723 LiDAR odometry under the settings that actually produced
# it, taken from raw_lio_replay_evidence.json rather than from any script.
#
# The first attempt used b22's settings instead, on the theory that 0723 had
# been replayed more loosely. It had not: rate 1.0 without flow control cannot
# process this recording at all here, dropping 8247 of 8253 lidar frames and
# reporting 219 million metres in ten seconds. The handshake was load-bearing.
#
# Nothing is varied. The point is to establish that the six-degree attitude
# wander during the stationary opening reproduces before trying to remove it.
# One thing does differ and cannot be helped: the binary now carries the S1a
# gravity-prior change, which was measured as a no-op. If the number comes back
# near 6.79 that measurement holds; if it does not, S1a mattered after all.
set -eo pipefail

root=/home/user/ros_ws/to_migrate_ws
raw=$root/artifacts/n3mapping_product_v1/20260723/0723_derived/full_ros2
params=$root/src/FAST_LIO/config/mid360.yaml
replay=/home/user/ros_ws/to_migrate_ws/install/n3mapping/lib/n3mapping/n3mapping_deterministic_ros2_replay.py
out=${OUT_DIR:?set OUT_DIR}

source /opt/ros/humble/setup.bash
source "$root/install/setup.bash"
set -u

if pgrep -x fastlio_mapping >/dev/null; then
  echo "a fastlio_mapping is already running; kill it before starting" >&2
  exit 3
fi

rm -rf "$out"
mkdir -p "$out/logs"

{
  echo "raw=$raw"
  echo "params_sha=$(sha256sum "$params" | cut -c1-16)"
  echo "fast_lio_head=$(git -C "$root/src/FAST_LIO" log --oneline -1)"
  echo "binary=$(stat -c %y "$root/install/fast_lio/lib/fast_lio/fastlio_mapping")"
  echo "replay=rate 0.5, min-subscriptions 2, lidar ack flow control (原始设置)"
} | tee "$out/provenance.txt"

lio_pid=""
rec_pid=""
cleanup() {
  for p in $rec_pid $lio_pid; do
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      kill -INT "$p"; wait "$p" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

"$root/install/fast_lio/lib/fast_lio/fastlio_mapping" \
  --ros-args \
  --params-file "$params" \
  -p common.lid_topic:=/go2w/livox/lidar \
  -p common.imu_topic:=/go2w/livox/imu \
  -p common.gravity_topic:=/lio_gravity \
  -p common.replay_lidar_ack_topic:=/n3mapping/replay_lidar_ack \
  -p pcd_save.pcd_save_en:=false \
  >"$out/logs/fast_lio.log" 2>&1 &
lio_pid=$!

sleep 8

ros2 bag record -o "$out/lio_ros2" \
  /go2w/livox/lidar /go2w/livox/imu /Odometry /cloud_registered_body /lio_gravity \
  >"$out/logs/recorder.log" 2>&1 &
rec_pid=$!

sleep 5

python3 -B "$replay" \
  --bag "$raw" \
  --topic-depth /go2w/livox/lidar=20 \
  --topic-depth /go2w/livox/imu=200 \
  --rate 0.5 \
  --min-subscriptions 2 \
  --match-timeout-s 120 \
  --ack-timeout-s 120 \
  --flow-control-lidar-topic /go2w/livox/lidar \
  --lidar-ack-topic /n3mapping/replay_lidar_ack \
  --lidar-ack-timeout-s 120 \
  >"$out/logs/replay.log" 2>&1

sleep 15
cleanup
trap - EXIT INT TERM
echo "EXACT REPRO DONE: $out"
