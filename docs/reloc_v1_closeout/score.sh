#!/usr/bin/env bash
# Scores a today20 output directory with the closeout aggregator.
#
# The aggregator wants a run laid out as <run>/today20_eval and
# <run>/f7_0723_map/dense_trajectory.csv, so each output directory is presented
# in that shape by link rather than copied. The dense trajectory is the map
# session's own optimized pose -- same reference for every arm, which is what a
# comparison needs; it is not independent ground truth and is not reported as one.
set -eo pipefail

out=$1        # e.g. /home/user/ros_ws/n3mapping_v1_closeout/today20_base
tag=$(basename "$out")
closeout=/home/user/ros_ws/n3mapping_v1_closeout
product=/home/user/ros_ws/to_migrate_ws/artifacts/n3mapping_product_v1/20260725/candidate_dd25f86

s=$closeout/score_$tag
rm -rf "$s"; mkdir -p "$s/f7_0723_map"
ln -s "$out" "$s/today20_eval"
ln -s "$product/f7_0723_map/dense_trajectory.csv" "$s/f7_0723_map/dense_trajectory.csv"

python3 "$closeout/aggregate_today20.py" "$s"
