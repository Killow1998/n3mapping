#!/usr/bin/env bash
# Runs the twenty-query relocalization evaluation against the current build.
#
# The stock runner points at n3mapping_v1_build and writes into the candidate
# directory, skipping queries that already have results. Both are wrong for
# comparing a descriptor change: the binary has to be the one being changed, and
# the baseline has to be re-run with it so the comparison isolates the weights
# rather than whatever else differs between two build trees.
set -eo pipefail

root=/home/user/ros_ws/to_migrate_ws
run=${RUN_DIR:-/home/user/ros_ws/n3mapping_v1_closeout/atlas_rebuild}
queries=$root/artifacts/n3mapping_product_v1/20260723/0723_gravity_v2/queries
out=${OUT_DIR:?set OUT_DIR}

source /opt/ros/humble/setup.bash
source "$root/install/setup.bash"
set -u

eval_bin=$root/build/n3mapping/n3mapping_relocalization_manifest_eval
rm -rf "$out"
mkdir -p "$out"

{
  echo "eval=$eval_bin"
  echo "built=$(stat -c %y "$eval_bin")"
  echo "map=$run/n3map.pbstream (atlas recompiled on this branch)"
  echo "atlas_built=$(stat -c %y "$run/n3map.pbstream.localization_atlas.pb")"
  echo "rhpd_part_a_scale=$(grep -oE 'rhpd_part_a_scale = [0-9.]+' "$root/src/n3mapping/include/n3mapping/config.h")"
  echo "rhpd_aux_scale=$(grep -oE 'rhpd_aux_scale = [0-9.]+' "$root/src/n3mapping/include/n3mapping/config.h")"
} | tee "$out/provenance.txt"

n=0
for d in "$queries"/*/; do
  q=$(basename "$d")
  n=$((n + 1))
  timeout 600 "$eval_bin" \
    --map "$run/n3map.pbstream" \
    --atlas "$run/n3map.pbstream.localization_atlas.pb" \
    --manifest "$d/manifest/frames.csv" \
    --output "$out/$q" \
    >"$out/$q.stdout.log" 2>"$out/$q.stderr.log" || echo "  $q rc=$?"
  echo "  [$n] $q done"
done
echo "TODAY20 DONE: $out"
