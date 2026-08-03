#!/usr/bin/env bash
# Measures the descriptor block weights against the twenty-query contract.
#
# The weights are compile-time defaults -- the evaluator takes no config file --
# so each variant is a rebuild. The atlas has to be rebuilt with it: the sidecar
# holds a tree over descriptor space, and changing the block scales changes the
# metric that tree was built under, so reusing the baseline atlas would search
# one metric and score another.
#
# Runs sequentially because both sides want the whole machine, and restores the
# tree to 1.0/1.0 at the end whatever happens.
set -eo pipefail

src=/home/user/ros_ws/to_migrate_ws/src/n3mapping
ws=/home/user/ros_ws/to_migrate_ws
closeout=/home/user/ros_ws/n3mapping_v1_closeout
map=$closeout/atlas_rebuild/n3map.pbstream

restore() {
  echo "== restoring 1.0/1.0"
  cd "$src" && git checkout -- include/n3mapping/config.h || true
}
trap restore EXIT

# Wait out the baseline if it is still going.
while tmux has-session -t t20base 2>/dev/null; do sleep 30; done
echo "== baseline finished: $(ls $closeout/today20_base/*/result.json 2>/dev/null | wc -l)/20"

source /opt/ros/humble/setup.bash
source "$ws/install/setup.bash"
set -u

for scale in 0.25 0.0; do
  tag=$(echo "$scale" | tr -d '.')
  echo "=============== part_a=$scale aux=$scale (tag $tag)"

  cd "$src"
  git checkout -- include/n3mapping/config.h
  sed -i "s/^    double rhpd_part_a_scale = 1.0;/    double rhpd_part_a_scale = $scale;/" include/n3mapping/config.h
  sed -i "s/^    double rhpd_aux_scale = 1.0;/    double rhpd_aux_scale = $scale;/" include/n3mapping/config.h
  grep -n "rhpd_part_a_scale\|rhpd_aux_scale" include/n3mapping/config.h

  cd "$ws"
  colcon build --packages-select n3mapping \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    >"$closeout/build_$tag.log" 2>&1 || { echo "BUILD FAILED $tag"; tail -20 "$closeout/build_$tag.log"; continue; }
  echo "  built"

  # Fresh atlas under the new metric, beside a link to the same map bytes.
  rundir=$closeout/atlas_v$tag
  rm -rf "$rundir"; mkdir -p "$rundir"
  ln -s "$map" "$rundir/n3map.pbstream"
  "$ws/build/n3mapping/n3mapping_localization_atlas_compile" \
    --map "$rundir/n3map.pbstream" \
    --output "$rundir/n3map.pbstream.localization_atlas.pb" --force \
    >"$closeout/atlas_$tag.log" 2>&1 || { echo "ATLAS FAILED $tag"; tail -20 "$closeout/atlas_$tag.log"; continue; }
  echo "  atlas built"

  RUN_DIR=$rundir OUT_DIR=$closeout/today20_v$tag bash /tmp/run_today20_weights.sh \
    >"$closeout/t20_v$tag.log" 2>&1 || echo "  run rc=$?"
  echo "  today20 done: $(ls $closeout/today20_v$tag/*/result.json 2>/dev/null | wc -l)/20"
done

echo "ALL VARIANTS DONE"
