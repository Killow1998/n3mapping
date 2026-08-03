#!/usr/bin/env bash
# Arms three and four of the occlusion experiment.
#
# Four is the control and runs first: the consistency gate with occlusion
# awareness off has to reproduce the baseline, because the threshold was derived
# as the first-order equivalent of the log-odds one at the baseline operating
# point. If it does not reproduce it, the conversion is not faithful and arm
# three cannot be read.
set -eo pipefail

src=/home/user/ros_ws/to_migrate_ws/src/n3mapping
ws=/home/user/ros_ws/to_migrate_ws
closeout=/home/user/ros_ws/n3mapping_v1_closeout

restore() {
  echo "== restoring defaults"
  cd "$src" && git checkout -- include/n3mapping/config.h || true
}
trap restore EXIT

source /opt/ros/humble/setup.bash
source "$ws/install/setup.bash"
set -u

run_arm() {
  local tag=$1 occl=$2 cmargin=$3
  echo "=============== arm $tag   occlusion=$occl  consistency_margin=$cmargin"
  cd "$src"
  git checkout -- include/n3mapping/config.h
  sed -i "s/^    bool reloc_visibility_occlusion_aware = false;/    bool reloc_visibility_occlusion_aware = $occl;/" include/n3mapping/config.h
  sed -i "s/^    double reloc_ambiguity_min_consistency_margin = 0.0;/    double reloc_ambiguity_min_consistency_margin = $cmargin;/" include/n3mapping/config.h
  grep -nE "occlusion_aware = |min_consistency_margin = " include/n3mapping/config.h

  cd "$ws"
  colcon build --packages-select n3mapping \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    >"$closeout/build_$tag.log" 2>&1 || { echo "BUILD FAILED $tag"; tail -20 "$closeout/build_$tag.log"; return 1; }
  echo "  built"

  RUN_DIR=$closeout/atlas_rebuild OUT_DIR=$closeout/today20_$tag \
    bash /tmp/run_today20_weights.sh >"$closeout/t20_$tag.log" 2>&1 || echo "  run rc=$?"
  echo "  done: $(ls $closeout/today20_$tag/*/result.json 2>/dev/null | wc -l)/20"
  bash /tmp/score.sh "$closeout/today20_$tag" 2>&1 | sed -n '/^cases:/,/no lock/p' || true
}

run_arm arm4_gateonly false 0.085
run_arm arm3_both     true  0.085

echo "ALL ARMS DONE"
