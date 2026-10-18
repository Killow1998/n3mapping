// Experimental candidate for baseline 114937a; not linked into production.
#include "n3mapping/world_localizing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>

namespace n3mapping {
void WorldLocalizing::killFreeSpaceDominatedHypotheses(
    const PointCloudT::Ptr &query_cloud, const Eigen::Isometry3d &odom_pose) {
  if (!config_.reloc_free_space_enable || !query_cloud || query_cloud->empty())
    return;
  // Experimental: geometry, temporal evidence and free space must nominate
  // the same physical pose before any candidate can be removed.
  const auto ranking = decision_policy_.rank(hypothesis_manager_.hypotheses(), odom_pose);
  const auto* free_best = freeSpaceBestHypothesis(query_cloud, odom_pose);
  auto guard_config = config_;
  guard_config.reloc_free_space_mode = "veto";
  const RelocalizationDecisionPolicy guard(guard_config);
  const auto decision = guard.evaluate(
      ranking, hypothesis_manager_.windowCount(), hypothesis_manager_.winnerStreak(),
      hypothesis_manager_.motionBaseline(odom_pose), free_best, odom_pose);
  if (decision.temporal_window_pending || !ranking.top1 || !free_best ||
      !decision.pass_free_space ||
      !decision.pass_log_likelihood || !decision.pass_winner_streak ||
      !decision.pass_converged_updates || !decision.pass_moving_visibility) return;
  auto &hypotheses = hypothesis_manager_.hypotheses();
  size_t alive = 0;
  for (const auto &hyp : hypotheses)
    if (hyp.alive)
      ++alive;
  if (alive < 2)
    return;
  rebuildFreeSpaceGridIfNeeded();
  if (!free_space_grid_.valid())
    return;

  std::vector<FreeSpaceGrid::Score> scores(hypotheses.size());
  int best = -1;
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    const auto &hyp = hypotheses[i];
    if (!hyp.alive)
      continue;
    scores[i] =
        free_space_grid_.score(*query_cloud, hyp.T_map_odom * odom_pose);
    if (!scores[i].valid)
      continue;
    if (best < 0 || scores[i].value > scores[best].value)
      best = static_cast<int>(i);
  }
  if (best < 0)
    return;

  for (size_t i = 0; i < hypotheses.size(); ++i) {
    if (!hypotheses[i].alive)
      continue;
    const Eigen::Isometry3d hp = hypotheses[i].T_map_odom * odom_pose;
    VLOG(1) << "[Reloc/FreeSpaceDump] i=" << i
            << " seed=" << hypotheses[i].seed_match_id
            << " conv=" << hypotheses[i].converged_updates
            << " value=" << scores[i].value
            << " occ=" << scores[i].occupied_fraction
            << " free=" << scores[i].free_fraction
            << " n=" << scores[i].scored_points << " sd=" << scores[i].stddev
            << " xyz=" << hp.translation().x() << "," << hp.translation().y()
            << "," << hp.translation().z();
  }

  // Retain the existing five-sigma heuristic for this experiment. Its sampling
  // variance assumes independent points; correlated returns and map coverage
  // can violate that assumption, so this is not a calibrated safety guarantee.
  const double sigmas = config_.reloc_free_space_kill_sigmas;
  const auto &top = scores[static_cast<size_t>(best)];
  size_t killed = 0;
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    auto &hyp = hypotheses[i];
    if (!hyp.alive || static_cast<int>(i) == best || !scores[i].valid)
      continue;
    const double noise = std::sqrt(top.stddev * top.stddev +
                                   scores[i].stddev * scores[i].stddev);
    if (top.value - scores[i].value > sigmas * noise) {
      hyp.alive = false;
      ++killed;
    }
  }
  if (killed > 0) {
    VLOG(1) << "[Reloc/FreeSpace] killed " << killed << " of " << alive
            << " hypotheses, best value=" << top.value
            << " occ=" << top.occupied_fraction << " free=" << top.free_fraction
            << " n=" << top.scored_points;
  }
}


}
