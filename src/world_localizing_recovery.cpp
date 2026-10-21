// A previous lock proposes one local search basin, never acceptance evidence.
#include "n3mapping/world_localizing.h"
#include "relocalization_observation.h"
#include <algorithm>
#include <chrono>
#include <utility>

namespace n3mapping {
void WorldLocalizing::appendTrackingRecovery(
    const Eigen::Isometry3d& odom_pose, RelocalizationObservation& observation,
    RelocalizationPerformance& cost, std::vector<RelocHypothesis>& hypotheses) {
  if (!has_ever_relocalized_ || is_relocalized_ || last_matched_id_ < 0 ||
      !config_.icp_refine_use_gicp || !localization_atlas_ ||
      !localization_atlas_->loaded()) return;
  for (const auto& hyp : hypothesis_manager_.hypotheses())
    if (hyp.alive && hyp.tracking_proposal) return;
  const Eigen::Isometry3d predicted = T_map_odom_ * odom_pose;
  if (!isFiniteRigidPose(predicted)) return;
  const auto anchor = findNearestKeyframe(predicted);
  if (anchor < 0) return;
  const auto crop = localization_atlas_->cropVisibilityIndices(predicted.translation());
  const auto visibility = [&](const Eigen::Isometry3d& pose) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = evaluatePreparedVisibilityConsistency(
        *localization_atlas_->globalMap(), observation.visibility, pose, crop.get());
    cost.visibility_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    ++cost.visibility_evaluations;
    return result;
  };
  const auto before = visibility(predicted);
  if (!before.valid || before.consistency_ratio < config_.loop_min_inlier_ratio) return;
  auto setting = matcher_.getSettings();
  setting.type = small_gicp::RegistrationSetting::GICP;
  setting.downsampling_resolution = config_.icp_refine_downsampling_resolution;
  setting.max_correspondence_distance = std::min(
      config_.gicp_downsampling_resolution, config_.icp_refine_max_correspondence_distance);
  setting.max_iterations = config_.gicp_max_iterations + config_.icp_refine_max_iterations;
  const auto start = std::chrono::steady_clock::now();
  const auto match = matcher_.alignPrepared(localization_atlas_->preparedTarget(),
                                           observation.registration, predicted, setting);
  cost.align_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  ++cost.alignments;
  if (!evaluateRelocMatchQuality(match, config_).accepted ||
      match.inlier_ratio < config_.loop_min_inlier_ratio) return;
  const auto delta = predicted.inverse() * match.T_target_source;
  if (delta.translation().norm() > setting.max_correspondence_distance ||
      Eigen::AngleAxisd(delta.rotation()).angle() > config_.reloc_track_max_rotation) return;
  const auto after = visibility(match.T_target_source);
  if (!after.valid || after.consistency_ratio < config_.loop_min_inlier_ratio) return;
  RelocHypothesis hyp;
  hyp.tracking_proposal = true;
  hyp.seed_match_id = hyp.last_match_id = anchor;
  hyp.T_map_odom = match.T_target_source * odom_pose.inverse();
  hyp.metric = match.metric;
  hyp.cumulative_log_likelihood = computeTrackLogLikelihood(match, predicted);
  hyp.num_updates = hyp.converged_updates = hyp.visibility_updates = 1;
  hyp.visibility_consistency_sum = after.consistency_ratio;
  hyp.visibility_evidence_sum = after.evidence_log_odds;
  hypotheses.push_back(std::move(hyp));
}
} // namespace n3mapping
