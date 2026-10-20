#include "loop_closure_diagnostics.h"
#include "n3mapping/cloud_utils.h"

namespace n3mapping::loop_diagnostics {

double rotationAngle(const Eigen::Isometry3d &transform) {
  return std::abs(Eigen::AngleAxisd(transform.rotation()).angle());
}

Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d &transform) {
  return transform.rotation().eulerAngles(0, 1, 2);
}





std::pair<double, double>
meanLoopResidual(const std::vector<EdgeInfo> &edges,
                 const std::map<int64_t, Eigen::Isometry3d> &poses) {
  if (edges.empty()) {
    return {0.0, 0.0};
  }

  double translation_sum = 0.0;
  double rotation_sum = 0.0;
  std::size_t count = 0;
  for (const auto &edge : edges) {
    auto from_it = poses.find(edge.from_id);
    auto to_it = poses.find(edge.to_id);
    if (from_it == poses.end() || to_it == poses.end()) {
      continue;
    }
    const Eigen::Isometry3d predicted =
        from_it->second.inverse() * to_it->second;
    const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
    translation_sum += residual.translation().norm();
    rotation_sum += rotationAngle(residual);
    ++count;
  }

  if (count == 0) {
    return {0.0, 0.0};
  }
  return {translation_sum / static_cast<double>(count),
          rotation_sum / static_cast<double>(count)};
}

LoopResidualAxisStats
meanLoopResidualAxes(const std::vector<EdgeInfo> &edges,
                     const std::map<int64_t, Eigen::Isometry3d> &poses) {
  LoopResidualAxisStats stats;
  std::size_t count = 0;
  for (const auto &edge : edges) {
    auto from_it = poses.find(edge.from_id);
    auto to_it = poses.find(edge.to_id);
    if (from_it == poses.end() || to_it == poses.end()) {
      continue;
    }
    const Eigen::Isometry3d predicted =
        from_it->second.inverse() * to_it->second;
    const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
    stats.translation += residual.translation().cwiseAbs();
    stats.rpy += rollPitchYaw(residual).cwiseAbs();
    ++count;
  }
  if (count == 0) {
    return stats;
  }
  const double inv_count = 1.0 / static_cast<double>(count);
  stats.translation *= inv_count;
  stats.rpy *= inv_count;
  return stats;
}

void accumulatePoseUpdateStats(
    const std::map<int64_t, Eigen::Isometry3d> &before,
    const std::map<int64_t, Eigen::Isometry3d> &after,
    CoreLoopClosureResult *result) {
  double translation_sum = 0.0;
  double rotation_sum = 0.0;
  std::size_t count = 0;

  for (const auto &[id, before_pose] : before) {
    auto after_it = after.find(id);
    if (after_it == after.end()) {
      continue;
    }
    const Eigen::Isometry3d delta = before_pose.inverse() * after_it->second;
    const double translation = delta.translation().norm();
    const double rotation = rotationAngle(delta);
    translation_sum += translation;
    rotation_sum += rotation;
    result->max_pose_update_translation =
        std::max(result->max_pose_update_translation, translation);
    result->max_pose_update_rotation =
        std::max(result->max_pose_update_rotation, rotation);
    ++count;
  }

  if (count == 0) {
    return;
  }
  const std::size_t previous_count = result->pose_update_count;
  const std::size_t total_count = previous_count + count;
  result->mean_pose_update_translation =
      (result->mean_pose_update_translation *
           static_cast<double>(previous_count) +
       translation_sum) /
      static_cast<double>(total_count);
  result->mean_pose_update_rotation =
      (result->mean_pose_update_rotation * static_cast<double>(previous_count) +
       rotation_sum) /
      static_cast<double>(total_count);
  result->pose_update_count = total_count;
}

void assignGraphTrialDiagnostics(LoopDebugCandidateEvent *event,
                                 const LoopGraphTrialDiagnostics &diagnostics) {
  if (!event) {
    return;
  }
  event->graph_trial_success = diagnostics.success;
  event->graph_trial_residual_x_after = diagnostics.residual_x_after;
  event->graph_trial_residual_y_after = diagnostics.residual_y_after;
  event->graph_trial_residual_z_after = diagnostics.residual_z_after;
  event->graph_trial_residual_roll_after = diagnostics.residual_roll_after;
  event->graph_trial_residual_pitch_after = diagnostics.residual_pitch_after;
  event->graph_trial_residual_yaw_after = diagnostics.residual_yaw_after;
  event->graph_trial_residual_translation_norm_after =
      diagnostics.residual_translation_norm_after;
  event->graph_trial_residual_rotation_norm_after =
      diagnostics.residual_rotation_norm_after;
  event->graph_trial_mean_pose_update_translation =
      diagnostics.mean_pose_update_translation;
  event->graph_trial_max_pose_update_translation =
      diagnostics.max_pose_update_translation;
  event->graph_trial_mean_pose_update_rotation =
      diagnostics.mean_pose_update_rotation;
  event->graph_trial_max_pose_update_rotation =
      diagnostics.max_pose_update_rotation;
  event->graph_trial_existing_loop_residual_delta =
      diagnostics.existing_loop_residual_delta;
  event->graph_trial_odom_residual_delta = diagnostics.odom_residual_delta;
  event->graph_trial_consistency_score = diagnostics.consistency_score;
  event->graph_trial_recommendation = diagnostics.recommendation;
}

void assignConsensusEstimatorTrialDiagnostics(
    LoopDebugCandidateEvent *event,
    const LoopGraphTrialDiagnostics &diagnostics) {
  if (!event) {
    return;
  }
  event->consensus_estimator_trial_success = diagnostics.success;
  event->consensus_estimator_trial_residual_x_after =
      diagnostics.residual_x_after;
  event->consensus_estimator_trial_residual_y_after =
      diagnostics.residual_y_after;
  event->consensus_estimator_trial_residual_z_after =
      diagnostics.residual_z_after;
  event->consensus_estimator_trial_residual_roll_after =
      diagnostics.residual_roll_after;
  event->consensus_estimator_trial_residual_pitch_after =
      diagnostics.residual_pitch_after;
  event->consensus_estimator_trial_residual_yaw_after =
      diagnostics.residual_yaw_after;
  event->consensus_estimator_trial_residual_translation_norm_after =
      diagnostics.residual_translation_norm_after;
  event->consensus_estimator_trial_residual_rotation_norm_after =
      diagnostics.residual_rotation_norm_after;
  event->consensus_estimator_trial_consistency_score =
      diagnostics.consistency_score;
  event->consensus_estimator_trial_recommendation = diagnostics.recommendation;
}

void assignSegmentDiagnostics(LoopDebugCandidateEvent *event,
                              const VerifiedLoop &loop) {
  if (!event) {
    return;
  }
  event->segment_pair_count = loop.segment_pair_count;
  event->segment_valid_pair_count = loop.segment_valid_pair_count;
  event->segment_consensus_inlier_count = loop.segment_consensus_inlier_count;
  event->segment_consensus_ratio = loop.segment_consensus_ratio;
  event->segment_translation_median = loop.segment_translation_median;
  event->segment_translation_std = loop.segment_translation_std;
  event->segment_yaw_median = loop.segment_yaw_median;
  event->segment_yaw_std = loop.segment_yaw_std;
  event->segment_z_std = loop.segment_z_std;
  event->segment_roll_pitch_std = loop.segment_roll_pitch_std;
  event->segment_direction = loop.segment_direction;
  event->segment_recommendation = loop.segment_recommendation;
}

void assignConsensusDiagnostics(LoopDebugCandidateEvent *event,
                                const VerifiedLoop &loop) {
  if (!event) {
    return;
  }
  event->consensus_shadow_decision = loop.consensus_shadow_decision;
  event->consensus_shadow_reason = loop.consensus_shadow_reason;
  event->consensus_valid_pair_count = loop.consensus_valid_pair_count;
  event->consensus_left_support_count = loop.consensus_left_support_count;
  event->consensus_right_support_count = loop.consensus_right_support_count;
  event->consensus_contradiction_count = loop.consensus_contradiction_count;
  event->consensus_median_translation_delta =
      loop.consensus_median_translation_delta;
  event->consensus_mad_translation_delta = loop.consensus_mad_translation_delta;
  event->consensus_median_rotation_delta = loop.consensus_median_rotation_delta;
  event->consensus_mad_rotation_delta = loop.consensus_mad_rotation_delta;
  event->consensus_estimator_valid = loop.consensus_estimator_valid;
  event->consensus_estimator_pair_count = loop.consensus_estimator_pair_count;
  event->consensus_estimator_inlier_count =
      loop.consensus_estimator_inlier_count;
  event->consensus_estimator_inlier_ratio =
      loop.consensus_estimator_inlier_ratio;
  event->consensus_estimator_translation_median =
      loop.consensus_estimator_translation_median;
  event->consensus_estimator_z_median = loop.consensus_estimator_z_median;
  event->consensus_estimator_yaw_median = loop.consensus_estimator_yaw_median;
  event->consensus_estimator_translation_mad =
      loop.consensus_estimator_translation_mad;
  event->consensus_estimator_z_mad = loop.consensus_estimator_z_mad;
  event->consensus_estimator_yaw_mad = loop.consensus_estimator_yaw_mad;
  event->consensus_estimator_measurement_delta_translation =
      loop.consensus_estimator_measurement_delta_translation;
  event->consensus_estimator_measurement_delta_rotation =
      loop.consensus_estimator_measurement_delta_rotation;
  event->consensus_estimator_recommendation =
      loop.consensus_estimator_recommendation;
}

void assignConsensusDiagnostics(LoopDebugCandidateEvent *event,
                                const LoopConsensusResult &consensus) {
  if (!event) {
    return;
  }
  event->consensus_shadow_decision =
      loopConsensusDecisionName(consensus.decision);
  event->consensus_shadow_reason = consensus.reason;
  event->consensus_valid_pair_count = consensus.valid_pair_count;
  event->consensus_left_support_count = consensus.left_support_count;
  event->consensus_right_support_count = consensus.right_support_count;
  event->consensus_contradiction_count = consensus.contradiction_count;
  event->consensus_median_translation_delta =
      consensus.median_translation_delta;
  event->consensus_mad_translation_delta = consensus.mad_translation_delta;
  event->consensus_median_rotation_delta = consensus.median_rotation_delta;
  event->consensus_mad_rotation_delta = consensus.mad_rotation_delta;
  event->consensus_estimator_valid = consensus.estimator_valid;
  event->consensus_estimator_pair_count = consensus.estimator_pair_count;
  event->consensus_estimator_inlier_count = consensus.estimator_inlier_count;
  event->consensus_estimator_inlier_ratio = consensus.estimator_inlier_ratio;
  event->consensus_estimator_translation_median =
      consensus.estimator_translation_median;
  event->consensus_estimator_z_median = consensus.estimator_z_median;
  event->consensus_estimator_yaw_median = consensus.estimator_yaw_median;
  event->consensus_estimator_translation_mad =
      consensus.estimator_translation_mad;
  event->consensus_estimator_z_mad = consensus.estimator_z_mad;
  event->consensus_estimator_yaw_mad = consensus.estimator_yaw_mad;
  event->consensus_estimator_measurement_delta_translation =
      consensus.estimator_measurement_delta_translation;
  event->consensus_estimator_measurement_delta_rotation =
      consensus.estimator_measurement_delta_rotation;
  event->consensus_estimator_recommendation =
      consensus.estimator_recommendation;
}

double processingTimeSeconds() {
  using Clock = std::chrono::system_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}



ZDistributionStats computeZStats(const core::LioFrame::PointCloud::Ptr &cloud) {
  ZDistributionStats stats;
  if (!cloud) {
    return stats;
  }
  for (const auto &point : cloud->points) {
    if (!isFinitePoint(point)) {
      continue;
    }
    const double z = static_cast<double>(point.z);
    stats.min_z = std::min(stats.min_z, z);
    stats.max_z = std::max(stats.max_z, z);
    stats.sum_z += z;
    stats.samples.push_back(z);
    ++stats.count;
  }
  return stats;
}

ZDistributionStats
computeTransformedZStats(const core::LioFrame::PointCloud::Ptr &cloud,
                         const Eigen::Isometry3d &transform) {
  ZDistributionStats stats;
  if (!cloud) {
    return stats;
  }
  for (const auto &point : cloud->points) {
    if (!isFinitePoint(point)) {
      continue;
    }
    const Eigen::Vector3d transformed =
        transform * Eigen::Vector3d(point.x, point.y, point.z);
    if (!std::isfinite(transformed.z())) {
      continue;
    }
    stats.min_z = std::min(stats.min_z, transformed.z());
    stats.max_z = std::max(stats.max_z, transformed.z());
    stats.sum_z += transformed.z();
    stats.samples.push_back(transformed.z());
    ++stats.count;
  }
  return stats;
}

double zOverlapRatio(const ZDistributionStats &a, const ZDistributionStats &b) {
  if (a.count == 0 || b.count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double denominator = std::min(a.span(), b.span());
  if (!std::isfinite(denominator) || denominator <= 1e-9) {
    return 0.0;
  }
  const double overlap =
      std::min(a.max_z, b.max_z) - std::max(a.min_z, b.min_z);
  return std::max(0.0, std::min(1.0, overlap / denominator));
}

double robustZOverlapRatio(const ZDistributionStats &a,
                           const ZDistributionStats &b) {
  if (a.count == 0 || b.count == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double a_min = a.robustMin();
  const double a_max = a.robustMax();
  const double b_min = b.robustMin();
  const double b_max = b.robustMax();
  if (!std::isfinite(a_min) || !std::isfinite(a_max) || !std::isfinite(b_min) ||
      !std::isfinite(b_max)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double denominator = std::min(a_max - a_min, b_max - b_min);
  if (denominator <= 1e-9) {
    return 0.0;
  }
  const double overlap = std::min(a_max, b_max) - std::max(a_min, b_min);
  return std::max(0.0, std::min(1.0, overlap / denominator));
}

double zCentroidDelta(const ZDistributionStats &source,
                      const ZDistributionStats &target) {
  const double source_mean = source.mean();
  const double target_mean = target.mean();
  if (!std::isfinite(source_mean) || !std::isfinite(target_mean)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return source_mean - target_mean;
}



bool hasUsableHypothesisFitness(const MatchResult &result) {
  return result.converged && std::isfinite(result.fitness_score);
}

VerticalHypothesisDiagnostics computeVerticalHypothesisDiagnostics(
    PointCloudMatcher &matcher, const core::LioFrame::PointCloud::Ptr &target,
    const core::LioFrame::PointCloud::Ptr &source,
    const MatchResult &zero_result) {
  VerticalHypothesisDiagnostics diagnostics;
  constexpr std::array<double, 7> kZOffsets = {-2.0, -1.0, -0.5, 0.0,
                                               0.5,  1.0,  2.0};
  struct Candidate {
    double z_offset = 0.0;
    double fitness = std::numeric_limits<double>::quiet_NaN();
  };
  std::vector<Candidate> usable;
  usable.reserve(kZOffsets.size());

  auto add_result = [&](double z_offset, const MatchResult &result) {
    if (!hasUsableHypothesisFitness(result)) {
      return;
    }
    ++diagnostics.count;
    usable.push_back({z_offset, result.fitness_score});
    if (!std::isfinite(diagnostics.best_z_offset_fitness) ||
        result.fitness_score < diagnostics.best_z_offset_fitness) {
      diagnostics.best_z_offset_m = z_offset;
      diagnostics.best_z_offset_fitness = result.fitness_score;
    }
  };

  diagnostics.zero_z_fitness = std::isfinite(zero_result.fitness_score)
                                   ? zero_result.fitness_score
                                   : std::numeric_limits<double>::quiet_NaN();
  add_result(0.0, zero_result);
  for (double z_offset : kZOffsets) {
    if (z_offset == 0.0) {
      continue;
    }
    Eigen::Isometry3d init = Eigen::Isometry3d::Identity();
    init.translation().z() = z_offset;
    try {
      add_result(z_offset, matcher.alignCloud(target, source, init));
    } catch (const std::exception &) {
      // Diagnostics must never affect loop-closure behavior.
    }
  }

  if (diagnostics.count == 0 ||
      !std::isfinite(diagnostics.best_z_offset_fitness)) {
    diagnostics.edge_recommendation = "reject";
    return diagnostics;
  }
  if (std::isfinite(diagnostics.zero_z_fitness)) {
    diagnostics.fitness_gap_zero_vs_best =
        diagnostics.zero_z_fitness - diagnostics.best_z_offset_fitness;
  }

  const double near_best_band =
      std::max(0.02, 0.10 * std::max(1e-6, diagnostics.best_z_offset_fitness));
  double near_min = std::numeric_limits<double>::infinity();
  double near_max = -std::numeric_limits<double>::infinity();
  for (const auto &candidate : usable) {
    if (candidate.fitness <=
        diagnostics.best_z_offset_fitness + near_best_band) {
      near_min = std::min(near_min, candidate.z_offset);
      near_max = std::max(near_max, candidate.z_offset);
    }
  }
  if (std::isfinite(near_min) && std::isfinite(near_max)) {
    diagnostics.z_hypothesis_spread_m = near_max - near_min;
    diagnostics.vertical_ambiguity_score =
        std::clamp(diagnostics.z_hypothesis_spread_m / 4.0, 0.0, 1.0);
  } else {
    diagnostics.z_hypothesis_spread_m = 0.0;
    diagnostics.vertical_ambiguity_score = 0.0;
  }

  const double zero_best_gap =
      std::isfinite(diagnostics.fitness_gap_zero_vs_best)
          ? diagnostics.fitness_gap_zero_vs_best
          : 0.0;
  if (diagnostics.vertical_ambiguity_score >= 0.25 &&
      zero_best_gap <=
          std::max(0.02, 0.10 * std::max(1e-6, diagnostics.zero_z_fitness))) {
    diagnostics.edge_recommendation = "planar_xy_yaw";
  } else {
    diagnostics.edge_recommendation = "full6dof";
  }
  return diagnostics;
}

} // namespace
