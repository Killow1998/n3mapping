#include "n3mapping/core/n3mapping_core.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/loop_verification_pipeline.h"

namespace n3mapping {
namespace {

double rotationAngle(const Eigen::Isometry3d &transform) {
  return std::abs(Eigen::AngleAxisd(transform.rotation()).angle());
}

Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d &transform) {
  return transform.rotation().eulerAngles(0, 1, 2);
}

struct LoopResidualAxisStats {
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
};

struct LoopRefereeDebugDecision {
  std::string recommendation = "not_available";
  std::string reason = "not_available";
  std::string risk_flags = "not_available";
};

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

struct ZDistributionStats {
  std::size_t count = 0;
  double min_z = std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();
  double sum_z = 0.0;
  std::vector<double> samples;

  double span() const {
    return count > 0 ? max_z - min_z : std::numeric_limits<double>::quiet_NaN();
  }

  double mean() const {
    return count > 0 ? sum_z / static_cast<double>(count)
                     : std::numeric_limits<double>::quiet_NaN();
  }

  double quantile(double q) const {
    if (samples.empty()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> values = samples;
    std::sort(values.begin(), values.end());
    const double index =
        std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(index));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(index));
    if (lo == hi) {
      return values[lo];
    }
    const double t = index - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
  }

  double robustMin() const { return quantile(0.05); }
  double robustMax() const { return quantile(0.95); }

  double robustSpan() const {
    const double lo = robustMin();
    const double hi = robustMax();
    return std::isfinite(lo) && std::isfinite(hi)
               ? hi - lo
               : std::numeric_limits<double>::quiet_NaN();
  }
};

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

struct VerticalHypothesisDiagnostics {
  int count = 0;
  double best_z_offset_m = std::numeric_limits<double>::quiet_NaN();
  double best_z_offset_fitness = std::numeric_limits<double>::quiet_NaN();
  double zero_z_fitness = std::numeric_limits<double>::quiet_NaN();
  double fitness_gap_zero_vs_best = std::numeric_limits<double>::quiet_NaN();
  double z_hypothesis_spread_m = std::numeric_limits<double>::quiet_NaN();
  double vertical_ambiguity_score = std::numeric_limits<double>::quiet_NaN();
  std::string edge_recommendation = "not_available";
};

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

void N3MappingCore::appendLoopDebugCandidate(
    const LoopDebugCandidateEvent &event) const {
  if (!config_.loop_debug_enable) {
    return;
  }
  std::lock_guard<std::mutex> lock(loop_debug_mutex_);
  LoopDebugLogger::appendCandidate(LoopDebugLogger::resolvePath(config_),
                                   event);
}

void N3MappingCore::appendLoopDebugOptimization(
    const LoopDebugOptimizationEvent &event) const {
  if (!config_.loop_debug_enable) {
    return;
  }
  std::lock_guard<std::mutex> lock(loop_debug_mutex_);
  LoopDebugLogger::appendOptimizationSummary(
      LoopDebugLogger::resolvePath(config_), event);
}

CoreLoopClosureResult N3MappingCore::processPendingLoopClosures() {
  CoreLoopClosureResult result;
  std::vector<int64_t> keyframes_to_check;
  {
    std::lock_guard<std::mutex> lock(loop_queue_mutex_);
    keyframes_to_check.swap(loop_detection_queue_);
  }
  result.queued_keyframe_count = keyframes_to_check.size();

  LoopVerificationPipeline verification_pipeline(
      config_, session_->keyframeManager(), session_->pointCloudMatcher(),
      session_->graphOptimizer(), session_->loopClosureManager());

  for (int64_t query_id : keyframes_to_check) {
    if (query_id - last_loop_check_id_ < config_.loop_kf_gap) {
      continue;
    }

    auto query_kf = session_->keyframeManager().getKeyframe(query_id);
    if (!query_kf) {
      continue;
    }

    std::map<int64_t, Keyframe::Ptr> keyframe_map;
    for (const auto &keyframe : session_->keyframeManager().getAllKeyframes()) {
      if (keyframe) {
        keyframe_map[keyframe->id] = keyframe;
      }
    }
    std::vector<LoopCandidate> candidates =
        session_->loopDetector().detectLoopCandidates(query_id, keyframe_map);
    if (config_.loop_spatial_candidates_enable) {
      auto spatial_candidates =
          session_->loopDetector().detectSpatialCandidates(query_id,
                                                           keyframe_map);
      for (const auto &spatial : spatial_candidates) {
        auto duplicate =
            std::find_if(candidates.begin(), candidates.end(),
                         [&](const LoopCandidate &existing) {
                           return existing.query_id == spatial.query_id &&
                                  existing.match_id == spatial.match_id;
                         });
        if (duplicate == candidates.end()) {
          candidates.push_back(spatial);
        } else {
          duplicate->source_flags |= spatial.source_flags;
          duplicate->spatial_score =
              std::max(duplicate->spatial_score, spatial.spatial_score);
        }
      }
    }
    result.detected_candidate_count += candidates.size();
    if (candidates.empty()) {
      continue;
    }
    last_loop_check_id_ = query_id;

    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());
    std::vector<LoopDebugCandidateEvent> debug_events;
    std::map<std::pair<int64_t, int64_t>, LoopGraphTrialDiagnostics>
        graph_trial_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopGraphTrialDiagnostics>
        consensus_estimator_trial_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopRefereeDebugDecision>
        referee_by_pair;
    std::map<std::pair<int64_t, int64_t>, LoopConsensusResult>
        consensus_by_pair;
    std::map<std::pair<int64_t, int64_t>, std::string> graph_reject_by_pair;
    const bool loop_debug_enabled = config_.loop_debug_enable;
    if (loop_debug_enabled) {
      debug_events.reserve(candidates.size());
    }

    auto flush_debug_events =
        [&](const std::set<std::pair<int64_t, int64_t>> &accepted_pairs,
            const std::string &not_selected_reason) {
          if (!loop_debug_enabled) {
            return;
          }
          for (auto &event : debug_events) {
            const std::pair<int64_t, int64_t> key(event.candidate.query_id,
                                                  event.candidate.match_id);
            auto trial_it = graph_trial_by_pair.find(key);
            if (trial_it != graph_trial_by_pair.end()) {
              assignGraphTrialDiagnostics(&event, trial_it->second);
            }
            auto consensus_trial_it =
                consensus_estimator_trial_by_pair.find(key);
            if (consensus_trial_it != consensus_estimator_trial_by_pair.end()) {
              assignConsensusEstimatorTrialDiagnostics(
                  &event, consensus_trial_it->second);
            }
            auto referee_it = referee_by_pair.find(key);
            if (referee_it != referee_by_pair.end()) {
              event.loop_referee_recommendation =
                  referee_it->second.recommendation;
              event.loop_referee_reason = referee_it->second.reason;
              event.loop_referee_risk_flags = referee_it->second.risk_flags;
            }
            auto consensus_it = consensus_by_pair.find(key);
            if (consensus_it != consensus_by_pair.end()) {
              assignConsensusDiagnostics(&event, consensus_it->second);
            }
            if (accepted_pairs.find(key) != accepted_pairs.end()) {
              event.gate_result = "accepted";
              event.reject_reason.clear();
            } else if (event.gate_result == "accepted") {
              event.gate_result = "rejected";
              auto graph_reject_it = graph_reject_by_pair.find(key);
              if (graph_reject_it != graph_reject_by_pair.end()) {
                event.reject_reason = graph_reject_it->second;
              } else {
                event.reject_reason = not_selected_reason.empty()
                                          ? "not_selected"
                                          : not_selected_reason;
              }
            }
            appendLoopDebugCandidate(event);
          }
        };

    auto make_rejected_event = [&](const LoopCandidate &candidate,
                                   const std::string &reject_reason) {
      if (!loop_debug_enabled) {
        return;
      }
      LoopDebugCandidateEvent event;
      event.processing_time = processingTimeSeconds();
      event.query_timestamp = query_kf->timestamp;
      event.candidate = candidate;
      event.gate_result = "rejected";
      event.reject_reason = reject_reason;
      debug_events.push_back(event);
    };

    for (const auto &candidate : candidates) {
      auto pipeline_result = verification_pipeline.evaluate(candidate);
      if (!pipeline_result.registration_attempted) {
        make_rejected_event(candidate, pipeline_result.reject_reason);
        continue;
      }
      const auto &source = pipeline_result.source_registration_cloud;
      const auto &target = pipeline_result.target_registration_cloud;
      ZDistributionStats source_z_before;
      ZDistributionStats target_z;
      if (loop_debug_enabled) {
        source_z_before = computeZStats(source);
        target_z = computeZStats(target);
      }

      LoopVerification verification = pipeline_result.verification;
      MatchResult match_result = verification.match_result;
      ZDistributionStats source_z_after;
      if (loop_debug_enabled) {
        source_z_after = computeTransformedZStats(
            source, verification.T_icp_correction_match);
      }

      VerifiedLoop loop = pipeline_result.loop;
      loop.source_z_span = source_z_before.span();
      loop.target_z_span = target_z.span();
      loop.z_overlap_ratio_before = zOverlapRatio(source_z_before, target_z);
      loop.z_overlap_ratio_after = zOverlapRatio(source_z_after, target_z);
      loop.source_z_robust_span = source_z_before.robustSpan();
      loop.target_z_robust_span = target_z.robustSpan();
      loop.z_robust_overlap_ratio_before =
          robustZOverlapRatio(source_z_before, target_z);
      loop.z_robust_overlap_ratio_after =
          robustZOverlapRatio(source_z_after, target_z);
      loop.source_target_z_centroid_delta_before =
          zCentroidDelta(source_z_before, target_z);
      loop.source_target_z_centroid_delta_after =
          zCentroidDelta(source_z_after, target_z);
      if (loop_debug_enabled && config_.loop_debug_vertical_hypotheses_enable &&
          match_result.converged && verification.fitness_ok &&
          verification.inlier_ok && verification.geometry_ok) {
        const auto diagnostics = computeVerticalHypothesisDiagnostics(
            session_->pointCloudMatcher(), target, source, match_result);
        loop.vertical_hypothesis_count = diagnostics.count;
        loop.best_z_offset_m = diagnostics.best_z_offset_m;
        loop.best_z_offset_fitness = diagnostics.best_z_offset_fitness;
        loop.zero_z_fitness = diagnostics.zero_z_fitness;
        loop.fitness_gap_zero_vs_best = diagnostics.fitness_gap_zero_vs_best;
        loop.z_hypothesis_spread_m = diagnostics.z_hypothesis_spread_m;
        loop.vertical_ambiguity_score = diagnostics.vertical_ambiguity_score;
        loop.vertical_hypothesis_edge_recommendation =
            diagnostics.edge_recommendation;
      }
      const std::string reject_reason = pipeline_result.reject_reason;
      if (loop_debug_enabled) {
        LoopDebugCandidateEvent event;
        event.processing_time = processingTimeSeconds();
        event.query_timestamp = query_kf->timestamp;
        event.candidate = candidate;
        event.icp_converged = match_result.converged;
        event.icp_iterations = match_result.iterations;
        event.icp_optimizer_error = match_result.optimizer_error;
        event.icp_termination = matchTerminationName(match_result.termination);
        event.fitness_score = match_result.fitness_score;
        event.inlier_ratio = match_result.inlier_ratio;
        event.icp_translation_norm = verification.icp_translation_norm;
        event.icp_rotation_norm = verification.icp_rotation_norm;
        event.residual = verification.T_measurement_residual;
        event.T_pred_match_query = verification.T_pred_match_query;
        event.T_icp_correction_match = verification.T_icp_correction_match;
        event.T_measured_match_query = verification.T_measured_match_query;
        event.has_loop_measurement = true;
        event.loop_measurement_match_query =
            verification.T_measured_match_query;
        event.loop_information = loop.information;
        event.edge_mode =
            loop.verified ? loopEdgeModeName(loop.edge_mode) : "not_applicable";
        event.vertical_observability_score =
            loop.verified ? loop.vertical_observability_score
                          : std::numeric_limits<double>::quiet_NaN();
        event.vertical_downweighted = loop.vertical_downweighted;
        event.source_z_span = loop.source_z_span;
        event.target_z_span = loop.target_z_span;
        event.z_overlap_ratio_before = loop.z_overlap_ratio_before;
        event.z_overlap_ratio_after = loop.z_overlap_ratio_after;
        event.source_z_robust_span = loop.source_z_robust_span;
        event.target_z_robust_span = loop.target_z_robust_span;
        event.z_robust_overlap_ratio_before =
            loop.z_robust_overlap_ratio_before;
        event.z_robust_overlap_ratio_after = loop.z_robust_overlap_ratio_after;
        event.source_target_z_centroid_delta_before =
            loop.source_target_z_centroid_delta_before;
        event.source_target_z_centroid_delta_after =
            loop.source_target_z_centroid_delta_after;
        event.vertical_information_ratio = loop.vertical_information_ratio;
        event.vertical_hypothesis_count = loop.vertical_hypothesis_count;
        event.best_z_offset_m = loop.best_z_offset_m;
        event.best_z_offset_fitness = loop.best_z_offset_fitness;
        event.zero_z_fitness = loop.zero_z_fitness;
        event.fitness_gap_zero_vs_best = loop.fitness_gap_zero_vs_best;
        event.z_hypothesis_spread_m = loop.z_hypothesis_spread_m;
        event.vertical_ambiguity_score = loop.vertical_ambiguity_score;
        event.vertical_hypothesis_edge_recommendation =
            loop.vertical_hypothesis_edge_recommendation;
        event.heightmap_overlap_cell_count = loop.heightmap_overlap_cell_count;
        event.heightmap_overlap_ratio = loop.heightmap_overlap_ratio;
        event.heightmap_ground_dz_median = loop.heightmap_ground_dz_median;
        event.heightmap_ground_dz_p90 = loop.heightmap_ground_dz_p90;
        event.heightmap_ground_dz_max = loop.heightmap_ground_dz_max;
        event.heightmap_ground_support_ratio =
            loop.heightmap_ground_support_ratio;
        event.heightmap_vertical_consistency_score =
            loop.heightmap_vertical_consistency_score;
        assignSegmentDiagnostics(&event, loop);
        assignConsensusDiagnostics(&event, loop);
        event.loop_referee_recommendation = loop.loop_referee_recommendation;
        event.loop_referee_reason = loop.loop_referee_reason;
        event.loop_referee_risk_flags = loop.loop_referee_risk_flags;
        event.gate_result = loop.verified ? "accepted" : "rejected";
        event.reject_reason = reject_reason;
        debug_events.push_back(event);
      }
      verified_loops.push_back(loop);
    }

    if (verified_loops.empty()) {
      flush_debug_events({}, "not_selected");
      continue;
    }

    auto valid_loops =
        session_->loopClosureManager().filterValidLoops(verified_loops);
    // Run every individual constraint gate before same-query consensus so a
    // candidate that is already rejected cannot manufacture or break a
    // majority. The legacy single-best mode retains its historical pre-gate
    // selection order.
    auto candidate_loops =
        config_.loop_keep_all_verified
            ? valid_loops
            : session_->loopClosureManager().selectBestPerQuery(valid_loops);
    if (candidate_loops.empty()) {
      flush_debug_events({}, "not_selected");
      continue;
    }

    const auto poses_before = session_->graphOptimizer().getOptimizedPoses();
    std::map<std::pair<int64_t, int64_t>, EdgeInfo> edge_by_pair;
    std::vector<VerifiedLoop> gated_loops;
    gated_loops.reserve(candidate_loops.size());

    for (const auto &gate : verification_pipeline.evaluateConstraints(
             candidate_loops, LoopEdgeDirection::MatchToQuery)) {
      const auto key = std::make_pair(gate.loop.query_id, gate.loop.match_id);
      consensus_by_pair[key] = gate.consensus;
      if (gate.has_edge) {
        graph_trial_by_pair[key] = gate.graph_trial;
      }
      if (gate.has_consensus_estimator_trial) {
        consensus_estimator_trial_by_pair[key] =
            gate.consensus_estimator_trial;
      }
      referee_by_pair[key] = {gate.loop.loop_referee_recommendation,
                              gate.loop.loop_referee_reason,
                              gate.loop.loop_referee_risk_flags};
      if (!gate.accepted || !gate.has_edge) {
        graph_reject_by_pair[key] = gate.reject_reason.empty()
                                        ? "verification_pipeline_rejected"
                                        : gate.reject_reason;
        continue;
      }
      edge_by_pair[key] = gate.edge;
      gated_loops.push_back(gate.loop);
    }

    if (gated_loops.empty()) {
      flush_debug_events({}, "verification_pipeline_rejected");
      continue;
    }

    // Preserve all measurements in the strict-majority same-query consensus
    // set. This retains redundant constraints without committing an isolated
    // corridor alias beside several agreeing measurements. Ambiguous sets
    // fall back to the historical best-per-query policy.
    std::vector<VerifiedLoop> best_loops;
    if (config_.loop_keep_all_verified) {
      auto selection = session_->loopClosureManager().selectSameQueryConsensus(
          gated_loops, keyframe_map);
      best_loops = std::move(selection.selected);
      graph_reject_by_pair.insert(selection.rejected.begin(),
                                  selection.rejected.end());
    } else {
      best_loops = std::move(gated_loops);
    }

    std::vector<EdgeInfo> edges;
    std::vector<VerifiedLoop> loops_with_edges;
    edges.reserve(best_loops.size());
    loops_with_edges.reserve(best_loops.size());
    for (const auto &loop : best_loops) {
      const auto key = std::make_pair(loop.query_id, loop.match_id);
      const auto edge_it = edge_by_pair.find(key);
      if (edge_it == edge_by_pair.end()) {
        graph_reject_by_pair[key] = "same_query_consensus_missing_edge";
        continue;
      }
      edges.push_back(edge_it->second);
      loops_with_edges.push_back(loop);
    }
    best_loops.swap(loops_with_edges);
    if (edges.empty()) {
      flush_debug_events({}, "same_query_consensus_rejected");
      continue;
    }
    result.place_candidate_count += best_loops.size();
    const auto residual_before = meanLoopResidual(edges, poses_before);
    const auto residual_axes_before = meanLoopResidualAxes(edges, poses_before);
    const bool optimization_committed =
        session_->loopClosureManager().applyEdges(edges,
                                                  session_->graphOptimizer());
    if (!optimization_committed) {
      flush_debug_events({}, "optimization_failed");
      continue;
    }

    std::set<std::pair<int64_t, int64_t>> accepted_debug_pairs;
    for (const auto &loop : best_loops) {
      accepted_debug_pairs.insert({loop.query_id, loop.match_id});
    }
    flush_debug_events(accepted_debug_pairs, "not_selected");

    loop_count_ += edges.size();
    refreshOptimizedPoses("loop_commit");
    const auto poses_after = session_->graphOptimizer().getOptimizedPoses();
    const auto residual_after = meanLoopResidual(edges, poses_after);
    const auto residual_axes_after = meanLoopResidualAxes(edges, poses_after);

    result.optimized = true;
    result.edge_count += edges.size();
    result.graph_edge_count += edges.size();
    result.loop_residual_translation_before = residual_before.first;
    result.loop_residual_rotation_before = residual_before.second;
    result.loop_residual_translation_after = residual_after.first;
    result.loop_residual_rotation_after = residual_after.second;
    accumulatePoseUpdateStats(poses_before, poses_after, &result);
    result.accepted_loops.insert(result.accepted_loops.end(),
                                 best_loops.begin(), best_loops.end());

    if (loop_debug_enabled) {
      LoopDebugOptimizationEvent event;
      event.processing_time = processingTimeSeconds();
      event.accepted_edge_count = edges.size();
      event.accepted_edges.reserve(edges.size());
      for (const auto &edge : edges) {
        event.accepted_edges.emplace_back(edge.from_id, edge.to_id);
      }
      event.loop_residual_translation_before = residual_before.first;
      event.loop_residual_rotation_before = residual_before.second;
      event.loop_residual_translation_after = residual_after.first;
      event.loop_residual_rotation_after = residual_after.second;
      event.loop_residual_translation_axes_before =
          residual_axes_before.translation;
      event.loop_residual_translation_axes_after =
          residual_axes_after.translation;
      event.loop_residual_rpy_axes_before = residual_axes_before.rpy;
      event.loop_residual_rpy_axes_after = residual_axes_after.rpy;
      event.mean_pose_update_translation = result.mean_pose_update_translation;
      event.max_pose_update_translation = result.max_pose_update_translation;
      event.mean_pose_update_rotation = result.mean_pose_update_rotation;
      event.max_pose_update_rotation = result.max_pose_update_rotation;
      appendLoopDebugOptimization(event);
    }
  }

  return result;
}

} // namespace n3mapping
