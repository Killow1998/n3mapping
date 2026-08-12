#include "n3mapping/core/n3mapping_core.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
#include "n3mapping/loop_heightmap_diagnostics.h"
#include "n3mapping/loop_verification_pipeline.h"
#include "n3mapping/floor_attitude.h"
#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/submap_graph_factor.h"
#include "n3mapping/submap_graph_trial_runtime.h"
#include "n3mapping/static_start_guard.h"
#include "n3mapping/pcl_compat.h"
#include <pcl/common/transforms.h>

namespace n3mapping {

namespace {

double rotationAngle(const Eigen::Isometry3d &transform) {
  return std::abs(Eigen::AngleAxisd(transform.rotation()).angle());
}

std::vector<core::AnchoredDenseTrajectorySample> anchorDenseTrajectory(
    const std::vector<core::DenseTrajectoryPose> &dense_optimized,
    const std::vector<Keyframe::Ptr> &keyframes) {
  std::vector<core::AnchoredDenseTrajectorySample> samples;
  samples.reserve(dense_optimized.size());

  for (const auto &dense_pose : dense_optimized) {
    Keyframe::Ptr anchor;
    double best_time = -std::numeric_limits<double>::infinity();
    for (const auto &kf : keyframes) {
      if (kf && kf->timestamp <= dense_pose.timestamp &&
          kf->timestamp >= best_time) {
        anchor = kf;
        best_time = kf->timestamp;
      }
    }
    if (!anchor && !keyframes.empty()) {
      anchor = keyframes.front();
    }

    core::AnchoredDenseTrajectorySample sample;
    sample.seq = dense_pose.seq;
    sample.timestamp = dense_pose.timestamp;
    sample.pose_world_lidar_raw = dense_pose.pose_world_lidar;
    if (anchor) {
      sample.anchor_keyframe_id = anchor->id;
      sample.anchor_pose_world_lidar_raw = anchor->pose_optimized;
      sample.has_anchor = true;
      sample.use_bracketing_correction = false;
    }
    samples.push_back(std::move(sample));
  }
  return samples;
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

Config validateOrThrow(const Config &config) {
  std::string error;
  if (!config.validate(&error)) {
    throw std::invalid_argument("Invalid N3MappingCore config: " + error);
  }
  return config;
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

N3MappingCore::N3MappingCore(const Config &config)
    : config_(validateOrThrow(config)),
      session_(std::make_unique<core::N3MappingSession>(config_)) {}

N3MappingCore::~N3MappingCore() = default;

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

CoreRunMode parseCoreRunMode(const std::string &mode) {
  if (mode == "mapping") {
    return CoreRunMode::MAPPING;
  }
  if (mode == "localization") {
    return CoreRunMode::LOCALIZATION;
  }
  if (mode == "map_extension") {
    return CoreRunMode::MAP_EXTENSION;
  }
  throw std::invalid_argument("Invalid n3mapping mode: " + mode);
}

const char *coreRunModeName(CoreRunMode mode) {
  switch (mode) {
  case CoreRunMode::LOCALIZATION:
    return "localization";
  case CoreRunMode::MAP_EXTENSION:
    return "map_extension";
  case CoreRunMode::MAPPING:
  default:
    return "mapping";
  }
}

bool coreRunModeLoadsMap(CoreRunMode mode) {
  return mode == CoreRunMode::LOCALIZATION ||
         mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeSavesMap(CoreRunMode mode) {
  return mode == CoreRunMode::MAPPING || mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeProcessesLoopClosures(CoreRunMode mode) {
  return mode == CoreRunMode::MAPPING;
}

core::BackendOutput N3MappingCore::processFrame(CoreRunMode mode,
                                                const core::LioFrame &frame) {
  switch (mode) {
  case CoreRunMode::LOCALIZATION:
    return processLocalizationFrame(frame);
  case CoreRunMode::MAP_EXTENSION:
    return processMapExtensionFrame(frame);
  case CoreRunMode::MAPPING:
  default:
    return processMappingFrame(frame);
  }
}

core::BackendOutput
N3MappingCore::processMappingFrame(const core::LioFrame &frame) {
  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty()) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  // Once the front end has run away nothing downstream can use what it sends,
  // and every frame that keeps being added replaces a usable partial map with
  // an unusable whole one. Stop taking them.
  if (config_.odom_sanity_enable) {
    if (!odometry_sanity_configured_) {
      OdometrySanityLimits limits;
      limits.max_speed_mps = config_.odom_sanity_max_speed_mps;
      limits.max_angular_rate_dps = config_.odom_sanity_max_angular_rate_dps;
      limits.max_consecutive_violations = config_.odom_sanity_max_consecutive;
      odometry_sanity_ = OdometrySanity(limits);
      odometry_sanity_configured_ = true;
    }
    if (odometry_sanity_.check(timestamp, frame.T_world_lidar).diverged) {
      return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }
  }

  // Nothing the estimator says while the platform is still is worth building
  // on, and the scan is the only thing here that can tell -- the odometry
  // reports 9.37 m of travel across a stationary opening it should report none
  // of.
  if (config_.mapping_static_start_guard_enable) {
    if (!static_start_guard_configured_) {
      static_start_guard_ =
          StaticStartGuard(staticStartGuardOptionsFromConfig(config_));
      static_start_guard_configured_ = true;
    }
    if (!static_start_guard_.update(timestamp, *frame.undistorted_cloud)) {
      return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }
  }

  auto output = makeOutput(true, frame.T_world_lidar, frame.undistorted_cloud);
  auto &keyframes = session_->keyframeManager();
  if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
    if (!external_dense_trajectory_recording_enabled_) {
      appendDenseTrajectorySampleWithLatestAnchor(timestamp,
                                                  frame.T_world_lidar);
    }
    return output;
  }

  const int64_t keyframe_id = keyframes.addKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud);
  session_->loopDetector().addDescriptor(keyframe_id, frame.undistorted_cloud);
  addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

  if (keyframe_id == 0) {
    session_->graphOptimizer().addPriorFactor(keyframe_id, frame.T_world_lidar);
  } else {
    addOdometryConstraint(keyframe_id, frame.T_world_lidar);
  }

  // The only absolute attitude the graph ever sees. Measured from this scan's
  // own returns, so it is independent of the pose it constrains.
  if (config_.floor_attitude_enable) {
    FloorNormalOptions floor_options;
    floor_options.max_radius_m = config_.floor_attitude_max_radius_m;
    floor_options.min_points = config_.floor_attitude_min_points;
    const auto floor = estimateFloorNormal(*frame.undistorted_cloud,
                                           frame.T_world_lidar, floor_options);
    if (floor.valid) {
      session_->graphOptimizer().addFloorAttitudeFactor(keyframe_id,
                                                        floor.normal_body);
      ++floor_attitude_accepted_;
    } else {
      ++floor_attitude_rejected_;
    }
  }

  session_->graphOptimizer().incrementalOptimize();
  refreshOptimizedPoses();

  if (session_->submapBuilder().enabled()) {
    const auto committed_keyframe = keyframes.getKeyframe(keyframe_id);
    if (!session_->submapBuilder().appendKeyframe(committed_keyframe)) {
      std::cerr << "Shadow submap append failed for committed keyframe id="
                << keyframe_id << '\n';
    } else {
      // graph_update necessarily runs before the just-committed keyframe can
      // join a shadow submap. Refresh once more so topology diagnostics see
      // complete ownership rather than remaining one keyframe behind.
      refreshSubmapPoses("submap_append");
    }
  }

  Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
  if (session_->graphOptimizer().hasNode(keyframe_id)) {
    try {
      optimized_pose = session_->graphOptimizer().getOptimizedPose(keyframe_id);
    } catch (const std::exception &) {
      optimized_pose = frame.T_world_lidar;
    }
  }

  output.accepted_keyframe = true;
  output.keyframe_id = keyframe_id;
  output.T_world_lidar = optimized_pose;
  output.cloud_world = makeWorldCloud(frame.undistorted_cloud, optimized_pose);
  if (!external_dense_trajectory_recording_enabled_) {
    if (auto kf = keyframes.getKeyframe(keyframe_id)) {
      appendDenseTrajectorySample(timestamp, frame.T_world_lidar, keyframe_id,
                                  kf->pose_odom);
    }
  }

  {
    std::lock_guard<std::mutex> lock(loop_queue_mutex_);
    loop_detection_queue_.push_back(keyframe_id);
  }
  return output;
}

core::BackendOutput
N3MappingCore::processLocalizationFrame(const core::LioFrame &frame) {
  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty()) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  if (!map_loaded_) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  Eigen::Isometry3d pose_map = frame.T_world_lidar;
  bool success = false;
  bool relocalization_locked = false;
  RelocalizationState relocalization_state = RelocalizationState::SEARCHING;
  PoseSource pose_source = PoseSource::NONE;
  std::string relocalization_decision = "not_attempted";
  int64_t seed_keyframe_id = -1;
  int64_t support_keyframe_id = -1;
  int64_t matched_keyframe_id = -1;
  auto &localizer = session_->worldLocalizing();

  if (localizer.isRelocalized()) {
    auto result = localizer.trackLocalization(frame.undistorted_cloud,
                                              frame.T_world_lidar);
    relocalization_state = result.state;
    pose_source = result.pose_source;
    if (result.success) {
      pose_map = result.pose_in_map;
      success = true;
      relocalization_decision = result.decision;
      seed_keyframe_id = result.seed_keyframe_id;
      support_keyframe_id = result.support_keyframe_id;
      matched_keyframe_id = result.matched_keyframe_id;
    }
  }

  if (!localizer.isRelocalized() || !success) {
    const double query_timestamp =
        static_cast<double>(frame.stamp.nsec) * 1e-9;
    auto result = localizer.relocalize(
        frame.undistorted_cloud, frame.T_world_lidar, query_timestamp);
    relocalization_decision = result.decision;
    relocalization_state = result.state;
    pose_source = result.pose_source;
    if (result.success) {
      pose_map = result.pose_in_map;
      success = true;
      relocalization_locked = true;
      seed_keyframe_id = result.seed_keyframe_id;
      support_keyframe_id = result.support_keyframe_id;
      matched_keyframe_id = result.matched_keyframe_id;
    }
  }

  if (!success) {
    pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
  }

  auto output = makeOutput(success, pose_map, frame.undistorted_cloud);
  output.relocalization_locked = relocalization_locked;
  output.relocalization_state = relocalization_state;
  output.pose_source = pose_source;
  output.relocalization_decision = relocalization_decision;
  output.relocalization_seed_keyframe_id = seed_keyframe_id;
  output.relocalization_support_keyframe_id = support_keyframe_id;
  output.matched_keyframe_id = matched_keyframe_id;
  return output;
}

RegistrationSeedProbeResult N3MappingCore::probeLocalizationRegistration(
    const core::LioFrame::PointCloud::Ptr &cloud,
    const Eigen::Isometry3d &odom_pose, const Eigen::Isometry3d &oracle_pose) {
  if (!map_loaded_) {
    RegistrationSeedProbeResult result;
    result.error = "map_not_loaded";
    return result;
  }
  return session_->worldLocalizing().probeRegistrationSeeds(cloud, odom_pose,
                                                            oracle_pose);
}

core::BackendOutput
N3MappingCore::processMapExtensionFrame(const core::LioFrame &frame) {
  if (!frame.pose_valid || !frame.undistorted_cloud ||
      frame.undistorted_cloud->empty() || !map_loaded_) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  auto &resuming = session_->mappingResuming();
  auto &localizer = session_->worldLocalizing();
  const auto state = resuming.getState();

  if (state == MappingResumingState::MAP_LOADED) {
    const bool locked = resuming.performInitialRelocalization(
        frame.undistorted_cloud, frame.T_world_lidar,
        frame.source_frame_id);
    auto output = makeOutput(
        locked, localizer.getMapToOdomTransform() * frame.T_world_lidar,
        frame.undistorted_cloud);
    output.relocalization_locked = locked;
    output.relocalization_state =
        locked ? RelocalizationState::FULL_6DOF_LOCKED
               : RelocalizationState::SEARCHING;
    output.pose_source =
        locked ? PoseSource::GEOMETRICALLY_CORRECTED : PoseSource::NONE;
    output.relocalization_decision =
        locked ? "accepted" : "initial_relocalization_rejected";
    if (locked) {
      const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
      const int64_t matched_id = localizer.getLastMatchedKeyframeId();
      output.relocalization_seed_keyframe_id = matched_id;
      output.relocalization_support_keyframe_id = matched_id;
      output.matched_keyframe_id = matched_id;
      auto matched_kf = session_->keyframeManager().getKeyframe(matched_id);
      if (!external_dense_trajectory_recording_enabled_ && matched_kf) {
        appendDenseTrajectorySample(timestamp, output.T_world_lidar, matched_id,
                                    matched_kf->pose_optimized, false);
      }
    }
    return output;
  }

  if (state != MappingResumingState::RELOCALIZED &&
      state != MappingResumingState::EXTENDING) {
    return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
  }

  // While old-map geometry is visible, extension must track that immutable
  // geometry continuously. A constant relocalization transform only carries
  // the session LIO's time-varying drift into the resumed map. This strict
  // path never falls back to a new global search or to self-created frames.
  const RelocResult tracking = localizer.trackLoadedMap(
      frame.undistorted_cloud, frame.T_world_lidar);
  if (!tracking.success ||
      tracking.state != RelocalizationState::FULL_6DOF_LOCKED ||
      tracking.pose_source != PoseSource::GEOMETRICALLY_CORRECTED ||
      tracking.matched_keyframe_id < 0) {
    auto output = makeOutput(
        false, tracking.pose_in_map, frame.undistorted_cloud);
    output.relocalization_state = tracking.state;
    output.pose_source = tracking.pose_source;
    output.relocalization_decision = tracking.decision;
    output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
    output.relocalization_support_keyframe_id =
        tracking.support_keyframe_id;
    output.matched_keyframe_id = tracking.matched_keyframe_id;
    return output;
  }

  Eigen::Isometry3d pose_map = tracking.pose_in_map;
  if (!resuming.shouldAddKeyframe(frame.T_world_lidar)) {
    if (!external_dense_trajectory_recording_enabled_) {
      const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
      appendDenseTrajectorySampleWithLatestAnchor(
          timestamp, frame.T_world_lidar, false);
    }
    auto output = makeOutput(true, pose_map, frame.undistorted_cloud);
    output.relocalization_state = tracking.state;
    output.pose_source = tracking.pose_source;
    output.relocalization_decision = tracking.decision;
    output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
    output.relocalization_support_keyframe_id =
        tracking.support_keyframe_id;
    output.matched_keyframe_id = tracking.matched_keyframe_id;
    return output;
  }

  const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
  const int64_t keyframe_id = resuming.processNewKeyframe(
      timestamp, frame.T_world_lidar, frame.undistorted_cloud,
      tracking.matched_keyframe_id, tracking.pose_in_map);
  if (keyframe_id >= 0) {
    // The strict tracker already supplied a loaded-map constraint for this
    // keyframe (the first one is represented by SESSION_ANCHOR). Running the
    // global descriptor pipeline again would be redundant and can only add a
    // less local alias.
    refreshOptimizedPoses();
    if (session_->graphOptimizer().hasNode(keyframe_id)) {
      try {
        pose_map = session_->graphOptimizer().getOptimizedPose(keyframe_id);
      } catch (const std::exception &) {
        pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
      }
    }
  }

  auto output = makeOutput(keyframe_id >= 0, pose_map, frame.undistorted_cloud);
  output.accepted_keyframe = keyframe_id >= 0;
  output.keyframe_id = keyframe_id;
  output.relocalization_state = tracking.state;
  output.pose_source = tracking.pose_source;
  output.relocalization_decision = tracking.decision;
  output.relocalization_seed_keyframe_id = tracking.seed_keyframe_id;
  output.relocalization_support_keyframe_id = tracking.support_keyframe_id;
  output.matched_keyframe_id = tracking.matched_keyframe_id;
  if (!external_dense_trajectory_recording_enabled_ && keyframe_id >= 0) {
    if (auto kf = session_->keyframeManager().getKeyframe(keyframe_id)) {
      appendDenseTrajectorySample(timestamp, kf->pose_odom, keyframe_id,
                                  kf->pose_odom, false);
    }
  }
  return output;
}

CoreLoopClosureResult N3MappingCore::processPendingLoopClosures() {
  CoreLoopClosureResult result;
  std::vector<int64_t> keyframes_to_check;
  {
    std::lock_guard<std::mutex> lock(loop_queue_mutex_);
    keyframes_to_check.swap(loop_detection_queue_);
  }

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
    // Every verified loop, not the single best-scoring one per query. The
    // bounded window is what keeps aliases out now; ranking by fitness only
    // discarded informative loops in favour of near neighbours.
    auto best_loops =
        config_.loop_keep_all_verified
            ? valid_loops
            : session_->loopClosureManager().selectBestPerQuery(valid_loops);
    if (best_loops.empty()) {
      flush_debug_events({}, "not_selected");
      continue;
    }

    const auto poses_before = session_->graphOptimizer().getOptimizedPoses();
    std::vector<EdgeInfo> edges;
    std::vector<VerifiedLoop> gated_best_loops;
    edges.reserve(best_loops.size());
    gated_best_loops.reserve(best_loops.size());

    for (const auto &loop : best_loops) {
      auto gate = verification_pipeline.evaluateConstraint(
          loop, LoopEdgeDirection::MatchToQuery);
      const auto key = std::make_pair(loop.query_id, loop.match_id);
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
      edges.push_back(gate.edge);
      gated_best_loops.push_back(gate.loop);
    }

    best_loops.swap(gated_best_loops);
    if (edges.empty()) {
      flush_debug_events({}, "verification_pipeline_rejected");
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

bool N3MappingCore::loadMap(const std::string &map_path) {
  try {
    auto candidate = std::make_unique<core::N3MappingSession>(config_);
    std::vector<core::DenseTrajectoryPose> loaded_dense_optimized;
    core::DenseTrajectoryMetadata loaded_dense_metadata;
    if (!candidate->mapSerializer().loadMap(
            map_path, candidate->keyframeManager(), candidate->loopDetector(),
            candidate->graphOptimizer(), &loaded_dense_optimized,
            &loaded_dense_metadata, &candidate->submapBuilder())) {
      return false;
    }
    if (!candidate->mappingResuming().initializeFromLoadedMap()) {
      return false;
    }

    std::string atlas_error;
    if (!candidate->worldLocalizing().loadLocalizationAtlas(map_path,
                                                            &atlas_error)) {
      std::cerr << "Failed to load localization atlas: " << atlas_error << '\n';
      return false;
    }

    auto loaded_dense_samples = anchorDenseTrajectory(
        loaded_dense_optimized,
        candidate->keyframeManager().getAllKeyframes());

    // Every fallible map-specific step completed against the candidate. The
    // session owns all cross-referenced components, so one pointer swap is the
    // transaction commit and cannot leave mixed revisions behind.
    session_.swap(candidate);
    dense_trajectory_samples_.swap(loaded_dense_samples);
    dense_trajectory_metadata_ = std::move(loaded_dense_metadata);
    {
      std::lock_guard<std::mutex> lock(loop_queue_mutex_);
      loop_detection_queue_.clear();
    }
    last_loop_check_id_ = -1000;
    loop_count_ = 0;
    floor_attitude_accepted_ = 0;
    floor_attitude_rejected_ = 0;
    map_loaded_ = true;
    return true;
  } catch (const std::exception &error) {
    std::cerr << "Failed to prepare map transaction: " << error.what() << '\n';
    return false;
  }
}

bool N3MappingCore::saveMap(const std::string &map_path) {
  if (!refreshSubmapPoses("save_map")) {
    return false;
  }
  const auto dense_optimized_trajectory = buildDenseOptimizedTrajectory();
  core::DenseTrajectoryMetadata metadata = dense_trajectory_metadata_;
  if (!dense_optimized_trajectory.empty() &&
      (metadata.source.empty() || metadata.source == "none")) {
    metadata.source = "native";
    metadata.degraded = false;
  }
  return session_->mapSerializer().saveMap(
      map_path, session_->keyframeManager(), session_->loopDetector(),
      session_->graphOptimizer(), dense_optimized_trajectory, metadata,
      &session_->submapBuilder());
}

bool N3MappingCore::saveGlobalMap(const std::string &pcd_path) {
  return session_->mapSerializer().saveGlobalMap(
      pcd_path, session_->keyframeManager(),
      config_.save_global_map_voxel_size);
}

bool N3MappingCore::saveMapSnapshot(std::string *error) {
  if (session_->keyframeManager().size() < 1) {
    if (error)
      *error = "no_keyframes";
    return false;
  }

  const std::string map_file = config_.map_save_path + "/n3map.pbstream";
  if (!saveMap(map_file)) {
    if (error)
      *error = "save_pbstream_failed";
    return false;
  }

  if (config_.save_global_map_on_shutdown) {
    const std::string global_map_file =
        config_.map_save_path + "/global_map.pcd";
    if (!saveGlobalMap(global_map_file)) {
      if (error)
        *error = "save_global_map_failed";
      return false;
    }
  }

  return true;
}

core::LioFrame::PointCloud::Ptr N3MappingCore::buildGlobalMap() const {
  return session_->mapSerializer().buildGlobalMap(
      session_->keyframeManager(), config_.global_map_voxel_size);
}

bool N3MappingCore::mapLoaded() const { return map_loaded_; }

Keyframe::Ptr N3MappingCore::getKeyframe(int64_t id) const {
  return session_->keyframeManager().getKeyframe(id);
}

std::vector<Keyframe::Ptr> N3MappingCore::getAllKeyframes() const {
  return session_->keyframeManager().getAllKeyframes();
}

KeyframeMapRevision N3MappingCore::mapRevision() const {
  return session_->keyframeManager().revision();
}

std::map<int64_t, Eigen::Isometry3d> N3MappingCore::getOptimizedPoses() const {
  return session_->graphOptimizer().getOptimizedPoses();
}

std::vector<core::DenseTrajectoryPose>
N3MappingCore::getDenseOptimizedTrajectory() const {
  return buildDenseOptimizedTrajectory();
}

void N3MappingCore::setExternalDenseTrajectoryRecordingEnabled(bool enabled) {
  external_dense_trajectory_recording_enabled_ = enabled;
}

void N3MappingCore::recordDenseTrajectoryPose(
    CoreRunMode mode, double timestamp,
    const Eigen::Isometry3d &pose_world_lidar) {
  if (!coreRunModeSavesMap(mode) || !std::isfinite(timestamp) ||
      !isFinitePose(pose_world_lidar)) {
    return;
  }

  if (mode == CoreRunMode::MAPPING) {
    appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_world_lidar);
    return;
  }

  if (mode != CoreRunMode::MAP_EXTENSION || !map_loaded_) {
    return;
  }

  const auto state = session_->mappingResuming().getState();
  if (state != MappingResumingState::RELOCALIZED &&
      state != MappingResumingState::EXTENDING) {
    return;
  }
  if (!session_->worldLocalizing().isRelocalized()) {
    return;
  }

  const Eigen::Isometry3d pose_map =
      session_->worldLocalizing().getMapToOdomTransform() * pose_world_lidar;
  if (!isFinitePose(pose_map)) {
    return;
  }
  appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_map, false);
}

core::BackendOutput
N3MappingCore::makeOutput(bool success, const Eigen::Isometry3d &pose,
                          const PointCloud::Ptr &cloud) const {
  core::BackendOutput output;
  output.success = success;
  output.T_world_lidar = pose;
  output.cloud_body = cloud;
  output.cloud_world = makeWorldCloud(cloud, pose);
  return output;
}

N3MappingCore::PointCloud::Ptr
N3MappingCore::makeWorldCloud(const PointCloud::Ptr &cloud,
                              const Eigen::Isometry3d &pose) const {
  if (!cloud || cloud->empty()) {
    return pcl::make_shared<PointCloud>();
  }
  auto transformed = pcl::make_shared<PointCloud>();
  pcl::transformPointCloud(*cloud, *transformed, pose.matrix().cast<float>());
  return transformed;
}

void N3MappingCore::appendDenseTrajectorySample(
    double timestamp, const Eigen::Isometry3d &raw_pose,
    int64_t anchor_keyframe_id, const Eigen::Isometry3d &anchor_raw_pose,
    bool use_bracketing_correction) {
  core::AnchoredDenseTrajectorySample sample;
  sample.seq = static_cast<uint64_t>(dense_trajectory_samples_.size());
  sample.timestamp = timestamp;
  sample.pose_world_lidar_raw = raw_pose;
  sample.anchor_keyframe_id = anchor_keyframe_id;
  sample.anchor_pose_world_lidar_raw = anchor_raw_pose;
  sample.has_anchor = anchor_keyframe_id >= 0;
  sample.use_bracketing_correction = use_bracketing_correction;
  if (dense_trajectory_metadata_.source == "keyframe_fallback") {
    dense_trajectory_metadata_.source = "mixed_keyframe_fallback_and_high_rate";
    dense_trajectory_metadata_.degraded = true;
  } else if (dense_trajectory_metadata_.source.empty() ||
             dense_trajectory_metadata_.source == "none") {
    dense_trajectory_metadata_.source = "native";
    dense_trajectory_metadata_.degraded = false;
  }
  dense_trajectory_samples_.push_back(sample);
}

void N3MappingCore::appendDenseTrajectorySampleWithLatestAnchor(
    double timestamp, const Eigen::Isometry3d &raw_pose,
    bool use_bracketing_correction) {
  auto latest = session_->keyframeManager().getLatestKeyframe();
  if (!latest) {
    appendDenseTrajectorySample(timestamp, raw_pose, -1,
                                Eigen::Isometry3d::Identity(),
                                use_bracketing_correction);
    return;
  }

  const Eigen::Isometry3d anchor_raw_pose =
      latest->is_from_loaded_map ? latest->pose_optimized : latest->pose_odom;
  appendDenseTrajectorySample(timestamp, raw_pose, latest->id, anchor_raw_pose,
                              use_bracketing_correction);
}

std::vector<core::DenseTrajectoryPose>
N3MappingCore::buildDenseOptimizedTrajectory() const {
  std::vector<core::DenseTrajectoryPose> dense_optimized;
  dense_optimized.reserve(dense_trajectory_samples_.size());
  for (const auto &sample : dense_trajectory_samples_) {
    core::DenseTrajectoryPose pose;
    pose.seq = sample.seq;
    pose.timestamp = sample.timestamp;
    pose.pose_world_lidar = sample.pose_world_lidar_raw;
    if (sample.use_bracketing_correction) {
      pose.pose_world_lidar = interpolateDenseCorrection(sample.timestamp) *
                              sample.pose_world_lidar_raw;
    } else if (sample.has_anchor) {
      auto anchor =
          session_->keyframeManager().getKeyframe(sample.anchor_keyframe_id);
      if (anchor) {
        pose.pose_world_lidar = anchor->pose_optimized *
                                sample.anchor_pose_world_lidar_raw.inverse() *
                                sample.pose_world_lidar_raw;
      }
    }
    dense_optimized.push_back(pose);
  }
  return dense_optimized;
}

Eigen::Isometry3d
N3MappingCore::interpolateDenseCorrection(double timestamp) const {
  const auto keyframes = session_->keyframeManager().getAllKeyframes();
  Keyframe::Ptr before;
  Keyframe::Ptr after;

  for (const auto &kf : keyframes) {
    if (!kf) {
      continue;
    }
    if (kf->timestamp <= timestamp &&
        (!before || kf->timestamp > before->timestamp)) {
      before = kf;
    }
    if (kf->timestamp >= timestamp &&
        (!after || kf->timestamp < after->timestamp)) {
      after = kf;
    }
  }

  if (!before && !after) {
    return Eigen::Isometry3d::Identity();
  }
  if (!before) {
    before = after;
  }
  if (!after) {
    after = before;
  }

  const Eigen::Isometry3d correction_before =
      before->pose_optimized * before->pose_odom.inverse();
  const Eigen::Isometry3d correction_after =
      after->pose_optimized * after->pose_odom.inverse();
  double alpha = 0.0;
  const double dt = after->timestamp - before->timestamp;
  if (std::isfinite(dt) && dt > 1e-9) {
    alpha = std::clamp((timestamp - before->timestamp) / dt, 0.0, 1.0);
  }

  Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
  correction.translation() = (1.0 - alpha) * correction_before.translation() +
                             alpha * correction_after.translation();
  Eigen::Quaterniond qb(correction_before.rotation());
  Eigen::Quaterniond qa(correction_after.rotation());
  qb.normalize();
  qa.normalize();
  correction.linear() = qb.slerp(alpha, qa).toRotationMatrix();
  return correction;
}

void N3MappingCore::addRhpdDescriptorForKeyframe(
    int64_t keyframe_id, const PointCloud::Ptr &fallback_cloud) {
  auto kf = session_->keyframeManager().getKeyframe(keyframe_id);
  if (!kf) {
    return;
  }

  const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
  PointCloud::Ptr rhpd_cloud = fallback_cloud;
  if (submap_radius > 0) {
    rhpd_cloud = session_->keyframeManager().buildCausalSubmapInRootFrame(
        keyframe_id, submap_radius, keyframe_id);
  }

  if (rhpd_cloud && !rhpd_cloud->empty() &&
      config_.rhpd_submap_voxel_size > 1e-4) {
    PointCloud::Ptr filtered;
    if (safeVoxelGridFilter<pcl::PointXYZI>(
            rhpd_cloud, config_.rhpd_submap_voxel_size, &filtered) &&
        filtered && !filtered->empty()) {
      rhpd_cloud = filtered;
    }
  }

  kf->rhpd_descriptor =
      session_->loopDetector().addRHPD(keyframe_id, rhpd_cloud);
}

bool N3MappingCore::addOdometryConstraint(int64_t keyframe_id,
                                          const Eigen::Isometry3d &pose) {
  auto prev_kf = session_->keyframeManager().getKeyframe(keyframe_id - 1);
  if (!prev_kf) {
    return false;
  }

  EdgeInfo edge;
  edge.from_id = keyframe_id - 1;
  edge.to_id = keyframe_id;
  edge.measurement = prev_kf->pose_odom.inverse() * pose;
  edge.information = Eigen::Matrix<double, 6, 6>::Identity();
  edge.information.block<3, 3>(0, 0) *=
      1.0 / (config_.odom_noise_position * config_.odom_noise_position);
  edge.information.block<3, 3>(3, 3) *=
      1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
  edge.type = EdgeType::ODOMETRY;
  session_->graphOptimizer().addOdometryEdge(edge);
  return true;
}

void N3MappingCore::refreshOptimizedPoses(const char* context) {
  session_->keyframeManager().updateOptimizedPoses(
      session_->graphOptimizer().getOptimizedPoses());
  refreshSubmapPoses(context);
}

bool N3MappingCore::refreshSubmapPoses(const char* context) {
  auto& submaps = session_->submapBuilder();
  if (!submaps.enabled()) {
    return true;
  }
  const auto keyframes = session_->keyframeManager().getAllKeyframes();
  const auto diagnostics = submaps.refreshMapPoses(keyframes);
  if (!diagnostics.valid) {
    LOG(WARNING) << "[SubmapShadow] pose projection refresh failed context="
                 << (context ? context : "unknown")
                 << " reason=" << diagnostics.failure_reason;
    return false;
  }
  VLOG(1) << "[SubmapShadow] pose projection context="
          << (context ? context : "unknown")
          << " submaps=" << diagnostics.submap_count
          << " projected_keyframes="
          << diagnostics.projected_keyframe_count
          << " unassigned_keyframes="
          << diagnostics.unassigned_keyframe_count
          << " refreshed_submaps="
          << diagnostics.refreshed_submap_count
          << " mean_translation_residual_m="
          << diagnostics.mean_translation_residual_m
          << " max_translation_residual_m="
          << diagnostics.max_translation_residual_m
          << " mean_rotation_residual_rad="
          << diagnostics.mean_rotation_residual_rad
          << " max_rotation_residual_rad="
          << diagnostics.max_rotation_residual_rad;

  const auto graph_snapshot = buildSubmapGraphSnapshot(
      submaps.getSubmaps(), keyframes, session_->graphOptimizer().getEdges(),
      session_->graphOptimizer().floorAttitudeConstraints());
  if (!graph_snapshot.valid) {
    // SG-02 remains observational: an invalid shadow snapshot is visible but
    // cannot change the reference keyframe-graph lifecycle.
    LOG(WARNING) << "[SubmapGraphShadow] snapshot failed context="
                 << (context ? context : "unknown")
                 << " reason=" << graph_snapshot.failure_reason;
  } else {
    VLOG(1) << "[SubmapGraphShadow] snapshot context="
            << (context ? context : "unknown")
            << " nodes=" << graph_snapshot.nodes.size()
            << " source_edges=" << graph_snapshot.source_edge_count
            << " intra_edges=" << graph_snapshot.intra_submap_edge_count
            << " cross_edges=" << graph_snapshot.cross_submap_edge_count
            << " unassigned_edges="
            << graph_snapshot.unassigned_endpoint_edge_count
            << " floor_assigned="
            << graph_snapshot.assigned_floor_constraint_count
            << " floor_unassigned="
            << graph_snapshot.unassigned_floor_constraint_count
            << " max_cross_translation_residual_m="
            << graph_snapshot.max_cross_edge_translation_residual_m
            << " max_cross_rotation_residual_rad="
            << graph_snapshot.max_cross_edge_rotation_residual_rad;
    const auto topology = evaluateSubmapGraphTopology(graph_snapshot);
    if (!topology.valid) {
      LOG(WARNING) << "[SubmapGraphShadow] topology failed context="
                   << (context ? context : "unknown")
                   << " reason=" << topology.failure_reason;
    } else {
      VLOG(1) << "[SubmapGraphShadow] topology context="
              << (context ? context : "unknown")
              << " nodes=" << topology.node_count
              << " components=" << topology.component_count
              << " isolated_submaps=" << topology.isolated_submap_count
              << " cross_edges=" << topology.cross_edge_count
              << " cross_session_edges="
              << topology.cross_session_edge_count
              << " keyframe_ownership_complete="
              << topology.keyframe_ownership_complete
              << " constraint_coverage_complete="
              << topology.constraint_coverage_complete
              << " connected=" << topology.connected
              << " shadow_graph_ready=" << topology.shadow_graph_ready;
    }
    const auto factors = evaluateSubmapGraphFactors(graph_snapshot);
    if (!factors.valid) {
      LOG(WARNING) << "[SubmapGraphShadow] factor semantics failed context="
                   << (context ? context : "unknown")
                   << " reason=" << factors.failure_reason;
    } else {
      VLOG(1) << "[SubmapGraphShadow] factor semantics context="
              << (context ? context : "unknown")
              << " cross_edges=" << factors.cross_edge_count
              << " full_6d_edges=" << factors.full_6d_edge_count
              << " xy_yaw_lifted_only="
              << factors.xy_yaw_exact_lifted_only_count
              << " direct_between_residual_equivalent="
              << factors.direct_between_residual_equivalent_count
              << " information_transportable="
              << factors.direct_between_information_transportable_count
              << " information_fallback_required="
              << factors.information_fallback_required_count
              << " assigned_floor_factors="
              << factors.assigned_floor_factor_count
              << " max_residual_transport_error="
              << factors.max_residual_transport_error_norm
              << " max_mahalanobis_squared_delta="
              << factors.max_mahalanobis_squared_delta
              << " max_floor_residual_error="
              << factors.max_floor_residual_error_norm;
    }
  }

  const std::string context_name = context ? context : "unknown";
  const auto trial = runSubmapGraphTrialCheckpoint(
      graph_snapshot, config_, "core", context_name);
  if (trial.checkpoint) {
    if (!trial.persisted) {
      LOG(WARNING) << "[SubmapGraphShadow] trial evidence write failed context="
                   << context_name << " path=" << trial.output_path;
    }
    LOG(INFO) << "[SubmapGraphShadow] trial checkpoint context="
              << context_name
              << " valid=" << trial.diagnostics.valid
              << " attempted=" << trial.diagnostics.attempted
              << " solved=" << trial.diagnostics.solved
              << " nodes=" << trial.diagnostics.node_count
              << " active_factors="
              << trial.diagnostics.active_edge_factor_count
              << " initial_error="
              << trial.diagnostics.initial_nonlinear_error
              << " final_error="
              << trial.diagnostics.final_nonlinear_error
              << " max_translation_delta_m="
              << trial.diagnostics.max_translation_delta_m
              << " max_rotation_delta_rad="
              << trial.diagnostics.max_rotation_delta_rad
              << " optimized_keyframe_p95_translation_error_m="
              << trial.diagnostics.optimized_keyframe_comparison
                     .p95_translation_error_m
              << " optimized_keyframe_max_translation_error_m="
              << trial.diagnostics.optimized_keyframe_comparison
                     .max_translation_error_m
              << " optimized_keyframe_p95_rotation_error_rad="
              << trial.diagnostics.optimized_keyframe_comparison
                     .p95_rotation_error_rad
              << " persisted=" << trial.persisted
              << " failure_reason="
              << (trial.diagnostics.failure_reason.empty()
                      ? "none"
                      : trial.diagnostics.failure_reason);
  }
  return true;
}

} // namespace n3mapping
