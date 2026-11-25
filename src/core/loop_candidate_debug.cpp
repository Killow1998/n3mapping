#include "loop_closure_diagnostics.h"
#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_verification_pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace n3mapping::loop_diagnostics {
namespace {

struct HeightRange {
  double low = std::numeric_limits<double>::quiet_NaN();
  double high = std::numeric_limits<double>::quiet_NaN();

  double span() const { return high - low; }

  double overlapRatio(const HeightRange& other) const {
    if (std::isnan(low) || std::isnan(other.low))
      return std::numeric_limits<double>::quiet_NaN();
    const double denominator = std::min(span(), other.span());
    if (denominator <= 1e-9) return 0.0;
    const double overlap = std::min(high, other.high) - std::max(low, other.low);
    return std::clamp(overlap / denominator, 0.0, 1.0);
  }
};

struct HeightStats {
  HeightRange range;
  HeightRange central_90_percent;
  double mean = std::numeric_limits<double>::quiet_NaN();
};

HeightStats cloudHeights(const core::LioFrame::PointCloud::Ptr& cloud,
                         const Eigen::Isometry3d* transform = nullptr) {
  if (!cloud) return {};
  std::vector<double> heights;
  heights.reserve(cloud->size());
  double sum = 0.0;
  for (const auto& point : cloud->points) {
    if (!isFinitePoint(point)) continue;
    double height = point.z;
    if (transform) {
      height = (*transform * Eigen::Vector3d(point.x, point.y, point.z)).z();
      if (!std::isfinite(height)) continue;
    }
    heights.push_back(height);
    sum += height;
  }
  if (heights.empty()) return {};
  std::sort(heights.begin(), heights.end());
  const auto quantile = [&](double q) {
    const double index = q * static_cast<double>(heights.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(index));
    const auto hi = static_cast<std::size_t>(std::ceil(index));
    if (lo == hi) return heights[lo];
    const double t = index - static_cast<double>(lo);
    return heights[lo] * (1.0 - t) + heights[hi] * t;
  };
  return {{heights.front(), heights.back()}, {quantile(0.05), quantile(0.95)},
          sum / static_cast<double>(heights.size())};
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
    if (!result.converged || !std::isfinite(result.fitness_score)) {
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

}

LoopDebugCandidateEvent makeCandidateDebugEvent(const LoopCandidate& candidate, double timestamp,
    LoopVerificationPipelineResult& result, PointCloudMatcher& matcher, bool vertical_hypotheses) {
  LoopDebugCandidateEvent event;
  event.processing_time = processingTimeSeconds();
  event.query_timestamp = timestamp;
  event.candidate = candidate;
  event.reject_reason = result.reject_reason;
  if (!result.registration_attempted) {
    event.gate_result = "rejected";
    return event;
  }
  const auto& verification = result.verification;
  const auto& match_result = verification.match_result;
  const auto& source = result.source_registration_cloud;
  const auto& target = result.target_registration_cloud;
  const auto before = cloudHeights(source);
  const auto after = cloudHeights(source, &verification.T_icp_correction_match);
  const auto reference = cloudHeights(target);
  auto& loop = result.loop;
  loop.source_z_span = before.range.span();
  loop.target_z_span = reference.range.span();
  loop.z_overlap_ratio_before = before.range.overlapRatio(reference.range);
  loop.z_overlap_ratio_after = after.range.overlapRatio(reference.range);
  loop.source_z_robust_span = before.central_90_percent.span();
  loop.target_z_robust_span = reference.central_90_percent.span();
  loop.z_robust_overlap_ratio_before = before.central_90_percent.overlapRatio(reference.central_90_percent);
  loop.z_robust_overlap_ratio_after = after.central_90_percent.overlapRatio(reference.central_90_percent);
  loop.source_target_z_centroid_delta_before = before.mean - reference.mean;
  loop.source_target_z_centroid_delta_after = after.mean - reference.mean;
  if (vertical_hypotheses &&
      match_result.converged && verification.fitness_ok &&
      verification.inlier_ok && verification.geometry_ok) {
    const auto diagnostics = computeVerticalHypothesisDiagnostics(
        matcher, target, source, match_result);
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
  assignSegmentDiagnostics(event, loop);
  assignConsensusDiagnostics(event, loop);
  event.loop_referee_recommendation = loop.loop_referee_recommendation;
  event.loop_referee_reason = loop.loop_referee_reason;
  event.loop_referee_risk_flags = loop.loop_referee_risk_flags;
  event.gate_result = loop.verified ? "accepted" : "rejected";
  return event;
}

}
