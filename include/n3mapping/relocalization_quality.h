#pragma once

#include <algorithm>
#include <cmath>

#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_state.h"

namespace n3mapping {

struct RelocMatchQuality {
  bool fitness_pass = false;
  bool inlier_pass = false;
  double confidence = 0.0;
  bool confidence_pass = false;
  bool accepted = false;
};

inline bool hasValidRegistrationResult(const MatchResult& match) {
  return (match.converged || match.termination == MatchTermination::FixedPoseEvaluation) &&
         std::isfinite(match.fitness_score) && std::isfinite(match.inlier_ratio) &&
         isFiniteRigidPose(match.T_target_source);
}

inline RelocMatchQuality evaluateRelocMatchQuality(const MatchResult& match,
                                                 const Config& config) {
  RelocMatchQuality quality;
  const bool finite = std::isfinite(match.fitness_score) && std::isfinite(match.inlier_ratio);
  quality.fitness_pass = finite && match.fitness_score < config.gicp_fitness_threshold;
  quality.inlier_pass = finite && match.inlier_ratio >= config.reloc_min_inlier_ratio;
  const double scale = std::max(1e-6, config.gicp_fitness_threshold * 0.5);
  quality.confidence = std::clamp(std::exp(-match.fitness_score / scale), 0.0, 1.0);
  quality.confidence_pass = quality.confidence >= config.reloc_min_confidence;
  quality.accepted = hasValidRegistrationResult(match) && quality.fitness_pass &&
                     quality.inlier_pass && quality.confidence_pass;
  return quality;
}

} // namespace n3mapping
