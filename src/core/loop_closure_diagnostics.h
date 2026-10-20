// Loop residual metrics, optional vertical hypotheses and debug record fields.
#pragma once
#include <algorithm>
#include <cmath>
#include "loop_closure_processing.h"
#include "n3mapping/loop_consensus_verifier.h"
#include "n3mapping/loop_graph_trial_diagnostics.h"
namespace n3mapping::loop_diagnostics {
struct LoopResidualAxisStats {
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Vector3d rpy = Eigen::Vector3d::Zero();
};

struct LoopRefereeDebugDecision {
  std::string recommendation = "not_available";
  std::string reason = "not_available";
  std::string risk_flags = "not_available";
};

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

double rotationAngle(const Eigen::Isometry3d& transform);
Eigen::Vector3d rollPitchYaw(const Eigen::Isometry3d& transform);
std::pair<double, double> meanLoopResidual(const std::vector<EdgeInfo>& edges,
    const std::map<int64_t, Eigen::Isometry3d>& poses);
LoopResidualAxisStats meanLoopResidualAxes(const std::vector<EdgeInfo>& edges,
    const std::map<int64_t, Eigen::Isometry3d>& poses);
void accumulatePoseUpdateStats(const std::map<int64_t, Eigen::Isometry3d>& before,
    const std::map<int64_t, Eigen::Isometry3d>& after, CoreLoopClosureResult* result);
void assignGraphTrialDiagnostics(LoopDebugCandidateEvent* event,
    const LoopGraphTrialDiagnostics& diagnostics);
void assignConsensusEstimatorTrialDiagnostics(LoopDebugCandidateEvent* event,
    const LoopGraphTrialDiagnostics& diagnostics);
void assignSegmentDiagnostics(LoopDebugCandidateEvent* event, const VerifiedLoop& loop);
void assignConsensusDiagnostics(LoopDebugCandidateEvent* event, const VerifiedLoop& loop);
void assignConsensusDiagnostics(LoopDebugCandidateEvent* event, const LoopConsensusResult& consensus);
double processingTimeSeconds();
ZDistributionStats computeZStats(const core::LioFrame::PointCloud::Ptr& cloud);
ZDistributionStats computeTransformedZStats(const core::LioFrame::PointCloud::Ptr& cloud,
    const Eigen::Isometry3d& transform);
double zOverlapRatio(const ZDistributionStats& a, const ZDistributionStats& b);
double robustZOverlapRatio(const ZDistributionStats& a, const ZDistributionStats& b);
double zCentroidDelta(const ZDistributionStats& source, const ZDistributionStats& target);
VerticalHypothesisDiagnostics computeVerticalHypothesisDiagnostics(PointCloudMatcher& matcher,
    const core::LioFrame::PointCloud::Ptr& target,
    const core::LioFrame::PointCloud::Ptr& source, const MatchResult& zero_result);
} // namespace n3mapping::loop_diagnostics
