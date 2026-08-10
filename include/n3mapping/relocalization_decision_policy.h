// Pure ranking and lock-gate evaluation for relocalization hypotheses.
#pragma once

#include <limits>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/relocalization_hypothesis_manager.h"

namespace n3mapping {

struct RelocalizationRanking {
  const RelocalizationHypothesis *top1 = nullptr;
  const RelocalizationHypothesis *top2 = nullptr;
  double top1_log_likelihood = -std::numeric_limits<double>::infinity();
  double top2_log_likelihood = -std::numeric_limits<double>::infinity();
  double top1_visibility = std::numeric_limits<double>::quiet_NaN();
  double top2_visibility = std::numeric_limits<double>::quiet_NaN();
  double top1_consistency = std::numeric_limits<double>::quiet_NaN();
  double top2_consistency = std::numeric_limits<double>::quiet_NaN();
  bool uses_visibility_evidence = false;
  double top1_decision_score = -std::numeric_limits<double>::infinity();
  double top2_decision_score = -std::numeric_limits<double>::infinity();
  double margin = std::numeric_limits<double>::infinity();
  double ratio = std::numeric_limits<double>::infinity();
  double basin_separation = 0.0;
  bool basin_separated = false;
};

struct RelocalizationDecision {
  int effective_temporal_window = 1;
  int effective_min_winner_streak = 1;
  int effective_min_converged_updates = 1;
  double effective_min_margin = 0.0;
  bool temporal_window_pending = false;
  bool pass_log_likelihood = false;
  bool pass_margin = false;
  bool pass_winner_streak = false;
  bool pass_converged_updates = false;
  bool moving_visibility_required = false;
  bool pass_moving_visibility = false;
  bool top2_viable = false;
  double consistency_margin = std::numeric_limits<double>::quiet_NaN();
  bool ambiguous = false;
  bool pass_free_space = false;
  bool accepted = false;
  std::string reject_reason;
};

class RelocalizationDecisionPolicy {
public:
  explicit RelocalizationDecisionPolicy(const Config &config);

  RelocalizationRanking
  rank(const std::vector<RelocalizationHypothesis> &hypotheses,
       const Eigen::Isometry3d &odom_pose) const;

  RelocalizationDecision
  evaluate(const RelocalizationRanking &ranking, int window_count,
           int winner_streak, const HypothesisMotionBaseline &motion_baseline,
           const RelocalizationHypothesis *free_space_best,
           const Eigen::Isometry3d &odom_pose) const;

private:
  static double
  meanVisibilityEvidence(const RelocalizationHypothesis *hypothesis);
  static double
  meanVisibilityConsistency(const RelocalizationHypothesis *hypothesis);
  bool samePhysicalPose(const RelocalizationHypothesis *lhs,
                        const RelocalizationHypothesis *rhs,
                        const Eigen::Isometry3d &odom_pose) const;

  Config config_;
};

} // namespace n3mapping
