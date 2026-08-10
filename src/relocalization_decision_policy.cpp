#include "n3mapping/relocalization_decision_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace n3mapping {

RelocalizationDecisionPolicy::RelocalizationDecisionPolicy(const Config &config)
    : config_(config) {}

RelocalizationRanking RelocalizationDecisionPolicy::rank(
    const std::vector<RelocalizationHypothesis> &hypotheses,
    const Eigen::Isometry3d &odom_pose) const {
  RelocalizationRanking ranking;
  std::vector<const RelocalizationHypothesis *> ranked_hypotheses;
  ranked_hypotheses.reserve(hypotheses.size());
  for (const auto &hypothesis : hypotheses) {
    if (hypothesis.alive) {
      ranked_hypotheses.push_back(&hypothesis);
    }
  }

  std::sort(ranked_hypotheses.begin(), ranked_hypotheses.end(),
            [](const RelocalizationHypothesis *lhs,
               const RelocalizationHypothesis *rhs) {
              const double lhs_visibility = meanVisibilityEvidence(lhs);
              const double rhs_visibility = meanVisibilityEvidence(rhs);
              const bool lhs_has_visibility = std::isfinite(lhs_visibility);
              const bool rhs_has_visibility = std::isfinite(rhs_visibility);
              if (lhs_has_visibility != rhs_has_visibility) {
                return lhs_has_visibility;
              }
              if (lhs_has_visibility &&
                  std::abs(lhs_visibility - rhs_visibility) > 1e-9) {
                return lhs_visibility > rhs_visibility;
              }
              return lhs->cumulative_log_likelihood >
                     rhs->cumulative_log_likelihood;
            });

  ranking.top1 =
      ranked_hypotheses.empty() ? nullptr : ranked_hypotheses.front();
  for (std::size_t i = 1; ranking.top1 && i < ranked_hypotheses.size(); ++i) {
    if (!samePhysicalPose(ranking.top1, ranked_hypotheses[i], odom_pose)) {
      ranking.top2 = ranked_hypotheses[i];
      break;
    }
  }

  ranking.top1_log_likelihood = ranking.top1
                                    ? ranking.top1->cumulative_log_likelihood
                                    : -std::numeric_limits<double>::infinity();
  ranking.top2_log_likelihood = ranking.top2
                                    ? ranking.top2->cumulative_log_likelihood
                                    : -std::numeric_limits<double>::infinity();
  ranking.top1_visibility = meanVisibilityEvidence(ranking.top1);
  ranking.top2_visibility = meanVisibilityEvidence(ranking.top2);
  ranking.top1_consistency = meanVisibilityConsistency(ranking.top1);
  ranking.top2_consistency = meanVisibilityConsistency(ranking.top2);
  ranking.uses_visibility_evidence = std::isfinite(ranking.top1_visibility);
  ranking.top1_decision_score = ranking.uses_visibility_evidence
                                    ? ranking.top1_visibility
                                    : ranking.top1_log_likelihood;
  ranking.top2_decision_score =
      ranking.top2 ? (ranking.uses_visibility_evidence &&
                              std::isfinite(ranking.top2_visibility)
                          ? ranking.top2_visibility
                          : (ranking.uses_visibility_evidence
                                 ? -std::numeric_limits<double>::infinity()
                                 : ranking.top2_log_likelihood))
                   : -std::numeric_limits<double>::infinity();
  ranking.margin =
      ranking.top2 ? ranking.top1_decision_score - ranking.top2_decision_score
                   : std::numeric_limits<double>::infinity();
  ranking.ratio =
      ranking.uses_visibility_evidence
          ? (ranking.top2 ? std::exp(std::clamp(ranking.margin, -700.0, 700.0))
                          : std::numeric_limits<double>::infinity())
          : (ranking.top2 && ranking.top2_decision_score > 1e-6
                 ? ranking.top1_decision_score / ranking.top2_decision_score
                 : std::numeric_limits<double>::infinity());

  if (ranking.top1 && ranking.top2) {
    const Eigen::Isometry3d top1_pose = ranking.top1->T_map_odom * odom_pose;
    const Eigen::Isometry3d top2_pose = ranking.top2->T_map_odom * odom_pose;
    ranking.basin_separation =
        (top1_pose.translation() - top2_pose.translation()).norm();
    ranking.basin_separated = ranking.basin_separation >=
                              config_.reloc_ambiguity_min_basin_separation;
  }
  return ranking;
}

RelocalizationDecision RelocalizationDecisionPolicy::evaluate(
    const RelocalizationRanking &ranking, int window_count, int winner_streak,
    const HypothesisMotionBaseline &motion_baseline,
    const RelocalizationHypothesis *free_space_best,
    const Eigen::Isometry3d &odom_pose) const {
  RelocalizationDecision decision;
  decision.effective_temporal_window =
      std::max(1, config_.reloc_temporal_window_size);
  decision.effective_min_winner_streak =
      std::min(std::max(1, config_.reloc_lock_min_winner_streak),
               decision.effective_temporal_window);
  decision.effective_min_converged_updates =
      std::min(std::max(1, config_.reloc_lock_min_converged_updates),
               decision.effective_temporal_window);
  decision.effective_min_margin = std::max(0.0, config_.reloc_lock_min_margin);
  decision.temporal_window_pending =
      window_count < decision.effective_temporal_window;

  decision.pass_log_likelihood =
      ranking.top1 && ranking.top1_log_likelihood >=
                          config_.reloc_lock_log_likelihood_threshold;
  decision.pass_margin =
      ranking.top1 && ranking.margin >= decision.effective_min_margin;
  decision.pass_winner_streak =
      ranking.top1 && winner_streak >= decision.effective_min_winner_streak;
  decision.pass_converged_updates =
      ranking.top1 && ranking.top1->converged_updates >=
                          decision.effective_min_converged_updates;
  decision.moving_visibility_required =
      motion_baseline.translation > config_.reloc_static_agg_max_translation ||
      motion_baseline.rotation > config_.reloc_static_agg_max_rotation;
  decision.pass_moving_visibility =
      ranking.top1 &&
      (!decision.moving_visibility_required ||
       !ranking.uses_visibility_evidence || ranking.top1_visibility >= 0.0);
  decision.top2_viable = ranking.top2 &&
                         ranking.top2_log_likelihood >=
                             config_.reloc_lock_log_likelihood_threshold &&
                         ranking.top2->converged_updates >=
                             decision.effective_min_converged_updates;

  decision.consistency_margin =
      ranking.top1_consistency - ranking.top2_consistency;
  const bool use_consistency_gate =
      config_.reloc_ambiguity_min_consistency_margin > 0.0 &&
      std::isfinite(decision.consistency_margin);
  const bool separation_too_small =
      use_consistency_gate
          ? decision.consistency_margin <
                config_.reloc_ambiguity_min_consistency_margin
          : ((ranking.margin < config_.reloc_ambiguity_min_margin) &&
             (ranking.ratio < config_.reloc_ambiguity_min_ratio));
  const bool competing_hypothesis =
      config_.reloc_ambiguity_ignore_basin_separation
          ? (ranking.top2 != nullptr)
          : ranking.basin_separated;
  decision.ambiguous =
      decision.top2_viable && competing_hypothesis && separation_too_small;

  decision.pass_free_space =
      config_.reloc_free_space_mode == "kill" || !free_space_best ||
      !ranking.top1 || free_space_best == ranking.top1 ||
      samePhysicalPose(free_space_best, ranking.top1, odom_pose);
  decision.accepted = !decision.ambiguous && decision.pass_log_likelihood &&
                      decision.pass_margin && decision.pass_winner_streak &&
                      decision.pass_converged_updates &&
                      decision.pass_moving_visibility &&
                      decision.pass_free_space;

  if (decision.ambiguous) {
    decision.reject_reason = "ambiguous";
  } else if (!ranking.top1) {
    decision.reject_reason = "no_active_hypothesis";
  } else if (!decision.pass_log_likelihood) {
    decision.reject_reason = "log_likelihood";
  } else if (!decision.pass_moving_visibility) {
    decision.reject_reason = "moving_visibility_disagreement";
  } else if (!decision.pass_margin) {
    decision.reject_reason = "margin";
  } else if (!decision.pass_winner_streak) {
    decision.reject_reason = "winner_streak";
  } else if (!decision.pass_converged_updates) {
    decision.reject_reason = "converged_updates";
  } else if (!decision.pass_free_space) {
    decision.reject_reason = "free_space_veto";
  } else if (!decision.accepted) {
    decision.reject_reason = "stability_guard";
  }
  return decision;
}

double RelocalizationDecisionPolicy::meanVisibilityEvidence(
    const RelocalizationHypothesis *hypothesis) {
  return hypothesis && hypothesis->visibility_updates > 0
             ? hypothesis->visibility_evidence_sum /
                   static_cast<double>(hypothesis->visibility_updates)
             : std::numeric_limits<double>::quiet_NaN();
}

double RelocalizationDecisionPolicy::meanVisibilityConsistency(
    const RelocalizationHypothesis *hypothesis) {
  return hypothesis && hypothesis->visibility_updates > 0
             ? hypothesis->visibility_consistency_sum /
                   static_cast<double>(hypothesis->visibility_updates)
             : std::numeric_limits<double>::quiet_NaN();
}

bool RelocalizationDecisionPolicy::samePhysicalPose(
    const RelocalizationHypothesis *lhs, const RelocalizationHypothesis *rhs,
    const Eigen::Isometry3d &odom_pose) const {
  if (!lhs || !rhs) {
    return false;
  }
  const Eigen::Isometry3d lhs_pose = lhs->T_map_odom * odom_pose;
  const Eigen::Isometry3d rhs_pose = rhs->T_map_odom * odom_pose;
  const double translation =
      (lhs_pose.translation() - rhs_pose.translation()).norm();
  const double rotation =
      Eigen::AngleAxisd(lhs_pose.rotation().transpose() * rhs_pose.rotation())
          .angle();
  return translation < config_.reloc_ambiguity_min_basin_separation &&
         rotation < config_.reloc_track_max_rotation;
}

} // namespace n3mapping
