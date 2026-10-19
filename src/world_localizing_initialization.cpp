#include "n3mapping/relocalization_order.h"
#include "n3mapping/world_localizing.h"
#include "relocalization_seed_plan.h"

#include <algorithm>
#include <future>
#include <glog/logging.h>

namespace n3mapping {
RelocalizationSeedPlan
makeRelocalizationSeedPlan(const std::vector<LoopCandidate> &candidates,
                           KeyframeManager &keyframes, const Config &config) {
  RelocalizationSeedPlan plan;
  auto &basins = plan.basins;
  basins.reserve(kRelocMaxBasinCount);

  // Candidates are already sorted by coarse descriptor distance.
  // Build up to top-K basins in XY, then verify per-basin with local-submap
  // ICP.
  for (const auto &candidate : candidates) {
    auto kf = keyframes.getKeyframe(candidate.match_id);
    if (!kf)
      continue;
    const Eigen::Vector3d pos = kf->pose_optimized.translation();

    int best_basin = -1;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < basins.size(); ++i) {
      const double d = (basins[i].center_pos.head<2>() - pos.head<2>()).norm();
      if (d < best_dist) {
        best_dist = d;
        best_basin = static_cast<int>(i);
      }
    }

    if (best_basin >= 0 && best_dist <= kRelocBasinAssignRadiusXY) {
      basins[best_basin].members.push_back(candidate);
    } else if (static_cast<int>(basins.size()) < kRelocMaxBasinCount) {
      RelocalizationSeedPlan::Basin bg;
      bg.center_match_id = candidate.match_id;
      bg.center_pos = pos;
      bg.members.push_back(candidate);
      basins.push_back(std::move(bg));
    }
  }
  auto &seeds = plan.seeds;
  for (const auto &basin : basins) {
    const auto count =
        std::min<std::size_t>(basin.members.size(), kRelocPerBasinVerifyCount);
    for (std::size_t i = 0; i < count; ++i) {
      for (const auto yaw :
           buildDescriptorYawHypotheses(basin.members[i], config))
        seeds.emplace_back(basin.members[i].match_id, yaw);
    }
  }
  std::sort(seeds.begin(), seeds.end());
  return plan;
}

void WorldLocalizing::initializeRelocalization(
    RelocalizationSeedPlan plan,
    const PointCloudMatcher::PreparedSource &prepared_query,
    const PreparedVisibilityObservation &visibility_observation,
    const Eigen::Isometry3d &odom_pose, RelocalizationPerformance &cost,
    RelocalizationDebugEvent *event) {
  auto &basins = plan.basins;
  using BasinGroup = RelocalizationSeedPlan::Basin;
  std::mutex performance_mutex;
  RelocalizationDebugEvent unused_event;
  auto &debug_event = event ? *event : unused_event;
  const bool reloc_debug_enabled = event != nullptr;
  // Workers share immutable requests. Rebuilding map revisions inside an
  // asynchronous basin would race with the other basin workers.
  for (auto &basin : basins) {
    const auto count =
        std::min<std::size_t>(basin.members.size(), kRelocPerBasinVerifyCount);
    basin.targets.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      basin.targets.push_back(
          makeRelocTargetRequest(basin.members[i].match_id));
    }
  }

  struct BasinBest {
    int64_t basin_center_id = -1;
    int64_t matched_kf_id = -1;
    LoopCandidate candidate;
    MatchResult match;
    VisibilityConsistencyResult visibility;
    RegistrationObservability registration_observability;
    double selection_score = std::numeric_limits<double>::max();
  };
  std::vector<BasinBest> basin_best_results;
  basin_best_results.reserve(basins.size() * kRelocPerBasinVerifyCount);
  const auto better_basin_mode = [](const BasinBest &candidate,
                                    const BasinBest &incumbent) {
    if (candidate.visibility.valid != incumbent.visibility.valid) {
      return candidate.visibility.valid;
    }
    if (candidate.visibility.valid) {
      const int order =
          compareRelocalizationScore(candidate.visibility.consistency_ratio,
                                     incumbent.visibility.consistency_ratio);
      if (order != 0)
        return order < 0;
    }
    const int quality_order = compareRelocalizationScore(
        -candidate.selection_score, -incumbent.selection_score);
    if (quality_order != 0)
      return quality_order < 0;
    if (candidate.matched_kf_id != incumbent.matched_kf_id)
      return candidate.matched_kf_id < incumbent.matched_kf_id;
    if (candidate.basin_center_id != incumbent.basin_center_id)
      return candidate.basin_center_id < incumbent.basin_center_id;
    return relocalizationPoseLess(candidate.match.T_target_source,
                                  incumbent.match.T_target_source);
  };

  const auto evaluate_basin = [&](const BasinGroup &basin) {
    std::vector<BasinBest> basin_modes;
    for (std::size_t i = 0; i < basin.targets.size(); ++i) {
      const auto &candidate = basin.members[i];
      RelocTargetRequest target_request = basin.targets[i];
      RelocalizationPerformance candidate_cost;
      const auto evaluations = candidate_evaluator_.evaluate(
          prepared_query, candidate, std::move(target_request),
          visibility_observation, &candidate_cost);
      {
        std::lock_guard<std::mutex> cost_lock(performance_mutex);
        cost.addCandidate(candidate_cost);
      }
      for (const auto &evaluation : evaluations) {
        const auto &selected_match = evaluation.match;
        const auto quality = evaluateRelocMatchQuality(selected_match);
        if (!quality.accepted)
          continue;

        BasinBest mode;
        mode.basin_center_id = basin.center_match_id;
        mode.matched_kf_id = evaluation.matched_keyframe_id;
        mode.candidate = candidate;
        mode.match = selected_match;
        mode.visibility = evaluation.visibility;
        if (reloc_debug_enabled) {
          mode.registration_observability = analyzeRegistrationObservability(
              selected_match,
              selected_match.converged ? "registration_endpoint"
                                       : "descriptor_initial",
              true, quality.accepted);
        }
        const double descriptor_score =
            std::isfinite(candidate.fused_score) ? candidate.fused_score : 1.0;
        mode.selection_score = selected_match.fitness_score + descriptor_score;

        auto duplicate = std::find_if(
            basin_modes.begin(), basin_modes.end(),
            [&](const BasinBest &existing) {
              const Eigen::Isometry3d delta =
                  existing.match.T_target_source.inverse() *
                  mode.match.T_target_source;
              return delta.translation().norm() <
                         config_.reloc_ambiguity_min_basin_separation &&
                     Eigen::AngleAxisd(delta.rotation()).angle() <
                         config_.reloc_track_max_rotation;
            });
        if (duplicate == basin_modes.end()) {
          basin_modes.push_back(std::move(mode));
        } else if (better_basin_mode(mode, *duplicate)) {
          *duplicate = std::move(mode);
        }
      }
    }

    std::sort(basin_modes.begin(), basin_modes.end(), better_basin_mode);
    if (basin_modes.size() >
        static_cast<std::size_t>(kRelocPerBasinVerifyCount)) {
      basin_modes.resize(kRelocPerBasinVerifyCount);
    }
    return basin_modes;
  };

  // Atlas registration shares one immutable target across all independent
  // spatial basins. Evaluate those basins concurrently, then merge in the
  // original order so ranking and tie-breaking remain deterministic. The
  // default path stays sequential.
  if (localization_atlas_ && localization_atlas_->loaded() &&
      basins.size() > 1) {
    std::vector<std::future<std::vector<BasinBest>>> futures;
    futures.reserve(basins.size());
    for (const auto &basin : basins) {
      futures.push_back(
          std::async(std::launch::async, evaluate_basin, std::cref(basin)));
    }
    for (auto &future : futures) {
      auto basin_modes = future.get();
      basin_best_results.insert(basin_best_results.end(), basin_modes.begin(),
                                basin_modes.end());
    }
  } else {
    for (const auto &basin : basins) {
      auto basin_modes = evaluate_basin(basin);
      basin_best_results.insert(basin_best_results.end(), basin_modes.begin(),
                                basin_modes.end());
    }
  }

  std::sort(basin_best_results.begin(), basin_best_results.end(),
            better_basin_mode);
  if (reloc_debug_enabled) {
    debug_event.basin_best_results.clear();
    debug_event.basin_best_results.reserve(basin_best_results.size());
    for (const auto &best : basin_best_results) {
      RelocDebugBasinBestSummary summary;
      summary.basin_center_id = best.basin_center_id;
      summary.matched_kf_id = best.matched_kf_id;
      summary.candidate = best.candidate;
      summary.pose_in_map = best.match.T_target_source;
      summary.fitness_score = best.match.fitness_score;
      summary.inlier_ratio = best.match.inlier_ratio;
      summary.selection_score = best.selection_score;
      summary.log_likelihood =
          computeRelocLogLikelihood(best.candidate, best.match);
      summary.visibility_consistency_ratio =
          best.visibility.valid ? best.visibility.consistency_ratio
                                : std::numeric_limits<double>::quiet_NaN();
      summary.visibility_observed_coverage =
          best.visibility.valid ? best.visibility.observed_coverage
                                : std::numeric_limits<double>::quiet_NaN();
      summary.visibility_foreground_conflict_ratio =
          best.visibility.valid ? best.visibility.foreground_conflict_ratio
                                : std::numeric_limits<double>::quiet_NaN();
      summary.visibility_evidence_log_odds =
          best.visibility.valid ? best.visibility.evidence_log_odds
                                : std::numeric_limits<double>::quiet_NaN();
      summary.visibility_known_bins = best.visibility.known_bins;
      summary.visibility_unknown_bins = best.visibility.unknown_bins;
      summary.visibility_known_fraction = best.visibility.known_fraction;
      summary.visibility_consistent_given_known =
          best.visibility.consistent_given_known;
      summary.visibility_foreground_conflict_given_known =
          best.visibility.foreground_conflict_given_known;
      summary.visibility_evidence_log_odds_given_known =
          best.visibility.evidence_log_odds_given_known;
      summary.registration_observability = best.registration_observability;
      debug_event.basin_best_results.push_back(std::move(summary));
    }
  }

  VLOG(1) << "Relocalization basin init: coarse_candidates="
          << cost.descriptor.returned_candidates << " basins=" << basins.size()
          << " retained_pose_modes=" << basin_best_results.size();

  std::vector<RelocHypothesis> seeded_hypotheses;
  seeded_hypotheses.reserve(basin_best_results.size());
  for (const auto &bb : basin_best_results) {
    RelocHypothesis hyp;
    hyp.seed_match_id = bb.basin_center_id;
    hyp.last_match_id = bb.matched_kf_id;
    hyp.T_map_odom = bb.match.T_target_source * odom_pose.inverse();
    hyp.cumulative_log_likelihood =
        computeRelocLogLikelihood(bb.candidate, bb.match);
    hyp.num_updates = 1;
    hyp.converged_updates = 1;
    if (bb.visibility.valid) {
      hyp.visibility_consistency_sum = bb.visibility.consistency_ratio;
      hyp.visibility_evidence_sum = bb.visibility.evidence_log_odds;
      hyp.visibility_updates = 1;
    }
    hyp.last_registration_observability = bb.registration_observability;
    hyp.alive = true;
    seeded_hypotheses.push_back(hyp);
  }
  hypothesis_manager_.start(std::move(seeded_hypotheses), odom_pose);
}
} // namespace n3mapping
