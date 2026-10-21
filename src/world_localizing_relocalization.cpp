#include "n3mapping/world_localizing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include "relocalization_seed_plan.h"
#include "relocalization_observation.h"
#include <iomanip>
#include <memory>
#include <utility>

namespace n3mapping {
namespace {

void fillQueryCloudDebugSummary(const PlaceObservation &observation,
                                RelocQueryCloudDebugSummary *summary) {
  if (!summary) {
    return;
  }
  summary->mode = observation.scale == ObservationScale::MOTION_SUBMAP
                      ? "motion_submap"
                      : "stationary";
  summary->frame_count = observation.frame_count;
  summary->motion_translation_m = observation.motion_translation;
  summary->motion_rotation_rad = observation.motion_rotation;
  summary->raw_points = observation.raw_points;
  summary->downsampled_points = observation.downsampled_points;
}
} // namespace

RelocResult WorldLocalizing::relocalize(const PointCloudT::Ptr &cloud,
                                        const Eigen::Isometry3d &odom_pose,
                                        double query_timestamp) {
  const double reloc_cpu_started = cpuTimeMs(CLOCK_PROCESS_CPUTIME_ID);
  const auto reloc_started = std::chrono::steady_clock::now();
  RelocResult result;
  result.success = false;
  auto &cost = result.performance;
  cost.available = true;
  const bool reloc_debug_enabled = debug_emitter_.enabled();
  RelocalizationDebugEvent debug_event = debug_emitter_.beginRelocalization();
  debug_event.query_timestamp = query_timestamp;
  const auto completed_result = [&]() {
    cost.total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - reloc_started).count();
    cost.process_cpu_ms = cpuTimeMs(CLOCK_PROCESS_CPUTIME_ID) - reloc_cpu_started;
    return result;
  };
  auto finish_debug = [&](const std::string &lock_result,
                          const std::string &reject_reason) {
    result.decision = reject_reason.empty() ? lock_result : reject_reason;
    const double total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - reloc_started)
            .count();
    cost.total_ms = total_ms;
    const auto cache = target_provider_->diagnostics();
    cost.cache_bytes = cache.cache_total_bytes;
    cost.cache_entries = cache.cache_entries;
    VLOG(1) << "[RelocTiming] total_ms=" << total_ms
            << " align_ms=" << cost.align_ms
            << " alignments=" << cost.alignments
            << " visibility_ms=" << cost.visibility_ms
            << " visibility_evaluations=" << cost.visibility_evaluations
            << " outcome=" << lock_result
            << " reason=" << (reject_reason.empty() ? "none" : reject_reason);
    if (!reloc_debug_enabled) {
      return;
    }
    debug_emitter_.finishRelocalization(debug_event, lock_result,
                                        reject_reason);
  };

  if (!cloud || cloud->empty()) {
    LOG(WARNING) << "Empty point cloud for relocalization.";
    finish_debug("rejected", "empty_cloud");
    return completed_result();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  result.state = has_ever_relocalized_ ? RelocalizationState::LOST
                                       : RelocalizationState::SEARCHING;

  if (keyframe_manager_.size() == 0) {
    LOG(WARNING) << "No keyframes available for relocalization.";
    finish_debug("rejected", "missing_keyframes");
    return completed_result();
  }

  const auto query_started = std::chrono::steady_clock::now();
  const PlaceObservation query_observation =
      query_builder_.buildStationary(cloud, odom_pose);
  cost.query_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - query_started).count();
  PointCloudT::Ptr query_cloud = query_observation.cloud;
  if (!query_cloud || query_cloud->empty()) {
    LOG(WARNING) << "Relocalization query cloud is empty after aggregation.";
    finish_debug("rejected", "empty_query_cloud");
    return completed_result();
  }
  if (reloc_debug_enabled) {
    const auto diagnostics_started = std::chrono::steady_clock::now();
    RelocQueryCloudDebugSummary query_cloud_debug;
    fillQueryCloudDebugSummary(query_observation, &query_cloud_debug);
    debug_event.query_cloud = query_cloud_debug;
    const PlaceObservation motion_observation =
        query_builder_.buildMotionSubmap(odom_pose);
    fillQueryCloudDebugSummary(motion_observation,
                               &debug_event.motion_query_cloud);
    auto motion_query_cloud = motion_observation.cloud;
    if (motion_query_cloud && !motion_query_cloud->empty()) {
      auto motion_candidates = place_index_.search(motion_query_cloud);
      debug_event.motion_query_cloud.candidate_count = motion_candidates.size();
      debug_event.motion_query_cloud.top_candidates =
          std::move(motion_candidates);
    }
    cost.diagnostics_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - diagnostics_started).count();
  }

  // Retrieval needs no registration representation. Prepare once on the
  // calling thread, only when candidates or hypotheses need verification.
  RelocalizationObservation prepared_observation;
  auto& visibility_observation = prepared_observation.visibility;
  const auto prepare_query = [&]() {
    prepared_observation.prepare(query_cloud, matcher_, config_, cost);
  };

  const auto candidates = place_index_.search(query_cloud, &cost.descriptor);
  // Only a current full registration can authorize publication after cheaper
  // intermediate confirmations. This is independent of historical scores.
  std::vector<const RelocHypothesis*> currently_registered;
  const bool starting = hypothesis_manager_.empty();
  if (reloc_debug_enabled) {
    debug_event.query_cloud.candidate_count = candidates.size();
    debug_event.query_cloud.top_candidates = candidates;
    debug_event.candidate_count = candidates.size();
    debug_event.top_candidates = candidates;
  }
  if (starting && candidates.empty()) {
    finish_debug("rejected", "no_candidates");
    return completed_result();
  }

  // New spatial regions enter independently. Established hypotheses keep their
  // poses and evidence; only the new regions need descriptor-seeded solves.
  updateRelocalizationCandidates(query_cloud, candidates, odom_pose,
      prepared_observation, cost, reloc_debug_enabled ? &debug_event : nullptr);
  if (hypothesis_manager_.empty()) {
    finish_debug("rejected", "no_valid_icp_hypothesis");
    return completed_result();
  }
  if (starting) {
    if (config_.reloc_temporal_window_size <= 1) {
      for (const auto& hypothesis : hypothesis_manager_.hypotheses())
        currently_registered.push_back(&hypothesis);
    }
  } else {
    currently_registered = confirmRelocalization(
        query_cloud, candidates, odom_pose, prepared_observation,
        cost, reloc_debug_enabled);
  }

  // Kill mode changes the hypothesis set before ranking. Veto mode is a final
  // acceptance condition and need not inspect geometry for an already rejected
  // or incomplete temporal window.
  if (config_.reloc_free_space_mode == "kill") {
    const auto started = std::chrono::steady_clock::now();
    killFreeSpaceDominatedHypotheses(query_cloud, odom_pose);
    cost.free_space_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  }

  if (reloc_debug_enabled) {
    debug_event.hypotheses.clear();
    debug_event.hypotheses.reserve(hypothesis_manager_.size());
    for (const auto &hyp : hypothesis_manager_.hypotheses()) {
      RelocDebugHypothesisSummary summary;
      summary.tracking_proposal = hyp.tracking_proposal;
      summary.seed_match_id = hyp.seed_match_id;
      summary.last_match_id = hyp.last_match_id;
      summary.pose_in_map = hyp.T_map_odom * odom_pose;
      summary.cumulative_log_likelihood = hyp.cumulative_log_likelihood;
      summary.num_updates = hyp.num_updates;
      summary.observations = hypothesis_manager_.windowCount() - hyp.first_observation + 1;
      summary.converged_updates = hyp.converged_updates;
      summary.visibility_updates = hyp.visibility_updates;
      summary.mean_visibility_consistency =
          hyp.visibility_updates > 0
              ? hyp.visibility_consistency_sum /
                    static_cast<double>(hyp.visibility_updates)
              : std::numeric_limits<double>::quiet_NaN();
      summary.mean_visibility_evidence =
          hyp.visibility_updates > 0
              ? hyp.visibility_evidence_sum /
                    static_cast<double>(hyp.visibility_updates)
              : std::numeric_limits<double>::quiet_NaN();
      // Initial hypotheses duplicate the basin-best registrations emitted in
      // the same event. Keep one complete copy to bound shadow logging cost.
      if (debug_event.basin_best_results.empty()) {
        summary.registration_observability =
            hyp.last_registration_observability;
      }
      summary.alive = hyp.alive;
      debug_event.hypotheses.push_back(std::move(summary));
    }
  }

  const RelocalizationRanking ranking =
      decision_policy_.rank(hypothesis_manager_.hypotheses(), odom_pose);
  const RelocHypothesis *top1 = ranking.top1;
  const RelocHypothesis *top2 = ranking.top2;
  const double top1_ll = ranking.top1_log_likelihood;
  const double top2_ll = ranking.top2_log_likelihood;
  const double top1_visibility = ranking.top1_visibility;
  const double top2_visibility = ranking.top2_visibility;
  const bool use_visibility_evidence = ranking.uses_visibility_evidence;
  const double top1_decision_score = ranking.top1_decision_score;
  const double margin = ranking.margin;
  const double ratio = ranking.ratio;

  if (top1) {
    result.state = RelocalizationState::PROVISIONAL;
    result.pose_source = PoseSource::NONE;
    if (reloc_debug_enabled) {
      debug_event.winner_seed_match_id = top1->seed_match_id;
      debug_event.winner_last_match_id = top1->last_match_id;
      if (top2) {
        debug_event.runner_up_seed_match_id = top2->seed_match_id;
        debug_event.runner_up_last_match_id = top2->last_match_id;
      }
      debug_event.temporal_hypothesis_score = top1_decision_score;
      debug_event.log_likelihood = top1_ll;
      debug_event.margin = margin;
      if (use_visibility_evidence) {
        debug_event.visibility_margin = margin;
        debug_event.visibility_ratio = ratio;
      }
    }

    // A relocalization solution is a physical pose trajectory, not the map
    // keyframe nearest to the moving sensor. Tracking nearest-keyframe ids here
    // resets the streak whenever a correct trajectory crosses a keyframe
    // Voronoi boundary. Comparing the raw map-to-odom translations is also
    // origin-dependent: a small rotation update around a large global
    // coordinate can look like a multi-meter translation. Apply both the
    // previous and current transforms to the current odom pose, then compare
    // the two physical LiDAR poses at the same instant.
    const WinnerStability winner_stability = hypothesis_manager_.observeWinner(
        top1->T_map_odom, odom_pose, config_.reloc_track_max_translation,
        config_.reloc_track_max_rotation);
    if (reloc_debug_enabled) {
      debug_event.winner_streak = winner_stability.streak;
      debug_event.winner_pose_translation_delta =
          winner_stability.translation_delta;
      debug_event.winner_pose_rotation_delta = winner_stability.rotation_delta;
    }

    VLOG(1) << "[Reloc/Stability] window=" << hypothesis_manager_.windowCount()
            << "/" << config_.reloc_temporal_window_size
            << " top1(seed=" << top1->seed_match_id
            << ",last_kf=" << top1->last_match_id << ",ll=" << top1_ll
            << ",conv_updates=" << top1->converged_updates
            << ",visibility=" << top1_visibility
            << ",updates=" << top1->num_updates << ")"
            << " top2(seed=" << (top2 ? top2->seed_match_id : -1)
            << ",last_kf=" << (top2 ? top2->last_match_id : -1)
            << ",ll=" << (top2 ? top2_ll : -1e9)
            << ",visibility=" << top2_visibility << ")"
            << " margin=" << margin << " evidence="
            << (use_visibility_evidence ? "visibility" : "legacy_loglik")
            << " winner_streak=" << winner_stability.streak
            << " winner_pose_delta=(" << winner_stability.translation_delta
            << "m," << winner_stability.rotation_delta << "rad)";
  }

  const HypothesisMotionBaseline motion_baseline =
      hypothesis_manager_.motionBaseline(top1, odom_pose);
  const int winner_observations = top1
      ? hypothesis_manager_.windowCount() - top1->first_observation + 1
      : hypothesis_manager_.windowCount();
  RelocalizationDecision decision =
      decision_policy_.evaluate(ranking, winner_observations,
                                hypothesis_manager_.winnerStreak(),
                                motion_baseline, nullptr, odom_pose);
  bool rejected_current_geometry = false;
  if (!decision.temporal_window_pending && decision.accepted &&
      std::find(currently_registered.begin(), currently_registered.end(), top1) ==
          currently_registered.end()) {
    decision.accepted = false;
    decision.reject_reason = "current_registration_unverified";
    rejected_current_geometry = true;
  }
  if (!decision.temporal_window_pending && decision.accepted) {
    const auto started = std::chrono::steady_clock::now();
    prepare_query();
    prepareVisibilityPlanarSupport(query_cloud,
        odom_pose.rotation().transpose() * Eigen::Vector3d::UnitZ(),
        &visibility_observation);
    const auto target = target_provider_->getTarget(makeRelocTargetRequest(top1->last_match_id));
    cost.directional_checked = true;
    VisibilityConsistencyResult witness;
    if (target.hasVisibility()) {
      witness = evaluatePreparedVisibilityConsistency(
          *target.visibility_target, visibility_observation,
          top1->T_map_odom * odom_pose, target.visibility_indices.get());
    }
    cost.directional_min_support = witness.planar_min_support;
    cost.directional_check_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    // A majority of support is required in EVERY planar motion direction.
    // Matching a large floor must not compensate for contradictory walls.
    if (!std::isfinite(witness.planar_min_support) || witness.planar_min_support <= 0.5) {
      decision.accepted = false;
      rejected_current_geometry = std::isfinite(witness.planar_min_support);
      decision.reject_reason = visibility_observation.planar_jacobians.empty()
          ? "planar_geometry_unobservable"
          : (std::isfinite(witness.planar_min_support)
              ? "directional_visibility_disagreement" : "directional_visibility_unavailable");
    }
    VLOG(1) << "[Reloc/Directional] min_support=" << cost.directional_min_support
            << " passed=" << decision.accepted << " ms=" << cost.directional_check_ms;
  }
  if (rejected_current_geometry) {
    // A rejected place must get fresh seed searches next time. Other spatial
    // basins remain reusable; ambiguity alone does not invalidate a proposal.
    const auto anchor = keyframe_manager_.getKeyframe(top1->seed_match_id);
    if (anchor)
      cost.invalidated_seed_poses += candidate_evaluator_.discardSeedsNear(
          anchor->pose_optimized.translation(), kRelocBasinAssignRadiusXY);
  }
  if (config_.reloc_free_space_mode != "kill" &&
      !decision.temporal_window_pending && decision.accepted) {
    const auto started = std::chrono::steady_clock::now();
    const auto *free_space_best = freeSpaceBestHypothesis(query_cloud, odom_pose);
    cost.free_space_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    decision = decision_policy_.evaluate(
        ranking, winner_observations, hypothesis_manager_.winnerStreak(),
        motion_baseline, free_space_best, odom_pose);
  }
  const int effective_temporal_window = decision.effective_temporal_window;
  if (decision.temporal_window_pending) {
    VLOG(1) << "Relocalization pending: winner observations "
            << winner_observations << "/"
            << effective_temporal_window
            << ", active hypotheses=" << hypothesis_manager_.size();
    finish_debug("pending", "temporal_window_pending");
    return completed_result();
  }

  const int effective_min_winner_streak = decision.effective_min_winner_streak;
  const int effective_min_converged_updates =
      decision.effective_min_converged_updates;
  const double effective_min_margin = decision.effective_min_margin;
  const bool pass_loglik = decision.pass_log_likelihood;
  const bool pass_margin = decision.pass_margin;
  const bool pass_winner_streak = decision.pass_winner_streak;
  const bool pass_converged_updates = decision.pass_converged_updates;
  const double evidence_motion_translation = motion_baseline.translation;
  const double evidence_motion_rotation = motion_baseline.rotation;
  // Repeated scans from one stationary viewpoint are not independent ray
  // evidence, so descriptor/registration evidence remains authoritative there.
  // Once the sensor has moved beyond the existing static-aggregation contract,
  // the new viewpoints must not cumulatively disconfirm the pose. Zero log odds
  // is the model's semantic support/opposition boundary, not a tuned score.
  VLOG(1) << "[Reloc/Baseline] window=" << hypothesis_manager_.windowCount()
          << " persisted=" << hypothesis_manager_.persistedFrames()
          << " motion_translation_m=" << evidence_motion_translation
          << " motion_rotation_deg=" << evidence_motion_rotation * 180.0 / M_PI;
  const bool moving_visibility_required = decision.moving_visibility_required;
  const bool pass_moving_visibility = decision.pass_moving_visibility;
  const bool top2_viable = decision.top2_viable;
  const double basin_separation = ranking.basin_separation;
  const bool ambiguous = decision.ambiguous;
  if (reloc_debug_enabled) {
    debug_event.temporal_hypothesis_score = top1_decision_score;
    debug_event.log_likelihood = top1_ll;
    debug_event.winner_streak = hypothesis_manager_.winnerStreak();
    debug_event.evidence_motion_translation = evidence_motion_translation;
    debug_event.evidence_motion_rotation = evidence_motion_rotation;
    debug_event.moving_visibility_required = moving_visibility_required;
    debug_event.moving_visibility_passed = pass_moving_visibility;
    debug_event.margin = margin;
    debug_event.ratio = ratio;
    debug_event.basin_separation = basin_separation;
    if (use_visibility_evidence) {
      debug_event.visibility_margin = margin;
      debug_event.visibility_ratio = ratio;
    }
  }

  // Free space never selects; it only refuses. When the hypothesis the map's
  // own free space favours is a different physical pose than the one the
  // visibility ranking put first, the two independent kinds of evidence
  // disagree and no lock is warranted. This can only remove locks, never
  // create one, which is the only form in which this evidence is safe: the
  // kill variant, which let free space pick the winner outright, locked a
  // 32.9 m alias on 11-58-53 that the visibility ranking had got right.
  const bool pass_free_space = decision.pass_free_space;
  if (!pass_free_space) {
    VLOG(1) << "[Reloc/FreeSpace] veto: free-space best disagrees with ranked "
               "top1";
  }

  if (decision.accepted) {
    const RelocHypothesis &best = *top1;
    T_map_odom_ = best.T_map_odom;
    is_relocalized_ = true;
    has_ever_relocalized_ = true;
    last_matched_id_ = best.last_match_id;
    relocalization_seed_id_ = best.seed_match_id;
    last_odom_pose_ = odom_pose;
    consecutive_track_failures_ = 0;

    result.success = true;
    result.state = RelocalizationState::FULL_6DOF_LOCKED;
    result.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    result.seed_keyframe_id = best.seed_match_id;
    result.support_keyframe_id = best.last_match_id;
    result.matched_keyframe_id = best.last_match_id;
    result.pose_in_map = T_map_odom_ * odom_pose;
    result.confidence =
        std::max(0.0, std::min(1.0, best.cumulative_log_likelihood / 10.0));
    result.fitness_score = 0.0;

    if (const auto skf = keyframe_manager_.getKeyframe(best.last_match_id)) {
      const Eigen::Quaterniond kq(skf->pose_optimized.rotation());
      const Eigen::Vector3d kt = skf->pose_optimized.translation();
      LOG(INFO) << std::fixed << std::setprecision(9)
                << "[Reloc/LockKF] id=" << skf->id
                << " stamp=" << skf->timestamp << " t=" << kt.x() << ","
                << kt.y() << "," << kt.z() << " q=" << kq.x() << "," << kq.y()
                << "," << kq.z() << "," << kq.w();
    }
    LOG(INFO) << "[Reloc/LockPose] pose="
              << (best.last_pose_is_refined ? "refined" : "predicted")
              << " prod_quality=" << best.last_production_quality
              << " rot_info_marginal_min=" << best.last_rot_info_marginal_min;
    LOG(INFO) << "[Reloc/LockTerm] iters=" << best.last_iterations << " term="
              << matchTerminationName(
                     static_cast<MatchTermination>(best.last_termination))
              << " inlier_ratio=" << best.last_inlier_ratio
              << " fitness=" << best.last_fitness;
    LOG(INFO) << "[Reloc/LockInfo] rot_info_min=" << best.last_rot_info_min
              << " trans_info_min=" << best.last_trans_info_min
              << " sigma_att_deg="
              << (best.last_rot_info_min > 0.0
                      ? 180.0 / M_PI / std::sqrt(best.last_rot_info_min)
                      : -1.0);
    LOG(INFO) << "Relocalization locked after temporal window. seed_kf="
              << result.seed_keyframe_id
              << " support_kf=" << result.support_keyframe_id
              << ", cumulative_loglik=" << best.cumulative_log_likelihood
              << ", guard(loglik=" << pass_loglik << ", margin=" << pass_margin
              << ", winner_streak=" << pass_winner_streak
              << ", converged_updates=" << pass_converged_updates
              << ", moving_visibility=" << pass_moving_visibility
              << ", ambiguous=" << ambiguous << ")"
              << ", margin_value=" << margin
              << ", winner_streak_value=" << hypothesis_manager_.winnerStreak()
              << ", converged_updates_value=" << best.converged_updates
              << ", ratio_value=" << ratio
              << ", basin_separation=" << basin_separation;
    finish_debug("accepted", "");
    releaseSearchPreparation();
  } else {
    const std::string &reject_reason = decision.reject_reason;
    if (ambiguous) {
      LOG(WARNING) << "AMBIGUOUS_REJECT relocalization lock: top1_seed="
                   << (top1 ? top1->seed_match_id : -1)
                   << " top1_kf=" << (top1 ? top1->last_match_id : -1)
                   << " top1_ll=" << top1_ll
                   << " top2_seed=" << (top2 ? top2->seed_match_id : -1)
                   << " top2_kf=" << (top2 ? top2->last_match_id : -1)
                   << " top2_ll=" << top2_ll << " margin=" << margin
                   << " ratio=" << ratio
                   << " basin_separation=" << basin_separation
                   << " top2_viable=" << top2_viable << " thresholds(margin>="
                   << config_.reloc_ambiguity_min_margin
                   << ", ratio>=" << config_.reloc_ambiguity_min_ratio
                   << ", basin_sep>="
                   << config_.reloc_ambiguity_min_basin_separation << ")";
    } else {
      LOG(WARNING)
          << "Relocalization rejected after temporal window by stability guard."
          << " top1_seed=" << (top1 ? top1->seed_match_id : -1)
          << " top1_kf=" << (top1 ? top1->last_match_id : -1)
          << " top1_ll=" << top1_ll
          << " top2_seed=" << (top2 ? top2->seed_match_id : -1)
          << " top2_ll=" << top2_ll << " margin=" << margin
          << " ratio=" << ratio << " basin_separation=" << basin_separation
          << " winner_streak=" << hypothesis_manager_.winnerStreak()
          << " top1_converged_updates=" << (top1 ? top1->converged_updates : 0)
          << " evidence_motion=(" << evidence_motion_translation << "m,"
          << evidence_motion_rotation << "rad)"
          << " moving_visibility_required=" << moving_visibility_required
          << " top1_visibility=" << top1_visibility
          << " require(loglik>=" << config_.reloc_lock_log_likelihood_threshold
          << ", margin>=" << effective_min_margin
          << ", winner_streak>=" << effective_min_winner_streak
          << ", converged_updates>=" << effective_min_converged_updates << ")"
          << " pass(loglik=" << pass_loglik << ", margin=" << pass_margin
          << ", winner_streak=" << pass_winner_streak
          << ", converged_updates=" << pass_converged_updates
          << ", moving_visibility=" << pass_moving_visibility
          << ", free_space=" << pass_free_space << ")"
          << " reason=" << reject_reason
          << " directional_support=" << cost.directional_min_support
          << " free_cells=" << free_space_grid_.freeCells();
    }
    finish_debug("rejected", reject_reason);
  }

  // The next window starts with fresh evidence for every retrieved seed.
  // Candidate poses survive separately, keyed by seed and map revision.
  const HypothesisPersistenceResult persistence =
      hypothesis_manager_.finishWindow(result.success,
                                       config_.reloc_persist_hypotheses,
                                       config_.reloc_persist_max_frames);
  if (persistence.reseeded) {
    VLOG(1) << "[Reloc/Persist] reseeding: alive=" << persistence.any_alive
            << " persisted_frames=" << persistence.persisted_frames;
  }

  return completed_result();
}

} // namespace n3mapping
