#include "n3mapping/relocalization_debug_logger.h"
#include "n3mapping/world_localizing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <Eigen/Eigenvalues>
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
  const auto prepare_query = [&]() {
    prepared_observation.prepare(query_cloud, matcher_, config_, cost);
  };

  // A new viewpoint must reconsider the best-ranked places before continuing.
  if (search_offset_ > 0) {
    const Eigen::Isometry3d motion = search_anchor_.inverse() * odom_pose;
    if (motion.translation().norm() > config_.reloc_static_agg_max_translation ||
        Eigen::AngleAxisd(motion.rotation()).angle() > config_.reloc_static_agg_max_rotation)
      search_offset_ = 0;
  }
  if (search_offset_ == 0) search_anchor_ = odom_pose;
  VLOG(1) << "[Reloc/Retrieval] offset=" << search_offset_;
  const auto candidates = place_index_.search(query_cloud, &cost.descriptor, search_offset_);
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
  const auto next_search_offset = candidates.size() == static_cast<size_t>(config_.reloc_num_candidates)
      ? search_offset_ + candidates.size() : 0;
  if (candidates.empty()) search_offset_ = 0;
  if (starting && candidates.empty()) {
    finish_debug("rejected", "no_candidates");
    return completed_result();
  }

  // New spatial regions enter independently. Established hypotheses keep their
  // poses and evidence; only the new regions need descriptor-seeded solves.
  updateRelocalizationCandidates(query_cloud, candidates, odom_pose,
      prepared_observation, cost, reloc_debug_enabled ? &debug_event : nullptr);
  if (hypothesis_manager_.empty()) {
    if (!has_ever_relocalized_) search_offset_ = next_search_offset;
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
    RelocalizationDebugEmitter::captureHypotheses(
        debug_event, hypothesis_manager_, odom_pose);
  }

  const RelocalizationRanking ranking =
      decision_policy_.rank(hypothesis_manager_.hypotheses(), odom_pose);
  const RelocHypothesis *top1 = ranking.top1;
  const RelocHypothesis *top2 = ranking.top2;

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
      debug_event.temporal_hypothesis_score = ranking.top1_decision_score;
      debug_event.log_likelihood = ranking.top1_log_likelihood;
      debug_event.margin = ranking.margin;
      if (ranking.uses_visibility_evidence) {
        debug_event.visibility_margin = ranking.margin;
        debug_event.visibility_ratio = ranking.ratio;
      }
    }

    // Compare physical LiDAR poses at the same instant to avoid keyframe
    // boundary changes and origin-dependent transform translation deltas.
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
            << ",last_kf=" << top1->last_match_id << ",ll=" << ranking.top1_log_likelihood
            << ",conv_updates=" << top1->converged_updates
            << ",visibility=" << ranking.top1_visibility
            << ",updates=" << top1->num_updates << ")"
            << " top2(seed=" << (top2 ? top2->seed_match_id : -1)
            << ",last_kf=" << (top2 ? top2->last_match_id : -1)
            << ",ll=" << (top2 ? ranking.top2_log_likelihood : -1e9)
            << ",visibility=" << ranking.top2_visibility << ")"
            << " margin=" << ranking.margin << " evidence="
            << (ranking.uses_visibility_evidence ? "visibility" : "legacy_loglik")
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
    const auto target = target_provider_->getTarget(makeRelocTargetRequest(top1->last_match_id));
    auto information = top1->last_registration_information;
    if (top1->metric.type == small_gicp::RegistrationSetting::GICP) {
      // GICP regularizes tangential covariance, which can make a bare floor
      // appear full rank. Test its surface constraints at the selected pose.
      const auto fixed_started = std::chrono::steady_clock::now();
      const MatchMetric metric{small_gicp::RegistrationSetting::PLANE_ICP,
          config_.gicp_downsampling_resolution, top1->metric.max_correspondence_distance};
      information = matcher_.evaluatePreparedPose(*target.registration_target,
          prepared_observation.registration, top1->T_map_odom * odom_pose, metric).information;
      cost.fixed_pose_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - fixed_started).count();
      ++cost.fixed_pose_evaluations;
    }
    // Information uses source-frame increments. Query structure alone cannot
    // establish a lock when the matched map leaves XY or yaw unconstrained.
    const Eigen::Vector3d up = (top1->T_map_odom * odom_pose).linear().transpose() *
                              Eigen::Vector3d::UnitZ();
    const Eigen::Vector3d horizontal = up.unitOrthogonal();
    Eigen::Matrix<double, 6, 3> basis = Eigen::Matrix<double, 6, 3>::Zero();
    basis.block<3, 1>(0, 0) = horizontal;
    basis.block<3, 1>(0, 1) = up.cross(horizontal);
    basis.block<3, 1>(3, 2) = up;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> spectrum(
        basis.transpose() * information * basis);
    const double numerical_zero = 6.0 * std::numeric_limits<double>::epsilon() *
                                  information.cwiseAbs().maxCoeff();
    if (spectrum.info() != Eigen::Success || spectrum.eigenvalues()[0] <= numerical_zero) {
      decision.accepted = false;
      decision.reject_reason = "matched_planar_geometry_unobservable";
      rejected_current_geometry = true;
    }
    cost.directional_checked = true;
    if (decision.accepted) {
      auto directional_observation = prepareVisibilityObservation(*query_cloud,
          visibilityOptionsFromConfig(config_),
          std::max(config_.global_map_voxel_size, config_.gicp_downsampling_resolution));
      prepareVisibilityPlanarSupport(query_cloud,
          odom_pose.rotation().transpose() * Eigen::Vector3d::UnitZ(),
          &directional_observation);
      VisibilityConsistencyResult witness;
      if (target.hasVisibility()) {
        witness = evaluatePreparedVisibilityConsistency(
            *target.visibility_target, directional_observation,
            top1->T_map_odom * odom_pose, target.visibility_indices.get());
      }
      bool directional_supported = witness.planar_min_support > 0.5;
      bool all_observations_conflict = witness.planar_max_conflict >= 0.5;
      if (std::isfinite(witness.planar_min_support) && witness.planar_min_support <= 0.5) {
        // A fused map can retain foreground from a different visit. Each stored
        // observation must independently support the candidate; views are not pooled.
        const int64_t references[] = {top1->seed_match_id, top1->last_match_id};
        const int reference_count = references[0] == references[1] ? 1 : 2;
        for (int i = 0; i < reference_count; ++i) {
          const auto reference = keyframe_manager_.getKeyframe(references[i]);
          if (!reference || !reference->cloud) continue;
          const auto observed_witness = evaluatePreparedVisibilityConsistency(
              *reference->cloud, directional_observation,
              reference->pose_optimized.inverse() * top1->T_map_odom * odom_pose);
          VLOG(1) << "[Reloc/DirectionalReference] keyframe=" << reference->id
                  << " support=" << observed_witness.planar_min_support
                  << " known_min=" << observed_witness.planar_min_known
                  << " agreement_given_known=" << observed_witness.planar_min_support_given_known
                  << " conflict_max=" << observed_witness.planar_max_conflict;
          all_observations_conflict = all_observations_conflict &&
              observed_witness.planar_max_conflict >= 0.5;
          if (observed_witness.planar_min_support > 0.5) {
            witness = observed_witness;
            directional_supported = true;
            break;
          }
          if (observed_witness.planar_min_support_given_known > 0.5) {
            // A changed foreground may dominate query geometry. Require the
            // stored observation to be explained independently in every axis.
            auto reciprocal = prepareVisibilityObservation(*reference->cloud,
                visibilityOptionsFromConfig(config_),
                std::max(config_.global_map_voxel_size, config_.gicp_downsampling_resolution));
            prepareVisibilityPlanarSupport(reference->cloud,
                reference->pose_optimized.rotation().transpose() *
                    top1->T_map_odom.rotation() * Eigen::Vector3d::UnitZ(),
                &reciprocal);
            const auto reverse = evaluatePreparedVisibilityConsistency(
                *query_cloud, reciprocal,
                (top1->T_map_odom * odom_pose).inverse() * reference->pose_optimized);
            VLOG(1) << "[Reloc/DirectionalReciprocal] keyframe=" << reference->id
                    << " forward_conditional=" << observed_witness.planar_min_support_given_known
                    << " reverse_support=" << reverse.planar_min_support;
            if (reverse.planar_min_support > 0.5) {
              witness = observed_witness;
              directional_supported = true;
              break;
            }
          }
        }
      }
      cost.directional_min_support = witness.planar_min_support;
      VLOG(1) << "[Reloc/DirectionalCoverage] known_min=" << witness.planar_min_known
              << " agreement_given_known=" << witness.planar_min_support_given_known
              << " conflict_max=" << witness.planar_max_conflict
              << " all_observations_conflict=" << all_observations_conflict;
      // A majority of support is required in EVERY planar motion direction.
      // Matching a large floor must not compensate for contradictory walls.
      if (!directional_supported) {
        decision.accepted = false;
        // H_support + H_conflict + H_unknown = I in the whitened query basis.
        // Only a majority of actual contradiction proves that even explaining
        // every unknown bearing cannot pass the unchanged support gate.
        rejected_current_geometry = all_observations_conflict;
        decision.reject_reason = directional_observation.planar_jacobians.empty()
            ? "planar_geometry_unobservable"
            : (std::isfinite(witness.planar_min_support)
                ? (rejected_current_geometry ? "directional_visibility_disagreement"
                                             : "directional_visibility_insufficient_coverage")
                : "directional_visibility_unavailable");
      }
    }
    cost.directional_check_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    VLOG(1) << "[Reloc/Directional] min_support=" << cost.directional_min_support
            << " passed=" << decision.accepted << " ms=" << cost.directional_check_ms;
  }
  if (rejected_current_geometry) {
    // A rejected place must get fresh seed searches next time. Other spatial
    // basins remain reusable; ambiguity alone does not invalidate a proposal.
    const auto anchor = keyframe_manager_.getKeyframe(top1->seed_match_id);
    if (anchor)
      cost.invalidated_seed_poses += candidate_evaluator_.discardSeedsNear(
          anchor->pose_optimized.translation(), kRelocBasinAssignRadius);
  }
  if (config_.reloc_free_space_mode != "kill" &&
      !decision.temporal_window_pending && decision.accepted) {
    const auto started = std::chrono::steady_clock::now();
    const auto *free_space_best = freeSpaceBestHypothesis(query_cloud, odom_pose, *ranking.top1);
    cost.free_space_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    decision = decision_policy_.evaluate(
        ranking, winner_observations, hypothesis_manager_.winnerStreak(),
        motion_baseline, free_space_best, odom_pose);
  }
  if (decision.temporal_window_pending) {
    VLOG(1) << "Relocalization pending: winner observations "
            << winner_observations << "/"
            << decision.effective_temporal_window
            << ", active hypotheses=" << hypothesis_manager_.size();
    finish_debug("pending", "temporal_window_pending");
    return completed_result();
  }

  // Stationary repeats do not add independent visibility evidence.
  VLOG(1) << "[Reloc/Baseline] window=" << hypothesis_manager_.windowCount()
          << " persisted=" << hypothesis_manager_.persistedFrames()
          << " motion_translation_m=" << motion_baseline.translation
          << " motion_rotation_deg=" << motion_baseline.rotation * 180.0 / M_PI;
  if (reloc_debug_enabled) {
    debug_event.temporal_hypothesis_score = ranking.top1_decision_score;
    debug_event.log_likelihood = ranking.top1_log_likelihood;
    debug_event.winner_streak = hypothesis_manager_.winnerStreak();
    debug_event.evidence_motion_translation = motion_baseline.translation;
    debug_event.evidence_motion_rotation = motion_baseline.rotation;
    debug_event.moving_visibility_required = decision.moving_visibility_required;
    debug_event.moving_visibility_passed = decision.pass_moving_visibility;
    debug_event.margin = ranking.margin;
    debug_event.ratio = ranking.ratio;
    debug_event.basin_separation = ranking.basin_separation;
    if (ranking.uses_visibility_evidence) {
      debug_event.visibility_margin = ranking.margin;
      debug_event.visibility_ratio = ranking.ratio;
    }
  }

  // Free-space disagreement vetoes the ranked winner.
  if (!decision.pass_free_space) {
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
              << ", guard(loglik=" << decision.pass_log_likelihood << ", margin=" << decision.pass_margin
              << ", winner_streak=" << decision.pass_winner_streak
              << ", converged_updates=" << decision.pass_converged_updates
              << ", moving_visibility=" << decision.pass_moving_visibility
              << ", ambiguous=" << decision.ambiguous << ")"
              << ", margin_value=" << ranking.margin
              << ", winner_streak_value=" << hypothesis_manager_.winnerStreak()
              << ", converged_updates_value=" << best.converged_updates
              << ", ratio_value=" << ranking.ratio
              << ", basin_separation=" << ranking.basin_separation;
    finish_debug("accepted", "");
    releaseSearchPreparation();
  } else {
    const std::string &reject_reason = decision.reject_reason;
    if (decision.ambiguous) {
      LOG(WARNING) << "AMBIGUOUS_REJECT relocalization lock: top1_seed="
                   << (top1 ? top1->seed_match_id : -1)
                   << " top1_kf=" << (top1 ? top1->last_match_id : -1)
                   << " top1_ll=" << ranking.top1_log_likelihood
                   << " top2_seed=" << (top2 ? top2->seed_match_id : -1)
                   << " top2_kf=" << (top2 ? top2->last_match_id : -1)
                   << " top2_ll=" << ranking.top2_log_likelihood << " margin=" << ranking.margin
                   << " ratio=" << ranking.ratio
                   << " basin_separation=" << ranking.basin_separation
                   << " top2_viable=" << decision.top2_viable << " thresholds(margin>="
                   << config_.reloc_ambiguity_min_margin
                   << ", ratio>=" << config_.reloc_ambiguity_min_ratio
                   << ", basin_sep>="
                   << config_.reloc_ambiguity_min_basin_separation << ")";
    } else {
      LOG(WARNING)
          << "Relocalization rejected after temporal window by stability guard."
          << " top1_seed=" << (top1 ? top1->seed_match_id : -1)
          << " top1_kf=" << (top1 ? top1->last_match_id : -1)
          << " top1_ll=" << ranking.top1_log_likelihood
          << " top2_seed=" << (top2 ? top2->seed_match_id : -1)
          << " top2_ll=" << ranking.top2_log_likelihood << " margin=" << ranking.margin
          << " ratio=" << ranking.ratio << " basin_separation=" << ranking.basin_separation
          << " winner_streak=" << hypothesis_manager_.winnerStreak()
          << " top1_converged_updates=" << (top1 ? top1->converged_updates : 0)
          << " evidence_motion=(" << motion_baseline.translation << "m,"
          << motion_baseline.rotation << "rad)"
          << " moving_visibility_required=" << decision.moving_visibility_required
          << " top1_visibility=" << ranking.top1_visibility
          << " require(loglik>=" << config_.reloc_lock_log_likelihood_threshold
          << ", margin>=" << decision.effective_min_margin
          << ", winner_streak>=" << decision.effective_min_winner_streak
          << ", converged_updates>=" << decision.effective_min_converged_updates << ")"
          << " pass(loglik=" << decision.pass_log_likelihood << ", margin=" << decision.pass_margin
          << ", winner_streak=" << decision.pass_winner_streak
          << ", converged_updates=" << decision.pass_converged_updates
          << ", moving_visibility=" << decision.pass_moving_visibility
          << ", free_space=" << decision.pass_free_space << ")"
          << " reason=" << reject_reason
          << " directional_support=" << cost.directional_min_support
          << " free_cells=" << free_space_grid_.freeCells();
    }
    finish_debug("rejected", reject_reason);
  }

  // Ambiguity alone must not abandon a viable place or its next confirmation.
  // Recovery already has a map pose prior; expand only failed initial geometry.
  if (!result.success && !has_ever_relocalized_ && !decision.moving_visibility_required &&
      (!decision.pass_log_likelihood || rejected_current_geometry))
    search_offset_ = next_search_offset;

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
