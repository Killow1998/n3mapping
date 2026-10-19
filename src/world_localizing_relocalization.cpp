#include "n3mapping/world_localizing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include "n3mapping/relocalization_order.h"
#include <Eigen/Eigenvalues>
#include <future>
#include <map>
#include <iomanip>
#include <memory>
#include <utility>

namespace n3mapping {
namespace {
constexpr int kRelocMaxBasinCount = 3;
constexpr int kRelocPerBasinVerifyCount = 3;
constexpr double kRelocBasinAssignRadiusXY = 4.0;

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
  std::mutex performance_mutex;
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
  PointCloudMatcher::PreparedSource prepared_query;
  PreparedVisibilityObservation visibility_observation;
  bool query_prepared = false;
  const auto prepare_query = [&]() {
    if (query_prepared) return;
    const auto source_started = std::chrono::steady_clock::now();
    prepared_query = matcher_.prepareSourceCloud(query_cloud);
    const auto visibility_started = std::chrono::steady_clock::now();
    visibility_observation = prepareVisibilityObservation(
        *query_cloud, visibilityOptionsFromConfig(config_));
    cost.source_preparations = 1;
    cost.visibility_preparations = 1;
    cost.source_prepare_ms = std::chrono::duration<double, std::milli>(
        visibility_started - source_started).count();
    cost.visibility_prepare_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - visibility_started).count();
    query_prepared = true;
    VLOG(1) << "[RelocQueryCost] source_preparations=1 source_prepare_ms="
            << cost.source_prepare_ms
            << " visibility_preparations=1 visibility_prepare_ms="
            << cost.visibility_prepare_ms;
  };
  const auto hypothesis_visibility = [&](const PreparedRelocTarget &target,
                                         const Eigen::Isometry3d &pose) {
    const auto started = std::chrono::steady_clock::now();
    auto visibility = evaluatePreparedVisibilityConsistency(
        *target.visibility_target, visibility_observation, pose, target.visibility_indices.get());
    cost.visibility_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++cost.visibility_evaluations;
    return visibility;
  };

  auto descriptor_supports_keyframe =
      [&](const std::vector<LoopCandidate> &candidates, int64_t keyframe_id) {
        auto target = keyframe_manager_.getKeyframe(keyframe_id);
        if (!target) {
          return false;
        }
        const Eigen::Vector2d target_xy =
            target->pose_optimized.translation().head<2>();
        for (const auto &candidate : candidates) {
          auto candidate_kf = keyframe_manager_.getKeyframe(candidate.match_id);
          if (!candidate_kf) {
            continue;
          }
          const double d_xy =
              (candidate_kf->pose_optimized.translation().head<2>() - target_xy)
                  .norm();
          if (d_xy <= kRelocBasinAssignRadiusXY) {
            return true;
          }
        }
        return false;
      };

  if (hypothesis_manager_.empty()) {
    auto candidates = place_index_.search(query_cloud, &cost.descriptor);
    if (reloc_debug_enabled) {
      debug_event.query_cloud.candidate_count = candidates.size();
      debug_event.query_cloud.top_candidates = candidates;
      debug_event.candidate_count = candidates.size();
      debug_event.top_candidates = candidates;
    }
    if (candidates.empty()) {
      LOG(WARNING) << "No relocalization candidates found. Map has "
                   << keyframe_manager_.size() << " keyframes, RHPD candidates="
                   << config_.rhpd_num_candidates
                   << " dist_thr=" << config_.rhpd_dist_threshold;
      finish_debug("rejected", "no_candidates");
      return completed_result();
    }

    struct BasinGroup {
      int64_t center_match_id = -1;
      Eigen::Vector3d center_pos = Eigen::Vector3d::Zero();
      std::vector<LoopCandidate> members;
      std::vector<RelocTargetRequest> targets;
    };
    std::vector<BasinGroup> basins;
    basins.reserve(kRelocMaxBasinCount);

    // Candidates are already sorted by coarse descriptor distance.
    // Build up to top-K basins in XY, then verify per-basin with local-submap
    // ICP.
    for (const auto &candidate : candidates) {
      auto kf = keyframe_manager_.getKeyframe(candidate.match_id);
      if (!kf)
        continue;
      const Eigen::Vector3d pos = kf->pose_optimized.translation();

      int best_basin = -1;
      double best_dist = std::numeric_limits<double>::max();
      for (size_t i = 0; i < basins.size(); ++i) {
        const double d =
            (basins[i].center_pos.head<2>() - pos.head<2>()).norm();
        if (d < best_dist) {
          best_dist = d;
          best_basin = static_cast<int>(i);
        }
      }

      if (best_basin >= 0 && best_dist <= kRelocBasinAssignRadiusXY) {
        basins[best_basin].members.push_back(candidate);
      } else if (static_cast<int>(basins.size()) < kRelocMaxBasinCount) {
        BasinGroup bg;
        bg.center_match_id = candidate.match_id;
        bg.center_pos = pos;
        bg.members.push_back(candidate);
        basins.push_back(std::move(bg));
      }
    }
    if (reloc_debug_enabled) {
      debug_event.basins.clear();
      debug_event.basins.reserve(basins.size());
      for (const auto &basin : basins) {
        RelocDebugBasinSummary summary;
        summary.center_match_id = basin.center_match_id;
        summary.member_match_ids.reserve(basin.members.size());
        for (const auto &member : basin.members) {
          summary.member_match_ids.push_back(member.match_id);
        }
        debug_event.basins.push_back(std::move(summary));
      }
    }

    if (!basins.empty()) prepare_query();
    // Workers share immutable requests. Rebuilding map revisions inside an
    // asynchronous basin would race with the other basin workers.
    for (auto &basin : basins) {
      const auto count = std::min<std::size_t>(
          basin.members.size(), kRelocPerBasinVerifyCount);
      basin.targets.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        basin.targets.push_back(makeRelocTargetRequest(basin.members[i].match_id));
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
        const int order = compareRelocalizationScore(
            candidate.visibility.consistency_ratio, incumbent.visibility.consistency_ratio);
        if (order != 0) return order < 0;
      }
      const int quality_order = compareRelocalizationScore(
          -candidate.selection_score, -incumbent.selection_score);
      if (quality_order != 0) return quality_order < 0;
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
                 prepared_query, candidate,
                 std::move(target_request), visibility_observation, &candidate_cost);
        {
          std::lock_guard<std::mutex> cost_lock(performance_mutex);
          cost.addCandidate(candidate_cost);
        }
        for (const auto &evaluation : evaluations) {
          const auto& selected_match = evaluation.match;
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
          const double descriptor_score = std::isfinite(candidate.fused_score)
                                              ? candidate.fused_score
                                              : 1.0;
          mode.selection_score =
              selected_match.fitness_score + descriptor_score;

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
            << candidates.size() << " basins=" << basins.size()
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

    if (hypothesis_manager_.empty()) {
      LOG(WARNING) << "Relocalization init failed: no valid ICP hypothesis.";
      finish_debug("rejected", "no_valid_icp_hypothesis");
      return completed_result();
    }
  } else {
    const auto current_candidates = place_index_.search(query_cloud, &cost.descriptor);
    if (reloc_debug_enabled) {
      debug_event.query_cloud.candidate_count = current_candidates.size();
      debug_event.query_cloud.top_candidates = current_candidates;
      debug_event.candidate_count = current_candidates.size();
      debug_event.top_candidates = current_candidates;
    }
    hypothesis_manager_.advanceWindow();
    // Hypotheses are independent until ranking. Evaluate those using the same
    // immutable map target together, without reordering their stored identities
    // or retaining an unbounded set of prepared targets across observations.
    std::map<int64_t, std::vector<RelocHypothesis*>> target_groups;
    for (auto &hyp : hypothesis_manager_.hypotheses()) {
      if (!hyp.alive)
        continue;
      if (reloc_debug_enabled) {
        // This field describes a registration produced by this query only.
        // A miss must be null instead of silently reusing the prior query's H.
        hyp.last_registration_observability = {};
      }

      Eigen::Isometry3d predicted_pose = hyp.T_map_odom * odom_pose;
      int64_t nearest_kf_id = findNearestKeyframe(predicted_pose);
      if (nearest_kf_id < 0) {
        hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
        continue;
      }

      if (!descriptor_supports_keyframe(current_candidates, nearest_kf_id)) {
        hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
        continue;
      }
      target_groups[nearest_kf_id].push_back(&hyp);
    }

    for (const auto &[nearest_kf_id, hypotheses] : target_groups) {
      RelocTargetRequest target_request = makeRelocTargetRequest(nearest_kf_id);
      const PreparedRelocTarget prepared_target =
          target_provider_->getTarget(target_request);
      cost.target_build_ms += prepared_target.metrics.target_build_ms;
      cost.target_prepare_ms += prepared_target.metrics.target_prepare_ms;
      cost.target_builds += prepared_target.metrics.target_builds;
      cost.target_preparations += prepared_target.metrics.target_preparations;
      cost.cache_hits += prepared_target.metrics.cache_hit;
      cost.cache_misses += prepared_target.metrics.cache_miss;
      cost.effective_cache_max_bytes = prepared_target.metrics.effective_max_bytes;
      cost.effective_cache_max_entries = prepared_target.metrics.effective_max_entries;
      for (auto *hypothesis : hypotheses) {
        auto &hyp = *hypothesis;
        const Eigen::Isometry3d predicted_pose = hyp.T_map_odom * odom_pose;
        ++cost.hypotheses;
        if (!prepared_target.hasVisibility()) {
          hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
          continue;
        }

        // Registration proposes a local correction; the LiDAR observation model
        // decides whether that correction actually explains the measured nearest
        // surface on each ray. This prevents one-way ICP from dragging a valid
        // pose onto a remote dense surface.
        prepare_query();
        const auto predicted_visibility =
            hypothesis_visibility(prepared_target, predicted_pose);
        MatchResult mr;
        if (prepared_target.valid()) {
          const auto align_started = std::chrono::steady_clock::now();
          mr = matcher_.alignPrepared(*prepared_target.registration_target,
                                      prepared_query, predicted_pose);
          cost.align_ms += std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - align_started).count();
          ++cost.alignments;
          if (!hasValidRegistrationResult(mr)) ++cost.rejected_alignments;
        }
        VisibilityConsistencyResult selected_visibility = predicted_visibility;
        Eigen::Isometry3d selected_pose = predicted_pose;
        bool selected_registration_endpoint = false;
        if (hasValidRegistrationResult(mr)) {
          const auto refined_visibility =
              hypothesis_visibility(prepared_target, mr.T_target_source);
          if (refined_visibility.valid &&
              (!selected_visibility.valid ||
               refined_visibility.consistency_ratio >
                   selected_visibility.consistency_ratio + 1e-9)) {
            selected_visibility = refined_visibility;
            selected_pose = mr.T_target_source;
            selected_registration_endpoint = true;
          }
          if (!selected_registration_endpoint) {
            const auto fixed_started = std::chrono::steady_clock::now();
            mr = matcher_.evaluatePreparedPose(*prepared_target.registration_target,
                                               prepared_query, selected_pose,
                                               mr.metric);
            cost.fixed_pose_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - fixed_started).count();
            ++cost.fixed_pose_evaluations;
          }
        }
        hyp.cumulative_log_likelihood +=
            computeTrackLogLikelihood(mr, predicted_pose);
        hyp.num_updates += 1;
        if (hasValidRegistrationResult(mr)) {
          hyp.T_map_odom = selected_pose * odom_pose.inverse();
          hyp.last_match_id = nearest_kf_id;
          hyp.converged_updates += 1;

          // Recorded here, not at alignment, so the numbers describe the pose
          // this hypothesis actually carries.
          const bool information_at_selected_pose =
              selected_pose.isApprox(mr.T_target_source);
          const Eigen::Matrix3d trans_info = mr.information.block<3, 3>(0, 0);
          const Eigen::Matrix3d rot_info = mr.information.block<3, 3>(3, 3);
          const Eigen::Matrix3d cross_info = mr.information.block<3, 3>(3, 0);
          const Eigen::Matrix3d rot_marginal =
              rot_info -
              cross_info *
                  trans_info.completeOrthogonalDecomposition().pseudoInverse() *
                  cross_info.transpose();
          const Eigen::Vector3d re =
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(rot_info)
                  .eigenvalues();
          const Eigen::Vector3d rm =
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(rot_marginal)
                  .eigenvalues();
          const Eigen::Vector3d te =
              Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(trans_info)
                  .eigenvalues();
          const bool production_quality =
              mr.success && mr.fitness_score <= config_.gicp_fitness_threshold &&
              mr.inlier_ratio >= config_.reloc_min_inlier_ratio;
          hyp.last_rot_info_min = re(0);
          hyp.last_rot_info_marginal_min = rm(0);
          hyp.last_trans_info_min = te(0);
          hyp.last_iterations = static_cast<int>(mr.iterations);
          hyp.last_termination = static_cast<int>(mr.termination);
          hyp.last_inlier_ratio = mr.inlier_ratio;
          hyp.last_fitness = mr.fitness_score;
          hyp.last_pose_is_refined = selected_registration_endpoint;
          hyp.last_production_quality = production_quality;
          if (reloc_debug_enabled) {
            hyp.last_registration_observability =
                analyzeRegistrationObservability(
                    mr,
                    selected_registration_endpoint ? "registration_endpoint"
                                                   : "motion_prediction",
                    information_at_selected_pose, production_quality);
          }
          VLOG(1) << "[Reloc/Info] seed=" << hyp.seed_match_id << " pose="
                  << (selected_registration_endpoint ? "refined" : "predicted")
                  << " prod_quality=" << production_quality
                  << " rot_eig=" << re(0) << "," << re(1) << "," << re(2)
                  << " rot_marg_eig=" << rm(0) << "," << rm(1) << "," << rm(2)
                  << " trans_eig=" << te(0) << "," << te(1) << "," << te(2)
                  << " inliers=" << mr.num_inliers
                  << " fitness=" << mr.fitness_score
                  << " inlier_ratio=" << mr.inlier_ratio;
        }
        if (selected_visibility.valid) {
          hyp.visibility_consistency_sum += selected_visibility.consistency_ratio;
          hyp.visibility_evidence_sum += selected_visibility.evidence_log_odds;
          hyp.visibility_updates += 1;
        }
      }
    }
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
      summary.seed_match_id = hyp.seed_match_id;
      summary.last_match_id = hyp.last_match_id;
      summary.pose_in_map = hyp.T_map_odom * odom_pose;
      summary.cumulative_log_likelihood = hyp.cumulative_log_likelihood;
      summary.num_updates = hyp.num_updates;
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
      hypothesis_manager_.motionBaseline(odom_pose);
  RelocalizationDecision decision =
      decision_policy_.evaluate(ranking, hypothesis_manager_.windowCount(),
                                hypothesis_manager_.winnerStreak(),
                                motion_baseline, nullptr, odom_pose);
  if (config_.reloc_free_space_mode != "kill" &&
      !decision.temporal_window_pending && decision.accepted) {
    const auto started = std::chrono::steady_clock::now();
    const auto *free_space_best = freeSpaceBestHypothesis(query_cloud, odom_pose);
    cost.free_space_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    decision = decision_policy_.evaluate(
        ranking, hypothesis_manager_.windowCount(), hypothesis_manager_.winnerStreak(),
        motion_baseline, free_space_best, odom_pose);
  }
  const int effective_temporal_window = decision.effective_temporal_window;
  if (decision.temporal_window_pending) {
    VLOG(1) << "Relocalization pending: window "
            << hypothesis_manager_.windowCount() << "/"
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
          << " free_cells=" << free_space_grid_.freeCells();
    }
    finish_debug("rejected", reject_reason);
  }

  // Card C/D: clearing after every rejected window capped the accumulated
  // viewpoint baseline at 1-3 cm. The manager keeps the seed pose across
  // rejected windows, so the baseline grows with the robot's motion.
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
