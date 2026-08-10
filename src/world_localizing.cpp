// WorldLocalizing: global relocalization via RHPD + ICP, and tracking
// localization with T_map_odom.
#include "n3mapping/world_localizing.h"

#include <cstdlib>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <string>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/pcl_compat.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <glog/logging.h>
#include <limits>
#include <memory>
#include <pcl/common/point_tests.h>

namespace n3mapping {
namespace {
constexpr int kRelocMaxBasinCount = 3;
constexpr int kRelocPerBasinVerifyCount = 3;
constexpr double kRelocBasinAssignRadiusXY = 4.0;

double processingTimeSeconds() {
  using Clock = std::chrono::system_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

bool hasValidRegistrationResult(const MatchResult &match) {
  return match.converged && std::isfinite(match.fitness_score) &&
         std::isfinite(match.inlier_ratio) &&
         isFiniteRigidPose(match.T_target_source);
}

bool sameGenerationAndPose(const KeyframeMapRevision &lhs,
                           const KeyframeMapRevision &rhs) {
  return lhs.generation == rhs.generation &&
         lhs.pose_revision == rhs.pose_revision;
}

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

WorldLocalizing::WorldLocalizing(const Config &config,
                                 KeyframeManager &keyframe_manager,
                                 LoopDetector &loop_detector,
                                 PointCloudMatcher &matcher)
    : config_(config), keyframe_manager_(keyframe_manager),
      loop_detector_(loop_detector), matcher_(matcher), query_builder_(config_),
      place_index_(config_, keyframe_manager_, loop_detector_),
      localization_atlas_(
          std::make_unique<LocalizationAtlas>(config_, matcher_)),
      candidate_evaluator_(config_, keyframe_manager_, matcher_,
                           *localization_atlas_),
      decision_policy_(config_),
      reloc_map_cache_(pcl::make_shared<PointCloudT>()),
      reloc_map_cached_keyframes_(0),
      loaded_map_visibility_cache_(pcl::make_shared<PointCloudT>()),
      is_relocalized_(false),
      T_map_odom_(Eigen::Isometry3d::Identity()), last_matched_id_(-1),
      relocalization_seed_id_(-1),
      last_odom_pose_(Eigen::Isometry3d::Identity()),
      consecutive_track_failures_(0), relocalize_debug_query_index_(0),
      track_debug_query_index_(0) {}

void WorldLocalizing::appendRelocalizationDebug(
    const RelocalizationDebugEvent &event) const {
  if (!config_.reloc_debug_enable) {
    return;
  }
  std::lock_guard<std::mutex> lock(debug_mutex_);
  RelocalizationDebugLogger::appendRelocalization(
      RelocalizationDebugLogger::resolvePath(config_), event);
}

void WorldLocalizing::appendTrackingDebug(
    const RelocTrackingDebugEvent &event) const {
  if (!config_.reloc_debug_enable) {
    return;
  }
  std::lock_guard<std::mutex> lock(debug_mutex_);
  RelocalizationDebugLogger::appendTracking(
      RelocalizationDebugLogger::resolvePath(config_), event);
}

RelocResult WorldLocalizing::relocalize(const PointCloudT::Ptr &cloud,
                                        const Eigen::Isometry3d &odom_pose) {
  RelocResult result;
  result.success = false;
  const bool reloc_debug_enabled = config_.reloc_debug_enable;
  RelocalizationDebugEvent debug_event;
  if (reloc_debug_enabled) {
    std::lock_guard<std::mutex> debug_lock(debug_mutex_);
    debug_event.query_index = ++relocalize_debug_query_index_;
    debug_event.processing_time = processingTimeSeconds();
  }
  auto finish_debug = [&](const std::string &lock_result,
                          const std::string &reject_reason) {
    result.decision = reject_reason.empty() ? lock_result : reject_reason;
    if (!reloc_debug_enabled) {
      return;
    }
    debug_event.processing_time = processingTimeSeconds();
    debug_event.lock_result = lock_result;
    debug_event.lock_accepted = (lock_result == "accepted");
    debug_event.reject_reason = reject_reason;
    appendRelocalizationDebug(debug_event);
  };

  if (!cloud || cloud->empty()) {
    LOG(WARNING) << "Empty point cloud for relocalization.";
    finish_debug("rejected", "empty_cloud");
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  if (keyframe_manager_.size() == 0) {
    LOG(WARNING) << "No keyframes available for relocalization.";
    finish_debug("rejected", "missing_keyframes");
    return result;
  }

  const PlaceObservation query_observation =
      query_builder_.buildStationary(cloud, odom_pose);
  PointCloudT::Ptr query_cloud = query_observation.cloud;
  if (!query_cloud || query_cloud->empty()) {
    LOG(WARNING) << "Relocalization query cloud is empty after aggregation.";
    finish_debug("rejected", "empty_query_cloud");
    return result;
  }
  if (reloc_debug_enabled) {
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
  }

  // The query is unchanged for every candidate and yaw seed in this call.
  // Prepare all registration representations once, then reuse them below.
  const auto prepared_query = matcher_.prepareSourceCloud(query_cloud);

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
    auto candidates = place_index_.search(query_cloud);
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
      return result;
    }

    struct BasinGroup {
      int64_t center_match_id = -1;
      Eigen::Vector3d center_pos = Eigen::Vector3d::Zero();
      std::vector<LoopCandidate> members;
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

    struct BasinBest {
      int64_t basin_center_id = -1;
      int64_t matched_kf_id = -1;
      LoopCandidate candidate;
      MatchResult match;
      VisibilityConsistencyResult visibility;
      double selection_score = std::numeric_limits<double>::max();
    };
    std::vector<BasinBest> basin_best_results;
    basin_best_results.reserve(basins.size() * kRelocPerBasinVerifyCount);
    const auto better_basin_mode = [](const BasinBest &candidate,
                                      const BasinBest &incumbent) {
      if (candidate.visibility.valid != incumbent.visibility.valid) {
        return candidate.visibility.valid;
      }
      if (candidate.visibility.valid &&
          std::abs(candidate.visibility.consistency_ratio -
                   incumbent.visibility.consistency_ratio) > 1e-9) {
        return candidate.visibility.consistency_ratio >
               incumbent.visibility.consistency_ratio;
      }
      return candidate.selection_score < incumbent.selection_score;
    };

    const auto evaluate_basin = [&](const BasinGroup &basin) {
      std::vector<BasinBest> basin_modes;
      int verified = 0;

      for (const auto &candidate : basin.members) {
        if (verified >= kRelocPerBasinVerifyCount)
          break;
        ++verified;

        const auto candidate_target = buildRelocTargetCloud(candidate.match_id);
        for (const auto &evaluation : candidate_evaluator_.evaluate(
                 query_cloud, prepared_query, candidate, candidate_target)) {
          const MatchResult selected_match =
              evaluation.registration.selectedMatch();
          const auto quality = evaluateRelocMatchQuality(selected_match);
          if (!quality.accepted)
            continue;

          BasinBest mode;
          mode.basin_center_id = basin.center_match_id;
          mode.matched_kf_id = evaluation.matched_keyframe_id;
          mode.candidate = candidate;
          mode.match = selected_match;
          mode.visibility = evaluation.visibility;
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
      rebuildRelocMapCacheIfNeeded();
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
      hyp.alive = true;
      seeded_hypotheses.push_back(hyp);
    }
    hypothesis_manager_.start(std::move(seeded_hypotheses), odom_pose);

    if (hypothesis_manager_.empty()) {
      LOG(WARNING) << "Relocalization init failed: no valid ICP hypothesis.";
      finish_debug("rejected", "no_valid_icp_hypothesis");
      return result;
    }
  } else {
    const auto current_candidates = place_index_.search(query_cloud);
    if (reloc_debug_enabled) {
      debug_event.query_cloud.candidate_count = current_candidates.size();
      debug_event.query_cloud.top_candidates = current_candidates;
      debug_event.candidate_count = current_candidates.size();
      debug_event.top_candidates = current_candidates;
    }
    hypothesis_manager_.advanceWindow();
    for (auto &hyp : hypothesis_manager_.hypotheses()) {
      if (!hyp.alive)
        continue;

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

      auto submap = buildRelocTargetCloud(nearest_kf_id);
      if (!submap || submap->empty()) {
        hyp.cumulative_log_likelihood -= config_.reloc_hypothesis_miss_penalty;
        continue;
      }

      // Registration proposes a local correction; the LiDAR observation model
      // decides whether that correction actually explains the measured nearest
      // surface on each ray. This prevents one-way ICP from dragging a valid
      // pose onto a remote dense surface.
      const auto predicted_visibility =
          evaluatePoseVisibility(submap, query_cloud, predicted_pose);
      PointCloudMatcher::PreparedTarget local_prepared_target;
      const PointCloudMatcher::PreparedTarget *prepared_target = nullptr;
      if (localization_atlas_ && localization_atlas_->loaded()) {
        prepared_target = &localization_atlas_->preparedTarget();
      } else {
        local_prepared_target = matcher_.prepareTargetCloud(submap);
        prepared_target = &local_prepared_target;
      }
      MatchResult mr = matcher_.alignPrepared(*prepared_target, prepared_query,
                                              predicted_pose);
      hyp.cumulative_log_likelihood +=
          computeTrackLogLikelihood(mr, predicted_pose);
      hyp.num_updates += 1;
      VisibilityConsistencyResult selected_visibility = predicted_visibility;
      Eigen::Isometry3d selected_pose = predicted_pose;
      if (hasValidRegistrationResult(mr)) {
        const auto refined_visibility =
            evaluatePoseVisibility(submap, query_cloud, mr.T_target_source);
        if (refined_visibility.valid &&
            (!selected_visibility.valid ||
             refined_visibility.consistency_ratio >
                 selected_visibility.consistency_ratio + 1e-9)) {
          selected_visibility = refined_visibility;
          selected_pose = mr.T_target_source;
        }
        hyp.T_map_odom = selected_pose * odom_pose.inverse();
        hyp.last_match_id = nearest_kf_id;
        hyp.converged_updates += 1;

        // Recorded here, not at alignment, so the numbers describe the pose
        // this hypothesis actually carries.
        const bool pose_is_refined = selected_pose.isApprox(mr.T_target_source);
        const Eigen::Matrix3d trans_info = mr.information.block<3, 3>(0, 0);
        const Eigen::Matrix3d rot_info = mr.information.block<3, 3>(3, 3);
        const Eigen::Matrix3d cross_info = mr.information.block<3, 3>(3, 0);
        const Eigen::Matrix3d rot_marginal =
            rot_info -
            cross_info * trans_info.completeOrthogonalDecomposition().pseudoInverse() *
                cross_info.transpose();
        const Eigen::Vector3d re =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(rot_info).eigenvalues();
        const Eigen::Vector3d rm =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(rot_marginal).eigenvalues();
        const Eigen::Vector3d te =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(trans_info).eigenvalues();
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
        hyp.last_pose_is_refined = pose_is_refined;
        hyp.last_production_quality = production_quality;
        VLOG(1) << "[Reloc/Info] seed=" << hyp.seed_match_id
                << " pose=" << (pose_is_refined ? "refined" : "predicted")
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

  const RelocHypothesis *free_space_best =
      freeSpaceBestHypothesis(query_cloud, odom_pose);
  if (config_.reloc_free_space_mode == "kill")
    killFreeSpaceDominatedHypotheses(query_cloud, odom_pose);

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
      summary.alive = hyp.alive;
      debug_event.hypotheses.push_back(std::move(summary));
    }
  }

  const RelocalizationRanking ranking = decision_policy_.rank(
      hypothesis_manager_.hypotheses(), odom_pose);
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
    result.state = RelocalizationState::REGION_HYPOTHESIS;
    result.pose_source = PoseSource::NONE;
    if (reloc_debug_enabled) {
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

    VLOG(1) << "[Reloc/Stability] window="
            << hypothesis_manager_.windowCount() << "/"
            << config_.reloc_temporal_window_size
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
  const RelocalizationDecision decision = decision_policy_.evaluate(
      ranking, hypothesis_manager_.windowCount(),
      hypothesis_manager_.winnerStreak(), motion_baseline, free_space_best,
      odom_pose);
  const int effective_temporal_window = decision.effective_temporal_window;
  if (decision.temporal_window_pending) {
    VLOG(1) << "Relocalization pending: window "
            << hypothesis_manager_.windowCount()
            << "/" << effective_temporal_window
            << ", active hypotheses=" << hypothesis_manager_.size();
    finish_debug("pending", "temporal_window_pending");
    return result;
  }

  const int effective_min_winner_streak =
      decision.effective_min_winner_streak;
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
  VLOG(1) << "[Reloc/Baseline] window="
          << hypothesis_manager_.windowCount()
          << " persisted=" << hypothesis_manager_.persistedFrames()
          << " motion_translation_m=" << evidence_motion_translation
          << " motion_rotation_deg="
          << evidence_motion_rotation * 180.0 / M_PI;
  const bool moving_visibility_required =
      decision.moving_visibility_required;
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
                << "[Reloc/LockKF] id=" << skf->id << " stamp=" << skf->timestamp
                << " t=" << kt.x() << "," << kt.y() << "," << kt.z()
                << " q=" << kq.x() << "," << kq.y() << "," << kq.z() << ","
                << kq.w();
    }
    LOG(INFO) << "[Reloc/LockPose] pose="
              << (best.last_pose_is_refined ? "refined" : "predicted")
              << " prod_quality=" << best.last_production_quality
              << " rot_info_marginal_min=" << best.last_rot_info_marginal_min;
    LOG(INFO) << "[Reloc/LockTerm] iters=" << best.last_iterations
              << " term=" << matchTerminationName(
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
              << ", winner_streak_value="
              << hypothesis_manager_.winnerStreak()
              << ", converged_updates_value=" << best.converged_updates
              << ", ratio_value=" << ratio
              << ", basin_separation=" << basin_separation;
    finish_debug("accepted", "");
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
      hypothesis_manager_.finishWindow(
          result.success, config_.reloc_persist_hypotheses,
          config_.reloc_persist_max_frames);
  if (persistence.reseeded) {
    VLOG(1) << "[Reloc/Persist] reseeding: alive=" << persistence.any_alive
            << " persisted_frames=" << persistence.persisted_frames;
  }

  return result;
}

RelocResult
WorldLocalizing::trackLocalization(const PointCloudT::Ptr &cloud,
                                   const Eigen::Isometry3d &odom_pose) {
  return trackLocalizationImpl(cloud, odom_pose, false);
}

RelocResult
WorldLocalizing::trackLoadedMap(const PointCloudT::Ptr &cloud,
                                const Eigen::Isometry3d &odom_pose) {
  return trackLocalizationImpl(cloud, odom_pose, true);
}

RelocResult
WorldLocalizing::trackLocalizationImpl(const PointCloudT::Ptr &cloud,
                                       const Eigen::Isometry3d &odom_pose,
                                       bool strict_loaded_map) {
  RelocResult result;
  result.success = false;
  const bool reloc_debug_enabled = config_.reloc_debug_enable;
  RelocTrackingDebugEvent debug_event;
  if (reloc_debug_enabled) {
    std::lock_guard<std::mutex> debug_lock(debug_mutex_);
    debug_event.query_index = ++track_debug_query_index_;
    debug_event.processing_time = processingTimeSeconds();
  }
  auto finish_tracking_debug = [&](const std::string &reject_reason) {
    result.decision = reject_reason.empty()
                          ? (strict_loaded_map
                                 ? "loaded_map_tracking_geometric"
                                 : "tracking_geometric")
                          : reject_reason;
    if (!reloc_debug_enabled) {
      return;
    }
    debug_event.processing_time = processingTimeSeconds();
    debug_event.result_success = result.success;
    debug_event.consecutive_track_failures = consecutive_track_failures_;
    debug_event.reject_reason = reject_reason;
    appendTrackingDebug(debug_event);
  };

  if (!cloud || cloud->empty()) {
    std::lock_guard<std::mutex> lock(mutex_);
    result.state = is_relocalized_ ? RelocalizationState::DEGRADED_TRACKING
                                   : RelocalizationState::SEARCHING;
    result.pose_source =
        is_relocalized_ ? PoseSource::ODOM_PREDICTED : PoseSource::NONE;
    result.seed_keyframe_id = relocalization_seed_id_;
    result.support_keyframe_id = last_matched_id_;
    result.matched_keyframe_id = last_matched_id_;
    result.pose_in_map = T_map_odom_ * odom_pose;
    debug_event.predicted_pose = result.pose_in_map;
    LOG(WARNING) << "Empty point cloud for tracking.";
    finish_tracking_debug("empty_cloud");
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  result.seed_keyframe_id = relocalization_seed_id_;

  Eigen::Isometry3d predicted_pose = T_map_odom_ * odom_pose;
  result.pose_in_map = predicted_pose;
  debug_event.predicted_pose = predicted_pose;

  if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
    is_relocalized_ = false;
    result.state = RelocalizationState::SEARCHING;
    result.pose_source = PoseSource::NONE;
    finish_tracking_debug("max_track_failures");
    return result;
  }

  int64_t nearest_kf_id = strict_loaded_map
                              ? findNearestLoadedKeyframe(predicted_pose)
                              : findNearestKeyframe(predicted_pose);
  debug_event.nearest_kf_id = nearest_kf_id;
  if (nearest_kf_id < 0) {
    consecutive_track_failures_++;
    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
      is_relocalized_ = false;
      result.success = false;
      result.state = RelocalizationState::SEARCHING;
      result.pose_source = PoseSource::NONE;
    } else {
      result.success = !strict_loaded_map;
      result.state = RelocalizationState::DEGRADED_TRACKING;
      result.pose_source = PoseSource::ODOM_PREDICTED;
    }
    result.matched_keyframe_id = last_matched_id_;
    result.support_keyframe_id = last_matched_id_;
    result.pose_in_map = predicted_pose;
    result.confidence = 0.5;
    last_odom_pose_ = odom_pose;
    finish_tracking_debug("nearest_keyframe_missing");
    return result;
  }

  // Build submap — use larger range when tracking is unstable. Extension
  // tracking must never use newly appended keyframes as its own reference.
  int submap_range = config_.gicp_submap_size;
  if (consecutive_track_failures_ > 0)
    submap_range =
        std::max(submap_range, config_.reloc_track_unstable_submap_size);

  PointCloudT::Ptr submap;
  if (strict_loaded_map) {
    rebuildLoadedMapVisibilityCacheIfNeeded();
    const auto nearest = keyframe_manager_.getKeyframe(nearest_kf_id);
    if (nearest) {
      submap = LocalizationAtlas::cropGlobalMap(
          config_, loaded_map_visibility_cache_,
          nearest->pose_optimized.translation());
    }
  } else {
    // Keep the established ordinary-localization target unchanged. The
    // loaded-map-only target is specific to map-extension safety.
    submap = keyframe_manager_.buildLocalSubmap(nearest_kf_id, submap_range);
  }
  debug_event.submap_size = submap ? submap->size() : 0;
  if (!submap || submap->empty()) {
    consecutive_track_failures_++;
    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
      is_relocalized_ = false;
      result.success = false;
      result.state = RelocalizationState::SEARCHING;
      result.pose_source = PoseSource::NONE;
    } else {
      result.success = !strict_loaded_map;
      result.state = RelocalizationState::DEGRADED_TRACKING;
      result.pose_source = PoseSource::ODOM_PREDICTED;
    }
    result.matched_keyframe_id = last_matched_id_;
    result.support_keyframe_id = last_matched_id_;
    result.pose_in_map = predicted_pose;
    result.confidence = 0.5;
    last_odom_pose_ = odom_pose;
    finish_tracking_debug("empty_submap");
    return result;
  }

  const auto prepared_target = matcher_.prepareTargetCloud(submap);
  const auto prepared_source = matcher_.prepareSourceCloud(cloud);
  MatchResult match_result =
      matcher_.alignPrepared(prepared_target, prepared_source, predicted_pose);
  bool retry_used = false;

  // If ICP failed with standard params and we have recent failures, retry with
  // wider search
  const bool icp_failed =
      !hasValidRegistrationResult(match_result) ||
      match_result.fitness_score >= config_.gicp_fitness_threshold ||
      match_result.inlier_ratio < config_.reloc_min_inlier_ratio;
  if (icp_failed &&
      consecutive_track_failures_ < config_.reloc_track_retry_max_failures) {
    auto saved = matcher_.getSettings();
    auto wide = saved;
    wide.max_correspondence_distance = saved.max_correspondence_distance *
                                       config_.reloc_track_retry_corr_scale;
    wide.max_iterations = config_.reloc_track_retry_max_iterations;
    auto retry = matcher_.alignPrepared(prepared_target, prepared_source,
                                        predicted_pose, wide);
    retry_used = true;
    if (hasValidRegistrationResult(retry) &&
        (!hasValidRegistrationResult(match_result) ||
         retry.fitness_score < match_result.fitness_score))
      match_result = retry;
  }
  debug_event.icp_converged = match_result.converged;
  debug_event.fitness_score = match_result.fitness_score;
  debug_event.inlier_ratio = match_result.inlier_ratio;
  debug_event.retry_used = retry_used;

  double scale = config_.gicp_fitness_threshold * 0.5;
  double current_confidence = std::exp(-match_result.fitness_score / scale);
  current_confidence = std::max(0.0, std::min(1.0, current_confidence));

  const bool icp_ok =
      hasValidRegistrationResult(match_result) &&
      match_result.fitness_score < config_.gicp_fitness_threshold &&
      match_result.inlier_ratio >= config_.reloc_min_inlier_ratio;

  double delta_translation = std::numeric_limits<double>::infinity();
  double delta_rotation = std::numeric_limits<double>::infinity();
  VisibilityConsistencyResult visibility;
  if (hasValidRegistrationResult(match_result)) {
    if (strict_loaded_map) {
      const auto predicted_visibility =
          evaluatePoseVisibility(submap, cloud, predicted_pose);
      visibility = evaluatePoseVisibility(
          submap, cloud, match_result.T_target_source);
      if (predicted_visibility.valid &&
          (!visibility.valid ||
           predicted_visibility.consistency_ratio >
               visibility.consistency_ratio + 1e-9)) {
        // ICP is a local proposal, not authority to slide along a repeated
        // surface. Retain the prediction when the immutable loaded map
        // explains it better than the optimizer endpoint.
        match_result.T_target_source = predicted_pose;
        visibility = predicted_visibility;
      }
    }
    const Eigen::Isometry3d delta =
        predicted_pose.inverse() * match_result.T_target_source;
    delta_translation = delta.translation().norm();
    delta_rotation = Eigen::AngleAxisd(delta.rotation()).angle();
  }

  const bool strict_geometry_ok =
      !strict_loaded_map ||
      (delta_translation <= config_.reloc_track_max_translation &&
       delta_rotation <= config_.reloc_track_max_rotation &&
       visibility.valid && visibility.evidence_log_odds > 0.0);

  const bool accepted_icp = icp_ok && strict_geometry_ok;

  if (accepted_icp) {
    Eigen::Isometry3d T_map_odom_icp =
        match_result.T_target_source * odom_pose.inverse();

    double alpha = 1.0;
    if (!strict_loaded_map && delta_translation <= 0.5) {
      alpha = std::min(current_confidence * 0.3, 0.2);
    } else if (!strict_loaded_map && delta_translation <= 2.0) {
      alpha = std::min(current_confidence * 0.15, 0.1);
    } else if (!strict_loaded_map) {
      alpha = std::min(current_confidence * 0.08, 0.05);
    }
    if (!strict_loaded_map) {
      alpha = std::max(alpha, 0.01);
    }

    if (strict_loaded_map) {
      T_map_odom_ = T_map_odom_icp;
    } else {
      Eigen::Vector3d t_current = T_map_odom_.translation();
      Eigen::Vector3d t_target = T_map_odom_icp.translation();
      Eigen::Vector3d t_new = t_current + alpha * (t_target - t_current);

      double alpha_z = alpha * 0.3;
      t_new.z() = t_current.z() + alpha_z * (t_target.z() - t_current.z());

      Eigen::Quaterniond q_current(T_map_odom_.rotation());
      Eigen::Quaterniond q_target(T_map_odom_icp.rotation());
      Eigen::Quaterniond q_new = q_current.slerp(alpha, q_target);

      T_map_odom_.translation() = t_new;
      T_map_odom_.linear() = q_new.toRotationMatrix();
    }

    result.success = true;
    result.state = RelocalizationState::FULL_6DOF_LOCKED;
    result.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
    result.matched_keyframe_id = nearest_kf_id;
    result.support_keyframe_id = nearest_kf_id;
    result.pose_in_map = T_map_odom_ * odom_pose;
    result.fitness_score = match_result.fitness_score;
    result.confidence = current_confidence;

    last_matched_id_ = nearest_kf_id;
    last_odom_pose_ = odom_pose;
    consecutive_track_failures_ = 0;

    VLOG(1) << (strict_loaded_map ? "Loaded-map tracking OK: fitness="
                                        : "Tracking OK: fitness=")
            << match_result.fitness_score
            << " conf=" << current_confidence << " alpha=" << alpha
            << " delta_t=" << delta_translation
            << " delta_r=" << delta_rotation
            << " visibility_logodds=" << visibility.evidence_log_odds;
    finish_tracking_debug("");
  } else {
    consecutive_track_failures_++;
    if (consecutive_track_failures_ <= 3 ||
        consecutive_track_failures_ % 5 == 0) {
      LOG(WARNING) << "Tracking failed (x" << consecutive_track_failures_
                   << "): converged=" << match_result.converged
                   << " fitness=" << match_result.fitness_score
                   << " inlier=" << match_result.inlier_ratio
                   << " nearest_kf=" << nearest_kf_id
                   << " submap_pts=" << (submap ? submap->size() : 0)
                   << " cloud_pts=" << cloud->size() << " pos=("
                   << predicted_pose.translation().x() << ","
                   << predicted_pose.translation().y() << ","
                   << predicted_pose.translation().z() << ")";
    }

    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
      is_relocalized_ = false;
      result.success = false;
      result.state = RelocalizationState::SEARCHING;
      result.pose_source = PoseSource::NONE;
    } else {
      // Keep odometry-based continuity for transient dropouts.
      result.success = !strict_loaded_map;
      result.state = RelocalizationState::DEGRADED_TRACKING;
      result.pose_source = PoseSource::ODOM_PREDICTED;
    }

    result.matched_keyframe_id = last_matched_id_;
    result.support_keyframe_id = last_matched_id_;
    result.pose_in_map = predicted_pose;
    result.confidence = 0.2;
    last_odom_pose_ = odom_pose;
    finish_tracking_debug(
        strict_loaded_map && icp_ok && !strict_geometry_ok
            ? "loaded_map_tracking_geometry_gate_failed"
            : "icp_gate_failed");
  }

  return result;
}

bool WorldLocalizing::isRelocalized() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return is_relocalized_;
}

Eigen::Isometry3d WorldLocalizing::getMapToOdomTransform() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return T_map_odom_;
}

void WorldLocalizing::resetLocalizationState() {
  std::lock_guard<std::mutex> lock(mutex_);
  is_relocalized_ = false;
  T_map_odom_ = Eigen::Isometry3d::Identity();
  last_matched_id_ = -1;
  relocalization_seed_id_ = -1;
  last_odom_pose_ = Eigen::Isometry3d::Identity();
  consecutive_track_failures_ = 0;
  query_builder_.reset();
  hypothesis_manager_.reset();
}

void WorldLocalizing::notifyMapReplaced() {
  std::lock_guard<std::mutex> lock(mutex_);
  is_relocalized_ = false;
  T_map_odom_ = Eigen::Isometry3d::Identity();
  last_matched_id_ = -1;
  relocalization_seed_id_ = -1;
  last_odom_pose_ = Eigen::Isometry3d::Identity();
  consecutive_track_failures_ = 0;
  query_builder_.reset();
  hypothesis_manager_.reset();
  // Map-derived state: none of it may survive into the next map. A map that
  // happens to have the same keyframe count as the previous one must still
  // rebuild its frame-RHPD index, reloc map cache, free-space grid and atlas.
  place_index_.resetMapDerivedState();
  reloc_map_cache_ = pcl::make_shared<PointCloudT>();
  reloc_map_cached_keyframes_ = 0;
  reloc_map_revision_ = {};
  loaded_map_visibility_cache_ = pcl::make_shared<PointCloudT>();
  loaded_map_visibility_cached_keyframes_ = 0;
  loaded_map_visibility_revision_ = {};
  free_space_grid_ = FreeSpaceGrid();
  free_space_grid_keyframes_ = 0;
  free_space_grid_revision_ = {};
  free_space_grid_failed_ = false;
  if (localization_atlas_) {
    localization_atlas_->clear();
  }
}

void WorldLocalizing::reset() {
  resetLocalizationState();
}

WorldLocalizing::WorldLocalizingCacheDiagnostics
WorldLocalizing::cacheDiagnostics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  WorldLocalizingCacheDiagnostics d;
  d.atlas_loaded = localization_atlas_ && localization_atlas_->loaded();
  d.frame_rhpd_indexed_keyframes = place_index_.indexedKeyframes();
  d.reloc_map_cached_keyframes = reloc_map_cached_keyframes_;
  d.free_space_grid_keyframes = free_space_grid_keyframes_;
  d.free_space_grid_valid = free_space_grid_.valid();
  d.free_space_grid_failed = free_space_grid_failed_;
  return d;
}

void WorldLocalizing::setMapToOdomTransform(
    const Eigen::Isometry3d &T_map_odom) {
  std::lock_guard<std::mutex> lock(mutex_);
  T_map_odom_ = T_map_odom;
  is_relocalized_ = true;
  relocalization_seed_id_ = -1;
  query_builder_.reset();
  hypothesis_manager_.reset();
}

int64_t WorldLocalizing::getLastMatchedKeyframeId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_matched_id_;
}

bool WorldLocalizing::loadLocalizationAtlas(const std::string &map_path,
                                            std::string *error) {
  if (!localization_atlas_) {
    if (error)
      *error = "localization atlas object is unavailable";
    return false;
  }
  localization_atlas_->clear();
  if (!config_.reloc_atlas_enable)
    return true;
  const std::string atlas_path =
      config_.reloc_atlas_path.empty()
          ? LocalizationAtlas::defaultAtlasPath(map_path)
          : config_.reloc_atlas_path;
  LocalizationAtlasStats stats;
  if (!localization_atlas_->load(map_path, atlas_path, &stats, error))
    return false;
  LOG(INFO) << "[LocalizationAtlas] Loaded " << atlas_path
            << " global_points=" << stats.global_point_count
            << " prepared_points=" << stats.prepared_point_count
            << " bytes=" << stats.sidecar_bytes << " load_ms=" << stats.load_ms
            << " kdtree_ms=" << stats.kdtree_ms;
  return true;
}

bool WorldLocalizing::localizationAtlasLoaded() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return localization_atlas_ && localization_atlas_->loaded();
}

RegistrationSeedProbeResult
WorldLocalizing::probeRegistrationSeeds(const PointCloudT::Ptr &cloud,
                                        const Eigen::Isometry3d &odom_pose,
                                        const Eigen::Isometry3d &oracle_pose) {
  RegistrationSeedProbeResult result;
  if (!cloud || cloud->empty()) {
    result.error = "empty_cloud";
    return result;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (keyframe_manager_.size() == 0) {
    result.error = "missing_keyframes";
    return result;
  }
  if (!localization_atlas_ || !localization_atlas_->loaded()) {
    result.error = "atlas_required";
    return result;
  }

  auto query_cloud = query_builder_.buildStationary(cloud, odom_pose).cloud;
  if (!query_cloud || query_cloud->empty()) {
    result.error = "empty_query_cloud";
    return result;
  }
  const auto prepared_query = matcher_.prepareSourceCloud(query_cloud);
  const auto &target = localization_atlas_->preparedTarget();

  const auto finish_attempt = [&](RegistrationSeedProbeAttempt *attempt) {
    if (!attempt)
      return;
    const auto quality = evaluateRelocMatchQuality(attempt->match);
    attempt->fitness_pass = quality.fitness_pass;
    attempt->inlier_pass = quality.inlier_pass;
    attempt->derived_confidence = quality.confidence;
    attempt->confidence_pass = quality.confidence_pass;
    attempt->production_quality_pass = quality.accepted;
    const auto visibility_target =
        buildRelocTargetCloud(attempt->seed_keyframe_id);
    attempt->initial_visibility = evaluatePoseVisibility(
        visibility_target, query_cloud, attempt->initial_pose);
    attempt->refined_visibility = evaluatePoseVisibility(
        visibility_target, query_cloud, attempt->match.T_target_source);
    attempt->production_kept_initial_pose =
        attempt->initial_visibility.valid &&
        (!attempt->refined_visibility.valid ||
         attempt->initial_visibility.consistency_ratio >
             attempt->refined_visibility.consistency_ratio + 1e-9);
    attempt->production_pose = attempt->production_kept_initial_pose
                                   ? attempt->initial_pose
                                   : attempt->match.T_target_source;
  };

  result.oracle_nearest_keyframe_id = findNearestKeyframe(oracle_pose);
  RegistrationSeedProbeAttempt oracle_attempt;
  oracle_attempt.seed_kind = "oracle_gt";
  oracle_attempt.seed_keyframe_id = result.oracle_nearest_keyframe_id;
  oracle_attempt.initial_pose = oracle_pose;
  oracle_attempt.match =
      matcher_.alignPrepared(target, prepared_query, oracle_pose);
  finish_attempt(&oracle_attempt);
  result.attempts.push_back(std::move(oracle_attempt));

  const auto candidates = place_index_.search(query_cloud);
  if (candidates.empty()) {
    result.error = "no_descriptor_candidates";
    result.valid = true;
    return result;
  }
  const auto &candidate = candidates.front();
  result.descriptor_candidate_keyframe_id = candidate.match_id;
  const auto keyframe = keyframe_manager_.getKeyframe(candidate.match_id);
  if (!keyframe) {
    result.error = "descriptor_keyframe_missing";
    result.valid = true;
    return result;
  }

  const auto yaw_hypotheses =
      buildDescriptorYawHypotheses(candidate, config_);
  for (const double yaw : yaw_hypotheses) {
    RegistrationSeedProbeAttempt attempt;
    attempt.seed_kind = "descriptor";
    attempt.seed_keyframe_id = candidate.match_id;
    attempt.yaw_offset_rad = yaw;
    attempt.initial_pose = keyframe->pose_optimized;
    attempt.initial_pose.linear() =
        keyframe->pose_optimized.linear() *
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    attempt.match =
        matcher_.alignPrepared(target, prepared_query, attempt.initial_pose);
    finish_attempt(&attempt);
    result.attempts.push_back(std::move(attempt));
  }
  result.valid = true;
  return result;
}

WorldLocalizing::RelocMatchQuality
WorldLocalizing::evaluateRelocMatchQuality(const MatchResult &match) const {
  RelocMatchQuality quality;
  const bool finite_metrics =
      std::isfinite(match.fitness_score) && std::isfinite(match.inlier_ratio);
  quality.fitness_pass =
      finite_metrics &&
      match.fitness_score < config_.gicp_fitness_threshold;
  quality.inlier_pass =
      finite_metrics &&
      match.inlier_ratio >= config_.reloc_min_inlier_ratio;
  const double scale = std::max(1e-6, config_.gicp_fitness_threshold * 0.5);
  quality.confidence = std::exp(-match.fitness_score / scale);
  quality.confidence = std::clamp(quality.confidence, 0.0, 1.0);
  quality.confidence_pass = quality.confidence >= config_.reloc_min_confidence;
  quality.accepted = hasValidRegistrationResult(match) &&
                     quality.fitness_pass &&
                     quality.inlier_pass && quality.confidence_pass;
  return quality;
}

void WorldLocalizing::rebuildFreeSpaceGridIfNeeded() {
  const KeyframeMapRevision current_revision = keyframe_manager_.revision();
  if (free_space_grid_revision_ != current_revision) {
    free_space_grid_ = FreeSpaceGrid();
    free_space_grid_keyframes_ = 0;
    free_space_grid_failed_ = false;
    free_space_grid_revision_ = current_revision;
  }
  if (free_space_grid_failed_)
    return;
  const size_t current_size = keyframe_manager_.size();
  if (free_space_grid_.valid())
    return;
  rebuildRelocMapCacheIfNeeded();
  PointCloudT::Ptr grid_source = reloc_map_cache_;
  if (!config_.reloc_free_space_map_pcd.empty()) {
    PointCloudT::Ptr dense(new PointCloudT);
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(config_.reloc_free_space_map_pcd,
                                             *dense) == 0 &&
        !dense->empty()) {
      grid_source = dense;
      LOG(INFO) << "[Reloc/FreeSpace] grid source overridden by "
                << config_.reloc_free_space_map_pcd
                << " points=" << dense->size();
    } else {
      LOG(WARNING) << "[Reloc/FreeSpace] failed to load "
                   << config_.reloc_free_space_map_pcd;
    }
  }
  if (!grid_source || grid_source->empty()) {
    free_space_grid_failed_ = true;
    return;
  }
  std::vector<Eigen::Vector3d> origins;
  origins.reserve(keyframe_manager_.size());
  for (const auto &kf : keyframe_manager_.getAllKeyframes()) {
    if (!kf || !kf->isValid())
      continue;
    origins.push_back(kf->pose_optimized.translation());
  }
  if (origins.empty()) {
    free_space_grid_failed_ = true;
    return;
  }
  const bool ok = free_space_grid_.build(
      *grid_source, origins, config_.reloc_free_space_resolution,
      config_.reloc_free_space_max_ray_length,
      config_.reloc_free_space_occupied_min_points);
  if (!ok) {
    free_space_grid_failed_ = true;
    return;
  }
  free_space_grid_keyframes_ = current_size;
  LOG(INFO) << "[Reloc/FreeSpace] grid built res="
            << free_space_grid_.resolution()
            << " occupied=" << free_space_grid_.occupiedCells()
            << " free=" << free_space_grid_.freeCells() << " in "
            << free_space_grid_.buildSeconds() << " s";
}

// Kills hypotheses whose observed cloud is contradicted by the map's own free
// space by more than the sampling noise of the measurement. This deliberately
// does not touch the ranking key or the acceptance gate: A0 showed that
// replacing the ranking key makes the gate margin measure something the
// ranking no longer orders, and the gate stops working. Here the existing
// visibility ranking and the existing gate run unchanged, on survivors.
const WorldLocalizing::RelocHypothesis *
WorldLocalizing::freeSpaceBestHypothesis(
    const PointCloudT::Ptr &query_cloud, const Eigen::Isometry3d &odom_pose) {
  if (!config_.reloc_free_space_enable || !query_cloud || query_cloud->empty())
    return nullptr;
  rebuildFreeSpaceGridIfNeeded();
  if (!free_space_grid_.valid())
    return nullptr;
  // A grid with no free cells has nothing to say about which hypothesis sits in
  // free space, so whichever it names would be an artefact of ranking noise.
  // Vetoing on that refuses a lock over no evidence at all -- and this is
  // reachable: a map with a single keyframe builds 28 occupied cells and zero
  // free ones.
  if (free_space_grid_.freeCells() == 0)
    return nullptr;
  const RelocHypothesis *best = nullptr;
  double best_value = 0.0;
  for (const auto &hyp : hypothesis_manager_.hypotheses()) {
    if (!hyp.alive)
      continue;
    const auto s = free_space_grid_.score(*query_cloud, hyp.T_map_odom * odom_pose);
    if (!s.valid)
      continue;
    VLOG(1) << "[Reloc/FreeSpaceCov] seed=" << hyp.seed_match_id
            << " value=" << s.value << " known_frac=" << s.known_fraction
            << " occ|known=" << s.occupied_given_known
            << " free|known=" << s.free_given_known
            << " n=" << s.scored_points << " known=" << s.known_points;
    if (!best || s.value > best_value) {
      best = &hyp;
      best_value = s.value;
    }
  }
  return best;
}

void WorldLocalizing::killFreeSpaceDominatedHypotheses(
    const PointCloudT::Ptr &query_cloud, const Eigen::Isometry3d &odom_pose) {
  if (!config_.reloc_free_space_enable || !query_cloud || query_cloud->empty())
    return;
  auto &hypotheses = hypothesis_manager_.hypotheses();
  size_t alive = 0;
  for (const auto &hyp : hypotheses)
    if (hyp.alive)
      ++alive;
  if (alive < 2)
    return;
  rebuildFreeSpaceGridIfNeeded();
  if (!free_space_grid_.valid())
    return;

  std::vector<FreeSpaceGrid::Score> scores(hypotheses.size());
  int best = -1;
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    const auto &hyp = hypotheses[i];
    if (!hyp.alive)
      continue;
    scores[i] =
        free_space_grid_.score(*query_cloud, hyp.T_map_odom * odom_pose);
    if (!scores[i].valid)
      continue;
    if (best < 0 || scores[i].value > scores[best].value)
      best = static_cast<int>(i);
  }
  if (best < 0)
    return;

  for (size_t i = 0; i < hypotheses.size(); ++i) {
    if (!hypotheses[i].alive)
      continue;
    const Eigen::Isometry3d hp = hypotheses[i].T_map_odom * odom_pose;
    VLOG(1) << "[Reloc/FreeSpaceDump] i=" << i
            << " seed=" << hypotheses[i].seed_match_id
            << " conv=" << hypotheses[i].converged_updates
            << " value=" << scores[i].value << " occ=" << scores[i].occupied_fraction
            << " free=" << scores[i].free_fraction << " n=" << scores[i].scored_points
            << " sd=" << scores[i].stddev << " xyz=" << hp.translation().x() << ","
            << hp.translation().y() << "," << hp.translation().z();
  }

  // The separation test is the sampling noise itself, not a tuned threshold:
  // two hypotheses are only distinguishable when their scores differ by more
  // than the binomial deviation of the difference.
  const double sigmas = config_.reloc_free_space_kill_sigmas;
  const auto &top = scores[static_cast<size_t>(best)];
  size_t killed = 0;
  for (size_t i = 0; i < hypotheses.size(); ++i) {
    auto &hyp = hypotheses[i];
    if (!hyp.alive || static_cast<int>(i) == best || !scores[i].valid)
      continue;
    const double noise = std::sqrt(top.stddev * top.stddev +
                                   scores[i].stddev * scores[i].stddev);
    if (top.value - scores[i].value > sigmas * noise) {
      hyp.alive = false;
      ++killed;
    }
  }
  if (killed > 0) {
    VLOG(1) << "[Reloc/FreeSpace] killed " << killed << " of " << alive
            << " hypotheses, best value=" << top.value
            << " occ=" << top.occupied_fraction
            << " free=" << top.free_fraction << " n=" << top.scored_points;
  }
}

void WorldLocalizing::rebuildRelocMapCacheIfNeeded() {
  const KeyframeMapRevision current_revision = keyframe_manager_.revision();
  if (localization_atlas_ && localization_atlas_->loaded()) {
    const auto &atlas_map = localization_atlas_->globalMap();
    if (reloc_map_cache_ != atlas_map) {
      reloc_map_cache_ = atlas_map;
      reloc_map_cached_keyframes_ = keyframe_manager_.size();
    }
    reloc_map_revision_ = current_revision;
    return;
  }
  const std::size_t current_size = keyframe_manager_.size();
  if (reloc_map_revision_ == current_revision) {
    return;
  }

  reloc_map_cache_ = LocalizationAtlas::buildGlobalMap(
      config_, keyframe_manager_.getAllKeyframes());
  reloc_map_cached_keyframes_ = current_size;
  reloc_map_revision_ = current_revision;
}

WorldLocalizing::PointCloudT::Ptr
WorldLocalizing::buildRelocTargetCloud(int64_t center_id) {
  rebuildRelocMapCacheIfNeeded();
  const auto center_keyframe = keyframe_manager_.getKeyframe(center_id);
  if (!center_keyframe)
    return pcl::make_shared<PointCloudT>();
  return LocalizationAtlas::cropGlobalMap(
      config_, reloc_map_cache_, center_keyframe->pose_optimized.translation());
}

void WorldLocalizing::rebuildLoadedMapVisibilityCacheIfNeeded() {
  const KeyframeMapRevision current_revision = keyframe_manager_.revision();
  std::vector<Keyframe::Ptr> loaded_keyframes;
  for (const auto &keyframe : keyframe_manager_.getAllKeyframes()) {
    if (keyframe && keyframe->is_from_loaded_map) {
      loaded_keyframes.push_back(keyframe);
    }
  }
  if (loaded_map_visibility_cache_ &&
      !loaded_map_visibility_cache_->empty() &&
      sameGenerationAndPose(loaded_map_visibility_revision_,
                            current_revision) &&
      loaded_map_visibility_cached_keyframes_ == loaded_keyframes.size()) {
    return;
  }
  loaded_map_visibility_cache_ =
      LocalizationAtlas::buildGlobalMap(config_, loaded_keyframes);
  loaded_map_visibility_cached_keyframes_ = loaded_keyframes.size();
  loaded_map_visibility_revision_ = current_revision;
}

VisibilityConsistencyResult
WorldLocalizing::evaluateLoadedMapPoseVisibility(
    const PointCloudT::Ptr &query_cloud,
    const Eigen::Isometry3d &T_map_lidar) {
  if (!query_cloud || query_cloud->empty() ||
      !T_map_lidar.matrix().allFinite()) {
    return {};
  }
  std::lock_guard<std::mutex> lock(mutex_);
  rebuildLoadedMapVisibilityCacheIfNeeded();
  auto target = LocalizationAtlas::cropGlobalMap(
      config_, loaded_map_visibility_cache_, T_map_lidar.translation());
  return evaluatePoseVisibility(target, query_cloud, T_map_lidar);
}

VisibilityConsistencyResult WorldLocalizing::evaluatePoseVisibility(
    const PointCloudT::Ptr &target_cloud, const PointCloudT::Ptr &query_cloud,
    const Eigen::Isometry3d &T_map_lidar) const {
  if (!target_cloud || !query_cloud)
    return {};
  return evaluateVisibilityConsistency(*target_cloud, *query_cloud, T_map_lidar,
                                       visibilityOptionsFromConfig(config_));
}

double WorldLocalizing::computeRelocLogLikelihood(
    const LoopCandidate &candidate, const MatchResult &match_result) const {
  if (!hasValidRegistrationResult(match_result))
    return -config_.reloc_hypothesis_not_converged_penalty;
  const double fit_scale = std::max(1e-6, config_.gicp_fitness_threshold);
  const double inlier_term = config_.reloc_reloc_inlier_weight *
                             std::max(0.0, match_result.inlier_ratio);
  const double fitness_term = -match_result.fitness_score / fit_scale;
  const double desc_distance = std::isfinite(candidate.rhpd_distance)
                                   ? candidate.rhpd_distance
                                   : candidate.sc_distance;
  const double desc_term =
      -config_.reloc_reloc_desc_dist_weight * desc_distance;
  return fitness_term + inlier_term + desc_term;
}

double WorldLocalizing::computeTrackLogLikelihood(
    const MatchResult &match_result,
    const Eigen::Isometry3d &predicted_pose) const {
  if (!hasValidRegistrationResult(match_result) ||
      !isFiniteRigidPose(predicted_pose))
    return -config_.reloc_hypothesis_not_converged_penalty;

  const double fit_scale = std::max(1e-6, config_.gicp_fitness_threshold);
  const double fitness_term = -match_result.fitness_score / fit_scale;
  const double inlier_term = config_.reloc_reloc_inlier_weight *
                             std::max(0.0, match_result.inlier_ratio);

  const Eigen::Isometry3d delta =
      predicted_pose.inverse() * match_result.T_target_source;
  const double delta_t = delta.translation().norm();
  const double motion_term = -config_.reloc_track_motion_weight * delta_t;

  return fitness_term + inlier_term + motion_term;
}

int64_t
WorldLocalizing::findNearestKeyframe(const Eigen::Isometry3d &pose) const {
  int64_t nearest_id = -1;
  double min_distance = std::numeric_limits<double>::max();

  for (const auto &kf : keyframe_manager_.getAllKeyframes()) {
    if (!kf)
      continue;
    double distance =
        (kf->pose_optimized.translation() - pose.translation()).norm();
    if (distance < config_.reloc_search_radius && distance < min_distance) {
      min_distance = distance;
      nearest_id = kf->id;
    }
  }

  return nearest_id;
}

int64_t
WorldLocalizing::findNearestLoadedKeyframe(
    const Eigen::Isometry3d &pose) const {
  int64_t nearest_id = -1;
  double min_distance = std::numeric_limits<double>::max();

  for (const auto &kf : keyframe_manager_.getAllKeyframes()) {
    if (!kf || !kf->is_from_loaded_map)
      continue;
    const double distance =
        (kf->pose_optimized.translation() - pose.translation()).norm();
    if (distance < config_.reloc_search_radius && distance < min_distance) {
      min_distance = distance;
      nearest_id = kf->id;
    }
  }
  return nearest_id;
}

} // namespace n3mapping
