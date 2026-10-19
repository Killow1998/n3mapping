#include "n3mapping/world_localizing.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <glog/logging.h>
#include <pcl/io/pcd_io.h>

namespace n3mapping {
WorldLocalizing::RelocMatchQuality
WorldLocalizing::evaluateRelocMatchQuality(const MatchResult &match) const {
  RelocMatchQuality quality;
  const bool finite_metrics =
      std::isfinite(match.fitness_score) && std::isfinite(match.inlier_ratio);
  quality.fitness_pass =
      finite_metrics && match.fitness_score < config_.gicp_fitness_threshold;
  quality.inlier_pass =
      finite_metrics && match.inlier_ratio >= config_.reloc_min_inlier_ratio;
  const double scale = std::max(1e-6, config_.gicp_fitness_threshold * 0.5);
  quality.confidence = std::exp(-match.fitness_score / scale);
  quality.confidence = std::clamp(quality.confidence, 0.0, 1.0);
  quality.confidence_pass = quality.confidence >= config_.reloc_min_confidence;
  quality.accepted = hasValidRegistrationResult(match) &&
                     quality.fitness_pass && quality.inlier_pass &&
                     quality.confidence_pass;
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
WorldLocalizing::freeSpaceBestHypothesis(const PointCloudT::Ptr &query_cloud,
                                         const Eigen::Isometry3d &odom_pose) {
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
    const auto s =
        free_space_grid_.score(*query_cloud, hyp.T_map_odom * odom_pose);
    if (!s.valid)
      continue;
    VLOG(1) << "[Reloc/FreeSpaceCov] seed=" << hyp.seed_match_id
            << " value=" << s.value << " known_frac=" << s.known_fraction
            << " occ|known=" << s.occupied_given_known
            << " free|known=" << s.free_given_known << " n=" << s.scored_points
            << " known=" << s.known_points;
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
            << " value=" << scores[i].value
            << " occ=" << scores[i].occupied_fraction
            << " free=" << scores[i].free_fraction
            << " n=" << scores[i].scored_points << " sd=" << scores[i].stddev
            << " xyz=" << hp.translation().x() << "," << hp.translation().y()
            << "," << hp.translation().z();
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
            << " occ=" << top.occupied_fraction << " free=" << top.free_fraction
            << " n=" << top.scored_points;
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

RelocTargetRequest WorldLocalizing::makeRelocTargetRequest(int64_t center_id) {
  const auto started = std::chrono::steady_clock::now();
  rebuildRelocMapCacheIfNeeded();
  RelocTargetRequest request;
  request.target_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  request.anchor_id = center_id;
  request.map_revision = keyframe_manager_.revision();
  if (const auto center_keyframe = keyframe_manager_.getKeyframe(center_id)) {
    request.visibility_from_atlas = localization_atlas_ && localization_atlas_->loaded() &&
        reloc_map_cache_ == localization_atlas_->globalMap();
    request.crop_center = center_keyframe->pose_optimized.translation();
    request.build_local_target = [this, map = reloc_map_cache_, center = request.crop_center]() {
      return LocalizationAtlas::cropGlobalMap(config_, map, center);
    };
  }
  return request;
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

int64_t WorldLocalizing::findNearestLoadedKeyframe(
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
