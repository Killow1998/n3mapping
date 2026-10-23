#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include "n3mapping/async_log.h"
#include <limits>
#include <utility>
#include <glog/logging.h>

namespace n3mapping {
namespace {

std::vector<core::AnchoredDenseTrajectorySample> anchorDenseTrajectory(
    const std::vector<core::DenseTrajectoryPose> &dense_optimized,
    const std::vector<Keyframe::Ptr> &keyframes) {
  std::vector<core::AnchoredDenseTrajectorySample> samples;
  samples.reserve(dense_optimized.size());

  for (const auto &dense_pose : dense_optimized) {
    if (dense_pose.correction_source) {
      samples.push_back(*dense_pose.correction_source);
      continue;
    }
    Keyframe::Ptr anchor;
    double best_time = -std::numeric_limits<double>::infinity();
    for (const auto &kf : keyframes) {
      if (kf && kf->timestamp <= dense_pose.timestamp &&
          kf->timestamp >= best_time) {
        anchor = kf;
        best_time = kf->timestamp;
      }
    }
    if (!anchor && !keyframes.empty()) {
      anchor = keyframes.front();
    }

    core::AnchoredDenseTrajectorySample sample;
    sample.seq = dense_pose.seq;
    sample.timestamp = dense_pose.timestamp;
    sample.pose_world_lidar_raw = dense_pose.pose_world_lidar;
    if (anchor) {
      sample.anchor_keyframe_id = anchor->id;
      sample.anchor_pose_world_lidar_raw = anchor->pose_optimized;
      sample.has_anchor = true;
      sample.use_bracketing_correction = false;
    }
    samples.push_back(std::move(sample));
  }
  return samples;
}

} // namespace

bool N3MappingCore::loadMap(const std::string &map_path) {
  LogOperation operation("map_load", map_path);
  try {
    auto candidate = std::make_unique<core::N3MappingSession>(config_);
    std::vector<core::DenseTrajectoryPose> loaded_dense_optimized;
    core::DenseTrajectoryMetadata loaded_dense_metadata;
    if (!candidate->mapSerializer().loadMap(
            map_path, candidate->keyframeManager(), candidate->loopDetector(),
            candidate->graphOptimizer(), &loaded_dense_optimized,
            &loaded_dense_metadata, &candidate->submapBuilder())) {
      return false;
    }
    if (!candidate->mappingResuming().initializeFromLoadedMap()) {
      return false;
    }

    std::string atlas_error;
    if (!candidate->worldLocalizing().loadLocalizationAtlas(map_path,
                                                            &atlas_error)) {
      LOG(ERROR) << "Failed to load localization atlas: " << atlas_error << '\n';
      return false;
    }

    auto loaded_dense_samples = anchorDenseTrajectory(
        loaded_dense_optimized,
        candidate->keyframeManager().getAllKeyframes());

    // Every fallible map-specific step completed against the candidate. The
    // session owns all cross-referenced components, so one pointer swap is the
    // transaction commit and cannot leave mixed revisions behind.
    session_.swap(candidate);
    dense_trajectory_samples_.swap(loaded_dense_samples);
    dense_trajectory_metadata_ = std::move(loaded_dense_metadata);
    {
      std::lock_guard<std::mutex> lock(loop_queue_mutex_);
      loop_detection_queue_.clear();
    }
    last_loop_check_id_ = -1000;
    loop_count_ = 0;
    floor_attitude_accepted_ = 0;
    floor_attitude_rejected_ = 0;
    localization_source_frame_id_.clear();
    map_loaded_ = true;
    operation.succeed();
    return true;
  } catch (const std::exception &error) {
    LOG(ERROR) << "Failed to prepare map transaction: " << error.what() << '\n';
    return false;
  }
}


} // namespace n3mapping
