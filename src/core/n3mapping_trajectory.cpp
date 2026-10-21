// Dense trajectory correction and checkpoint provenance.
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/n3mapping_session.h"
#include <algorithm>
#include <cmath>

namespace n3mapping {
void N3MappingCore::appendDenseTrajectorySample(
    double timestamp, const Eigen::Isometry3d &raw_pose,
    int64_t anchor_keyframe_id, const Eigen::Isometry3d &anchor_raw_pose,
    bool use_bracketing_correction) {
  core::AnchoredDenseTrajectorySample sample;
  sample.seq = static_cast<uint64_t>(dense_trajectory_samples_.size());
  sample.timestamp = timestamp;
  sample.pose_world_lidar_raw = raw_pose;
  sample.anchor_keyframe_id = anchor_keyframe_id;
  sample.anchor_pose_world_lidar_raw = anchor_raw_pose;
  sample.has_anchor = anchor_keyframe_id >= 0;
  sample.use_bracketing_correction = use_bracketing_correction;
  if (dense_trajectory_metadata_.source == "keyframe_fallback") {
    dense_trajectory_metadata_.source = "mixed_keyframe_fallback_and_high_rate";
    dense_trajectory_metadata_.degraded = true;
  } else if (dense_trajectory_metadata_.source.empty() ||
             dense_trajectory_metadata_.source == "none") {
    dense_trajectory_metadata_.source = "native";
    dense_trajectory_metadata_.degraded = false;
  }
  dense_trajectory_samples_.push_back(sample);
}

void N3MappingCore::appendDenseTrajectorySampleWithLatestAnchor(
    double timestamp, const Eigen::Isometry3d &raw_pose,
    bool use_bracketing_correction) {
  auto latest = session_->keyframeManager().getLatestKeyframe();
  if (!latest) {
    appendDenseTrajectorySample(timestamp, raw_pose, -1,
                                Eigen::Isometry3d::Identity(),
                                use_bracketing_correction);
    return;
  }

  const Eigen::Isometry3d anchor_raw_pose =
      latest->is_from_loaded_map ? latest->pose_optimized : latest->pose_odom;
  appendDenseTrajectorySample(timestamp, raw_pose, latest->id, anchor_raw_pose,
                              use_bracketing_correction);
}

std::vector<core::DenseTrajectoryPose>
N3MappingCore::buildDenseOptimizedTrajectory() const {
  std::vector<core::DenseTrajectoryPose> dense_optimized;
  dense_optimized.reserve(dense_trajectory_samples_.size());
  std::map<MapSessionId, std::vector<Keyframe::Ptr>> time_indices;
  if (std::any_of(dense_trajectory_samples_.begin(), dense_trajectory_samples_.end(),
                  [](const auto& sample) { return sample.use_bracketing_correction; })) {
    for (const auto& kf : session_->keyframeManager().getAllKeyframes())
      if (kf && std::isfinite(kf->timestamp)) time_indices[kf->session_id].push_back(kf);
    for (auto& [session, index] : time_indices) {
      std::stable_sort(index.begin(), index.end(),
          [](const auto& lhs, const auto& rhs) { return lhs->timestamp < rhs->timestamp; });
      index.erase(std::unique(index.begin(), index.end(),
          [](const auto& lhs, const auto& rhs) { return lhs->timestamp == rhs->timestamp; }), index.end());
    }
  }
  for (const auto &sample : dense_trajectory_samples_) {
    core::DenseTrajectoryPose pose;
    pose.seq = sample.seq;
    pose.timestamp = sample.timestamp;
    pose.pose_world_lidar = sample.pose_world_lidar_raw;
    pose.correction_source = sample;
    if (sample.use_bracketing_correction) {
      const auto anchor = session_->keyframeManager().getKeyframe(sample.anchor_keyframe_id);
      const auto index = anchor ? time_indices.find(anchor->session_id) : time_indices.begin();
      if (index != time_indices.end())
        pose.pose_world_lidar = interpolateDenseCorrection(sample.timestamp, index->second) *
                                sample.pose_world_lidar_raw;
    } else if (sample.has_anchor) {
      auto anchor =
          session_->keyframeManager().getKeyframe(sample.anchor_keyframe_id);
      if (anchor) {
        pose.pose_world_lidar = anchor->pose_optimized *
                                sample.anchor_pose_world_lidar_raw.inverse() *
                                sample.pose_world_lidar_raw;
      }
    }
    dense_optimized.push_back(pose);
  }
  return dense_optimized;
}

Eigen::Isometry3d
N3MappingCore::interpolateDenseCorrection(
    double timestamp, const std::vector<Keyframe::Ptr>& time_index) const {
  if (time_index.empty() || std::isnan(timestamp)) {
    return Eigen::Isometry3d::Identity();
  }
  const auto next = std::lower_bound(time_index.begin(), time_index.end(), timestamp,
      [](const auto& kf, double time) { return kf->timestamp < time; });
  const auto after = next == time_index.end() ? time_index.back() : *next;
  const auto before = next == time_index.begin() || after->timestamp == timestamp
                          ? after : *std::prev(next);

  const Eigen::Isometry3d correction_before =
      before->pose_optimized * before->pose_odom.inverse();
  const Eigen::Isometry3d correction_after =
      after->pose_optimized * after->pose_odom.inverse();
  double alpha = 0.0;
  const double dt = after->timestamp - before->timestamp;
  if (std::isfinite(dt) && dt > 1e-9) {
    alpha = std::clamp((timestamp - before->timestamp) / dt, 0.0, 1.0);
  }

  Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
  correction.translation() = (1.0 - alpha) * correction_before.translation() +
                             alpha * correction_after.translation();
  Eigen::Quaterniond qb(correction_before.rotation());
  Eigen::Quaterniond qa(correction_after.rotation());
  qb.normalize();
  qa.normalize();
  correction.linear() = qb.slerp(alpha, qa).normalized().toRotationMatrix();
  return correction;
}

} // namespace n3mapping
