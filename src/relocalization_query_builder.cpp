#include "n3mapping/relocalization_query_builder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <glog/logging.h>
#include <pcl/common/transforms.h>

#include "n3mapping/cloud_utils.h"

namespace n3mapping {

RelocalizationQueryBuilder::RelocalizationQueryBuilder(const Config &config)
    : config_(config) {}

PlaceObservation RelocalizationQueryBuilder::singleScanObservation(
    const QueryFrame &frame) const {
  PlaceObservation observation;
  observation.scale = ObservationScale::SINGLE_SCAN;
  observation.support_begin = frame.sequence;
  observation.support_end = frame.sequence;
  observation.frame_count = 1;
  observation.cloud = frame.cloud;
  observation.raw_points = frame.cloud ? frame.cloud->size() : 0;
  observation.downsampled_points = observation.raw_points;
  return observation;
}

PlaceObservation RelocalizationQueryBuilder::buildStationary(
    const PointCloudT::Ptr &cloud, const Eigen::Isometry3d &odom_pose) {
  QueryFrame frame;
  frame.sequence = next_sequence_++;
  frame.cloud = pcl::make_shared<PointCloudT>(*cloud);
  frame.odom_pose = odom_pose;
  frames_.push_back(std::move(frame));

  const int max_frames = std::max(1, config_.reloc_static_agg_max_frames);
  while (static_cast<int>(frames_.size()) > max_frames) {
    frames_.pop_front();
  }

  const PlaceObservation current = singleScanObservation(frames_.back());
  if (!config_.reloc_static_agg_enable || max_frames <= 1) {
    return current;
  }

  std::vector<const QueryFrame *> selected;
  selected.reserve(max_frames);
  for (auto it = frames_.rbegin();
       it != frames_.rend() && static_cast<int>(selected.size()) < max_frames;
       ++it) {
    const Eigen::Isometry3d delta = odom_pose.inverse() * it->odom_pose;
    const double delta_t = delta.translation().norm();
    const double delta_r = Eigen::AngleAxisd(delta.rotation()).angle();
    if (delta_t <= config_.reloc_static_agg_max_translation &&
        delta_r <= config_.reloc_static_agg_max_rotation) {
      selected.push_back(&(*it));
    } else {
      break;
    }
  }

  if (static_cast<int>(selected.size()) <
      std::max(1, config_.reloc_static_agg_min_frames)) {
    return current;
  }

  double max_dr = 0.0;
  double max_dt = 0.0;
  for (const QueryFrame *source : selected) {
    const Eigen::Isometry3d delta = odom_pose.inverse() * source->odom_pose;
    max_dt = std::max(max_dt, delta.translation().norm());
    max_dr = std::max(max_dr, Eigen::AngleAxisd(delta.rotation()).angle());
  }
  VLOG(1) << "[Reloc/QueryAgg] frames=" << selected.size()
          << " max_dt_m=" << max_dt << " max_dr_deg=" << max_dr * 180.0 / M_PI;

  auto merged = pcl::make_shared<PointCloudT>();
  std::size_t raw_points = 0;
  for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
    const QueryFrame *source = *it;
    raw_points += source->cloud->size();
    const Eigen::Matrix4f current_from_source =
        (odom_pose.inverse() * source->odom_pose).matrix().cast<float>();
    PointCloudT transformed;
    pcl::transformPointCloud(*source->cloud, transformed, current_from_source);
    *merged += transformed;
  }

  if (config_.reloc_static_agg_voxel_size > 1e-4 && !merged->empty()) {
    PointCloudT::Ptr downsampled;
    if (safeVoxelGridFilter<pcl::PointXYZI>(
            merged, config_.reloc_static_agg_voxel_size, &downsampled) &&
        downsampled) {
      *merged = *downsampled;
    }
  }

  PlaceObservation observation;
  observation.scale = ObservationScale::STATIONARY_AGGREGATE;
  observation.support_begin = selected.back()->sequence;
  observation.support_end = selected.front()->sequence;
  observation.frame_count = static_cast<int>(selected.size());
  observation.cloud = merged->empty() ? current.cloud : merged;
  const Eigen::Isometry3d oldest_delta =
      odom_pose.inverse() * selected.back()->odom_pose;
  observation.motion_translation = oldest_delta.translation().norm();
  observation.motion_rotation =
      Eigen::AngleAxisd(oldest_delta.rotation()).angle();
  observation.raw_points = raw_points;
  observation.downsampled_points = observation.cloud->size();

  VLOG(1) << "[Reloc/Aggregation] static_frames=" << selected.size()
          << " raw_points=" << raw_points
          << " aggregated_points=" << merged->size()
          << " motion_gate(t<=" << config_.reloc_static_agg_max_translation
          << ", r<=" << config_.reloc_static_agg_max_rotation << ")";
  return observation;
}

PlaceObservation RelocalizationQueryBuilder::buildMotionSubmap(
    const Eigen::Isometry3d &odom_pose) const {
  PlaceObservation observation;
  observation.scale = ObservationScale::MOTION_SUBMAP;
  observation.motion_translation = std::numeric_limits<double>::quiet_NaN();
  observation.motion_rotation = std::numeric_limits<double>::quiet_NaN();
  if (frames_.empty()) {
    return observation;
  }

  const int max_frames = std::max(1, config_.reloc_static_agg_max_frames);
  std::vector<const QueryFrame *> selected;
  selected.reserve(max_frames);
  for (auto it = frames_.rbegin();
       it != frames_.rend() && static_cast<int>(selected.size()) < max_frames;
       ++it) {
    selected.push_back(&(*it));
  }

  auto merged = pcl::make_shared<PointCloudT>();
  std::size_t raw_points = 0;
  for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
    const QueryFrame *source = *it;
    if (!source || !source->cloud) {
      continue;
    }
    raw_points += source->cloud->size();
    const Eigen::Matrix4f current_from_source =
        (odom_pose.inverse() * source->odom_pose).matrix().cast<float>();
    PointCloudT transformed;
    pcl::transformPointCloud(*source->cloud, transformed, current_from_source);
    *merged += transformed;
  }

  if (config_.reloc_static_agg_voxel_size > 1e-4 && !merged->empty()) {
    PointCloudT::Ptr downsampled;
    if (safeVoxelGridFilter<pcl::PointXYZI>(
            merged, config_.reloc_static_agg_voxel_size, &downsampled) &&
        downsampled) {
      *merged = *downsampled;
    }
  }

  const std::size_t point_budget =
      frames_.back().cloud
          ? std::max<std::size_t>(1, frames_.back().cloud->size())
          : merged->size();
  double adaptive_voxel = std::max(1e-3, config_.reloc_static_agg_voxel_size);
  for (int iteration = 0; iteration < 6 && merged->size() > point_budget;
       ++iteration) {
    const double ratio =
        static_cast<double>(merged->size()) / static_cast<double>(point_budget);
    adaptive_voxel *= std::max(1.1, 1.05 * std::cbrt(ratio));
    PointCloudT::Ptr downsampled;
    if (!safeVoxelGridFilter<pcl::PointXYZI>(merged, adaptive_voxel,
                                             &downsampled) ||
        !downsampled || downsampled->size() >= merged->size()) {
      break;
    }
    *merged = *downsampled;
  }

  observation.support_begin = selected.back()->sequence;
  observation.support_end = selected.front()->sequence;
  observation.frame_count = static_cast<int>(selected.size());
  observation.cloud = merged;
  observation.raw_points = raw_points;
  observation.downsampled_points = merged->size();
  const Eigen::Isometry3d delta =
      odom_pose.inverse() * selected.back()->odom_pose;
  observation.motion_translation = delta.translation().norm();
  observation.motion_rotation = Eigen::AngleAxisd(delta.rotation()).angle();
  return observation;
}

void RelocalizationQueryBuilder::reset() {
  frames_.clear();
  next_sequence_ = 0;
}

} // namespace n3mapping
