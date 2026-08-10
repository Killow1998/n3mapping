#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/keyframe_manager.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping {

struct SpatialKeyframeMatch {
  int64_t keyframe_id = -1;
  double squared_distance = 0.0;
};

struct KeyframeSpatialSearchResult {
  std::vector<SpatialKeyframeMatch> matches;
  KeyframeMapRevision map_revision;
  bool index_rebuilt = false;
  double index_build_ms = 0.0;
  double query_ms = 0.0;
};

class KeyframeSpatialIndex {
public:
  KeyframeSpatialSearchResult
  radiusSearch(const std::vector<Keyframe::Ptr> &keyframes,
               const KeyframeMapRevision &map_revision,
               const Eigen::Vector3d &center, double radius, int max_keyframes);

private:
  void rebuildLocked(const std::vector<Keyframe::Ptr> &keyframes,
                     const KeyframeMapRevision &map_revision, double *build_ms);

  std::mutex mutex_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr positions_ =
      pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::KdTreeFLANN<pcl::PointXYZ> tree_;
  std::vector<int64_t> keyframe_ids_;
  KeyframeMapRevision indexed_revision_;
  bool have_indexed_revision_ = false;
};

struct LocalMapSelectionRequest {
  Eigen::Isometry3d predicted_pose = Eigen::Isometry3d::Identity();
  int64_t anchor_id = -1;
  double spatial_radius = 0.0;
  int max_spatial_keyframes = 0;
  int recent_tail_count = 0;
  KeyframeMapRevision map_revision;
};

struct LocalMapSelection {
  std::vector<int64_t> keyframe_ids;
  std::string reason;
  KeyframeMapRevision map_revision;
  bool requested_revision_matched = false;
  bool index_rebuilt = false;
  double index_build_ms = 0.0;
  double query_ms = 0.0;
  std::size_t selected_points = 0;
  std::size_t loaded_keyframes = 0;
  std::size_t current_keyframes = 0;
  double z_min = 0.0;
  double z_max = 0.0;
};

class LocalMapSelector {
public:
  explicit LocalMapSelector(KeyframeManager &keyframe_manager)
      : keyframe_manager_(keyframe_manager) {}

  LocalMapSelection select(const LocalMapSelectionRequest &request);

private:
  KeyframeManager &keyframe_manager_;
  KeyframeSpatialIndex spatial_index_;
};

} // namespace n3mapping
