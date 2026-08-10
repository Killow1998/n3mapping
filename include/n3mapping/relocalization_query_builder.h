// Builds bounded relocalization observations from the recent odometry-aligned
// scan history. This module owns query-history lifetime only; it does not
// search candidates, register poses, rank hypotheses, or make lock decisions.
#pragma once

#include <cstdint>
#include <deque>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"

namespace n3mapping {

enum class ObservationScale {
  SINGLE_SCAN,
  STATIONARY_AGGREGATE,
  MOTION_SUBMAP,
  CAUSAL_SUBMAP,
  CLOSED_SUBMAP,
};

struct PlaceObservation {
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  ObservationScale scale = ObservationScale::SINGLE_SCAN;
  int64_t support_begin = -1;
  int64_t support_end = -1;
  int frame_count = 0;
  PointCloudT::Ptr cloud = pcl::make_shared<PointCloudT>();
  double motion_translation = 0.0;
  double motion_rotation = 0.0;
  std::size_t raw_points = 0;
  std::size_t downsampled_points = 0;
};

class RelocalizationQueryBuilder {
public:
  using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

  explicit RelocalizationQueryBuilder(const Config &config);

  PlaceObservation buildStationary(const PointCloudT::Ptr &cloud,
                                   const Eigen::Isometry3d &odom_pose);
  PlaceObservation buildMotionSubmap(const Eigen::Isometry3d &odom_pose) const;
  void reset();

private:
  struct QueryFrame {
    int64_t sequence = -1;
    PointCloudT::Ptr cloud;
    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
  };

  PlaceObservation singleScanObservation(const QueryFrame &frame) const;

  Config config_;
  std::deque<QueryFrame> frames_;
  int64_t next_sequence_ = 0;
};

} // namespace n3mapping
