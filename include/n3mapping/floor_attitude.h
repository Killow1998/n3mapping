#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct FloorNormalObservation {
  bool valid = false;

  Eigen::Vector3d normal_body = Eigen::Vector3d::UnitZ();
  int support_points = 0;

  double planarity = 1.0;
};

struct FloorNormalOptions {

  double max_radius_m = 8.0;

  double band_m = 0.35;
  double band_below_m = 0.10;
  int min_points = 400;

  double max_planarity = 0.20;

  double max_tilt_from_sensor_up_deg = 35.0;
};

FloorNormalObservation
estimateFloorNormal(const pcl::PointCloud<pcl::PointXYZI> &cloud_in_sensor,
                    const Eigen::Isometry3d &T_world_sensor,
                    const FloorNormalOptions &options = {});

}
