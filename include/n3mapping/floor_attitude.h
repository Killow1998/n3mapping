// floor_attitude.h - the floor under a scan, as an absolute attitude observation.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

// Assumes a level floor when using its normal as absolute roll/pitch evidence.
// Yaw remains unconstrained.
struct FloorNormalObservation {
  bool valid = false;
  // Unit normal in the sensor/body frame, pointing away from the floor.
  Eigen::Vector3d normal_body = Eigen::Vector3d::UnitZ();
  int support_points = 0;
  // Smallest singular value over the middle one: how plane-like the fit is.
  double planarity = 1.0;
};

struct FloorNormalOptions {
  // Only the patch the robot stands on. Distant floor returns arrive at grazing
  // incidence and bend the fit.
  double max_radius_m = 8.0;
  // Depth of the band above the lowest returns that counts as floor.
  double band_m = 0.35;
  double band_below_m = 0.10;
  int min_points = 400;
  // Above this the patch is not a plane and the normal means nothing.
  double max_planarity = 0.20;
  // A floor cannot be steep. Rejecting implausible fits here keeps a wall or a
  // ramp from being handed to the optimiser as if it were level ground.
  double max_tilt_from_sensor_up_deg = 35.0;
};

// Fits the floor under one scan. The cloud is in the sensor frame; the pose is
// used only to know which way the sensor thinks down is, so the fit stays an
// independent observation rather than an echo of the pose it will constrain.
FloorNormalObservation
estimateFloorNormal(const pcl::PointCloud<pcl::PointXYZI> &cloud_in_sensor,
                    const Eigen::Isometry3d &T_world_sensor,
                    const FloorNormalOptions &options = {});

} // namespace n3mapping
