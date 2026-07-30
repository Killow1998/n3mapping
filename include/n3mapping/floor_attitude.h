// floor_attitude.h - the floor under a scan, as an absolute attitude observation.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

// Every pose in the graph is currently tied only to its neighbours: of the 258
// edges the 0723 session produced, 256 are odometry and 2 are loops, and not one
// of them says which way is up. So when the estimator's world frame tilts away
// from gravity -- measured at 2.4-5.4 deg early in that session and 1.6-1.9 deg
// late -- nothing downstream can notice, and the height error it produces
// (2.77 m between two keyframes 0.46 m apart) is permanent.
//
// The floor is the observation that closes that gap. It is level, the robot
// drives on it, and it is fittable in 249 of the session's 257 keyframes. Its
// normal in the body frame is an absolute measurement of roll and pitch, leaving
// yaw untouched, which is exactly the pair of degrees of freedom that drifts.
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
