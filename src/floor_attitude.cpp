// floor_attitude.cpp
#include "n3mapping/floor_attitude.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <vector>

namespace n3mapping {

FloorNormalObservation
estimateFloorNormal(const pcl::PointCloud<pcl::PointXYZI> &cloud_in_sensor,
                    const Eigen::Isometry3d &T_world_sensor,
                    const FloorNormalOptions &options) {
  FloorNormalObservation out;
  if (static_cast<int>(cloud_in_sensor.size()) < options.min_points)
    return out;

  // Which way the sensor currently believes down to be, expressed in the sensor
  // frame. Only used to pick out the floor returns; the fitted normal itself
  // comes from the geometry.
  const Eigen::Vector3d down_sensor =
      T_world_sensor.rotation().transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);

  // Height along that direction, and horizontal radius across it.
  std::vector<Eigen::Vector3d> near;
  std::vector<double> height;
  near.reserve(cloud_in_sensor.size());
  height.reserve(cloud_in_sensor.size());
  const double r2 = options.max_radius_m * options.max_radius_m;
  for (const auto &pt : cloud_in_sensor.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
      continue;
    const Eigen::Vector3d p(pt.x, pt.y, pt.z);
    const double h = -p.dot(down_sensor);
    const double horiz2 = p.squaredNorm() - h * h;
    if (horiz2 > r2)
      continue;
    near.push_back(p);
    height.push_back(h);
  }
  if (static_cast<int>(near.size()) < options.min_points)
    return out;

  std::vector<double> sorted = height;
  const std::size_t k = static_cast<std::size_t>(0.02 * (sorted.size() - 1));
  std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end());
  const double h_low = sorted[k];

  std::vector<Eigen::Vector3d> floor;
  floor.reserve(near.size());
  for (std::size_t i = 0; i < near.size(); ++i) {
    if (height[i] > h_low - options.band_below_m &&
        height[i] < h_low + options.band_m) {
      floor.push_back(near[i]);
    }
  }
  if (static_cast<int>(floor.size()) < options.min_points)
    return out;

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto &p : floor)
    centroid += p;
  centroid /= static_cast<double>(floor.size());

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto &p : floor) {
    const Eigen::Vector3d d = p - centroid;
    cov += d * d.transpose();
  }
  cov /= static_cast<double>(floor.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
  const Eigen::Vector3d evals = solver.eigenvalues();
  Eigen::Vector3d normal = solver.eigenvectors().col(0);
  const double planarity =
      std::sqrt(std::max(evals(0), 0.0)) /
      std::max(std::sqrt(std::max(evals(1), 0.0)), 1e-9);
  if (planarity > options.max_planarity)
    return out;

  // Point it away from the floor, i.e. against the direction the sensor calls
  // down.
  if (normal.dot(-down_sensor) < 0.0)
    normal = -normal;
  normal.normalize();

  const double tilt =
      std::acos(std::clamp(normal.dot(-down_sensor), -1.0, 1.0)) * 180.0 / M_PI;
  if (tilt > options.max_tilt_from_sensor_up_deg)
    return out;

  out.valid = true;
  out.normal_body = normal;
  out.support_points = static_cast<int>(floor.size());
  out.planarity = planarity;
  return out;
}

} // namespace n3mapping
