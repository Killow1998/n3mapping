#include "n3mapping/visibility_consistency.h"
#include "visibility_angular_bins.h"

#include <Eigen/Eigenvalues>
#include <pcl/common/point_tests.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace n3mapping {

void prepareVisibilityPlanarSupport(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &query,
    const Eigen::Vector3d &up_in_query,
    PreparedVisibilityObservation *observation) {
  if (!observation) return;
  observation->planar_jacobians.clear();
  if (!query || query->empty() || observation->ranges.empty() ||
      !up_in_query.allFinite() || up_in_query.norm() < 1e-6) return;
  const auto &options = observation->options;
  const double resolution = observation->resolution_deg;
  if (!std::isfinite(resolution) || resolution <= 0.0 ||
      360.0 / resolution > std::numeric_limits<int>::max()) return;
  const int azimuth_bins = std::ceil(360.0 / resolution);
  const int elevation_bins = std::ceil(180.0 / resolution);
  std::vector<int> nearest(observation->ranges.size(), -1);
  for (std::size_t i = 0; i < query->size(); ++i) {
    const auto &point = (*query)[i];
    if (!pcl::isFinite(point)) continue;
    const Eigen::Vector3d p(point.x, point.y, point.z);
    const double range = p.norm();
    if (range < options.range_min_m || range > options.range_max_m) continue;
    const auto azimuth = visibility_detail::originalAngularBin(
        p.y(), p.x(), 180.0, resolution, azimuth_bins, true);
    const auto elevation = visibility_detail::originalAngularBin(
        p.z(), std::hypot(p.x(), p.y()), 90.0, resolution, elevation_bins, false);
    const auto index = static_cast<std::size_t>(elevation) * azimuth_bins + azimuth;
    if (index < nearest.size() && range == observation->ranges[index])
      nearest[index] = static_cast<int>(i);
  }

  const Eigen::Vector3d up = up_in_query.normalized();
  const Eigen::Vector3d x = up.unitOrthogonal();
  const Eigen::Vector3d y = up.cross(x);
  pcl::KdTreeFLANN<pcl::PointXYZI> tree;
  tree.setInputCloud(query);
  std::vector<Eigen::Vector3d> jacobians(nearest.size(), Eigen::Vector3d::Zero());
  Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
  std::vector<int> neighbors(20);
  std::vector<float> distances(20);
  std::size_t valid_normals = 0;
  for (std::size_t i = 0; i < nearest.size(); ++i) {
    if (nearest[i] < 0) continue;
    const auto &point = (*query)[nearest[i]];
    const int count = tree.nearestKSearch(point, 20, neighbors, distances);
    if (count < 3) continue;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (int j = 0; j < count; ++j)
      mean += (*query)[neighbors[j]].getVector3fMap().cast<double>();
    mean /= count;
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int j = 0; j < count; ++j) {
      const Eigen::Vector3d delta =
          (*query)[neighbors[j]].getVector3fMap().cast<double>() - mean;
      covariance.noalias() += delta * delta.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success || solver.eigenvalues()[1] <= 1e-12) continue;
    const Eigen::Vector3d normal = solver.eigenvectors().col(0);
    const Eigen::Vector3d p(point.x, point.y, point.z);
    auto &j = jacobians[i];
    // Point-to-plane residual derivative for yaw and two translations in the
    // odometry horizontal plane. A horizontal surface contributes no planar
    // information, however numerous its returns or however tilted the sensor.
    j = Eigen::Vector3d(p.cross(normal).dot(up), normal.dot(x), normal.dot(y));
    information.noalias() += j * j.transpose();
    ++valid_normals;
  }

  // Do not whiten numerical noise into apparent observability (e.g. a floor
  // alone), or silently discard an unconstrained horizontal direction.
  if (valid_normals < 3 || information.diagonal().minCoeff() <= 1e-6 * valid_normals)
    return;
  const Eigen::Matrix3d scale = information.diagonal().cwiseSqrt().cwiseInverse().asDiagonal();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scale * information * scale);
  if (solver.info() != Eigen::Success ||
      solver.eigenvalues()[0] <= 1e-6 * solver.eigenvalues()[2]) return;
  const Eigen::Matrix3d whitening = scale * solver.eigenvectors() *
      solver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal();
  // Sum(j*j^T) is now I over the whole observation. The least eigenvalue over
  // only the matching bearings is min_v (v^T H_match v)/(v^T H_all v): matching
  // a strongly constrained axis cannot pay for disagreement in another axis.
  for (auto &j : jacobians) j = (whitening.transpose() * j).eval();
  observation->planar_jacobians = std::move(jacobians);
}
} // namespace n3mapping
