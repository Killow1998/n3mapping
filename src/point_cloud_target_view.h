#pragma once

#include <small_gicp/points/point_cloud.hpp>

namespace n3mapping::registration_detail {
inline Eigen::Matrix4d covarianceFromNormal(const Eigen::Vector4d& normal) {
  Eigen::Matrix4d result = Eigen::Matrix4d::Zero();
  result.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() -
      .999 * normal.head<3>() * normal.head<3>().transpose();
  return result;
}
// small_gicp's fixed spectrum (.001, 1, 1) is fully determined by the normal.
// Retain arbitrary explicitly supplied covariances; compact prepared targets
// reconstruct only the matrix requested by the current correspondence.
struct TargetView {
  const small_gicp::PointCloud& cloud;
  Eigen::Matrix4d covariance(std::size_t i) const {
    if (!cloud.covs.empty()) return cloud.cov(i);
    return covarianceFromNormal(cloud.normal(i));
  }
};
}  // namespace n3mapping::registration_detail

namespace small_gicp::traits {
template <>
struct Traits<n3mapping::registration_detail::TargetView> {
  using View = n3mapping::registration_detail::TargetView;
  static std::size_t size(const View& view) { return view.cloud.size(); }
  static bool has_points(const View& view) { return !view.cloud.points.empty(); }
  static bool has_normals(const View& view) { return !view.cloud.normals.empty(); }
  static bool has_covs(const View& view) {
    return !view.cloud.covs.empty() || has_normals(view);
  }
  static const Eigen::Vector4d& point(const View& view, std::size_t i) {
    return view.cloud.point(i);
  }
  static const Eigen::Vector4d& normal(const View& view, std::size_t i) {
    return view.cloud.normal(i);
  }
  static Eigen::Matrix4d cov(const View& view, std::size_t i) {
    return view.covariance(i);
  }
};
}  // namespace small_gicp::traits
