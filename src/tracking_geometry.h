#pragma once

#include <optional>
#include <Eigen/Eigenvalues>
#include <small_gicp/points/point_cloud.hpp>
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

// Count geometric support from admitted correspondences. Source-only normals
// also describe new surfaces absent from the reference map.
inline std::optional<Eigen::Isometry3d> supportedTrackingPose(
    const small_gicp::PointCloud& cloud, int normal_neighbors,
    const Eigen::Isometry3d& prediction, const MatchResult& match) {
  const auto& endpoint = match.T_target_source;
  Eigen::Matrix3d normal_information = match.information.topLeftCorner<3, 3>();
  if (match.metric.type == small_gicp::RegistrationSetting::GICP) {
    // Prepared covariances have spectrum (epsilon, 1, 1). Two aligned
    // surfaces contribute 1/(2*epsilon) normally but only 1/2 tangentially.
    // Remove that artificial tangential stiffness before counting support.
    const double epsilon = cloud.cov(0).topLeftCorner<3, 3>().trace() - 2.0;
    normal_information = (normal_information -
        0.5 * match.num_inliers * Eigen::Matrix3d::Identity()) / (0.5 / epsilon - 0.5);
  } else if (match.metric.type != small_gicp::RegistrationSetting::PLANE_ICP) {
    return endpoint;
  }
  const Eigen::Matrix3d frame = endpoint.rotation().transpose() * prediction.rotation();
  normal_information = frame.transpose() * normal_information * frame;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(normal_information);
  if (solver.info() != Eigen::Success) return std::nullopt;
  if ((solver.eigenvalues().array() >= normal_neighbors).all()) return endpoint;
  Eigen::Isometry3d delta = prediction.inverse() * endpoint;
  Eigen::Vector3d components = solver.eigenvectors().transpose() * delta.translation();
  for (int k = 0; k < 3; ++k)
    if (solver.eigenvalues()[k] < normal_neighbors) components[k] = 0.0;
  delta.translation() = solver.eigenvectors() * components;
  return prediction * delta;
}

}  // namespace n3mapping
