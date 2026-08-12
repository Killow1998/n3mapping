// Shadow-only observability diagnostics derived from matcher-owned information.
#pragma once

#include <cstddef>
#include <limits>
#include <string>

#include <Eigen/Core>

#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

struct RegistrationObservability {
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  bool available = false;
  Matrix6d information = Matrix6d::Zero();
  bool information_finite = false;
  double symmetry_max_abs = std::numeric_limits<double>::quiet_NaN();

  bool spectrum_valid = false;
  Vector6d information_eigenvalues =
      Vector6d::Constant(std::numeric_limits<double>::quiet_NaN());
  Matrix6d information_eigenvectors =
      Matrix6d::Constant(std::numeric_limits<double>::quiet_NaN());
  bool positive_definite = false;
  int numerical_rank = 0;
  double numerical_rank_tolerance =
      std::numeric_limits<double>::quiet_NaN();
  double condition_number = std::numeric_limits<double>::quiet_NaN();

  Eigen::Vector3d translation_block_eigenvalues = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d rotation_block_eigenvalues = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());

  bool rotational_marginal_valid = false;
  Eigen::Matrix3d rotational_marginal_information = Eigen::Matrix3d::Zero();
  Eigen::Vector3d rotational_marginal_eigenvalues = Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d rotational_marginal_eigenvectors =
      Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());

  double optimizer_error = std::numeric_limits<double>::quiet_NaN();
  // RMS-like scale in the matcher's objective units, not a metric covariance.
  double residual_scale = std::numeric_limits<double>::quiet_NaN();
  double fitness_score = std::numeric_limits<double>::quiet_NaN();
  std::size_t num_inliers = 0;
  double inlier_ratio = std::numeric_limits<double>::quiet_NaN();
  std::size_t iterations = 0;
  MatchTermination termination = MatchTermination::Invalid;
  bool converged = false;
  bool success = false;
  bool production_quality = false;

  // The information matrix always describes the registration endpoint. The
  // selected pose can instead be the descriptor seed or motion prediction
  // when visibility rejects that endpoint, so retain both facts explicitly.
  std::string selected_pose_source;
  bool information_at_selected_pose = false;
};

RegistrationObservability analyzeRegistrationObservability(
    const MatchResult &match, const std::string &selected_pose_source,
    bool information_at_selected_pose, bool production_quality);

} // namespace n3mapping
