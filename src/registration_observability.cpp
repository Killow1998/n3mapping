#include "n3mapping/registration_observability.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace n3mapping {

RegistrationObservability analyzeRegistrationObservability(
    const MatchResult &match, const std::string &selected_pose_source,
    bool information_at_selected_pose, bool production_quality) {
  RegistrationObservability result;
  result.available = true;
  result.information = match.information;
  result.optimizer_error = match.optimizer_error;
  result.fitness_score = match.fitness_score;
  result.num_inliers = match.num_inliers;
  result.inlier_ratio = match.inlier_ratio;
  result.iterations = match.iterations;
  result.termination = match.termination;
  result.converged = match.converged;
  result.success = match.success;
  result.production_quality = production_quality;
  result.selected_pose_source = selected_pose_source;
  result.information_at_selected_pose = information_at_selected_pose;
  if (std::isfinite(match.optimizer_error) && match.optimizer_error >= 0.0 &&
      match.num_inliers > 0) {
    result.residual_scale = std::sqrt(
        match.optimizer_error / static_cast<double>(match.num_inliers));
  }

  result.information_finite = match.information.array().isFinite().all();
  if (!result.information_finite) {
    return result;
  }

  result.symmetry_max_abs =
      (match.information - match.information.transpose())
          .cwiseAbs()
          .maxCoeff();
  const RegistrationObservability::Matrix6d symmetric_information =
      0.5 * (match.information + match.information.transpose());

  Eigen::SelfAdjointEigenSolver<RegistrationObservability::Matrix6d>
      full_solver(symmetric_information);
  if (full_solver.info() == Eigen::Success &&
      full_solver.eigenvalues().array().isFinite().all() &&
      full_solver.eigenvectors().array().isFinite().all()) {
    result.spectrum_valid = true;
    result.information_eigenvalues = full_solver.eigenvalues();
    result.information_eigenvectors = full_solver.eigenvectors();
    result.positive_definite =
        result.information_eigenvalues.minCoeff() > 0.0;

    const auto abs_eigenvalues =
        result.information_eigenvalues.cwiseAbs();
    const double max_abs = abs_eigenvalues.maxCoeff();
    result.numerical_rank_tolerance =
        std::numeric_limits<double>::epsilon() * 6.0 * max_abs;
    result.numerical_rank = static_cast<int>(
        (abs_eigenvalues.array() > result.numerical_rank_tolerance).count());
    const double min_abs = abs_eigenvalues.minCoeff();
    result.condition_number =
        min_abs > 0.0 ? max_abs / min_abs
                      : std::numeric_limits<double>::infinity();
  }

  const Eigen::Matrix3d translation_information =
      symmetric_information.block<3, 3>(0, 0);
  const Eigen::Matrix3d rotation_information =
      symmetric_information.block<3, 3>(3, 3);
  const Eigen::Matrix3d rotation_translation =
      symmetric_information.block<3, 3>(3, 0);

  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> translation_solver(
      translation_information);
  if (translation_solver.info() == Eigen::Success) {
    result.translation_block_eigenvalues = translation_solver.eigenvalues();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> rotation_solver(
      rotation_information);
  if (rotation_solver.info() == Eigen::Success) {
    result.rotation_block_eigenvalues = rotation_solver.eigenvalues();
  }

  const Eigen::Matrix3d translation_pseudoinverse =
      translation_information.completeOrthogonalDecomposition()
          .pseudoInverse();
  const Eigen::Matrix3d rotational_marginal =
      rotation_information -
      rotation_translation * translation_pseudoinverse *
          rotation_translation.transpose();
  result.rotational_marginal_information =
      0.5 * (rotational_marginal + rotational_marginal.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> marginal_solver(
      result.rotational_marginal_information);
  if (marginal_solver.info() == Eigen::Success &&
      marginal_solver.eigenvalues().array().isFinite().all() &&
      marginal_solver.eigenvectors().array().isFinite().all()) {
    result.rotational_marginal_valid = true;
    result.rotational_marginal_eigenvalues = marginal_solver.eigenvalues();
    result.rotational_marginal_eigenvectors = marginal_solver.eigenvectors();
  }

  return result;
}

} // namespace n3mapping
