#pragma once

#include <limits>
#include <Eigen/Eigenvalues>

namespace n3mapping {

inline double minimumPlanarRegistrationInformation(
    const Eigen::Matrix<double, 6, 6> &information, const Eigen::Vector3d &up) {
  const Eigen::Vector3d horizontal = up.unitOrthogonal();
  Eigen::Matrix<double, 6, 3> basis = Eigen::Matrix<double, 6, 3>::Zero();
  basis.block<3, 1>(0, 0) = horizontal;
  basis.block<3, 1>(0, 1) = up.cross(horizontal);
  basis.block<3, 1>(3, 2) = up;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> spectrum(
      basis.transpose() * information * basis);
  return spectrum.info() == Eigen::Success
             ? spectrum.eigenvalues()[0]
             : std::numeric_limits<double>::quiet_NaN();
}

}
