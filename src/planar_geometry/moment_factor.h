// Shared moment factor. A shared plane retains cross-scan dependence;
// its Jacobian contains no information along the plane's tangent directions.
#pragma once
#include <Eigen/Eigenvalues>
#include "n3mapping/planar_geometry.h"
#include "n3mapping/graph_factor_noise.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <gtsam/geometry/OrientedPlane3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace n3mapping::planar_geometry {
using Moment = PlaneMoment;

inline gtsam::SharedNoiseModel momentNoise(const Config& config, double sigma,
                                         double information_weight = 1.) {
  if (!std::isfinite(information_weight) || information_weight <= 0. || information_weight > 1.)
    throw std::invalid_argument("Invalid geometry information weight");
  auto geometry = config;
  const double scale = std::sqrt(information_weight);
  geometry.robust_kernel_type = "Cauchy";
  // Scale cost and Jacobians without changing the robust transition in metres.
  geometry.robust_kernel_delta = 3. * scale;
  return makeGraphRobustNoise(
      gtsam::noiseModel::Isotropic::Sigma(4, sigma / scale), geometry);
}

class MomentPlaneFactor
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::OrientedPlane3> {
  std::array<Eigen::Vector3d, 4> points_;
  Eigen::Vector3d origin_;

public:
  MomentPlaneFactor(gtsam::Key pose, gtsam::Key plane, const Moment &patch,
                    const Eigen::Vector3d &origin,
                    const gtsam::SharedNoiseModel &model)
      : NoiseModelFactor2(model, pose, plane), origin_(origin) {
    points_[0] = patch.mean();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(patch.covariance());
    for (int i = 0; i < 3; ++i)
      points_[i + 1] = eigen.eigenvectors().col(i) *
                       std::sqrt(std::max(0., eigen.eigenvalues()[i]));
  }
  gtsam::Vector evaluateError(
      const gtsam::Pose3 &pose, const gtsam::OrientedPlane3 &plane,
      boost::optional<gtsam::Matrix &> H1 = boost::none,
      boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
    gtsam::Vector4 residual;
    const auto rotation = pose.rotation().matrix();
    const auto normal = plane.normal().unitVector();
    const auto basis = plane.normal().basis();
    if (H1)
      H1->setZero(4, 6);
    if (H2)
      H2->setZero(4, 3);
    for (int i = 0; i < 4; ++i) {
      Eigen::Vector3d world = rotation * points_[i];
      if (i == 0)
        world += pose.translation() - origin_;
      residual[i] = normal.dot(world) - (i == 0 ? plane.distance() : 0.);
      if (H1) {
        H1->block<1, 3>(i, 0) =
            -normal.transpose() * rotation * gtsam::skewSymmetric(points_[i]);
        if (i == 0)
          H1->block<1, 3>(i, 3) = normal.transpose() * rotation;
      }
      if (H2) {
        H2->block<1, 2>(i, 0) = world.transpose() * basis;
        if (i == 0)
          (*H2)(i, 2) = -1.;
      }
    }
    return residual;
  }
};

} // namespace n3mapping::planar_geometry
