#pragma once
#include "moment_factor.h"
#include <gtsam/linear/JacobianFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <json/json.h>
#include <stdexcept>
namespace n3mapping::planar_geometry {
void auditFactor(const MomentPlaneFactor &factor, const gtsam::Pose3 &pose,
                 const gtsam::OrientedPlane3 &plane, Json::Value &audit) {
  gtsam::Matrix h1, h2;
  factor.evaluateError(pose, plane, h1, h2);
  double worst = 0;
  for (int i = 0; i < 9; ++i) {
    gtsam::Vector numerical;
    if (i < 6) {
      gtsam::Vector6 d = gtsam::Vector6::Zero();
      d[i] = 1e-6;
      numerical = (factor.evaluateError(pose.retract(d), plane) -
                   factor.evaluateError(pose.retract(-d), plane)) /
                  2e-6;
      worst = std::max(worst, (numerical - h1.col(i)).cwiseAbs().maxCoeff());
    } else {
      gtsam::Vector3 d = gtsam::Vector3::Zero();
      d[i - 6] = 1e-6;
      numerical = (factor.evaluateError(pose, plane.retract(d)) -
                   factor.evaluateError(pose, plane.retract(-d))) /
                  2e-6;
      worst =
          std::max(worst, (numerical - h2.col(i - 6)).cwiseAbs().maxCoeff());
    }
  }
  const Eigen::Matrix<double, 3, 2> tangent =
      pose.rotation().matrix().transpose() * plane.normal().basis();
  const double null = (h1.rightCols(3) * tangent).norm();
  audit["jacobian_max_abs"] = worst;
  audit["tangent_translation_information_norm"] = null;
  if (worst > 1e-6 || null > 1e-10)
    throw std::runtime_error(
        "Plane Jacobian or physical null-space audit failed");
  gtsam::Vector6 perturb = gtsam::Vector6::Zero();
  perturb.tail<3>() =
      pose.rotation().matrix().transpose() * plane.normal().unitVector() * .07;
  gtsam::Values values;
  values.insert(factor.keys()[0], pose.retract(perturb));
  values.insert(factor.keys()[1], plane);
  auto jac = boost::dynamic_pointer_cast<gtsam::JacobianFactor>(
      factor.linearize(values));
  const gtsam::Vector gradient = -jac->getA().transpose() * jac->getb();
  double robust_error = 0;
  for (int i = 0; i < 9; ++i) {
    auto a = values, b = values;
    if (i < 6) {
      gtsam::Vector6 dx = gtsam::Vector6::Zero();
      dx[i] = 1e-6;
      a.update(factor.keys()[0], pose.retract(perturb).retract(dx));
      b.update(factor.keys()[0], pose.retract(perturb).retract(-dx));
    } else {
      gtsam::Vector3 dx = gtsam::Vector3::Zero();
      dx[i - 6] = 1e-6;
      a.update(factor.keys()[1], plane.retract(dx));
      b.update(factor.keys()[1], plane.retract(-dx));
    }
    const double derivative = (factor.error(a) - factor.error(b)) / 2e-6;
    robust_error =
        std::max(robust_error, std::abs(derivative - gradient[i]) /
                                   std::max(1., std::abs(derivative)));
  }
  audit["robust_cost_gradient_relative_max"] = robust_error;
  if (robust_error > 1e-5)
    throw std::runtime_error("Robust objective and IRLS gradient disagree");
}

} // namespace n3mapping::planar_geometry
