#pragma once
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose3.h>
#include <cmath>
#include <functional>
namespace n3mapping {

inline double normalizeAngle(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

class Pose3XYYawFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>;

    Pose3XYYawFactor(gtsam::Key key1,
                     gtsam::Key key2,
                     const gtsam::Pose3& measured,
                     const gtsam::SharedNoiseModel& noise_model)
      : Base(noise_model, key1, key2)
      , measured_(measured)
    {
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override
    {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new Pose3XYYawFactor(*this)));
    }

    gtsam::Vector evaluateError(const gtsam::Pose3& pose1,
                                const gtsam::Pose3& pose2,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none) const override
    {
        if (H1) {
            *H1 = numericalJacobian1(pose1, pose2);
        }
        if (H2) {
            *H2 = numericalJacobian2(pose1, pose2);
        }
        return errorVector(pose1, pose2);
    }

  private:
    gtsam::Pose3 measured_;

    Eigen::Vector3d errorVector(const gtsam::Pose3& pose1, const gtsam::Pose3& pose2) const
    {
        const gtsam::Pose3 predicted = pose1.between(pose2);
        Eigen::Vector3d error;
        error << predicted.translation().x() - measured_.translation().x(),
                 predicted.translation().y() - measured_.translation().y(),
                 normalizeAngle(predicted.rotation().yaw() - measured_.rotation().yaw());
        return error;
    }

    Eigen::Matrix<double, 3, 6> numericalJacobian1(const gtsam::Pose3& pose1,
                                                  const gtsam::Pose3& pose2) const
    {
        return numericalJacobian(
            [&](const gtsam::Pose3& p) { return errorVector(p, pose2); }, pose1);
    }

    Eigen::Matrix<double, 3, 6> numericalJacobian2(const gtsam::Pose3& pose1,
                                                  const gtsam::Pose3& pose2) const
    {
        return numericalJacobian(
            [&](const gtsam::Pose3& p) { return errorVector(pose1, p); }, pose2);
    }

    Eigen::Matrix<double, 3, 6> numericalJacobian(
        const std::function<Eigen::Vector3d(const gtsam::Pose3&)>& fn,
        const gtsam::Pose3& pose) const
    {
        constexpr double kDelta = 1e-5;
        Eigen::Matrix<double, 3, 6> H;
        H.setZero();
        for (int i = 0; i < 6; ++i) {
            gtsam::Vector6 d;
            d.setZero();
            d(i) = kDelta;
            const Eigen::Vector3d plus = fn(pose.retract(d));
            d(i) = -kDelta;
            const Eigen::Vector3d minus = fn(pose.retract(d));
            H.col(i) = (plus - minus) / (2.0 * kDelta);
        }
        return H;
    }
};


} // namespace n3mapping
