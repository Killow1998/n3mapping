#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "n3mapping/core/types.h"

namespace n3mapping {

enum class CoreRunMode;

enum class InputLinearVelocityFrame { CHILD, PARENT };

inline InputLinearVelocityFrame parseInputLinearVelocityFrame(
    const std::string& value) {
    if (value == "child") return InputLinearVelocityFrame::CHILD;
    if (value == "parent") return InputLinearVelocityFrame::PARENT;
    throw std::invalid_argument("input_linear_velocity_frame must be child or parent");
}

// ROS 1 and ROS 2 Odometry have the same field layout. Keep the transport
// conversion shared; motion measurements never enter the mapping algorithm.
template <typename Odometry>
bool forwardOdometryTwist(const Odometry& input,
                          InputLinearVelocityFrame linear_frame,
                          Odometry* output) {
    if (!output || input.child_frame_id.empty() ||
        input.child_frame_id != output->child_frame_id) return false;
    const auto& twist = input.twist.twist;
    Eigen::Matrix<double, 6, 1> velocity;
    velocity << twist.linear.x, twist.linear.y, twist.linear.z,
                twist.angular.x, twist.angular.y, twist.angular.z;
    Eigen::Matrix<double, 6, 6> covariance;
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 6; ++col)
            covariance(row, col) = input.twist.covariance[row * 6 + col];
    if (!velocity.allFinite() || !covariance.allFinite()) return false;

    Eigen::Matrix<double, 6, 6> transform =
        Eigen::Matrix<double, 6, 6>::Identity();
    if (linear_frame == InputLinearVelocityFrame::PARENT) {
        const auto& q = input.pose.pose.orientation;
        Eigen::Quaterniond orientation(q.w, q.x, q.y, q.z);
        if (!orientation.coeffs().allFinite() || orientation.norm() < 1e-9)
            return false;
        transform.template topLeftCorner<3, 3>() =
            orientation.normalized().toRotationMatrix().transpose();
    }
    velocity = (transform * velocity).eval();
    covariance = (transform * covariance * transform.transpose()).eval();
    if (!velocity.allFinite() || !covariance.allFinite()) return false;

    auto& result = output->twist.twist;
    result.linear.x = velocity(0);
    result.linear.y = velocity(1);
    result.linear.z = velocity(2);
    result.angular.x = velocity(3);
    result.angular.y = velocity(4);
    result.angular.z = velocity(5);
    for (int row = 0; row < 6; ++row)
        for (int col = 0; col < 6; ++col)
            output->twist.covariance[row * 6 + col] = covariance(row, col);
    return true;
}

struct RealtimeLocalizationStatus {
    double estimate_stamp = 0.0;
    double observation_stamp = 0.0;
    double correction_stamp = 0.0;
    // True only when this input frame requested an Odometry estimate. A
    // lost/initializing backend may publish status without attempting one.
    bool estimate_attempted = false;
    bool estimate_available = false;
    std::string input_reason;
};

std::string localizationStatusJson(CoreRunMode mode,
                                   const core::BackendOutput& output,
                                   double stamp,
                                   const std::string& frame_id,
                                   const RealtimeLocalizationStatus* realtime = nullptr);

}  // namespace n3mapping
