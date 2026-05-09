// ROS-free DLIO IMU integration primitive used by deskew and prior prediction.
#pragma once

#include <vector>

#include <Eigen/Geometry>

namespace n3mapping {
namespace lio {
namespace dlio {

struct ImuMeasurement {
    double stamp = 0.0;
    double dt = 0.0;
    Eigen::Vector3f angular_velocity = Eigen::Vector3f::Zero();
    Eigen::Vector3f linear_acceleration = Eigen::Vector3f::Zero();
};

struct ImuIntegrationRequest {
    double start_time = 0.0;
    Eigen::Quaternionf q_init = Eigen::Quaternionf::Identity();
    Eigen::Vector3f p_init = Eigen::Vector3f::Zero();
    Eigen::Vector3f v_init = Eigen::Vector3f::Zero();
    std::vector<double> sorted_timestamps;
    double gravity = 9.80665;
};

struct IntegratedPose {
    double stamp = 0.0;
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
};

std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>
integrateImuStates(const std::vector<ImuMeasurement>& imu_samples,
                   const ImuIntegrationRequest& request);

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
integrateImu(const std::vector<ImuMeasurement>& imu_samples,
             const ImuIntegrationRequest& request);

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
