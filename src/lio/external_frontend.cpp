#include "n3mapping/lio/external_frontend.h"

namespace n3mapping {
namespace lio {
namespace {

bool isValidCloud(const core::LioFrame::PointCloud::Ptr& cloud) {
    return cloud && !cloud->empty();
}

bool isValidPose(const Eigen::Isometry3d& pose) {
    return pose.matrix().allFinite();
}

Eigen::Matrix<double, 6, 6> sanitizeCovariance(
    const Eigen::Matrix<double, 6, 6>& covariance) {
    if (!covariance.allFinite()) {
        return Eigen::Matrix<double, 6, 6>::Identity();
    }
    return covariance;
}

}  // namespace

void ExternalLioFrontend::addImu(const core::ImuSample&) {
    // External LIO mode receives already fused odometry, so raw IMU is ignored.
}

std::optional<core::LioFrame> ExternalLioFrontend::addLidar(
    const core::RawLidarFrame& frame) {
    if (!latest_frame_ || !isValidCloud(frame.points)) {
        return std::nullopt;
    }

    core::LioFrame output = *latest_frame_;
    output.stamp = frame.stamp_end.nsec != 0 ? frame.stamp_end : frame.stamp_begin;
    output.undistorted_cloud = frame.points;
    latest_frame_ = output;
    return output;
}

void ExternalLioFrontend::reset() {
    latest_frame_.reset();
}

std::optional<core::LioFrame> ExternalLioFrontend::addSynchronizedFrame(
    const core::TimeStamp& stamp,
    const Eigen::Isometry3d& T_world_lidar,
    const PointCloud::Ptr& undistorted_cloud,
    const Eigen::Matrix<double, 6, 6>& covariance,
    bool covariance_valid) {
    if (!isValidCloud(undistorted_cloud) || !isValidPose(T_world_lidar)) {
        return std::nullopt;
    }

    core::LioFrame output;
    output.stamp = stamp;
    output.T_world_lidar = T_world_lidar;
    output.undistorted_cloud = undistorted_cloud;
    output.covariance = sanitizeCovariance(covariance);
    output.covariance_valid = covariance_valid && covariance.allFinite();
    output.pose_valid = true;
    latest_frame_ = output;
    return output;
}

}  // namespace lio
}  // namespace n3mapping
