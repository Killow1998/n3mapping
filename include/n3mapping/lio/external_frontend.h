// Adapter for the compatibility mode where another LIO publishes odometry and
// an undistorted/registerd cloud, and n3mapping consumes both in-process.
#pragma once

#include <optional>

#include "n3mapping/lio/frontend.h"

namespace n3mapping {
namespace lio {

class ExternalLioFrontend : public LioFrontend {
public:
    using PointCloud = core::LioFrame::PointCloud;

    FrontendCapability capability() const override {
        return FrontendCapability::ExternalFrameAdapter;
    }
    void addImu(const core::ImuSample& imu) override;
    std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame) override;
    void reset() override;

    std::optional<core::LioFrame> addSynchronizedFrame(
        const core::TimeStamp& stamp,
        const Eigen::Isometry3d& T_world_lidar,
        const PointCloud::Ptr& undistorted_cloud,
        const Eigen::Matrix<double, 6, 6>& covariance =
            Eigen::Matrix<double, 6, 6>::Identity(),
        bool covariance_valid = false);

private:
    std::optional<core::LioFrame> latest_frame_;
};

}  // namespace lio
}  // namespace n3mapping
