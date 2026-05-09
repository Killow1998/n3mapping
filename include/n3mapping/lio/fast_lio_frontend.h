// Placeholder FAST-LIO2 frontend adapter. The algorithm extraction is not
// implemented here yet; this class only reserves the in-process interface.
#pragma once

#include "n3mapping/lio/fast_lio_cloud_adapter.h"
#include "n3mapping/lio/frontend.h"
#include "n3mapping/lio/frontend_config.h"

namespace n3mapping {
namespace lio {

class FastLioFrontend : public LioFrontend {
public:
    explicit FastLioFrontend(const LioFrontendConfig& config = LioFrontendConfig());

    void addImu(const core::ImuSample& imu) override;
    std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame) override;
    void reset() override;

    bool implemented() const { return false; }
    const LioFrontendConfig& config() const { return config_; }
    size_t imuSamplesSeen() const { return imu_samples_seen_; }
    size_t lidarFramesSeen() const { return lidar_frames_seen_; }
    const fast_lio::CloudAdapterStats& lastCloudStats() const {
        return last_cloud_stats_;
    }

private:
    LioFrontendConfig config_;
    size_t imu_samples_seen_ = 0;
    size_t lidar_frames_seen_ = 0;
    fast_lio::CloudAdapterStats last_cloud_stats_;
};

}  // namespace lio
}  // namespace n3mapping
