// Placeholder DLIO frontend adapter. The algorithm extraction is not
// implemented here yet; this class only reserves the in-process interface.
#pragma once

#include <cstddef>

#include "n3mapping/lio/dlio_core.h"
#include "n3mapping/lio/frontend.h"
#include "n3mapping/lio/frontend_config.h"

namespace n3mapping {
namespace lio {

class DlioFrontend : public LioFrontend {
public:
    explicit DlioFrontend(const LioFrontendConfig& config = LioFrontendConfig());

    void addImu(const core::ImuSample& imu) override;
    std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame) override;
    void reset() override;
    void setDebugCallbacks(const LioDebugCallbacks& callbacks) override;

    FrontendCapability capability() const override {
        return FrontendCapability::PredictionOnly;
    }
    bool implemented() const { return false; }
    const LioFrontendConfig& config() const { return config_; }
    size_t imuSamplesSeen() const { return core_.imuSamplesSeen(); }
    size_t lidarFramesSeen() const { return core_.lidarFramesSeen(); }
    const dlio::CloudAdapterStats& lastCloudStats() const {
        return core_.lastInputPacket().cloud_stats;
    }
    dlio::TimeEncoding lastTimeEncoding() const {
        return core_.lastInputPacket().time_encoding;
    }
    bool lastInputHadCompleteImuWindow() const {
        return core_.lastInputPacket().has_complete_imu_window;
    }
    size_t lastInputImuSamples() const { return core_.lastInputPacket().imu_samples.size(); }
    const std::optional<LioCoreState>& predictedState() const {
        return core_.predictedState();
    }
    LioLocalMap::PointCloud::ConstPtr localMapCloud() const {
        return core_.localMapCloud();
    }
    const LioLocalMap::AlignmentStats& lastAlignmentStats() const {
        return core_.lastAlignmentStats();
    }

private:
    LioFrontendConfig config_;
    dlio::Core core_;
    LioDebugCallbacks debug_callbacks_;
};

}  // namespace lio
}  // namespace n3mapping
