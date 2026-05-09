// Placeholder DLIO frontend adapter. The algorithm extraction is not
// implemented here yet; this class only reserves the in-process interface.
#pragma once

#include <cstddef>

#include "n3mapping/lio/core_state.h"
#include "n3mapping/lio/dlio_cloud_adapter.h"
#include "n3mapping/lio/frontend.h"
#include "n3mapping/lio/frontend_config.h"
#include "n3mapping/lio/imu_propagator.h"
#include "n3mapping/lio/imu_sample_buffer.h"
#include "n3mapping/lio/local_map.h"
#include "n3mapping/lio/dlio_input_adapter.h"

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
    size_t imuSamplesSeen() const { return imu_buffer_.size(); }
    size_t lidarFramesSeen() const { return lidar_frames_seen_; }
    const dlio::CloudAdapterStats& lastCloudStats() const {
        return last_cloud_stats_;
    }
    dlio::TimeEncoding lastTimeEncoding() const { return last_time_encoding_; }
    bool lastInputHadCompleteImuWindow() const {
        return last_complete_imu_window_;
    }
    size_t lastInputImuSamples() const { return last_input_imu_samples_; }
    const std::optional<LioCoreState>& predictedState() const {
        return predicted_state_;
    }
    LioLocalMap::PointCloud::ConstPtr localMapCloud() const {
        return local_map_.cloud();
    }
    const LioLocalMap::AlignmentStats& lastAlignmentStats() const {
        return last_alignment_stats_;
    }

private:
    LioFrontendConfig config_;
    ImuSampleBuffer imu_buffer_;
    size_t lidar_frames_seen_ = 0;
    dlio::CloudAdapterStats last_cloud_stats_;
    dlio::TimeEncoding last_time_encoding_ = dlio::TimeEncoding::OusterOffsetNs;
    bool last_complete_imu_window_ = false;
    size_t last_input_imu_samples_ = 0;
    std::optional<LioCoreState> predicted_state_;
    LioLocalMap local_map_;
    LioLocalMap::AlignmentStats last_alignment_stats_;
    LioDebugCallbacks debug_callbacks_;
};

}  // namespace lio
}  // namespace n3mapping
