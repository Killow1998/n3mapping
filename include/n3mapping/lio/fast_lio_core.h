// Minimal ROS-free FAST-LIO2 core boundary. The scan-to-map algorithm is
// extracted in later steps; this class owns the input buffering contract now.
#pragma once

#include <cstddef>
#include <optional>

#include "n3mapping/core/types.h"
#include "n3mapping/lio/fast_lio_input_adapter.h"
#include "n3mapping/lio/frontend_config.h"
#include "n3mapping/lio/imu_sample_buffer.h"

namespace n3mapping {
namespace lio {
namespace fast_lio {

class Core {
public:
    explicit Core(const LioFrontendConfig& config = LioFrontendConfig());

    void addImu(const core::ImuSample& imu);
    std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame);
    void reset();

    bool implemented() const { return false; }
    size_t imuSamplesSeen() const { return imu_buffer_.size(); }
    size_t lidarFramesSeen() const { return lidar_frames_seen_; }
    const InputPacket& lastInputPacket() const { return last_input_packet_; }

private:
    CloudAdapterOptions cloudOptions() const;

    LioFrontendConfig config_;
    ImuSampleBuffer imu_buffer_;
    size_t lidar_frames_seen_ = 0;
    InputPacket last_input_packet_;
};

const char* coreStatus();

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
