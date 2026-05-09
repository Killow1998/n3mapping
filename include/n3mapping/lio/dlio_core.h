// Minimal ROS-free DLIO core boundary. The odometry algorithm is extracted in
// later steps; this class owns the input buffering contract now.
#pragma once

#include <cstddef>
#include <optional>

#include "n3mapping/core/types.h"
#include "n3mapping/lio/dlio_imu_integration.h"
#include "n3mapping/lio/dlio_input_adapter.h"
#include "n3mapping/lio/dlio_map_accumulator.h"
#include "n3mapping/lio/dlio_scan_timing.h"
#include "n3mapping/lio/core_state.h"
#include "n3mapping/lio/frontend.h"
#include "n3mapping/lio/frontend_config.h"
#include "n3mapping/lio/imu_propagator.h"
#include "n3mapping/lio/imu_sample_buffer.h"
#include "n3mapping/lio/local_map.h"

namespace n3mapping {
namespace lio {
namespace dlio {

class Core {
public:
    explicit Core(const LioFrontendConfig& config = LioFrontendConfig());

    void addImu(const core::ImuSample& imu);
    std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame);
    void reset();

    FrontendCapability capability() const { return FrontendCapability::PredictionOnly; }
    bool implemented() const { return false; }
    size_t imuSamplesSeen() const { return imu_buffer_.size(); }
    size_t lidarFramesSeen() const { return lidar_frames_seen_; }
    const InputPacket& lastInputPacket() const { return last_input_packet_; }
    const ScanTiming& lastScanTiming() const { return last_scan_timing_; }
    const std::optional<ImuPropagationState>& lastImuPropagation() const {
        return last_imu_propagation_;
    }
    const std::optional<LioCoreState>& predictedState() const {
        return predicted_state_;
    }
    LioLocalMap::PointCloud::ConstPtr localMapCloud() const {
        return local_map_.cloud();
    }
    MapAccumulator::PointCloud::ConstPtr denseMapCloud() const {
        return dense_map_.map();
    }
    const MapAccumulator::AddResult& lastDenseMapAddResult() const {
        return last_dense_map_add_result_;
    }
    const LioLocalMap::AlignmentStats& lastAlignmentStats() const {
        return last_alignment_stats_;
    }

private:
    CloudAdapterOptions cloudOptions() const;

    LioFrontendConfig config_;
    ImuSampleBuffer imu_buffer_;
    size_t lidar_frames_seen_ = 0;
    InputPacket last_input_packet_;
    ScanTiming last_scan_timing_;
    std::optional<ImuPropagationState> last_imu_propagation_;
    std::optional<LioCoreState> predicted_state_;
    LioLocalMap local_map_;
    MapAccumulator dense_map_;
    MapAccumulator::AddResult last_dense_map_add_result_;
    LioLocalMap::AlignmentStats last_alignment_stats_;
};

const char* coreStatus();

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
