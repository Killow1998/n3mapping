#include "n3mapping/lio/fast_lio_frontend.h"

namespace n3mapping {
namespace lio {

FastLioFrontend::FastLioFrontend(const LioFrontendConfig& config)
    : config_(config) {}

void FastLioFrontend::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> FastLioFrontend::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    const auto adapted = fast_lio::cloudFromRawLidar(frame);
    last_cloud_stats_ = adapted.stats;
    return std::nullopt;
}

void FastLioFrontend::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_cloud_stats_ = fast_lio::CloudAdapterStats{};
}

}  // namespace lio
}  // namespace n3mapping
