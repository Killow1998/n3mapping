#include "n3mapping/lio/fast_lio_frontend.h"

namespace n3mapping {
namespace lio {

FastLioFrontend::FastLioFrontend(const LioFrontendConfig& config)
    : config_(config),
      imu_buffer_(config.imu_buffer_max_samples) {}

void FastLioFrontend::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> FastLioFrontend::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    fast_lio::CloudAdapterOptions options;
    options.point_filter_num = config_.point_filter_num;
    options.scan_lines = config_.scan_lines;
    options.blind = config_.blind;
    options.max_abs_coordinate = config_.max_abs_coordinate;
    const auto adapted = fast_lio::cloudFromRawLidar(frame, options);
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
