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
    const auto packet = fast_lio::buildInputPacket(frame, imu_buffer_, options);
    last_cloud_stats_ = packet.cloud_stats;
    last_complete_imu_window_ = packet.has_complete_imu_window;
    last_input_imu_samples_ = packet.imu_samples.size();
    return std::nullopt;
}

void FastLioFrontend::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_cloud_stats_ = fast_lio::CloudAdapterStats{};
    last_complete_imu_window_ = false;
    last_input_imu_samples_ = 0;
}

}  // namespace lio
}  // namespace n3mapping
