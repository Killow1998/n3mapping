#include "n3mapping/lio/dlio_frontend.h"

namespace n3mapping {
namespace lio {

DlioFrontend::DlioFrontend(const LioFrontendConfig& config)
    : config_(config) {}

void DlioFrontend::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> DlioFrontend::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    const auto adapted = dlio::cloudFromRawLidar(frame);
    last_cloud_stats_ = adapted.stats;
    last_time_encoding_ = adapted.resolved_time_encoding;
    return std::nullopt;
}

void DlioFrontend::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_cloud_stats_ = dlio::CloudAdapterStats{};
    last_time_encoding_ = dlio::TimeEncoding::OusterOffsetNs;
}

}  // namespace lio
}  // namespace n3mapping
