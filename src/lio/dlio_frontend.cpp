#include "n3mapping/lio/dlio_frontend.h"

#include <algorithm>
#include <cctype>

namespace n3mapping {
namespace lio {
namespace {

dlio::TimeEncoding parseDlioTimeEncoding(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "velodyne" || value == "velodyne_seconds" ||
        value == "velodyne_offset_seconds") {
        return dlio::TimeEncoding::VelodyneOffsetSeconds;
    }
    if (value == "livox" || value == "livox_ns" || value == "livox_offset_ns") {
        return dlio::TimeEncoding::LivoxOffsetNs;
    }
    if (value == "ouster" || value == "ouster_ns" || value == "ouster_offset_ns") {
        return dlio::TimeEncoding::OusterOffsetNs;
    }
    return dlio::TimeEncoding::Auto;
}

}  // namespace

DlioFrontend::DlioFrontend(const LioFrontendConfig& config)
    : config_(config),
      imu_buffer_(config.imu_buffer_max_samples) {}

void DlioFrontend::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> DlioFrontend::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    dlio::CloudAdapterOptions options;
    options.time_encoding = parseDlioTimeEncoding(config_.dlio_time_encoding);
    options.blind = config_.blind;
    options.max_abs_coordinate = config_.max_abs_coordinate;
    const auto adapted = dlio::cloudFromRawLidar(frame, options);
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
