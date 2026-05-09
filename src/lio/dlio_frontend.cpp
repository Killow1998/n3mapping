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
    LioTimingStats timing;
    dlio::CloudAdapterOptions options;
    options.time_encoding = parseDlioTimeEncoding(config_.dlio_time_encoding);
    options.blind = config_.blind;
    options.max_abs_coordinate = config_.max_abs_coordinate;
    const auto packet = dlio::buildInputPacket(frame, imu_buffer_, options);
    last_cloud_stats_ = packet.cloud_stats;
    last_time_encoding_ = packet.time_encoding;
    last_complete_imu_window_ = packet.has_complete_imu_window;
    last_input_imu_samples_ = packet.imu_samples.size();
    if (!packet.imu_samples.empty()) {
        const auto propagation = propagateImu(packet.imu_samples, ImuPropagationState{});
        predicted_state_ = stateFromImuPropagation(propagation);
    } else {
        predicted_state_.reset();
    }
    if (config_.prediction_only_output && predicted_state_ && packet.cloud &&
        !packet.cloud->empty() && frame.points && !frame.points->empty()) {
        auto output = frameFromState(*predicted_state_);
        output.undistorted_cloud = frame.points;
        output.pose_valid = predicted_state_->initialized;
        if (config_.debug_publish_odom && debug_callbacks_.odom) {
            debug_callbacks_.odom(output);
        }
        if (config_.debug_publish_deskewed_cloud &&
            debug_callbacks_.deskewed_cloud) {
            debug_callbacks_.deskewed_cloud(output.undistorted_cloud);
        }
        if (config_.debug_publish_timing && debug_callbacks_.timing) {
            debug_callbacks_.timing(timing);
        }
        return output;
    }
    if (config_.debug_publish_timing && debug_callbacks_.timing) {
        debug_callbacks_.timing(timing);
    }
    return std::nullopt;
}

void DlioFrontend::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_cloud_stats_ = dlio::CloudAdapterStats{};
    last_time_encoding_ = dlio::TimeEncoding::OusterOffsetNs;
    last_complete_imu_window_ = false;
    last_input_imu_samples_ = 0;
    predicted_state_.reset();
}

void DlioFrontend::setDebugCallbacks(const LioDebugCallbacks& callbacks) {
    debug_callbacks_ = callbacks;
}

}  // namespace lio
}  // namespace n3mapping
