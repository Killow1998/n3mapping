#include "n3mapping/lio/dlio_frontend.h"

#include "n3mapping/lio/dlio_settings.h"

namespace n3mapping {
namespace lio {

DlioFrontend::DlioFrontend(const LioFrontendConfig& config)
    : config_(config),
      imu_buffer_(config.imu_buffer_max_samples) {}

void DlioFrontend::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> DlioFrontend::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    LioTimingStats timing;
    const auto settings = dlio::makeSettings(config_);
    const auto options = dlio::makeCloudAdapterOptions(settings);
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
        last_alignment_stats_ =
            local_map_.estimateAlignmentCorrection(
                *predicted_state_,
                frame.points,
                config_.alignment_max_correspondence_distance);
        if (last_alignment_stats_.valid) {
            predicted_state_->T_world_lidar.translation() +=
                last_alignment_stats_.centroid_correction_world;
        }
        local_map_.addFrame(*predicted_state_, frame.points);
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
        if (config_.debug_publish_local_map && debug_callbacks_.local_map) {
            debug_callbacks_.local_map(local_map_.cloud());
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
    local_map_.clear();
    last_alignment_stats_ = LioLocalMap::AlignmentStats{};
}

void DlioFrontend::setDebugCallbacks(const LioDebugCallbacks& callbacks) {
    debug_callbacks_ = callbacks;
}

}  // namespace lio
}  // namespace n3mapping
