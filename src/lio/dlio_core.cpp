#include "n3mapping/lio/dlio_core.h"

#include <algorithm>
#include <cctype>

namespace n3mapping {
namespace lio {
namespace dlio {
namespace {

TimeEncoding parseTimeEncoding(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (value == "velodyne" || value == "velodyne_seconds" ||
        value == "velodyne_offset_seconds") {
        return TimeEncoding::VelodyneOffsetSeconds;
    }
    if (value == "livox" || value == "livox_ns" || value == "livox_offset_ns") {
        return TimeEncoding::LivoxOffsetNs;
    }
    if (value == "ouster" || value == "ouster_ns" || value == "ouster_offset_ns") {
        return TimeEncoding::OusterOffsetNs;
    }
    return TimeEncoding::Auto;
}

}  // namespace

Core::Core(const LioFrontendConfig& config)
    : config_(config),
      imu_buffer_(config.imu_buffer_max_samples) {}

void Core::addImu(const core::ImuSample& imu) {
    imu_buffer_.add(imu);
}

std::optional<core::LioFrame> Core::addLidar(const core::RawLidarFrame& frame) {
    ++lidar_frames_seen_;
    last_input_packet_ = buildInputPacket(frame, imu_buffer_, cloudOptions());
    if (!last_input_packet_.imu_samples.empty()) {
        ImuPropagationState initial_state;
        if (predicted_state_) {
            initial_state.stamp = predicted_state_->stamp;
            initial_state.orientation =
                Eigen::Quaterniond(predicted_state_->T_world_lidar.linear());
            initial_state.position = predicted_state_->T_world_lidar.translation();
            initial_state.velocity = predicted_state_->velocity_world;
            initial_state.valid = predicted_state_->initialized;
        }
        last_imu_propagation_ =
            propagateImu(last_input_packet_.imu_samples, initial_state);
        predicted_state_ = stateFromImuPropagation(*last_imu_propagation_);
    } else {
        last_imu_propagation_.reset();
        predicted_state_.reset();
    }
    if (config_.prediction_only_output && predicted_state_ && frame.points &&
        !frame.points->empty()) {
        last_alignment_stats_ =
            local_map_.estimateAlignmentCorrection(*predicted_state_, frame.points, 1.0);
        if (last_alignment_stats_.valid) {
            predicted_state_->T_world_lidar.translation() +=
                last_alignment_stats_.centroid_correction_world;
        }
        local_map_.addFrame(*predicted_state_, frame.points);
        auto output = frameFromState(*predicted_state_);
        output.undistorted_cloud = frame.points;
        output.pose_valid = predicted_state_->initialized;
        return output;
    }
    return std::nullopt;
}

void Core::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_input_packet_ = InputPacket{};
    last_imu_propagation_.reset();
    predicted_state_.reset();
    local_map_.clear();
    last_alignment_stats_ = LioLocalMap::AlignmentStats{};
}

CloudAdapterOptions Core::cloudOptions() const {
    CloudAdapterOptions options;
    options.time_encoding = parseTimeEncoding(config_.dlio_time_encoding);
    options.blind = config_.blind;
    options.max_abs_coordinate = config_.max_abs_coordinate;
    return options;
}

const char* coreStatus() {
    return "dlio_core input boundary ready: odometry algorithm extraction pending";
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
