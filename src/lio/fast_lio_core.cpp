#include "n3mapping/lio/fast_lio_core.h"

#include "n3mapping/lio/fast_lio_settings.h"

namespace n3mapping {
namespace lio {
namespace fast_lio {
namespace {

bool seedOrientationFromFirstImu(ImuPropagationState& state,
                                 const std::vector<core::ImuSample>& samples) {
    if (samples.empty() || !samples.front().has_orientation) {
        return false;
    }
    Eigen::Quaterniond q = samples.front().orientation;
    if (!q.coeffs().allFinite() || q.norm() <= 1e-12) {
        return false;
    }
    state.orientation = q.normalized();
    state.valid = true;
    return true;
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
        } else {
            seedOrientationFromFirstImu(initial_state,
                                        last_input_packet_.imu_samples);
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
    return makeCloudAdapterOptions(makeSettings(config_));
}

const char* coreStatus() {
    return "fast_lio_core input boundary ready: scan-to-map algorithm extraction pending";
}

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
