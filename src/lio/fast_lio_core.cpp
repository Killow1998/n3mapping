#include "n3mapping/lio/fast_lio_core.h"

namespace n3mapping {
namespace lio {
namespace fast_lio {

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
        last_imu_propagation_ =
            propagateImu(last_input_packet_.imu_samples, ImuPropagationState{});
        predicted_state_ = stateFromImuPropagation(*last_imu_propagation_);
    } else {
        last_imu_propagation_.reset();
        predicted_state_.reset();
    }
    return std::nullopt;
}

void Core::reset() {
    imu_buffer_.clear();
    lidar_frames_seen_ = 0;
    last_input_packet_ = InputPacket{};
    last_imu_propagation_.reset();
    predicted_state_.reset();
}

CloudAdapterOptions Core::cloudOptions() const {
    CloudAdapterOptions options;
    options.point_filter_num = config_.point_filter_num;
    options.scan_lines = config_.scan_lines;
    options.blind = config_.blind;
    options.max_abs_coordinate = config_.max_abs_coordinate;
    return options;
}

const char* coreStatus() {
    return "fast_lio_core input boundary ready: scan-to-map algorithm extraction pending";
}

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
