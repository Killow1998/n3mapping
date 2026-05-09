#include "n3mapping/lio/dlio_frontend.h"

#include <chrono>

namespace n3mapping {
namespace lio {
namespace {

double elapsedMs(const std::chrono::steady_clock::time_point& start,
                 const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

DlioFrontend::DlioFrontend(const LioFrontendConfig& config)
    : config_(config),
      core_(config) {}

void DlioFrontend::addImu(const core::ImuSample& imu) {
    core_.addImu(applyTimeOffset(imu, config_.time_offset));
}

std::optional<core::LioFrame> DlioFrontend::addLidar(const core::RawLidarFrame& frame) {
    const auto start = std::chrono::steady_clock::now();
    auto output = core_.addLidar(applyTimeOffset(frame, config_.time_offset));
    const auto end = std::chrono::steady_clock::now();
    LioTimingStats timing;
    timing.odometry_ms = elapsedMs(start, end);
    timing.total_ms = timing.odometry_ms;
    if (output) {
        if (config_.debug_publish_odom && debug_callbacks_.odom) {
            debug_callbacks_.odom(*output);
        }
        if (config_.debug_publish_deskewed_cloud &&
            debug_callbacks_.deskewed_cloud) {
            debug_callbacks_.deskewed_cloud(output->undistorted_cloud);
        }
        if (config_.debug_publish_timing && debug_callbacks_.timing) {
            debug_callbacks_.timing(timing);
        }
        if (config_.debug_publish_local_map && debug_callbacks_.local_map) {
            debug_callbacks_.local_map(core_.localMapCloud());
        }
        return output;
    }
    if (config_.debug_publish_timing && debug_callbacks_.timing) {
        debug_callbacks_.timing(timing);
    }
    return std::nullopt;
}

void DlioFrontend::reset() {
    core_.reset();
}

void DlioFrontend::setDebugCallbacks(const LioDebugCallbacks& callbacks) {
    debug_callbacks_ = callbacks;
}

}  // namespace lio
}  // namespace n3mapping
