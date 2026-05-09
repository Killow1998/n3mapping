#include "n3mapping/lio/dlio_frontend.h"

namespace n3mapping {
namespace lio {

DlioFrontend::DlioFrontend(const LioFrontendConfig& config)
    : config_(config),
      core_(config) {}

void DlioFrontend::addImu(const core::ImuSample& imu) {
    core_.addImu(applyTimeOffset(imu, config_.time_offset));
}

std::optional<core::LioFrame> DlioFrontend::addLidar(const core::RawLidarFrame& frame) {
    LioTimingStats timing;
    auto output = core_.addLidar(applyTimeOffset(frame, config_.time_offset));
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
