#include "n3mapping/lio/fast_lio_frontend.h"

namespace n3mapping {
namespace lio {

FastLioFrontend::FastLioFrontend(const LioFrontendConfig& config)
    : config_(config) {}

void FastLioFrontend::addImu(const core::ImuSample&) {}

std::optional<core::LioFrame> FastLioFrontend::addLidar(const core::RawLidarFrame&) {
    return std::nullopt;
}

void FastLioFrontend::reset() {}

}  // namespace lio
}  // namespace n3mapping
