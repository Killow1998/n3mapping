#include "n3mapping/lio/dlio_frontend.h"

namespace n3mapping {
namespace lio {

DlioFrontend::DlioFrontend(const LioFrontendConfig& config)
    : config_(config) {}

void DlioFrontend::addImu(const core::ImuSample&) {}

std::optional<core::LioFrame> DlioFrontend::addLidar(const core::RawLidarFrame&) {
    return std::nullopt;
}

void DlioFrontend::reset() {}

}  // namespace lio
}  // namespace n3mapping
