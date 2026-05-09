// Placeholder DLIO frontend adapter. The algorithm extraction is not
// implemented here yet; this class only reserves the in-process interface.
#pragma once

#include "n3mapping/lio/frontend.h"
#include "n3mapping/lio/frontend_config.h"

namespace n3mapping {
namespace lio {

class DlioFrontend : public LioFrontend {
public:
    explicit DlioFrontend(const LioFrontendConfig& config = LioFrontendConfig());

    void addImu(const core::ImuSample& imu) override;
    std::optional<core::LioFrame> addLidar(const core::RawLidarFrame& frame) override;
    void reset() override;

    bool implemented() const { return false; }
    const LioFrontendConfig& config() const { return config_; }

private:
    LioFrontendConfig config_;
};

}  // namespace lio
}  // namespace n3mapping
