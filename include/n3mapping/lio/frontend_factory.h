// Factory for selecting a LIO frontend implementation from Config.
#pragma once

#include <memory>
#include <string>

#include "n3mapping/config.h"
#include "n3mapping/lio/frontend.h"

namespace n3mapping {
namespace lio {

enum class FrontendMode {
    External,
    FastLio,
    Dlio,
    Unknown,
};

struct FrontendFactoryResult {
    std::unique_ptr<LioFrontend> frontend;
    FrontendMode mode = FrontendMode::Unknown;
    std::string error;

    bool ok() const { return static_cast<bool>(frontend); }
};

FrontendMode parseFrontendMode(const std::string& mode);
const char* frontendModeName(FrontendMode mode);
FrontendFactoryResult createLioFrontend(const Config& config);

}  // namespace lio
}  // namespace n3mapping
