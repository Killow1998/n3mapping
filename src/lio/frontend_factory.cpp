#include "n3mapping/lio/frontend_factory.h"

#include <algorithm>
#include <cctype>

#ifdef N3MAPPING_BUILD_DLIO_CORE
#include "n3mapping/lio/dlio_frontend.h"
#endif
#include "n3mapping/lio/external_frontend.h"
#include "n3mapping/lio/frontend_config.h"
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
#include "n3mapping/lio/fast_lio_frontend.h"
#endif

namespace n3mapping {
namespace lio {
namespace {

std::string normalized(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

FrontendFactoryResult unsupported(FrontendMode mode, const std::string& detail) {
    FrontendFactoryResult result;
    result.mode = mode;
    result.error = detail;
    return result;
}

}  // namespace

FrontendMode parseFrontendMode(const std::string& mode) {
    const std::string value = normalized(mode);
    if (value == "external") return FrontendMode::External;
    if (value == "fast_lio" || value == "fastlio" || value == "fast_lio2" ||
        value == "fastlio2") {
        return FrontendMode::FastLio;
    }
    if (value == "dlio" || value == "direct_lidar_inertial_odometry") {
        return FrontendMode::Dlio;
    }
    return FrontendMode::Unknown;
}

const char* frontendModeName(FrontendMode mode) {
    switch (mode) {
        case FrontendMode::External:
            return "external";
        case FrontendMode::FastLio:
            return "fast_lio";
        case FrontendMode::Dlio:
            return "dlio";
        case FrontendMode::Unknown:
        default:
            return "unknown";
    }
}

FrontendFactoryResult createLioFrontend(const Config& config) {
    FrontendFactoryResult result;
    result.mode = parseFrontendMode(config.frontend_mode);
    switch (result.mode) {
        case FrontendMode::External:
            result.frontend = std::make_unique<ExternalLioFrontend>();
            return result;
        case FrontendMode::FastLio:
#ifdef N3MAPPING_BUILD_FAST_LIO_CORE
            result.frontend = std::make_unique<FastLioFrontend>(makeLioFrontendConfig(config));
            return result;
#else
            return unsupported(result.mode,
                               "frontend_mode=fast_lio is configured, but fast_lio_core is not built into this refactor stage yet");
#endif
        case FrontendMode::Dlio:
#ifdef N3MAPPING_BUILD_DLIO_CORE
            result.frontend = std::make_unique<DlioFrontend>(makeLioFrontendConfig(config));
            return result;
#else
            return unsupported(result.mode,
                               "frontend_mode=dlio is configured, but dlio_core is not built into this refactor stage yet");
#endif
        case FrontendMode::Unknown:
        default:
            return unsupported(result.mode,
                               "unsupported frontend_mode='" + config.frontend_mode +
                                   "'; expected external, fast_lio, or dlio");
    }
}

}  // namespace lio
}  // namespace n3mapping
