#pragma once

#include <string>

namespace n3mapping {

enum class CoreRunMode {
    MAPPING,
    LOCALIZATION,
    MAP_EXTENSION,
};

CoreRunMode parseCoreRunMode(const std::string& mode);
const char* coreRunModeName(CoreRunMode mode);
bool coreRunModeLoadsMap(CoreRunMode mode);
bool coreRunModeSavesMap(CoreRunMode mode);
bool coreRunModeProcessesLoopClosures(CoreRunMode mode);

}  // namespace n3mapping
