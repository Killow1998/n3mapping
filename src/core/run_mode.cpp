#include "n3mapping/core/run_mode.h"

#include <stdexcept>

namespace n3mapping {

CoreRunMode parseCoreRunMode(const std::string &mode) {
  if (mode == "mapping") {
    return CoreRunMode::MAPPING;
  }
  if (mode == "localization") {
    return CoreRunMode::LOCALIZATION;
  }
  if (mode == "map_extension") {
    return CoreRunMode::MAP_EXTENSION;
  }
  throw std::invalid_argument("Invalid n3mapping mode: " + mode);
}

const char *coreRunModeName(CoreRunMode mode) {
  switch (mode) {
  case CoreRunMode::LOCALIZATION:
    return "localization";
  case CoreRunMode::MAP_EXTENSION:
    return "map_extension";
  case CoreRunMode::MAPPING:
  default:
    return "mapping";
  }
}

bool coreRunModeLoadsMap(CoreRunMode mode) {
  return mode == CoreRunMode::LOCALIZATION ||
         mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeSavesMap(CoreRunMode mode) {
  return mode == CoreRunMode::MAPPING || mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeProcessesLoopClosures(CoreRunMode mode) {
  return mode == CoreRunMode::MAPPING || mode == CoreRunMode::MAP_EXTENSION;
}

} // namespace n3mapping
