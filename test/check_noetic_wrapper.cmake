set(REQUIRED_FILES
  "${ROOT_DIR}/noetic/CMakeLists.txt"
  "${ROOT_DIR}/noetic/package.xml"
  "${ROOT_DIR}/noetic/include/n3mapping_noetic/config_noetic.h"
  "${ROOT_DIR}/noetic/include/n3mapping_noetic/conversions.h"
  "${ROOT_DIR}/noetic/src/config_noetic.cpp"
  "${ROOT_DIR}/noetic/src/conversions.cpp"
  "${ROOT_DIR}/noetic/src/n3mapping_noetic_node.cpp"
)

foreach(REQUIRED_FILE IN LISTS REQUIRED_FILES)
  if(NOT EXISTS "${REQUIRED_FILE}")
    message(FATAL_ERROR "Missing Noetic wrapper file: ${REQUIRED_FILE}")
  endif()
endforeach()

set(FORBIDDEN_PATTERNS
  "detectLoopCandidates("
  "incrementalOptimize("
  "WorldLocalizing"
  "MappingResuming"
  "LoopClosureManager"
  "GraphOptimizer"
  "PointCloudMatcher"
  "MapSerializer"
  "KeyframeManager"
  "RelocalizationOutputAuthority"
  "RealtimeOdometry"
  "updateCorrection("
  "localizationStatusJson("
  "appendProjectionMetadata("
)

set(CHECK_FILES
  "${ROOT_DIR}/noetic/src/n3mapping_noetic_node.cpp"
  "${ROOT_DIR}/noetic/src/conversions.cpp"
  "${ROOT_DIR}/noetic/include/n3mapping_noetic/conversions.h"
)

set(OFFENDING_FILES "")
foreach(CHECK_FILE IN LISTS CHECK_FILES)
  file(READ "${CHECK_FILE}" FILE_CONTENTS)
  foreach(FORBIDDEN_PATTERN IN LISTS FORBIDDEN_PATTERNS)
    string(FIND "${FILE_CONTENTS}" "${FORBIDDEN_PATTERN}" PATTERN_INDEX)
    if(NOT PATTERN_INDEX EQUAL -1)
      list(APPEND OFFENDING_FILES
        "${CHECK_FILE}: forbidden backend token '${FORBIDDEN_PATTERN}'")
    endif()
  endforeach()
endforeach()

if(OFFENDING_FILES)
  list(JOIN OFFENDING_FILES "\n" OFFENDING_OUTPUT)
  message(FATAL_ERROR
    "Noetic wrapper directly references backend internals:\n${OFFENDING_OUTPUT}")
endif()

file(READ "${ROOT_DIR}/noetic/src/n3mapping_noetic_node.cpp" NODE_CONTENTS)
string(FIND "${NODE_CONTENTS}" "void liveCloudCallback(" CLOUD_START)
if(CLOUD_START EQUAL -1)
  message(FATAL_ERROR "Noetic live cloud callback is missing")
endif()
string(SUBSTRING "${NODE_CONTENTS}" ${CLOUD_START} -1 CLOUD_CONTENTS)
string(FIND "${CLOUD_CONTENTS}" "realtime_output_->observeCloud(" RESERVATION_INDEX)
string(FIND "${CLOUD_CONTENTS}" "cloud_world_pub_.getNumSubscribers() == 0" SUBSCRIBERS_INDEX)
if(RESERVATION_INDEX EQUAL -1 OR SUBSCRIBERS_INDEX EQUAL -1 OR
    RESERVATION_INDEX GREATER SUBSCRIBERS_INDEX)
  message(FATAL_ERROR "Body-only subscribers must still reserve and publish projection association")
endif()
foreach(QUEUE_WIRING IN ITEMS
    "ros::VoidPtr(), &live_odom_queue_)"
    "live_nh.setCallbackQueue(&live_cloud_queue_)"
    "ros::AsyncSpinner>(1, &live_odom_queue_)"
    "ros::AsyncSpinner>(1, &live_cloud_queue_)"
    "realtime_output_->isCurrent(*projected)")
  string(FIND "${NODE_CONTENTS}" "${QUEUE_WIRING}" WIRING_INDEX)
  if(WIRING_INDEX EQUAL -1)
    message(FATAL_ERROR "Noetic live output queue/projection contract missing: ${QUEUE_WIRING}")
  endif()
endforeach()
