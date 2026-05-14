set(REQUIRED_FILES
  "${ROOT_DIR}/ros1/CMakeLists.txt"
  "${ROOT_DIR}/ros1/package.xml"
  "${ROOT_DIR}/ros1/include/n3mapping_ros1/config_ros1.h"
  "${ROOT_DIR}/ros1/include/n3mapping_ros1/conversions.h"
  "${ROOT_DIR}/ros1/src/config_ros1.cpp"
  "${ROOT_DIR}/ros1/src/conversions.cpp"
  "${ROOT_DIR}/ros1/src/n3mapping_ros1_node.cpp"
)

foreach(REQUIRED_FILE IN LISTS REQUIRED_FILES)
  if(NOT EXISTS "${REQUIRED_FILE}")
    message(FATAL_ERROR "Missing ROS1 wrapper skeleton file: ${REQUIRED_FILE}")
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
)

set(CHECK_FILES
  "${ROOT_DIR}/ros1/src/n3mapping_ros1_node.cpp"
  "${ROOT_DIR}/ros1/src/conversions.cpp"
  "${ROOT_DIR}/ros1/include/n3mapping_ros1/conversions.h"
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
    "ROS1 wrapper directly references backend internals:\n${OFFENDING_OUTPUT}")
endif()
