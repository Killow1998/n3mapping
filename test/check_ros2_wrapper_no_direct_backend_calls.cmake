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
  "${ROOT_DIR}/src/n3mapping_node.cpp"
)

foreach(CHECK_DIR
    "${ROOT_DIR}/include/n3mapping/ros2"
    "${ROOT_DIR}/src/ros2")
  if(EXISTS "${CHECK_DIR}")
    file(GLOB_RECURSE WRAPPER_FILES
      "${CHECK_DIR}/*.h"
      "${CHECK_DIR}/*.hpp"
      "${CHECK_DIR}/*.hh"
      "${CHECK_DIR}/*.c"
      "${CHECK_DIR}/*.cc"
      "${CHECK_DIR}/*.cpp"
    )
    list(APPEND CHECK_FILES ${WRAPPER_FILES})
  endif()
endforeach()

set(OFFENDING_FILES "")

foreach(CHECK_FILE IN LISTS CHECK_FILES)
  if(NOT EXISTS "${CHECK_FILE}")
    continue()
  endif()

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
    "ROS2 wrapper directly references backend internals:\n${OFFENDING_OUTPUT}")
endif()
