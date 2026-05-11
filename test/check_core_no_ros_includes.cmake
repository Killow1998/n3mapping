set(FORBIDDEN_PATTERNS
  "rclcpp"
  "sensor_msgs"
  "nav_msgs"
  "geometry_msgs"
  "visualization_msgs"
  "tf2"
  "ros/ros.h"
  "message_filters"
)

set(CHECK_DIRS
  "${ROOT_DIR}/include/n3mapping/core"
  "${ROOT_DIR}/src/core"
)

set(OFFENDING_FILES "")

foreach(CHECK_DIR IN LISTS CHECK_DIRS)
  if(NOT EXISTS "${CHECK_DIR}")
    continue()
  endif()

  file(GLOB_RECURSE CANDIDATE_FILES
    "${CHECK_DIR}/*.h"
    "${CHECK_DIR}/*.hpp"
    "${CHECK_DIR}/*.hh"
    "${CHECK_DIR}/*.c"
    "${CHECK_DIR}/*.cc"
    "${CHECK_DIR}/*.cpp"
  )

  foreach(CANDIDATE_FILE IN LISTS CANDIDATE_FILES)
    file(READ "${CANDIDATE_FILE}" FILE_CONTENTS)
    foreach(FORBIDDEN_PATTERN IN LISTS FORBIDDEN_PATTERNS)
      string(FIND "${FILE_CONTENTS}" "${FORBIDDEN_PATTERN}" PATTERN_INDEX)
      if(NOT PATTERN_INDEX EQUAL -1)
        list(APPEND OFFENDING_FILES
          "${CANDIDATE_FILE}: forbidden token '${FORBIDDEN_PATTERN}'")
      endif()
    endforeach()
  endforeach()
endforeach()

if(OFFENDING_FILES)
  list(JOIN OFFENDING_FILES "\n" OFFENDING_OUTPUT)
  message(FATAL_ERROR
    "ROS includes leaked into core files:\n${OFFENDING_OUTPUT}")
endif()
