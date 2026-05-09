if(NOT DEFINED N3MAPPING_SOURCE_DIR)
  message(FATAL_ERROR "N3MAPPING_SOURCE_DIR is required")
endif()

set(wrapper_dirs
  "${N3MAPPING_SOURCE_DIR}/src/ros2"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/ros2"
  "${N3MAPPING_SOURCE_DIR}/ros1"
)

set(forbidden_algorithm_names
  "RHPDescriptor"
  "HybridScancontext"
  "LoopDetector"
  "LoopClosureManager"
  "GraphOptimizer"
  "MapSerializer"
  "PointCloudMatcher"
  "WorldLocalizing"
  "MappingResuming"
  "KeyframeManager"
  "N3MappingSession"
)

set(violations "")
foreach(dir IN LISTS wrapper_dirs)
  if(NOT EXISTS "${dir}")
    continue()
  endif()
  file(GLOB_RECURSE wrapper_files
    "${dir}/*.h"
    "${dir}/*.hpp"
    "${dir}/*.cpp"
    "${dir}/*.cc"
  )
  foreach(path IN LISTS wrapper_files)
    get_filename_component(stem "${path}" NAME_WE)
    foreach(name IN LISTS forbidden_algorithm_names)
      if(stem STREQUAL "${name}")
        list(APPEND violations "${path}: wrapper file duplicates algorithm class name ${name}")
      endif()
    endforeach()
  endforeach()
endforeach()

if(violations)
  string(REPLACE ";" "\n" formatted "${violations}")
  message(FATAL_ERROR
    "ROS wrappers must call n3mapping_core/n3mapping_lio instead of duplicating algorithm files:\n${formatted}")
endif()

message(STATUS "ROS wrappers do not duplicate known algorithm implementation file names")
