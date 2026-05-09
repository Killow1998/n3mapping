if(NOT DEFINED N3MAPPING_SOURCE_DIR)
  message(FATAL_ERROR "N3MAPPING_SOURCE_DIR is required")
endif()

set(scan_dirs
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/core"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/lio"
  "${N3MAPPING_SOURCE_DIR}/src/core"
  "${N3MAPPING_SOURCE_DIR}/src/lio"
)

set(scan_files
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/config.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/keyframe.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/keyframe_manager.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/loop_detector.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/loop_closure_manager.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/graph_optimizer.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/map_serializer.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/mapping_resuming.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/point_cloud_matcher.h"
  "${N3MAPPING_SOURCE_DIR}/include/n3mapping/world_localizing.h"
  "${N3MAPPING_SOURCE_DIR}/src/keyframe_manager.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/loop_detector.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/loop_closure_manager.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/graph_optimizer.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/map_serializer.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/mapping_resuming.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/point_cloud_matcher.cpp"
  "${N3MAPPING_SOURCE_DIR}/src/world_localizing.cpp"
)

foreach(dir IN LISTS scan_dirs)
  file(GLOB_RECURSE files_in_dir
    "${dir}/*.h"
    "${dir}/*.hpp"
    "${dir}/*.cpp"
  )
  list(APPEND scan_files ${files_in_dir})
endforeach()

list(REMOVE_DUPLICATES scan_files)

set(forbidden_patterns
  "#[ \t]*include[ \t]*[<\"]rclcpp/"
  "#[ \t]*include[ \t]*[<\"]ros/"
  "#[ \t]*include[ \t]*[<\"]roscpp/"
  "#[ \t]*include[ \t]*[<\"]std_msgs/"
  "#[ \t]*include[ \t]*[<\"]sensor_msgs/"
  "#[ \t]*include[ \t]*[<\"]nav_msgs/"
  "#[ \t]*include[ \t]*[<\"]geometry_msgs/"
  "#[ \t]*include[ \t]*[<\"]visualization_msgs/"
  "#[ \t]*include[ \t]*[<\"]tf2"
  "#[ \t]*include[ \t]*[<\"]message_filters/"
  "#[ \t]*include[ \t]*[<\"]cv_bridge/"
  "#[ \t]*include[ \t]*[<\"]pcl_conversions/"
  "#[ \t]*include[ \t]*[<\"]livox_ros_driver2/"
)

set(violations "")
foreach(path IN LISTS scan_files)
  if(NOT EXISTS "${path}")
    continue()
  endif()
  file(READ "${path}" content)
  foreach(pattern IN LISTS forbidden_patterns)
    if(content MATCHES "${pattern}")
      list(APPEND violations "${path}: ${pattern}")
    endif()
  endforeach()
endforeach()

if(violations)
  string(REPLACE ";" "\n" formatted "${violations}")
  message(FATAL_ERROR
    "n3mapping core/lio sources must not include ROS or ROS-message headers:\n${formatted}")
endif()

message(STATUS "n3mapping core/lio sources have no ROS-message includes")
