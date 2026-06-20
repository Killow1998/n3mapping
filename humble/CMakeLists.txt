cmake_minimum_required(VERSION 3.10)
project(n3mapping)

set(N3MAPPING_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
get_filename_component(N3MAPPING_ROOT "${N3MAPPING_ROOT}" ABSOLUTE)

add_compile_definitions(N3MAPPING_SOURCE_DIR="${N3MAPPING_ROOT}")
option(BUILD_TESTING "Build the testing tree." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Protobuf REQUIRED)
find_package(small_gicp REQUIRED)
find_package(Glog QUIET)
if(NOT Glog_FOUND AND NOT TARGET glog::glog)
  find_library(GLOG_LIBRARY glog REQUIRED)
  add_library(glog::glog UNKNOWN IMPORTED)
  set_target_properties(glog::glog PROPERTIES
    IMPORTED_LOCATION "${GLOG_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/include"
  )
endif()
if(NOT DEFINED GLOG_INCLUDE_DIRS)
  if(DEFINED GOOGLE_GLOG_INCLUDE_DIR)
    set(GLOG_INCLUDE_DIRS ${GOOGLE_GLOG_INCLUDE_DIR})
  elseif(TARGET glog::glog)
    get_target_property(GLOG_INCLUDE_DIRS glog::glog INTERFACE_INCLUDE_DIRECTORIES)
  else()
    set(GLOG_INCLUDE_DIRS "/usr/include")
  endif()
endif()
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(PROTO_DIR ${N3MAPPING_ROOT}/proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto_gen)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})
file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")
set(PROTO_SRCS)
set(PROTO_HDRS)
foreach(PROTO_FILE ${PROTO_FILES})
  get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
  list(APPEND PROTO_SRCS "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.cc")
  list(APPEND PROTO_HDRS "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.h")
  add_custom_command(
    OUTPUT "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.cc" "${PROTO_GEN_DIR}/${PROTO_NAME}.pb.h"
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${PROTO_GEN_DIR} -I${PROTO_DIR} ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf code for ${PROTO_NAME}"
  )
endforeach()

include_directories(
  ${N3MAPPING_ROOT}/include
  ${N3MAPPING_ROOT}/include/Scancontext
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PROTOBUF_INCLUDE_DIRS}
  ${PROTO_GEN_DIR}
  ${GLOG_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include(${N3MAPPING_ROOT}/cmake/n3mapping_core.cmake)
include(${N3MAPPING_ROOT}/cmake/n3mapping_nav_resource_reader.cmake)
include(${N3MAPPING_ROOT}/cmake/n3mapping_humble.cmake)
include(${N3MAPPING_ROOT}/cmake/n3mapping_tests.cmake)
include(${N3MAPPING_ROOT}/cmake/n3mapping_checks.cmake)

set(N3MAPPING_CORE_SOURCES
  ${PROTO_SRCS}
  ${N3MAPPING_ROOT}/src/config.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_core.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_session.cpp
  ${N3MAPPING_ROOT}/src/keyframe_manager.cpp
  ${N3MAPPING_ROOT}/src/global_map_cache.cpp
  ${N3MAPPING_ROOT}/src/point_cloud_matcher.cpp
  ${N3MAPPING_ROOT}/src/loop_detector.cpp
  ${N3MAPPING_ROOT}/src/loop_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/loop_heightmap_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/loop_graph_trial_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/loop_closure_manager.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer.cpp
  ${N3MAPPING_ROOT}/src/n3map_proto_utils.cpp
  ${N3MAPPING_ROOT}/src/map_serializer.cpp
  ${N3MAPPING_ROOT}/src/world_localizing.cpp
  ${N3MAPPING_ROOT}/src/relocalization_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming.cpp
  ${N3MAPPING_ROOT}/src/RHPDescriptor.cpp
  ${N3MAPPING_ROOT}/include/Scancontext/Scancontext.cpp
  ${N3MAPPING_ROOT}/include/Scancontext/HybridScancontext.cpp
)

set(N3MAPPING_HUMBLE_WRAPPER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/config_humble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/conversions.cpp
)

add_library(n3mapping_core SHARED
  ${N3MAPPING_CORE_SOURCES}
)
add_library(${PROJECT_NAME}_lib ALIAS n3mapping_core)
n3mapping_configure_core_target(n3mapping_core)

n3mapping_add_nav_resource_reader(n3map_nav_resource_reader)

add_library(n3mapping_humble_wrapper SHARED
  ${N3MAPPING_HUMBLE_WRAPPER_SOURCES}
)
n3mapping_configure_humble_wrapper_target(n3mapping_humble_wrapper)

add_executable(n3mapping_node
  ${CMAKE_CURRENT_SOURCE_DIR}/src/n3mapping_node.cpp
)
n3mapping_configure_humble_node_target(n3mapping_node)

add_executable(n3mapping_synthetic_relocalization_eval
  ${N3MAPPING_ROOT}/tools/n3mapping_synthetic_relocalization_eval.cpp
)
target_link_libraries(n3mapping_synthetic_relocalization_eval
  n3mapping_core
  ${PCL_LIBRARIES}
  glog::glog
)

add_executable(n3mapping_filter_nav_pbstream
  ${N3MAPPING_ROOT}/tools/n3mapping_filter_nav_pbstream.cpp
)
target_link_libraries(n3mapping_filter_nav_pbstream
  n3mapping_core
  ${PCL_LIBRARIES}
)

add_executable(n3mapping_kitti360_reader
  ${N3MAPPING_ROOT}/tools/n3mapping_kitti360_reader.cpp
)

add_executable(n3mapping_kitti360_eval
  ${N3MAPPING_ROOT}/tools/n3mapping_kitti360_eval.cpp
)
target_link_libraries(n3mapping_kitti360_eval
  n3mapping_core
  ${PCL_LIBRARIES}
  glog::glog
)

add_executable(n3mapping_m2dgr_eval
  ${N3MAPPING_ROOT}/tools/n3mapping_m2dgr_eval.cpp
)
target_link_libraries(n3mapping_m2dgr_eval
  n3mapping_core
  ${PCL_LIBRARIES}
  glog::glog
)

add_executable(n3mapping_synthetic_relocalization_publisher
  ${N3MAPPING_ROOT}/tools/n3mapping_synthetic_relocalization_publisher.cpp
)
target_link_libraries(n3mapping_synthetic_relocalization_publisher
  n3mapping_core
  ${PCL_LIBRARIES}
)
ament_target_dependencies(n3mapping_synthetic_relocalization_publisher
  rclcpp
  sensor_msgs
  nav_msgs
  pcl_conversions
)

add_executable(n3mapping_synthetic_relocalization_visualizer
  ${N3MAPPING_ROOT}/tools/n3mapping_synthetic_relocalization_visualizer.cpp
)
target_link_libraries(n3mapping_synthetic_relocalization_visualizer
  n3mapping_core
  ${PCL_LIBRARIES}
)
ament_target_dependencies(n3mapping_synthetic_relocalization_visualizer
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

install(TARGETS
  n3mapping_core
  n3map_nav_resource_reader
  n3mapping_humble_wrapper
  n3mapping_node
  n3mapping_synthetic_relocalization_eval
  n3mapping_filter_nav_pbstream
  n3mapping_kitti360_reader
  n3mapping_kitti360_eval
  n3mapping_m2dgr_eval
  n3mapping_synthetic_relocalization_publisher
  n3mapping_synthetic_relocalization_visualizer
  EXPORT export_n3mapping
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  ${N3MAPPING_ROOT}/tools/run_synthetic_relocalization_matrix.py
  ${N3MAPPING_ROOT}/tools/n3mapping_loop_debug_analyze.py
  ${N3MAPPING_ROOT}/tools/n3mapping_loop_candidate_benchmark.py
  ${N3MAPPING_ROOT}/tools/n3mapping_eval_matrix.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_n3mapping HAS_LIBRARY_TARGET)
ament_export_dependencies(
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_srvs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  small_gicp
)

install(DIRECTORY ${N3MAPPING_ROOT}/include/
  DESTINATION include
)

install(DIRECTORY include/
  DESTINATION include
)

install(CODE "
  file(REMOVE
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/launch/mapping.launch\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/launch/localization.launch\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/launch/map_extension.launch\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/launch/synthetic_relocalization_visualization.launch\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/launch/n3_noetic.rviz\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/launch/synthetic_relocalization_visualization_noetic.rviz\"
  )
")

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY ${N3MAPPING_ROOT}/config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  n3mapping_find_test_dependencies()

  ament_add_gtest(test_keyframe_manager ${N3MAPPING_ROOT}/test/test_keyframe_manager.cpp)
  if(TARGET test_keyframe_manager)
    target_link_libraries(test_keyframe_manager ${PROJECT_NAME}_lib ${PCL_LIBRARIES})
    ament_target_dependencies(test_keyframe_manager rclcpp)
  endif()

  ament_add_gtest(test_global_map_cache ${N3MAPPING_ROOT}/test/test_global_map_cache.cpp)
  if(TARGET test_global_map_cache)
    target_link_libraries(test_global_map_cache ${PROJECT_NAME}_lib ${PCL_LIBRARIES})
    ament_target_dependencies(test_global_map_cache rclcpp)
  endif()

  ament_add_gtest(test_config ${N3MAPPING_ROOT}/test/test_config.cpp)
  if(TARGET test_config)
    target_link_libraries(test_config ${PROJECT_NAME}_lib)
    ament_target_dependencies(test_config rclcpp)
  endif()

  ament_add_gtest(test_n3mapping_core ${N3MAPPING_ROOT}/test/test_n3mapping_core.cpp)
  if(TARGET test_n3mapping_core)
    target_link_libraries(test_n3mapping_core ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
    ament_target_dependencies(test_n3mapping_core rclcpp)
  endif()

  ament_add_gtest(test_point_cloud_matcher ${N3MAPPING_ROOT}/test/test_point_cloud_matcher.cpp)
  if(TARGET test_point_cloud_matcher)
    target_link_libraries(test_point_cloud_matcher ${PROJECT_NAME}_lib ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_point_cloud_matcher rclcpp)
  endif()

  ament_add_gtest(test_point_cloud_matcher_pbt ${N3MAPPING_ROOT}/test/test_point_cloud_matcher_pbt.cpp)
  if(TARGET test_point_cloud_matcher_pbt)
    target_link_libraries(test_point_cloud_matcher_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_point_cloud_matcher_pbt rclcpp)
  endif()

  ament_add_gtest(test_loop_detector ${N3MAPPING_ROOT}/test/test_loop_detector.cpp)
  if(TARGET test_loop_detector)
    target_link_libraries(test_loop_detector ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_loop_detector rclcpp)
  endif()

  ament_add_gtest(test_loop_debug_logger ${N3MAPPING_ROOT}/test/test_loop_debug_logger.cpp)
  if(TARGET test_loop_debug_logger)
    target_link_libraries(test_loop_debug_logger ${PROJECT_NAME}_lib)
    ament_target_dependencies(test_loop_debug_logger rclcpp)
  endif()

  ament_add_gtest(test_loop_heightmap_diagnostics ${N3MAPPING_ROOT}/test/test_loop_heightmap_diagnostics.cpp)
  if(TARGET test_loop_heightmap_diagnostics)
    target_link_libraries(test_loop_heightmap_diagnostics ${PROJECT_NAME}_lib)
    ament_target_dependencies(test_loop_heightmap_diagnostics rclcpp)
  endif()

  ament_add_gtest(test_loop_graph_trial_diagnostics ${N3MAPPING_ROOT}/test/test_loop_graph_trial_diagnostics.cpp)
  if(TARGET test_loop_graph_trial_diagnostics)
    target_link_libraries(test_loop_graph_trial_diagnostics ${PROJECT_NAME}_lib)
    ament_target_dependencies(test_loop_graph_trial_diagnostics rclcpp)
  endif()

  ament_add_gtest(test_loop_detector_pbt ${N3MAPPING_ROOT}/test/test_loop_detector_pbt.cpp)
  if(TARGET test_loop_detector_pbt)
    target_link_libraries(test_loop_detector_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_loop_detector_pbt rclcpp)
  endif()

  ament_add_gtest(test_graph_optimizer ${N3MAPPING_ROOT}/test/test_graph_optimizer.cpp)
  if(TARGET test_graph_optimizer)
    target_link_libraries(test_graph_optimizer ${PROJECT_NAME}_lib ${PCL_LIBRARIES} gtsam)
    ament_target_dependencies(test_graph_optimizer rclcpp)
  endif()

  ament_add_gtest(test_graph_optimizer_pbt ${N3MAPPING_ROOT}/test/test_graph_optimizer_pbt.cpp)
  if(TARGET test_graph_optimizer_pbt)
    target_link_libraries(test_graph_optimizer_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} gtsam)
    ament_target_dependencies(test_graph_optimizer_pbt rclcpp)
  endif()

  ament_add_gtest(test_map_serializer ${N3MAPPING_ROOT}/test/test_map_serializer.cpp)
  if(TARGET test_map_serializer)
    target_link_libraries(test_map_serializer ${PROJECT_NAME}_lib n3map_nav_resource_reader ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
    ament_target_dependencies(test_map_serializer rclcpp)
  endif()

  ament_add_gtest(test_n3mapping_filter_nav_pbstream ${N3MAPPING_ROOT}/test/test_n3mapping_filter_nav_pbstream.cpp)
  if(TARGET test_n3mapping_filter_nav_pbstream)
    target_link_libraries(test_n3mapping_filter_nav_pbstream ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
    ament_target_dependencies(test_n3mapping_filter_nav_pbstream rclcpp)
  endif()

  ament_add_gtest(test_kitti360_reader ${N3MAPPING_ROOT}/test/test_kitti360_reader.cpp)

  ament_add_gtest(test_kitti360_eval ${N3MAPPING_ROOT}/test/test_kitti360_eval.cpp)
  if(TARGET test_kitti360_eval)
    target_link_libraries(test_kitti360_eval ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
  endif()

  ament_add_gtest(test_m2dgr_eval ${N3MAPPING_ROOT}/test/test_m2dgr_eval.cpp)
  if(TARGET test_m2dgr_eval)
    target_link_libraries(test_m2dgr_eval ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
  endif()

  ament_add_gtest(test_map_serializer_pbt ${N3MAPPING_ROOT}/test/test_map_serializer_pbt.cpp)
  if(TARGET test_map_serializer_pbt)
    target_link_libraries(test_map_serializer_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
    ament_target_dependencies(test_map_serializer_pbt rclcpp)
  endif()

  ament_add_gtest(test_world_localizing ${N3MAPPING_ROOT}/test/test_world_localizing.cpp)
  if(TARGET test_world_localizing)
    target_link_libraries(test_world_localizing ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_world_localizing rclcpp)
  endif()

  ament_add_gtest(test_relocalization_debug_logger ${N3MAPPING_ROOT}/test/test_relocalization_debug_logger.cpp)
  if(TARGET test_relocalization_debug_logger)
    target_link_libraries(test_relocalization_debug_logger ${PROJECT_NAME}_lib)
    ament_target_dependencies(test_relocalization_debug_logger rclcpp)
  endif()

  ament_add_gtest(test_world_localizing_pbt ${N3MAPPING_ROOT}/test/test_world_localizing_pbt.cpp)
  if(TARGET test_world_localizing_pbt)
    target_link_libraries(test_world_localizing_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_world_localizing_pbt rclcpp)
  endif()

  ament_add_gtest(test_mapping_resuming ${N3MAPPING_ROOT}/test/test_mapping_resuming.cpp)
  if(TARGET test_mapping_resuming)
    target_link_libraries(test_mapping_resuming ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_mapping_resuming rclcpp)
  endif()

  ament_add_gtest(test_mapping_resuming_pbt ${N3MAPPING_ROOT}/test/test_mapping_resuming_pbt.cpp)
  if(TARGET test_mapping_resuming_pbt)
    target_link_libraries(test_mapping_resuming_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam OpenMP::OpenMP_CXX)
    ament_target_dependencies(test_mapping_resuming_pbt rclcpp)
  endif()

  ament_add_gmock(test_loop_closure_manager ${N3MAPPING_ROOT}/test/test_loop_closure_manager.cpp)
  if(TARGET test_loop_closure_manager)
    target_link_libraries(test_loop_closure_manager ${PROJECT_NAME}_lib)
    ament_target_dependencies(test_loop_closure_manager rclcpp)
  endif()

  ament_add_gtest(test_n3mapping_node_pbt ${N3MAPPING_ROOT}/test/test_n3mapping_node_pbt.cpp)
  if(TARGET test_n3mapping_node_pbt)
    target_link_libraries(test_n3mapping_node_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES})
    ament_target_dependencies(test_n3mapping_node_pbt rclcpp)
  endif()

  n3mapping_add_static_checks()
endif()

ament_package()
