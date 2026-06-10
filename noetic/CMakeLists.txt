cmake_minimum_required(VERSION 3.10)
project(n3mapping)

set(N3MAPPING_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
get_filename_component(N3MAPPING_ROOT "${N3MAPPING_ROOT}" ABSOLUTE)

add_compile_definitions(N3MAPPING_SOURCE_DIR="${N3MAPPING_ROOT}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  std_srvs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  message_filters
  pcl_ros
  pcl_conversions
)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PROTOBUF_INCLUDE_DIRS}
  ${PROTO_GEN_DIR}
  ${GLOG_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include(${N3MAPPING_ROOT}/cmake/n3mapping_core.cmake)
include(${N3MAPPING_ROOT}/cmake/n3mapping_nav_resource_reader.cmake)

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
  ${N3MAPPING_ROOT}/src/loop_closure_manager.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer.cpp
  ${N3MAPPING_ROOT}/src/n3map_proto_utils.cpp
  ${N3MAPPING_ROOT}/src/map_serializer.cpp
  ${N3MAPPING_ROOT}/src/world_localizing.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming.cpp
  ${N3MAPPING_ROOT}/src/RHPDescriptor.cpp
  ${N3MAPPING_ROOT}/include/Scancontext/Scancontext.cpp
  ${N3MAPPING_ROOT}/include/Scancontext/HybridScancontext.cpp
)

add_library(n3mapping_core SHARED
  ${N3MAPPING_CORE_SOURCES}
)
add_library(${PROJECT_NAME}_lib ALIAS n3mapping_core)
n3mapping_configure_core_target(n3mapping_core)
target_include_directories(n3mapping_core PUBLIC ${catkin_INCLUDE_DIRS})

n3mapping_add_nav_resource_reader(n3map_nav_resource_reader)

add_library(n3mapping_noetic_wrapper SHARED
  src/config_noetic.cpp
  src/conversions.cpp
)
target_include_directories(n3mapping_noetic_wrapper PUBLIC
  ${N3MAPPING_ROOT}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${catkin_INCLUDE_DIRS}
  ${PROTO_GEN_DIR}
)
target_link_libraries(n3mapping_noetic_wrapper
  n3mapping_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
set_target_properties(n3mapping_core n3mapping_noetic_wrapper PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib
)
set_target_properties(n3map_nav_resource_reader PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib
)

add_executable(n3mapping_node
  src/n3mapping_noetic_node.cpp
)
target_link_libraries(n3mapping_node
  n3mapping_noetic_wrapper
  ${catkin_LIBRARIES}
)
set_target_properties(n3mapping_node PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

add_executable(n3mapping_synthetic_relocalization_eval
  ${N3MAPPING_ROOT}/tools/n3mapping_synthetic_relocalization_eval.cpp
)
target_link_libraries(n3mapping_synthetic_relocalization_eval
  n3mapping_core
  ${PCL_LIBRARIES}
)
set_target_properties(n3mapping_synthetic_relocalization_eval PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

add_executable(n3mapping_filter_nav_pbstream
  ${N3MAPPING_ROOT}/tools/n3mapping_filter_nav_pbstream.cpp
)
target_link_libraries(n3mapping_filter_nav_pbstream
  n3mapping_core
  ${PCL_LIBRARIES}
)
set_target_properties(n3mapping_filter_nav_pbstream PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

add_executable(n3mapping_synthetic_relocalization_visualizer
  src/synthetic_relocalization_visualizer.cpp
)
target_link_libraries(n3mapping_synthetic_relocalization_visualizer
  n3mapping_core
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
set_target_properties(n3mapping_synthetic_relocalization_visualizer PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

catkin_package(
  INCLUDE_DIRS include ../include
  LIBRARIES n3mapping_core n3map_nav_resource_reader n3mapping_noetic_wrapper
  CATKIN_DEPENDS
    roscpp
    std_msgs
    std_srvs
    sensor_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    tf2_ros
    message_filters
    pcl_ros
    pcl_conversions
)

install(TARGETS
  n3mapping_core
  n3map_nav_resource_reader
  n3mapping_noetic_wrapper
  n3mapping_node
  n3mapping_synthetic_relocalization_eval
  n3mapping_filter_nav_pbstream
  n3mapping_synthetic_relocalization_visualizer
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY ${N3MAPPING_ROOT}/include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(CODE "
  file(REMOVE
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/mapping.launch.py\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/localization.launch.py\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/map_extension.launch.py\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/synthetic_relocalization_visualization.launch.py\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/n3.rviz\"
    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CATKIN_PACKAGE_SHARE_DESTINATION}/launch/synthetic_relocalization_visualization.rviz\"
  )
")

install(DIRECTORY ${N3MAPPING_ROOT}/config ${CMAKE_CURRENT_SOURCE_DIR}/launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  include(${N3MAPPING_ROOT}/cmake/n3mapping_checks.cmake)
  n3mapping_add_static_checks()

  catkin_add_gtest(test_loop_detector
    ${N3MAPPING_ROOT}/test/test_loop_detector.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_loop_detector)
    target_link_libraries(test_loop_detector n3mapping_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
  endif()

  catkin_add_gtest(test_loop_debug_logger
    ${N3MAPPING_ROOT}/test/test_loop_debug_logger.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_loop_debug_logger)
    target_link_libraries(test_loop_debug_logger n3mapping_core)
  endif()

  catkin_add_gtest(test_global_map_cache
    ${N3MAPPING_ROOT}/test/test_global_map_cache.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_global_map_cache)
    target_link_libraries(test_global_map_cache n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_world_localizing
    ${N3MAPPING_ROOT}/test/test_world_localizing.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_world_localizing)
    target_link_libraries(test_world_localizing n3mapping_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
  endif()

  catkin_add_gtest(test_synthetic_relocalization
    ${N3MAPPING_ROOT}/test/test_synthetic_relocalization.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_synthetic_relocalization)
    target_link_libraries(test_synthetic_relocalization n3mapping_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
  endif()
endif()
