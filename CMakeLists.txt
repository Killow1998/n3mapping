cmake_minimum_required(VERSION 3.10)
project(n3mapping)

# 定义包源目录宏，用于地图保存路径
add_compile_definitions(N3MAPPING_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
option(BUILD_TESTING "Build the testing tree." OFF)

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
)

# PCL
find_package(PCL REQUIRED COMPONENTS common io filters)

# Eigen3
find_package(Eigen3 REQUIRED)

# OpenCV (for ScanContext)
find_package(OpenCV REQUIRED)

# GTSAM
find_package(GTSAM REQUIRED)

# Protobuf
find_package(Protobuf REQUIRED)

# small_gicp
find_package(small_gicp REQUIRED)

# GLOG (Ubuntu Noetic 的 libgoogle-glog-dev 仅提供 pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

message(STATUS "GLOG Include Dirs: ${GLOG_INCLUDE_DIRS}")

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Protobuf 生成
set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto_gen)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# 查找所有 proto 文件
file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")

# 生成 protobuf 代码
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

catkin_package(
  INCLUDE_DIRS include ${PROTO_GEN_DIR}
  LIBRARIES ${PROJECT_NAME}_lib
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs nav_msgs geometry_msgs visualization_msgs message_filters tf2 tf2_ros tf2_geometry_msgs pcl_ros pcl_conversions cv_bridge
)

# 包含目录
include_directories(
  include
  include/Scancontext
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PROTOBUF_INCLUDE_DIRS}
  ${PROTO_GEN_DIR}
  ${GLOG_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# 库文件
add_library(${PROJECT_NAME}_lib SHARED
  ${PROTO_SRCS}
  src/config.cpp
  src/keyframe_manager.cpp
  src/point_cloud_matcher.cpp
  src/loop_detector.cpp
  src/loop_closure_manager.cpp
  src/graph_optimizer.cpp
  src/map_serializer.cpp
  src/mode_handlers.cpp
  src/world_localizing.cpp
  src/mapping_resuming.cpp
  src/RHPDescriptor.cpp
  include/Scancontext/Scancontext.cpp
  include/Scancontext/HybridScancontext.cpp
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  gtsam
  OpenMP::OpenMP_CXX
  small_gicp::small_gicp
  ${GLOG_LIBRARIES}
)

# 主节点
add_executable(n3mapping_node
  src/n3mapping_node.cpp
)

target_link_libraries(n3mapping_node
  ${PROJECT_NAME}_lib
)

# 安装
install(TARGETS ${PROJECT_NAME}_lib n3mapping_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

# ==================== 测试 ====================
if(CATKIN_ENABLE_TESTING AND BUILD_TESTING)
  find_package(GTest REQUIRED)

  catkin_add_gtest(test_keyframe_manager test/test_keyframe_manager.cpp)
  if(TARGET test_keyframe_manager)
    target_link_libraries(test_keyframe_manager ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_point_cloud_matcher test/test_point_cloud_matcher.cpp)
  if(TARGET test_point_cloud_matcher)
    target_link_libraries(test_point_cloud_matcher ${PROJECT_NAME}_lib ${PCL_LIBRARIES} OpenMP::OpenMP_CXX ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_point_cloud_matcher_pbt test/test_point_cloud_matcher_pbt.cpp)
  if(TARGET test_point_cloud_matcher_pbt)
    target_link_libraries(test_point_cloud_matcher_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} OpenMP::OpenMP_CXX ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_loop_detector test/test_loop_detector.cpp)
  if(TARGET test_loop_detector)
    target_link_libraries(test_loop_detector ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_loop_detector_pbt test/test_loop_detector_pbt.cpp)
  if(TARGET test_loop_detector_pbt)
    target_link_libraries(test_loop_detector_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_graph_optimizer test/test_graph_optimizer.cpp)
  if(TARGET test_graph_optimizer)
    target_link_libraries(test_graph_optimizer ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_graph_optimizer_pbt test/test_graph_optimizer_pbt.cpp)
  if(TARGET test_graph_optimizer_pbt)
    target_link_libraries(test_graph_optimizer_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_map_serializer test/test_map_serializer.cpp)
  if(TARGET test_map_serializer)
    target_link_libraries(test_map_serializer ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_map_serializer_pbt test/test_map_serializer_pbt.cpp)
  if(TARGET test_map_serializer_pbt)
    target_link_libraries(test_map_serializer_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_n3mapping_node_pbt test/test_n3mapping_node_pbt.cpp)
  if(TARGET test_n3mapping_node_pbt)
    target_link_libraries(test_n3mapping_node_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_world_localizing test/test_world_localizing.cpp)
  if(TARGET test_world_localizing)
    target_link_libraries(test_world_localizing ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_world_localizing_pbt test/test_world_localizing_pbt.cpp)
  if(TARGET test_world_localizing_pbt)
    target_link_libraries(test_world_localizing_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_mapping_resuming test/test_mapping_resuming.cpp)
  if(TARGET test_mapping_resuming)
    target_link_libraries(test_mapping_resuming ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()

  catkin_add_gtest(test_mapping_resuming_pbt test/test_mapping_resuming_pbt.cpp)
  if(TARGET test_mapping_resuming_pbt)
    target_link_libraries(test_mapping_resuming_pbt ${PROJECT_NAME}_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES})
  endif()
endif()
