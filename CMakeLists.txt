cmake_minimum_required(VERSION 3.10)
project(n3mapping)

# 定义包源目录宏，用于地图保存路径
add_compile_definitions(N3MAPPING_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
option(BUILD_TESTING "Build the testing tree." OFF)
option(BUILD_ROS2_WRAPPER "Build the ROS 2 wrapper target and node." ON)
option(BUILD_ROS1_WRAPPER "Build the ROS 1 wrapper package skeleton." OFF)
option(BUILD_FAST_LIO_CORE "Build the in-process FAST-LIO2 frontend core." OFF)
option(BUILD_DLIO_CORE "Build the in-process DLIO frontend core." OFF)
set(N3MAPPING_LIVOX_CUSTOM_MSG "AUTO" CACHE STRING "Livox CustomMsg support: AUTO, ON, or OFF.")
set_property(CACHE N3MAPPING_LIVOX_CUSTOM_MSG PROPERTY STRINGS AUTO ON OFF)

if(BUILD_ROS1_WRAPPER)
  message(FATAL_ERROR "BUILD_ROS1_WRAPPER is a source-layout placeholder in this ament package; build the ros1/ skeleton with catkin after the wrapper is implemented.")
endif()

# C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译优化
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall -Wextra)

# 查找 ROS2 依赖包
find_package(ament_cmake REQUIRED)
if(BUILD_ROS2_WRAPPER)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(message_filters REQUIRED)
  find_package(tf2 REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(tf2_geometry_msgs REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(cv_bridge REQUIRED)
  string(TOUPPER "${N3MAPPING_LIVOX_CUSTOM_MSG}" N3MAPPING_LIVOX_CUSTOM_MSG_MODE)
  if(NOT N3MAPPING_LIVOX_CUSTOM_MSG_MODE MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "N3MAPPING_LIVOX_CUSTOM_MSG must be AUTO, ON, or OFF")
  endif()
  if(NOT N3MAPPING_LIVOX_CUSTOM_MSG_MODE STREQUAL "OFF")
    if(N3MAPPING_LIVOX_CUSTOM_MSG_MODE STREQUAL "ON")
      find_package(livox_ros_driver2 REQUIRED)
    else()
      find_package(livox_ros_driver2 QUIET)
    endif()
  endif()
endif()

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

# GLOG
find_package(Glog REQUIRED)
if(NOT Glog_FOUND)
  message(FATAL_ERROR "glog not found!")
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

# 包含目录
include_directories(
  include
  include/Scancontext
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

# ROS-free LIO frontend interface/adapters. This stays separate from the
# mapping core so FAST-LIO2/DLIO extraction can evolve behind one boundary.
set(N3MAPPING_LIO_SOURCES
  src/lio/dlio_cloud_adapter.cpp
  src/lio/dlio_input_adapter.cpp
  src/lio/external_frontend.cpp
  src/lio/fast_lio_cloud_adapter.cpp
  src/lio/fast_lio_input_adapter.cpp
  src/lio/frontend_config.cpp
  src/lio/frontend_factory.cpp
  src/lio/imu_propagator.cpp
  src/lio/imu_sample_buffer.cpp
)

if(BUILD_FAST_LIO_CORE)
  list(APPEND N3MAPPING_LIO_SOURCES src/lio/fast_lio_frontend.cpp)
endif()
if(BUILD_DLIO_CORE)
  list(APPEND N3MAPPING_LIO_SOURCES src/lio/dlio_frontend.cpp)
endif()

add_library(${PROJECT_NAME}_lio SHARED
  ${N3MAPPING_LIO_SOURCES}
)

target_include_directories(${PROJECT_NAME}_lio PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}_lio
  ${PCL_LIBRARIES}
)

if(BUILD_FAST_LIO_CORE)
  target_compile_definitions(${PROJECT_NAME}_lio PUBLIC N3MAPPING_BUILD_FAST_LIO_CORE)
endif()
if(BUILD_DLIO_CORE)
  target_compile_definitions(${PROJECT_NAME}_lio PUBLIC N3MAPPING_BUILD_DLIO_CORE)
endif()

# ROS-free core library. This is the first boundary for future ROS1/ROS2 wrappers
# and in-process LIO frontend adapters.
set(N3MAPPING_CORE_SOURCES
  ${PROTO_SRCS}
  src/core/localization_mode_processor.cpp
  src/core/map_extension_mode_processor.cpp
  src/core/n3mapping_core.cpp
  src/core/n3mapping_session.cpp
  src/core/mapping_mode_processor.cpp
  src/core/pipeline_coordinator.cpp
  src/keyframe_manager.cpp
  src/point_cloud_matcher.cpp
  src/loop_detector.cpp
  src/loop_closure_manager.cpp
  src/graph_optimizer.cpp
  src/map_serializer.cpp
  src/world_localizing.cpp
  src/mapping_resuming.cpp
  src/RHPDescriptor.cpp
  include/Scancontext/Scancontext.cpp
  include/Scancontext/HybridScancontext.cpp
)

add_library(${PROJECT_NAME}_core SHARED
  ${N3MAPPING_CORE_SOURCES}
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}_core
  ${PROJECT_NAME}_lio
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  gtsam
  OpenMP::OpenMP_CXX
  small_gicp::small_gicp
  glog::glog
)

if(BUILD_FAST_LIO_CORE)
  add_library(${PROJECT_NAME}_fast_lio_core SHARED
    src/lio/fast_lio_core.cpp
  )
  target_include_directories(${PROJECT_NAME}_fast_lio_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${PROJECT_NAME}_fast_lio_core
    ${PROJECT_NAME}_lio
  )
  set_target_properties(${PROJECT_NAME}_fast_lio_core PROPERTIES
    OUTPUT_NAME fast_lio_core
  )
endif()

if(BUILD_DLIO_CORE)
  add_library(${PROJECT_NAME}_dlio_core SHARED
    src/lio/dlio_core.cpp
  )
  target_include_directories(${PROJECT_NAME}_dlio_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${PROJECT_NAME}_dlio_core
    ${PROJECT_NAME}_lio
  )
  set_target_properties(${PROJECT_NAME}_dlio_core PROPERTIES
    OUTPUT_NAME dlio_core
  )
endif()

if(BUILD_ROS2_WRAPPER)
  set(N3MAPPING_ROS2_SOURCES
    src/ros2/config_ros.cpp
    src/ros2/conversions.cpp
    src/ros2/mode_handlers.cpp
  )

  add_library(${PROJECT_NAME}_ros2 SHARED
    ${N3MAPPING_ROS2_SOURCES}
  )

  add_library(${PROJECT_NAME}_lib ALIAS ${PROJECT_NAME}_ros2)

  target_include_directories(${PROJECT_NAME}_ros2 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
    $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(${PROJECT_NAME}_ros2
    ${PROJECT_NAME}_core
  )

  ament_target_dependencies(${PROJECT_NAME}_ros2
    rclcpp
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    message_filters
    tf2
    tf2_ros
    tf2_geometry_msgs
    pcl_conversions
    cv_bridge
  )
  if(livox_ros_driver2_FOUND)
    target_compile_definitions(${PROJECT_NAME}_ros2 PUBLIC N3MAPPING_HAS_LIVOX_ROS_DRIVER2)
    ament_target_dependencies(${PROJECT_NAME}_ros2
      livox_ros_driver2
    )
    message(STATUS "Livox CustomMsg input enabled for n3mapping_ros2")
  elseif(N3MAPPING_LIVOX_CUSTOM_MSG_MODE STREQUAL "OFF")
    message(STATUS "Livox CustomMsg input disabled by N3MAPPING_LIVOX_CUSTOM_MSG=OFF")
  else()
    message(STATUS "livox_ros_driver2 not found; Livox CustomMsg input is disabled. Use -DN3MAPPING_LIVOX_CUSTOM_MSG=ON to require it.")
  endif()

  # 主节点
  add_executable(n3mapping_node
    src/ros2/n3mapping_node.cpp
  )

  target_link_libraries(n3mapping_node
    ${PROJECT_NAME}_ros2
  )
endif()

# 安装
set(N3MAPPING_EXPORT_TARGETS ${PROJECT_NAME}_lio ${PROJECT_NAME}_core)
if(BUILD_FAST_LIO_CORE)
  list(APPEND N3MAPPING_EXPORT_TARGETS ${PROJECT_NAME}_fast_lio_core)
endif()
if(BUILD_DLIO_CORE)
  list(APPEND N3MAPPING_EXPORT_TARGETS ${PROJECT_NAME}_dlio_core)
endif()

install(TARGETS ${N3MAPPING_EXPORT_TARGETS}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_ROS2_WRAPPER)
  install(TARGETS ${PROJECT_NAME}_ros2
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )

  install(TARGETS n3mapping_node
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY ${PROTO_GEN_DIR}/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
)

if(BUILD_ROS2_WRAPPER)
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )

  install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
  )
endif()

# ==================== 测试 ====================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_gmock REQUIRED)

  add_test(NAME test_core_no_ros_dependencies
    COMMAND ${CMAKE_COMMAND}
      -DN3MAPPING_CHECK_LIBRARY=$<TARGET_FILE:${PROJECT_NAME}_core>
      -DN3MAPPING_CHECK_TARGET_NAME=${PROJECT_NAME}_core
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_core_no_ros_deps.cmake
  )
  add_test(NAME test_lio_no_ros_dependencies
    COMMAND ${CMAKE_COMMAND}
      -DN3MAPPING_CHECK_LIBRARY=$<TARGET_FILE:${PROJECT_NAME}_lio>
      -DN3MAPPING_CHECK_TARGET_NAME=${PROJECT_NAME}_lio
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_core_no_ros_deps.cmake
  )
  add_test(NAME test_core_no_ros_includes
    COMMAND ${CMAKE_COMMAND}
      -DN3MAPPING_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_core_no_ros_includes.cmake
  )
  add_test(NAME test_wrapper_no_algorithm_duplication
    COMMAND ${CMAKE_COMMAND}
      -DN3MAPPING_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_wrapper_no_algorithm_duplication.cmake
  )
  ament_add_gtest(test_config_defaults test/test_config_defaults.cpp)
  if(TARGET test_config_defaults)
    target_link_libraries(test_config_defaults
      ${PROJECT_NAME}_lio
    )
  endif()

  # ROS-free core facade smoke test
  ament_add_gtest(test_external_lio_frontend test/test_external_lio_frontend.cpp)
  if(TARGET test_external_lio_frontend)
    target_link_libraries(test_external_lio_frontend
      ${PROJECT_NAME}_lio
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_lio_frontend_factory test/test_lio_frontend_factory.cpp)
  if(TARGET test_lio_frontend_factory)
    target_link_libraries(test_lio_frontend_factory
      ${PROJECT_NAME}_lio
      ${PCL_LIBRARIES}
    )
  endif()

  if(BUILD_DLIO_CORE)
    ament_add_gtest(test_dlio_core test/test_dlio_core.cpp)
    if(TARGET test_dlio_core)
      target_link_libraries(test_dlio_core
        ${PROJECT_NAME}_dlio_core
        ${PROJECT_NAME}_lio
        ${PCL_LIBRARIES}
      )
    endif()
  endif()

  if(BUILD_FAST_LIO_CORE)
    ament_add_gtest(test_fast_lio_core test/test_fast_lio_core.cpp)
    if(TARGET test_fast_lio_core)
      target_link_libraries(test_fast_lio_core
        ${PROJECT_NAME}_fast_lio_core
        ${PROJECT_NAME}_lio
        ${PCL_LIBRARIES}
      )
    endif()
  endif()

  ament_add_gtest(test_fast_lio_cloud_adapter test/test_fast_lio_cloud_adapter.cpp)
  if(TARGET test_fast_lio_cloud_adapter)
    target_link_libraries(test_fast_lio_cloud_adapter
      ${PROJECT_NAME}_lio
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_fast_lio_input_adapter test/test_fast_lio_input_adapter.cpp)
  if(TARGET test_fast_lio_input_adapter)
    target_link_libraries(test_fast_lio_input_adapter
      ${PROJECT_NAME}_lio
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_dlio_cloud_adapter test/test_dlio_cloud_adapter.cpp)
  if(TARGET test_dlio_cloud_adapter)
    target_link_libraries(test_dlio_cloud_adapter
      ${PROJECT_NAME}_lio
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_dlio_input_adapter test/test_dlio_input_adapter.cpp)
  if(TARGET test_dlio_input_adapter)
    target_link_libraries(test_dlio_input_adapter
      ${PROJECT_NAME}_lio
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_imu_sample_buffer test/test_imu_sample_buffer.cpp)
  if(TARGET test_imu_sample_buffer)
    target_link_libraries(test_imu_sample_buffer
      ${PROJECT_NAME}_lio
    )
  endif()

  ament_add_gtest(test_imu_propagator test/test_imu_propagator.cpp)
  if(TARGET test_imu_propagator)
    target_link_libraries(test_imu_propagator
      ${PROJECT_NAME}_lio
    )
  endif()

  ament_add_gtest(test_localization_mode_processor test/test_localization_mode_processor.cpp)
  if(TARGET test_localization_mode_processor)
    target_link_libraries(test_localization_mode_processor
      ${PROJECT_NAME}_core
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
  endif()

  ament_add_gtest(test_n3mapping_core test/test_n3mapping_core.cpp)
  if(TARGET test_n3mapping_core)
    target_link_libraries(test_n3mapping_core
      ${PROJECT_NAME}_core
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_n3mapping_session test/test_n3mapping_session.cpp)
  if(TARGET test_n3mapping_session)
    target_link_libraries(test_n3mapping_session
      ${PROJECT_NAME}_core
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_mapping_mode_processor test/test_mapping_mode_processor.cpp)
  if(TARGET test_mapping_mode_processor)
    target_link_libraries(test_mapping_mode_processor
      ${PROJECT_NAME}_core
      ${PCL_LIBRARIES}
    )
  endif()

  ament_add_gtest(test_map_extension_mode_processor test/test_map_extension_mode_processor.cpp)
  if(TARGET test_map_extension_mode_processor)
    target_link_libraries(test_map_extension_mode_processor
      ${PROJECT_NAME}_core
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      gtsam
      OpenMP::OpenMP_CXX
    )
  endif()

  ament_add_gtest(test_pipeline_coordinator test/test_pipeline_coordinator.cpp)
  if(TARGET test_pipeline_coordinator)
    target_link_libraries(test_pipeline_coordinator
      ${PROJECT_NAME}_core
      ${PCL_LIBRARIES}
    )
  endif()

  if(BUILD_ROS2_WRAPPER)
  ament_add_gtest(test_ros2_conversions test/test_ros2_conversions.cpp)
  if(TARGET test_ros2_conversions)
    target_link_libraries(test_ros2_conversions
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
    )
    ament_target_dependencies(test_ros2_conversions
      sensor_msgs
      nav_msgs
      pcl_conversions
    )
    if(livox_ros_driver2_FOUND)
      ament_target_dependencies(test_ros2_conversions
        livox_ros_driver2
      )
    endif()
  endif()
  
  # KeyframeManager 单元测试
  ament_add_gtest(test_keyframe_manager test/test_keyframe_manager.cpp)
  if(TARGET test_keyframe_manager)
    target_link_libraries(test_keyframe_manager
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
    )
    ament_target_dependencies(test_keyframe_manager
      rclcpp
    )
  endif()

  # PointCloudMatcher 单元测试
  ament_add_gtest(test_point_cloud_matcher test/test_point_cloud_matcher.cpp)
  if(TARGET test_point_cloud_matcher)
    target_link_libraries(test_point_cloud_matcher
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_point_cloud_matcher
      rclcpp
    )
  endif()

  # PointCloudMatcher 属性测试 (Property-Based Test)
  ament_add_gtest(test_point_cloud_matcher_pbt test/test_point_cloud_matcher_pbt.cpp)
  if(TARGET test_point_cloud_matcher_pbt)
    target_link_libraries(test_point_cloud_matcher_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_point_cloud_matcher_pbt
      rclcpp
    )
  endif()

  # LoopDetector 单元测试
  ament_add_gtest(test_loop_detector test/test_loop_detector.cpp)
  if(TARGET test_loop_detector)
    target_link_libraries(test_loop_detector
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_loop_detector
      rclcpp
    )
  endif()

  # LoopDetector 属性测试 (Property-Based Test)
  ament_add_gtest(test_loop_detector_pbt test/test_loop_detector_pbt.cpp)
  if(TARGET test_loop_detector_pbt)
    target_link_libraries(test_loop_detector_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_loop_detector_pbt
      rclcpp
    )
  endif()

  # GraphOptimizer 单元测试
  ament_add_gtest(test_graph_optimizer test/test_graph_optimizer.cpp)
  if(TARGET test_graph_optimizer)
    target_link_libraries(test_graph_optimizer
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      gtsam
    )
    ament_target_dependencies(test_graph_optimizer
      rclcpp
    )
  endif()

  # GraphOptimizer 属性测试 (Property-Based Test)
  ament_add_gtest(test_graph_optimizer_pbt test/test_graph_optimizer_pbt.cpp)
  if(TARGET test_graph_optimizer_pbt)
    target_link_libraries(test_graph_optimizer_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      gtsam
    )
    ament_target_dependencies(test_graph_optimizer_pbt
      rclcpp
    )
  endif()

  # MapSerializer 单元测试
  ament_add_gtest(test_map_serializer test/test_map_serializer.cpp)
  if(TARGET test_map_serializer)
    target_link_libraries(test_map_serializer
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      gtsam
    )
    ament_target_dependencies(test_map_serializer
      rclcpp
    )
  endif()

  # MapSerializer 属性测试 (Property-Based Test)
  ament_add_gtest(test_map_serializer_pbt test/test_map_serializer_pbt.cpp)
  if(TARGET test_map_serializer_pbt)
    target_link_libraries(test_map_serializer_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      gtsam
    )
    ament_target_dependencies(test_map_serializer_pbt
      rclcpp
    )
  endif()

  # WorldLocalizing 单元测试
  ament_add_gtest(test_world_localizing test/test_world_localizing.cpp)
  if(TARGET test_world_localizing)
    target_link_libraries(test_world_localizing
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_world_localizing
      rclcpp
    )
  endif()

  # WorldLocalizing 属性测试 (Property-Based Test)
  ament_add_gtest(test_world_localizing_pbt test/test_world_localizing_pbt.cpp)
  if(TARGET test_world_localizing_pbt)
    target_link_libraries(test_world_localizing_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_world_localizing_pbt
      rclcpp
    )
  endif()

  # MappingResuming 单元测试
  ament_add_gtest(test_mapping_resuming test/test_mapping_resuming.cpp)
  if(TARGET test_mapping_resuming)
    target_link_libraries(test_mapping_resuming
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      gtsam
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_mapping_resuming
      rclcpp
    )
  endif()

  # MappingResuming 属性测试 (Property-Based Test)
  ament_add_gtest(test_mapping_resuming_pbt test/test_mapping_resuming_pbt.cpp)
  if(TARGET test_mapping_resuming_pbt)
    target_link_libraries(test_mapping_resuming_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
      ${OpenCV_LIBRARIES}
      ${PROTOBUF_LIBRARIES}
      gtsam
      OpenMP::OpenMP_CXX
    )
    ament_target_dependencies(test_mapping_resuming_pbt
      rclcpp
    )
  endif()

  # LoopClosureManager 单元测试 (Mock)
  ament_add_gmock(test_loop_closure_manager test/test_loop_closure_manager.cpp)
  if(TARGET test_loop_closure_manager)
    target_link_libraries(test_loop_closure_manager
      ${PROJECT_NAME}_ros2
    )
    ament_target_dependencies(test_loop_closure_manager
      rclcpp
    )
  endif()

  # N3MappingNode 属性测试 (Property-Based Test)
  ament_add_gtest(test_n3mapping_node_pbt test/test_n3mapping_node_pbt.cpp)
  if(TARGET test_n3mapping_node_pbt)
    target_link_libraries(test_n3mapping_node_pbt
      ${PROJECT_NAME}_ros2
      ${PCL_LIBRARIES}
    )
    ament_target_dependencies(test_n3mapping_node_pbt
      rclcpp
    )
  endif()
  endif()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
if(BUILD_ROS2_WRAPPER)
  ament_export_libraries(${PROJECT_NAME}_lio ${PROJECT_NAME}_core ${PROJECT_NAME}_ros2)
  ament_export_dependencies(
    rclcpp
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    visualization_msgs
    message_filters
    tf2
    tf2_ros
    tf2_geometry_msgs
    pcl_conversions
    cv_bridge
  )
  if(livox_ros_driver2_FOUND)
    ament_export_dependencies(livox_ros_driver2)
  endif()
else()
  ament_export_libraries(${PROJECT_NAME}_lio ${PROJECT_NAME}_core)
endif()

ament_package()
