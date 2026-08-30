cmake_minimum_required(VERSION 3.10)
project(n3mapping)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(N3MAPPING_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
get_filename_component(N3MAPPING_ROOT "${N3MAPPING_ROOT}" ABSOLUTE)

add_compile_definitions(N3MAPPING_SOURCE_DIR="${N3MAPPING_ROOT}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(N3MAPPING_BUILD_RESEARCH_TOOLS
  "Build and install offline synthetic/dataset/research tools." OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
include(${N3MAPPING_ROOT}/cmake/n3mapping_product_identity.cmake)
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
  message_generation
)

add_message_files(
  DIRECTORY msg
  BASE_DIR ${N3MAPPING_ROOT}/msg
  FILES RelocalizationStatus.msg
)
generate_messages(
  DEPENDENCIES std_msgs geometry_msgs
)

find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(TBB REQUIRED)
find_package(small_gicp REQUIRED)
include(${N3MAPPING_ROOT}/cmake/n3mapping_dependency_abi.cmake)
n3mapping_verify_dependency_abi()
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

add_library(n3mapping_core SHARED
  ${N3MAPPING_CORE_SOURCES}
)
n3mapping_guard_product_target(n3mapping_core)
add_library(${PROJECT_NAME}_lib ALIAS n3mapping_core)
n3mapping_configure_core_target(n3mapping_core)
target_include_directories(n3mapping_core PUBLIC ${catkin_INCLUDE_DIRS})

n3mapping_add_nav_resource_reader(n3map_nav_resource_reader)

add_library(n3mapping_noetic_wrapper SHARED
  src/config_noetic.cpp
  src/conversions.cpp
)
n3mapping_guard_product_target(n3mapping_noetic_wrapper)
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
n3mapping_guard_product_target(n3mapping_node)
target_link_libraries(n3mapping_node
  n3mapping_noetic_wrapper
  ${catkin_LIBRARIES}
)
add_dependencies(n3mapping_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
set_target_properties(n3mapping_node PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

add_executable(n3mapping_localization_atlas_compile
  ${N3MAPPING_ROOT}/tools/n3mapping_localization_atlas_compile.cpp
)
target_link_libraries(n3mapping_localization_atlas_compile
  n3mapping_core
  glog::glog
)
set_target_properties(n3mapping_localization_atlas_compile PROPERTIES
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

add_executable(n3mapping_dense_trajectory_export
  ${N3MAPPING_ROOT}/tools/n3mapping_dense_trajectory_export.cpp
)
target_link_libraries(n3mapping_dense_trajectory_export
  n3map_nav_resource_reader
  OpenSSL::Crypto
)
set_target_properties(n3mapping_dense_trajectory_export PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

add_executable(n3mapping_relocalization_manifest_eval
  ${N3MAPPING_ROOT}/tools/n3mapping_relocalization_manifest_eval.cpp
)
n3mapping_guard_product_target(n3mapping_relocalization_manifest_eval)
target_link_libraries(n3mapping_relocalization_manifest_eval
  n3mapping_core
  ${PCL_LIBRARIES}
  glog::glog
)
set_target_properties(n3mapping_relocalization_manifest_eval PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
)

if(N3MAPPING_BUILD_RESEARCH_TOOLS)
  add_executable(n3mapping_synthetic_relocalization_eval
    ${N3MAPPING_ROOT}/tools/n3mapping_synthetic_relocalization_eval.cpp
  )
  target_link_libraries(n3mapping_synthetic_relocalization_eval
    n3mapping_core
    ${PCL_LIBRARIES}
    glog::glog
  )
  set_target_properties(n3mapping_synthetic_relocalization_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
  )

  add_executable(n3mapping_kitti360_reader
    ${N3MAPPING_ROOT}/tools/n3mapping_kitti360_reader.cpp
  )
  set_target_properties(n3mapping_kitti360_reader PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
  )

  add_executable(n3mapping_kitti360_eval
    ${N3MAPPING_ROOT}/tools/n3mapping_kitti360_eval.cpp
  )
  target_link_libraries(n3mapping_kitti360_eval
    n3mapping_core
    ${PCL_LIBRARIES}
    glog::glog
  )
  set_target_properties(n3mapping_kitti360_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
  )

  add_executable(n3mapping_m2dgr_eval
    ${N3MAPPING_ROOT}/tools/n3mapping_m2dgr_eval.cpp
  )
  target_link_libraries(n3mapping_m2dgr_eval
    n3mapping_core
    ${PCL_LIBRARIES}
    glog::glog
    OpenSSL::Crypto
  )
  set_target_properties(n3mapping_m2dgr_eval PROPERTIES
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
endif()

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
    message_runtime
)

install(TARGETS
  n3mapping_core
  n3map_nav_resource_reader
  n3mapping_noetic_wrapper
  n3mapping_node
  n3mapping_localization_atlas_compile
  n3mapping_filter_nav_pbstream
  n3mapping_dense_trajectory_export
  n3mapping_relocalization_manifest_eval
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  ${N3MAPPING_ROOT}/tools/extract_ros1_relocalization_manifest.py
  ${N3MAPPING_ROOT}/tools/n3mapping_ros1_raw_bag_inventory.py
  ${N3MAPPING_ROOT}/tools/n3mapping_same_frame_reference_join.py
  ${N3MAPPING_ROOT}/tools/n3mapping_product_bundle.py
  ${N3MAPPING_ROOT}/tools/n3mapping_support_bundle.py
  ${N3MAPPING_ROOT}/tools/n3mapping_product_gate.py
  ${N3MAPPING_ROOT}/tools/n3mapping_product_identity.py
  ${N3MAPPING_ROOT}/tools/n3mapping_product_preflight.py
  ${N3MAPPING_ROOT}/tools/n3mapping_product_runtime.py
  ${N3MAPPING_ROOT}/tools/n3mapping_deterministic_ros2_replay.py
  ${N3MAPPING_ROOT}/tools/n3mapping_raw_lio_replay_evidence.py
  ${N3MAPPING_ROOT}/tools/n3mapping_ros1_raw_bag_slice.py
  ${N3MAPPING_ROOT}/tools/n3mapping_submap_graph_trial_qualify.py
  ${N3MAPPING_ROOT}/tools/n3mapping_submap_graph_trial_compare.py
  ${N3MAPPING_ROOT}/tools/n3mapping_runtime_performance_analyze.py
  ${N3MAPPING_ROOT}/tools/n3mapping_process_resource_sample.py
  ${N3MAPPING_ROOT}/tools/n3mapping_runtime_performance_run.py
  ${N3MAPPING_ROOT}/tools/n3mapping_runtime_performance_gate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(N3MAPPING_BUILD_RESEARCH_TOOLS)
  install(TARGETS
    n3mapping_synthetic_relocalization_eval
    n3mapping_kitti360_reader
    n3mapping_kitti360_eval
    n3mapping_m2dgr_eval
    n3mapping_synthetic_relocalization_visualizer
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(PROGRAMS
    ${N3MAPPING_ROOT}/tools/n3mapping_loop_debug_analyze.py
    ${N3MAPPING_ROOT}/tools/n3mapping_loop_candidate_benchmark.py
    ${N3MAPPING_ROOT}/tools/n3mapping_dataset_readiness.py
    ${N3MAPPING_ROOT}/tools/n3mapping_episode_freeze.py
    ${N3MAPPING_ROOT}/tools/n3mapping_episode_benchmark.py
    ${N3MAPPING_ROOT}/tools/n3mapping_episode_diagnose.py
    ${N3MAPPING_ROOT}/tools/n3mapping_episode_review_prepare.py
    ${N3MAPPING_ROOT}/tools/n3mapping_surface_overlap_audit.py
    ${N3MAPPING_ROOT}/tools/n3mapping_runtime_signal_audit.py
    ${N3MAPPING_ROOT}/tools/n3mapping_keyframe_surface_support_audit.py
    ${N3MAPPING_ROOT}/tools/n3mapping_multiview_free_space_audit.py
    ${N3MAPPING_ROOT}/tools/n3mapping_verifier_pair_freeze.py
    ${N3MAPPING_ROOT}/tools/n3mapping_verifier_split_freeze.py
    ${N3MAPPING_ROOT}/tools/n3mapping_frozen_bev_verifier_audit.py
    ${N3MAPPING_ROOT}/tools/n3mapping_eval_matrix.py
    ${N3MAPPING_ROOT}/tools/n3mapping_fa02_freeze.py
    ${N3MAPPING_ROOT}/tools/n3mapping_fa02_gate.py
    ${N3MAPPING_ROOT}/tools/n3mapping_fa02_run.py
    ${N3MAPPING_ROOT}/tools/n3mapping_fa02b_gate.py
    ${N3MAPPING_ROOT}/tools/n3mapping_fa02b_run.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

install(DIRECTORY ${N3MAPPING_ROOT}/include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY ${N3MAPPING_ROOT}/config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES
  launch/localization.launch
  launch/map_extension.launch
  launch/mapping.launch
  launch/n3_noetic.rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
if(N3MAPPING_BUILD_RESEARCH_TOOLS)
  install(FILES
    launch/synthetic_relocalization_visualization.launch
    launch/synthetic_relocalization_visualization_noetic.rviz
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  )
endif()

install(FILES ${N3MAPPING_ROOT}/requirements-product-tools.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  include(${N3MAPPING_ROOT}/cmake/n3mapping_checks.cmake)
  n3mapping_add_static_checks()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(
    NAME test_n3mapping_support_bundle
    COMMAND ${Python3_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/test_n3mapping_support_bundle.py
  )
  set_tests_properties(test_n3mapping_support_bundle PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_bundle
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_bundle.py
  )
  set_tests_properties(test_n3mapping_product_bundle PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_gate
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_gate.py
  )
  set_tests_properties(test_n3mapping_product_gate PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_runtime
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_runtime.py
  )
  set_tests_properties(test_n3mapping_product_runtime PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_preflight
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_preflight.py
  )
  set_tests_properties(test_n3mapping_product_preflight PROPERTIES TIMEOUT 60)

  add_test(
    NAME test_n3mapping_source_tree_guard
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_source_tree_guard.py
  )
  set_tests_properties(test_n3mapping_source_tree_guard PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_raw_lio_replay_evidence
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_raw_lio_replay_evidence.py
  )
  set_tests_properties(test_n3mapping_raw_lio_replay_evidence PROPERTIES TIMEOUT 60)

  add_test(
    NAME test_n3mapping_ros1_raw_bag_slice
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${Python3_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_ros1_raw_bag_slice.py
  )
  set_tests_properties(test_n3mapping_ros1_raw_bag_slice PROPERTIES TIMEOUT 60)

  catkin_add_gtest(test_config
    ${N3MAPPING_ROOT}/test/test_config.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_config)
    target_link_libraries(test_config n3mapping_core)
  endif()

  catkin_add_gtest(test_static_start_guard
    ${N3MAPPING_ROOT}/test/test_static_start_guard.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_static_start_guard)
    target_link_libraries(test_static_start_guard n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_odometry_sanity
    ${N3MAPPING_ROOT}/test/test_odometry_sanity.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_odometry_sanity)
    target_link_libraries(test_odometry_sanity n3mapping_core)
  endif()

  catkin_add_gtest(test_core_types
    ${N3MAPPING_ROOT}/test/test_core_types.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_core_types)
    target_link_libraries(test_core_types n3mapping_core ${PCL_LIBRARIES})
  endif()

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

  catkin_add_gtest(test_localization_atlas
    ${N3MAPPING_ROOT}/test/test_localization_atlas.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_localization_atlas)
    target_link_libraries(test_localization_atlas n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_world_localizing
    ${N3MAPPING_ROOT}/test/test_world_localizing.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_world_localizing)
    target_link_libraries(test_world_localizing n3mapping_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
  endif()

  catkin_add_gtest(test_config_wiring
    ${N3MAPPING_ROOT}/test/test_config_wiring.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_config_wiring)
    target_link_libraries(test_config_wiring n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_visibility_consistency
    ${N3MAPPING_ROOT}/test/test_visibility_consistency.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_visibility_consistency)
    target_link_libraries(test_visibility_consistency n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_relocalization_debug_logger
    ${N3MAPPING_ROOT}/test/test_relocalization_debug_logger.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_relocalization_debug_logger)
    target_link_libraries(test_relocalization_debug_logger n3mapping_core)
  endif()

  catkin_add_gtest(test_relocalization_target_provider
    ${N3MAPPING_ROOT}/test/test_relocalization_target_provider.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_relocalization_target_provider)
    target_link_libraries(test_relocalization_target_provider n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_local_map_selector
    ${N3MAPPING_ROOT}/test/test_local_map_selector.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_local_map_selector)
    target_link_libraries(test_local_map_selector n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_submap_builder
    ${N3MAPPING_ROOT}/test/test_submap_builder.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_submap_builder)
    target_link_libraries(test_submap_builder n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_submap_graph_trial
    ${N3MAPPING_ROOT}/test/test_submap_graph_trial.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_submap_graph_trial)
    target_link_libraries(test_submap_graph_trial n3mapping_core ${PCL_LIBRARIES})
  endif()

  catkin_add_gtest(test_submap_graph_trial_runtime
    ${N3MAPPING_ROOT}/test/test_submap_graph_trial_runtime.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_submap_graph_trial_runtime)
    target_link_libraries(test_submap_graph_trial_runtime n3mapping_core ${PCL_LIBRARIES})
  endif()

  if(N3MAPPING_BUILD_RESEARCH_TOOLS)
    catkin_add_gtest(test_kitti360_reader
      ${N3MAPPING_ROOT}/test/test_kitti360_reader.cpp
      ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
    )

    catkin_add_gtest(test_kitti360_eval
      ${N3MAPPING_ROOT}/test/test_kitti360_eval.cpp
      ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
    )
    if(TARGET test_kitti360_eval)
      target_link_libraries(test_kitti360_eval n3mapping_core ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
    endif()

    catkin_add_gtest(test_m2dgr_eval
      ${N3MAPPING_ROOT}/test/test_m2dgr_eval.cpp
      ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
    )
    if(TARGET test_m2dgr_eval)
      target_link_libraries(test_m2dgr_eval n3mapping_core ${PCL_LIBRARIES} ${PROTOBUF_LIBRARIES} gtsam)
    endif()

    add_test(
      NAME test_n3mapping_fa02_tools
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${PYTHON_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/test_n3mapping_fa02_tools.py
    )
    set_tests_properties(test_n3mapping_fa02_tools PROPERTIES TIMEOUT 60)

    add_test(
      NAME test_n3mapping_fa02b_tools
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${PYTHON_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/test_n3mapping_fa02b_tools.py
    )
    set_tests_properties(test_n3mapping_fa02b_tools PROPERTIES TIMEOUT 60)
  endif()

  catkin_add_gtest(test_synthetic_relocalization
    ${N3MAPPING_ROOT}/test/test_synthetic_relocalization.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_synthetic_relocalization)
    target_link_libraries(test_synthetic_relocalization n3mapping_core ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX)
  endif()

  catkin_add_gtest(test_dense_trajectory_export
    ${N3MAPPING_ROOT}/test/test_dense_trajectory_export.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_dense_trajectory_export)
    target_link_libraries(test_dense_trajectory_export
      n3map_nav_resource_reader
    )
  endif()

  add_test(
    NAME test_config_reaches_behaviour
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_config_reaches_behaviour.py
  )
  set_tests_properties(test_config_reaches_behaviour PROPERTIES TIMEOUT 60)

  add_test(
    NAME test_n3mapping_data_contract_tools
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_data_contract_tools.py
  )
  set_tests_properties(test_n3mapping_data_contract_tools PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_submap_graph_trial_qualify
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_submap_graph_trial_qualify.py
  )
  set_tests_properties(test_n3mapping_submap_graph_trial_qualify PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_submap_graph_trial_compare
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_submap_graph_trial_compare.py
  )
  set_tests_properties(test_n3mapping_submap_graph_trial_compare PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_runtime_performance_analyze
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_runtime_performance_analyze.py
  )
  set_tests_properties(test_n3mapping_runtime_performance_analyze PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_runtime_performance_run
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_runtime_performance_run.py
  )
  set_tests_properties(test_n3mapping_runtime_performance_run PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_runtime_performance_gate
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_runtime_performance_gate.py
  )
  set_tests_properties(test_n3mapping_runtime_performance_gate PROPERTIES TIMEOUT 30)
endif()
