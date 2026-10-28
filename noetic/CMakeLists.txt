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
add_compile_options(-Wall -Wextra)

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

include(${N3MAPPING_ROOT}/cmake/n3mapping_dependencies.cmake)

add_custom_target(n3mapping_proto_headers ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CATKIN_DEVEL_PREFIX}/include
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PROTO_HDRS} ${CATKIN_DEVEL_PREFIX}/include
  DEPENDS ${PROTO_HDRS})

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
n3mapping_configure_core_target(n3mapping_core)

n3mapping_add_nav_resource_reader(n3map_nav_resource_reader)
add_dependencies(n3map_nav_resource_reader n3mapping_proto_headers)

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

add_library(n3mapping_noetic_node_impl STATIC
  src/n3mapping_node_lifecycle.cpp
  src/n3mapping_node_logging.cpp
  src/n3mapping_node_live.cpp
  src/n3mapping_node_backend.cpp
  src/n3mapping_node_visualization.cpp
  src/n3mapping_node_map.cpp
)
n3mapping_guard_product_target(n3mapping_noetic_node_impl)
target_link_libraries(n3mapping_noetic_node_impl PUBLIC
  log4cxx
  n3mapping_noetic_wrapper
  ${catkin_LIBRARIES}
)
add_dependencies(n3mapping_noetic_node_impl
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
add_executable(n3mapping_node src/n3mapping_node.cpp)
n3mapping_guard_product_target(n3mapping_node)
target_link_libraries(n3mapping_node n3mapping_noetic_node_impl)
add_dependencies(n3mapping_node ${${PROJECT_NAME}_EXPORTED_TARGETS})
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
  add_executable(n3mapping_planar_geometry_eval
    ${N3MAPPING_ROOT}/tools/n3mapping_planar_geometry_eval.cpp
  )
  target_include_directories(n3mapping_planar_geometry_eval PRIVATE ${JSONCPP_INCLUDE_DIRS})
  target_link_libraries(n3mapping_planar_geometry_eval n3mapping_core ${JSONCPP_LIBRARIES} TBB::tbb)
  set_target_properties(n3mapping_planar_geometry_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
  )

  add_executable(n3mapping_registration_cross_eval
    ${N3MAPPING_ROOT}/tools/n3mapping_registration_cross_eval.cpp
  )
  target_link_libraries(n3mapping_registration_cross_eval
    n3mapping_core
    ${PCL_LIBRARIES}
    glog::glog
  )
  set_target_properties(n3mapping_registration_cross_eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
  )

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
install(FILES ${PROTO_HDRS} DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION})

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
  add_test(NAME test_logging_ros_e2e
    COMMAND ${PYTHON_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/logging_ros_e2e.py
      --ros noetic --node $<TARGET_FILE:n3mapping_node>)
  set_tests_properties(test_logging_ros_e2e PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)

  add_test(NAME test_map_extension_ros_e2e
    COMMAND ${PYTHON_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/map_extension_ros_e2e.py
      --ros noetic --node $<TARGET_FILE:n3mapping_node>)
  set_tests_properties(test_map_extension_ros_e2e PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)

  add_test(
    NAME test_noetic_output_smoke
    COMMAND ${PYTHON_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/noetic_output_smoke.py
      $<TARGET_FILE:n3mapping_node>
  )
  set_tests_properties(test_noetic_output_smoke PROPERTIES TIMEOUT 60)

  catkin_add_gtest(test_noetic_live_output_queues
    ${N3MAPPING_ROOT}/test/test_noetic_live_output_queues.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp
  )
  if(TARGET test_noetic_live_output_queues)
    target_link_libraries(test_noetic_live_output_queues ${catkin_LIBRARIES})
  endif()
  catkin_add_gtest(test_ros_output_contract
    ${N3MAPPING_ROOT}/test/test_ros_output_contract.cpp
    ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp)
  if(TARGET test_ros_output_contract)
    target_include_directories(test_ros_output_contract PRIVATE ${JSONCPP_INCLUDE_DIRS})
    target_link_libraries(test_ros_output_contract n3mapping_noetic_wrapper ${JSONCPP_LIBRARIES})
  endif()

  include(${N3MAPPING_ROOT}/cmake/n3mapping_checks.cmake)
  include(${N3MAPPING_ROOT}/cmake/n3mapping_core_tests.cmake)

  n3mapping_add_static_checks()

  add_test(
    NAME test_n3mapping_support_bundle
    COMMAND ${PYTHON_EXECUTABLE} -B ${N3MAPPING_ROOT}/test/test_n3mapping_support_bundle.py
  )
  set_tests_properties(test_n3mapping_support_bundle PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_bundle
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_bundle.py
  )
  set_tests_properties(test_n3mapping_product_bundle PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_gate
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_gate.py
  )
  set_tests_properties(test_n3mapping_product_gate PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_runtime
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_runtime.py
  )
  set_tests_properties(test_n3mapping_product_runtime PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_product_preflight
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_product_preflight.py
  )
  set_tests_properties(test_n3mapping_product_preflight PROPERTIES TIMEOUT 60)

  add_test(
    NAME test_n3mapping_source_tree_guard
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_source_tree_guard.py
  )
  set_tests_properties(test_n3mapping_source_tree_guard PROPERTIES TIMEOUT 30)

  add_test(
    NAME test_n3mapping_raw_lio_replay_evidence
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_raw_lio_replay_evidence.py
  )
  set_tests_properties(test_n3mapping_raw_lio_replay_evidence PROPERTIES TIMEOUT 60)

  add_test(
    NAME test_n3mapping_ros1_raw_bag_slice
    COMMAND ${CMAKE_COMMAND} -E env
      PYTHONDONTWRITEBYTECODE=1
      ${PYTHON_EXECUTABLE} -B
      ${N3MAPPING_ROOT}/test/test_n3mapping_ros1_raw_bag_slice.py
  )
  set_tests_properties(test_n3mapping_ros1_raw_bag_slice PROPERTIES TIMEOUT 60)

  if(N3MAPPING_BUILD_RESEARCH_TOOLS)

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
