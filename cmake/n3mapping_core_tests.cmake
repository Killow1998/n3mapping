# One ROS-free test catalog, registered through each distribution's native tool.
function(n3mapping_add_core_test name)
  cmake_parse_arguments(TEST "MOCK" "" "SOURCES;LIBRARIES" ${ARGN})
  if(NOT TEST_SOURCES)
    set(TEST_SOURCES ${N3MAPPING_ROOT}/test/${name}.cpp)
  endif()
  list(APPEND TEST_SOURCES ${N3MAPPING_ROOT}/test/n3mapping_gtest_main.cpp)
  if(COMMAND ament_add_gtest)
    if(TEST_MOCK)
      ament_add_gmock(${name} ${TEST_SOURCES} TIMEOUT 60 SKIP_LINKING_MAIN_LIBRARIES)
    else()
      ament_add_gtest(${name} ${TEST_SOURCES} TIMEOUT 60 SKIP_LINKING_MAIN_LIBRARIES)
    endif()
  else()
    if(TEST_MOCK)
      catkin_add_gmock(${name} ${TEST_SOURCES})
    else()
      catkin_add_gtest(${name} ${TEST_SOURCES})
    endif()
  endif()
  if(TARGET ${name})
    # Core tests must not inherit either wrapper's ROS include directories.
    set_property(TARGET ${name} PROPERTY INCLUDE_DIRECTORIES "")
    target_include_directories(${name} PRIVATE
      ${N3MAPPING_ROOT}/include ${PROTO_GEN_DIR} ${PCL_INCLUDE_DIRS}
      ${GTEST_INCLUDE_DIRS} ${GMOCK_INCLUDE_DIRS})
    if(TEST_MOCK)
      target_compile_definitions(${name} PRIVATE N3MAPPING_TEST_GMOCK)
    endif()
    target_link_libraries(${name} ${TEST_LIBRARIES})
  endif()
endfunction()

foreach(name
    test_async_log
    test_config
    test_config_wiring
    test_core_types
    test_global_map_cache
    test_graph_optimizer
    test_graph_optimizer_pbt
    test_graph_solver_execution
    test_keyframe_manager
    test_local_map_selector
    test_localization_atlas
    test_loop_consensus_verifier
    test_loop_debug_logger
    test_loop_detector
    test_loop_detector_pbt
    test_loop_graph_trial_diagnostics
    test_loop_heightmap_diagnostics
    test_map_serializer_pbt
    test_mapping_resuming
    test_mapping_resuming_pbt
    test_n3mapping_core
    test_n3mapping_filter_nav_pbstream
    test_odometry_sanity
    test_point_cloud_matcher_pbt
    test_realtime_projection_contract
    test_relocalization_debug_logger
    test_relocalization_target_provider
    test_static_start_guard
    test_submap_builder
    test_submap_graph_trial
    test_submap_graph_trial_runtime
    test_synthetic_relocalization
    test_trajectory_checkpoint
    test_visibility_consistency
    test_world_localizing
)
  n3mapping_add_core_test(${name} LIBRARIES n3mapping_core)
endforeach()

n3mapping_add_core_test(test_realtime_odometry LIBRARIES ${PCL_LIBRARIES})
n3mapping_add_core_test(test_dense_trajectory_export LIBRARIES n3map_nav_resource_reader)
n3mapping_add_core_test(test_map_serializer LIBRARIES n3mapping_core n3map_nav_resource_reader)
n3mapping_add_core_test(test_loop_closure_manager MOCK LIBRARIES n3mapping_core)
n3mapping_add_core_test(test_point_cloud_matcher
  SOURCES ${N3MAPPING_ROOT}/test/test_point_cloud_matcher.cpp
          ${N3MAPPING_ROOT}/test/test_relocalization_candidate_evaluator.cpp
          ${N3MAPPING_ROOT}/test/test_registration_search.cpp
  LIBRARIES n3mapping_core)
n3mapping_add_core_test(test_localization_episode
  SOURCES ${N3MAPPING_ROOT}/test/test_localization_episode.cpp
          ${N3MAPPING_ROOT}/test/test_localization_recovery.cpp
  LIBRARIES n3mapping_core ${JSONCPP_LIBRARIES})
n3mapping_add_core_test(test_realtime_output LIBRARIES n3mapping_core ${JSONCPP_LIBRARIES})
n3mapping_add_core_test(test_joint_geometry LIBRARIES n3mapping_core ${JSONCPP_LIBRARIES})
foreach(name test_localization_episode test_realtime_output test_joint_geometry)
  if(TARGET ${name})
    target_include_directories(${name} PRIVATE ${JSONCPP_INCLUDE_DIRS})
  endif()
endforeach()

if(N3MAPPING_BUILD_RESEARCH_TOOLS)
  n3mapping_add_core_test(test_kitti360_reader LIBRARIES ${PCL_LIBRARIES})
  n3mapping_add_core_test(test_kitti360_eval LIBRARIES n3mapping_core)
  n3mapping_add_core_test(test_m2dgr_eval LIBRARIES n3mapping_core)
endif()
