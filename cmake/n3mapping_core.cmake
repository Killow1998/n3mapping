find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

set(N3MAPPING_CORE_SOURCES
  ${PROTO_SRCS}
  ${N3MAPPING_ROOT}/src/logging/async_log.cpp
  ${N3MAPPING_ROOT}/src/logging/log_store.cpp
  ${N3MAPPING_ROOT}/src/logging/process_logging.cpp
  ${N3MAPPING_ROOT}/src/logging/runtime_events.cpp
  ${N3MAPPING_ROOT}/src/logging/runtime_performance_fields.cpp
  ${N3MAPPING_ROOT}/src/config.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_core.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_mapping.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_extension.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_load.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_save.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_trajectory.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_constraints.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_localization.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_loop_closure.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_loop_async.cpp
  ${N3MAPPING_ROOT}/src/core/loop_closure_processing.cpp
  ${N3MAPPING_ROOT}/src/core/loop_closure_commit.cpp
  ${N3MAPPING_ROOT}/src/core/loop_closure_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/core/run_mode.cpp
  ${N3MAPPING_ROOT}/src/core/map_artifacts.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_session.cpp
  ${N3MAPPING_ROOT}/src/core/realtime_output.cpp
  ${N3MAPPING_ROOT}/src/ros_output_contract.cpp
  ${N3MAPPING_ROOT}/src/causal_submap.cpp
  ${N3MAPPING_ROOT}/src/keyframe_manager.cpp
  ${N3MAPPING_ROOT}/src/local_map_selector.cpp
  ${N3MAPPING_ROOT}/src/submap_builder.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_projection.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_factor.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_trial.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_trial_runtime.cpp
  ${N3MAPPING_ROOT}/src/global_map_cache.cpp
  ${N3MAPPING_ROOT}/src/point_cloud_matcher.cpp
  ${N3MAPPING_ROOT}/src/point_cloud_matcher_alignment.cpp
  ${N3MAPPING_ROOT}/src/point_cloud_registration.cpp
  ${N3MAPPING_ROOT}/src/loop_detector.cpp
  ${N3MAPPING_ROOT}/src/loop_detector_spatial.cpp
  ${N3MAPPING_ROOT}/src/loop_verifier.cpp
  ${N3MAPPING_ROOT}/src/loop_verification_pipeline.cpp
  ${N3MAPPING_ROOT}/src/loop_constraint_pipeline.cpp
  ${N3MAPPING_ROOT}/src/loop_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/loop_heightmap_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/loop_graph_trial_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/loop_segment_consistency.cpp
  ${N3MAPPING_ROOT}/src/loop_consensus_verifier.cpp
  ${N3MAPPING_ROOT}/src/loop_consensus_registration.cpp
  ${N3MAPPING_ROOT}/src/loop_closure_manager.cpp
  ${N3MAPPING_ROOT}/src/graph_factor_noise.cpp
  ${N3MAPPING_ROOT}/src/graph_solver_execution.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer_snapshot.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer_restore.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer_factors.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer_geometry.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer_append.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer_plane_append.cpp
  ${N3MAPPING_ROOT}/src/planar_geometry/association.cpp
  ${N3MAPPING_ROOT}/src/planar_geometry/append_association.cpp
  ${N3MAPPING_ROOT}/src/planar_geometry/merging.cpp
  ${N3MAPPING_ROOT}/src/planar_geometry/plane_partition.cpp
  ${N3MAPPING_ROOT}/src/planar_geometry/builder.cpp
  ${N3MAPPING_ROOT}/src/planar_geometry/serialization.cpp
  ${N3MAPPING_ROOT}/src/n3map_proto_utils.cpp
  ${N3MAPPING_ROOT}/src/map_serializer.cpp
  ${N3MAPPING_ROOT}/src/map_serializer_save.cpp
  ${N3MAPPING_ROOT}/src/map_serializer_load.cpp
  ${N3MAPPING_ROOT}/src/localization_atlas.cpp
  ${N3MAPPING_ROOT}/src/localization_atlas_descriptors.cpp
  ${N3MAPPING_ROOT}/src/localization_atlas_map.cpp
  ${N3MAPPING_ROOT}/src/localization_crop.cpp
  ${N3MAPPING_ROOT}/src/registration_observability.cpp
  ${N3MAPPING_ROOT}/src/relocalization_candidate_evaluator.cpp
  ${N3MAPPING_ROOT}/src/relocalization_decision_policy.cpp
  ${N3MAPPING_ROOT}/src/relocalization_debug_emitter.cpp
  ${N3MAPPING_ROOT}/src/relocalization_hypothesis_manager.cpp
  ${N3MAPPING_ROOT}/src/relocalization_place_index.cpp
  ${N3MAPPING_ROOT}/src/relocalization_place_prepare.cpp
  ${N3MAPPING_ROOT}/src/relocalization_place_catalog.cpp
  ${N3MAPPING_ROOT}/src/relocalization_query_builder.cpp
  ${N3MAPPING_ROOT}/src/relocalization_seed_plan.cpp
  ${N3MAPPING_ROOT}/src/relocalization_target_provider.cpp
  ${N3MAPPING_ROOT}/src/tracking_target_cache.cpp
  ${N3MAPPING_ROOT}/src/world_localizing.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_relocalization.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_initialization.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_confirmation.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_recovery.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_tracking.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_tracking_registration.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_evidence.cpp
  ${N3MAPPING_ROOT}/src/world_localizing_probe.cpp
  ${N3MAPPING_ROOT}/src/free_space_grid.cpp
  ${N3MAPPING_ROOT}/src/free_space_grid_serialization.cpp
  ${N3MAPPING_ROOT}/src/floor_attitude.cpp
  ${N3MAPPING_ROOT}/src/odometry_sanity.cpp
  ${N3MAPPING_ROOT}/src/static_start_guard.cpp
  ${N3MAPPING_ROOT}/src/visibility_consistency.cpp
  ${N3MAPPING_ROOT}/src/visibility_planar_support.cpp
  ${N3MAPPING_ROOT}/src/relocalization_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/runtime_performance_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/relocalization_performance.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming_keyframes.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming_loops.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming_submaps.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming_acquisition.cpp
  ${N3MAPPING_ROOT}/src/RHPDescriptor.cpp
  ${N3MAPPING_ROOT}/src/rhpd_heading.cpp
  ${N3MAPPING_ROOT}/src/rhpd_place.cpp
  ${N3MAPPING_ROOT}/src/rhpd_polar.cpp
  ${N3MAPPING_ROOT}/src/rhpd_manager.cpp
)

function(n3mapping_configure_core_target target_name)
  if(NOT DEFINED N3MAPPING_ROOT)
    set(N3MAPPING_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  # Directory-level wrapper includes must not leak into the core build.
  set_property(TARGET ${target_name} PROPERTY INCLUDE_DIRECTORIES "")
  target_include_directories(${target_name} PUBLIC
    $<BUILD_INTERFACE:${N3MAPPING_ROOT}/include>
    $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
    $<INSTALL_INTERFACE:include>
  )
  target_include_directories(${target_name} SYSTEM PUBLIC
    ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIRS})
  target_include_directories(${target_name} PRIVATE
    ${JSONCPP_INCLUDE_DIRS} ${N3MAPPING_ROOT}/include/Scancontext)

  target_link_libraries(${target_name}
    PUBLIC
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    gtsam
    OpenMP::OpenMP_CXX
    small_gicp::small_gicp
    glog::glog
    OpenSSL::Crypto
    PRIVATE
    ${CMAKE_DL_LIBS}
    TBB::tbb
    ${JSONCPP_LIBRARIES}
  )
endfunction()
