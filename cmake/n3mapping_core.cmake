set(N3MAPPING_CORE_SOURCES
  ${PROTO_SRCS}
  ${N3MAPPING_ROOT}/src/config.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_core.cpp
  ${N3MAPPING_ROOT}/src/core/n3mapping_session.cpp
  ${N3MAPPING_ROOT}/src/keyframe_manager.cpp
  ${N3MAPPING_ROOT}/src/local_map_selector.cpp
  ${N3MAPPING_ROOT}/src/submap_builder.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_projection.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_factor.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_trial.cpp
  ${N3MAPPING_ROOT}/src/submap_graph_trial_runtime.cpp
  ${N3MAPPING_ROOT}/src/global_map_cache.cpp
  ${N3MAPPING_ROOT}/src/point_cloud_matcher.cpp
  ${N3MAPPING_ROOT}/src/loop_detector.cpp
  ${N3MAPPING_ROOT}/src/loop_verifier.cpp
  ${N3MAPPING_ROOT}/src/loop_verification_pipeline.cpp
  ${N3MAPPING_ROOT}/src/loop_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/loop_heightmap_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/loop_graph_trial_diagnostics.cpp
  ${N3MAPPING_ROOT}/src/loop_segment_consistency.cpp
  ${N3MAPPING_ROOT}/src/loop_consensus_verifier.cpp
  ${N3MAPPING_ROOT}/src/loop_closure_manager.cpp
  ${N3MAPPING_ROOT}/src/graph_factor_noise.cpp
  ${N3MAPPING_ROOT}/src/graph_optimizer.cpp
  ${N3MAPPING_ROOT}/src/n3map_proto_utils.cpp
  ${N3MAPPING_ROOT}/src/map_serializer.cpp
  ${N3MAPPING_ROOT}/src/localization_atlas.cpp
  ${N3MAPPING_ROOT}/src/registration_observability.cpp
  ${N3MAPPING_ROOT}/src/relocalization_candidate_evaluator.cpp
  ${N3MAPPING_ROOT}/src/relocalization_decision_policy.cpp
  ${N3MAPPING_ROOT}/src/relocalization_debug_emitter.cpp
  ${N3MAPPING_ROOT}/src/relocalization_hypothesis_manager.cpp
  ${N3MAPPING_ROOT}/src/relocalization_place_index.cpp
  ${N3MAPPING_ROOT}/src/relocalization_query_builder.cpp
  ${N3MAPPING_ROOT}/src/relocalization_target_provider.cpp
  ${N3MAPPING_ROOT}/src/world_localizing.cpp
  ${N3MAPPING_ROOT}/src/free_space_grid.cpp
  ${N3MAPPING_ROOT}/src/floor_attitude.cpp
  ${N3MAPPING_ROOT}/src/odometry_sanity.cpp
  ${N3MAPPING_ROOT}/src/static_start_guard.cpp
  ${N3MAPPING_ROOT}/src/visibility_consistency.cpp
  ${N3MAPPING_ROOT}/src/relocalization_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/runtime_performance_debug_logger.cpp
  ${N3MAPPING_ROOT}/src/mapping_resuming.cpp
  ${N3MAPPING_ROOT}/src/RHPDescriptor.cpp
  ${N3MAPPING_ROOT}/include/Scancontext/Scancontext.cpp
  ${N3MAPPING_ROOT}/include/Scancontext/HybridScancontext.cpp
)

function(n3mapping_configure_core_target target_name)
  if(NOT DEFINED N3MAPPING_ROOT)
    set(N3MAPPING_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  target_include_directories(${target_name} PUBLIC
    $<BUILD_INTERFACE:${N3MAPPING_ROOT}/include>
    $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
    $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(${target_name}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    gtsam
    TBB::tbb
    OpenMP::OpenMP_CXX
    small_gicp::small_gicp
    glog::glog
    OpenSSL::Crypto
  )
endfunction()
