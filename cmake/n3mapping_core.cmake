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
    OpenMP::OpenMP_CXX
    small_gicp::small_gicp
    glog::glog
  )
endfunction()
