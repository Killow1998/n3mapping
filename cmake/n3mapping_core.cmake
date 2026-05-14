function(n3mapping_configure_core_target target_name)
  target_include_directories(${target_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
