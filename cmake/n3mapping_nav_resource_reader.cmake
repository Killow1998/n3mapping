function(n3mapping_add_nav_resource_reader target_name)
  if(NOT DEFINED N3MAPPING_ROOT)
    set(N3MAPPING_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  add_library(${target_name} SHARED
    ${PROTO_SRCS}
    ${N3MAPPING_ROOT}/src/n3map_nav_resource_reader.cpp
  )

  target_include_directories(${target_name} PUBLIC
    $<BUILD_INTERFACE:${N3MAPPING_ROOT}/include>
    $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
  )

  target_link_libraries(${target_name}
    ${PCL_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
  )
endfunction()
