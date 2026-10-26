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
include(${N3MAPPING_ROOT}/cmake/n3mapping_graph_solver.cmake)
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
