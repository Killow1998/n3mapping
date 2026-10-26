# GTSAM PR #1158 protects childFactors during parallel tree elimination.
# Probe the actual selected headers, including 4.1.1 builds with the upstream
# backport. Unknown implementations retain the local serial containment.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES gtsam TBB::tbb)
unset(N3MAPPING_GTSAM_LOCKED_ELIMINATION CACHE)
check_cxx_source_compiles("
  #include <gtsam/linear/GaussianBayesTree.h>
  #include <gtsam/linear/GaussianFactorGraph.h>
  #include <gtsam/inference/ClusterTree.h>
  #include <utility>
  using Tree = gtsam::EliminatableClusterTree<
      gtsam::GaussianBayesTree, gtsam::GaussianFactorGraph>;
  int main() {
    return sizeof(decltype(std::declval<gtsam::EliminationData<Tree>&>().writeLock)) == 0;
  }" N3MAPPING_GTSAM_LOCKED_ELIMINATION)
cmake_pop_check_state()
set(N3MAPPING_GTSAM_BUILD_ID "")
if(N3MAPPING_GTSAM_LOCKED_ELIMINATION AND CMAKE_READELF)
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _graph_build_type)
  foreach(_configuration IN ITEMS ${_graph_build_type} RELEASE RELWITHDEBINFO NOCONFIG)
    get_target_property(_graph_library gtsam IMPORTED_LOCATION_${_configuration})
    if(_graph_library AND EXISTS "${_graph_library}")
      break()
    endif()
  endforeach()
  if(NOT _graph_library)
    get_target_property(_graph_library gtsam IMPORTED_LOCATION)
  endif()
  if(_graph_library AND EXISTS "${_graph_library}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env LC_ALL=C
      "${CMAKE_READELF}" --notes "${_graph_library}"
      RESULT_VARIABLE _graph_readelf_status OUTPUT_VARIABLE _graph_notes)
    if(_graph_readelf_status EQUAL 0 AND _graph_notes MATCHES "Build ID: ([0-9a-f]+)")
      set(N3MAPPING_GTSAM_BUILD_ID "${CMAKE_MATCH_1}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_graph_library}")
    endif()
  endif()
endif()
if(NOT N3MAPPING_GTSAM_BUILD_ID)
  set(N3MAPPING_GTSAM_LOCKED_ELIMINATION 0)
endif()
configure_file("${N3MAPPING_ROOT}/src/graph_solver_dependency.h.in"
  "${PROTO_GEN_DIR}/graph_solver_dependency.h")
if(N3MAPPING_GTSAM_LOCKED_ELIMINATION)
  message(STATUS "N3M graph solves: lock-protected GTSAM ${N3MAPPING_GTSAM_BUILD_ID}, bounded by num_threads")
else()
  message(STATUS "N3M graph solves: unqualified GTSAM, local serial containment")
endif()
