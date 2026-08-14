# A build tree inside the source tree makes the repository dirty and defeats
# the clean-source requirement used by verified Product builds. Check both
# resolved and literal paths: the former catches a real nested build tree,
# while the latter also catches a symlink placed inside the repository.
function(_n3mapping_path_is_within out_var candidate root)
  set(_prefix "${root}/")
  string(LENGTH "${_prefix}" _prefix_length)
  string(LENGTH "${candidate}" _candidate_length)
  set(_result 0)
  if("${candidate}" STREQUAL "${root}")
    set(_result 1)
  elseif(NOT _candidate_length LESS _prefix_length)
    string(SUBSTRING "${candidate}" 0 ${_prefix_length} _head)
    if("${_head}" STREQUAL "${_prefix}")
      set(_result 1)
    endif()
  endif()
  set(${out_var} ${_result} PARENT_SCOPE)
endfunction()

get_filename_component(_n3mapping_real_source "${N3MAPPING_ROOT}" REALPATH)
get_filename_component(_n3mapping_real_binary "${CMAKE_BINARY_DIR}" REALPATH)
_n3mapping_path_is_within(_n3mapping_in_source_resolved
  "${_n3mapping_real_binary}" "${_n3mapping_real_source}")
_n3mapping_path_is_within(_n3mapping_in_source_literal
  "${CMAKE_BINARY_DIR}" "${N3MAPPING_ROOT}")
if(_n3mapping_in_source_resolved OR _n3mapping_in_source_literal)
  message(FATAL_ERROR
    "Refusing to build inside the n3mapping source tree.\n"
    "  source tree: ${_n3mapping_real_source}\n"
    "  build tree:  ${_n3mapping_real_binary}\n"
    "Use build, install, and log directories outside the repository.")
endif()

# A production identity is only emitted when CMake is given the exact clean
# Git commit being built. Development builds remain usable but identify
# themselves as UNVERIFIED and are rejected by Product Bundle/runtime/Gate.
set(N3MAPPING_PRODUCT_COMMIT "" CACHE STRING
  "Clean lowercase Git commit embedded in Product V1 executables.")

file(SHA256
  "${N3MAPPING_ROOT}/config/product_v1.yaml"
  N3MAPPING_PRODUCT_PROFILE_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${N3MAPPING_ROOT}/config/product_v1.yaml"
  "${N3MAPPING_ROOT}/.git/HEAD"
  "${N3MAPPING_ROOT}/.git/index")

set(_n3mapping_product_compiled_commit "UNVERIFIED")
set(_n3mapping_product_build_verified 0)
if(N3MAPPING_BUILD_RESEARCH_TOOLS)
  set(_n3mapping_product_research_tools "ON")
else()
  set(_n3mapping_product_research_tools "OFF")
endif()
if(N3MAPPING_PRODUCT_COMMIT)
  string(LENGTH "${N3MAPPING_PRODUCT_COMMIT}"
    _n3mapping_product_commit_length)
  if(NOT _n3mapping_product_commit_length EQUAL 40 OR
     NOT N3MAPPING_PRODUCT_COMMIT MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "N3MAPPING_PRODUCT_COMMIT must be a lowercase 40-character Git SHA")
  endif()

  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${N3MAPPING_ROOT}"
    RESULT_VARIABLE _n3mapping_git_head_result
    OUTPUT_VARIABLE _n3mapping_git_head
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _n3mapping_git_head_result EQUAL 0 OR
     NOT _n3mapping_git_head STREQUAL N3MAPPING_PRODUCT_COMMIT)
    message(FATAL_ERROR
      "N3MAPPING_PRODUCT_COMMIT does not match the source-tree HEAD")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=all
    WORKING_DIRECTORY "${N3MAPPING_ROOT}"
    RESULT_VARIABLE _n3mapping_git_status_result
    OUTPUT_VARIABLE _n3mapping_git_status
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _n3mapping_git_status_result EQUAL 0 OR _n3mapping_git_status)
    message(FATAL_ERROR
      "A verified Product V1 binary requires a clean source tree")
  endif()
  set(_n3mapping_product_compiled_commit "${N3MAPPING_PRODUCT_COMMIT}")
  set(_n3mapping_product_build_verified 1)
  add_custom_target(n3mapping_product_identity_guard
    COMMAND "${CMAKE_COMMAND}"
      -DN3MAPPING_GIT_EXECUTABLE=${GIT_EXECUTABLE}
      -DN3MAPPING_PRODUCT_SOURCE_DIR=${N3MAPPING_ROOT}
      -DN3MAPPING_EXPECTED_PRODUCT_COMMIT=${N3MAPPING_PRODUCT_COMMIT}
      -P
      "${N3MAPPING_ROOT}/cmake/n3mapping_verify_clean_product_build.cmake"
    VERBATIM
  )
endif()

add_compile_definitions(
  N3MAPPING_PRODUCT_COMMIT="${_n3mapping_product_compiled_commit}"
  N3MAPPING_PRODUCT_PROFILE_SHA256="${N3MAPPING_PRODUCT_PROFILE_SHA256}"
  N3MAPPING_PRODUCT_BUILD_VERIFIED=${_n3mapping_product_build_verified}
  N3MAPPING_PRODUCT_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  N3MAPPING_PRODUCT_RESEARCH_TOOLS="${_n3mapping_product_research_tools}"
)

function(n3mapping_guard_product_target target)
  if(TARGET n3mapping_product_identity_guard)
    add_dependencies(${target} n3mapping_product_identity_guard)
  endif()
endfunction()
