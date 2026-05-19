function(n3mapping_add_static_checks)
  if(NOT DEFINED N3MAPPING_ROOT)
    set(N3MAPPING_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  add_test(
    NAME test_core_no_ros_includes
    COMMAND ${CMAKE_COMMAND}
      -DROOT_DIR=${N3MAPPING_ROOT}
      -P ${N3MAPPING_ROOT}/test/check_core_no_ros_includes.cmake
  )

  add_test(
    NAME test_humble_wrapper_no_direct_backend_calls
    COMMAND ${CMAKE_COMMAND}
      -DROOT_DIR=${N3MAPPING_ROOT}
      -P ${N3MAPPING_ROOT}/test/check_humble_wrapper_no_direct_backend_calls.cmake
  )

  add_test(
    NAME test_noetic_wrapper
    COMMAND ${CMAKE_COMMAND}
      -DROOT_DIR=${N3MAPPING_ROOT}
      -P ${N3MAPPING_ROOT}/test/check_noetic_wrapper.cmake
  )
endfunction()
