function(n3mapping_add_static_checks)
  add_test(
    NAME test_core_no_ros_includes
    COMMAND ${CMAKE_COMMAND}
      -DROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_core_no_ros_includes.cmake
  )

  add_test(
    NAME test_ros2_wrapper_no_direct_backend_calls
    COMMAND ${CMAKE_COMMAND}
      -DROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_ros2_wrapper_no_direct_backend_calls.cmake
  )

  add_test(
    NAME test_ros1_wrapper_skeleton
    COMMAND ${CMAKE_COMMAND}
      -DROOT_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/test/check_ros1_wrapper_skeleton.cmake
  )
endfunction()
