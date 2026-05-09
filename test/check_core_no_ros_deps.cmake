if(DEFINED N3MAPPING_CHECK_LIBRARY)
  set(check_library "${N3MAPPING_CHECK_LIBRARY}")
elseif(DEFINED N3MAPPING_CORE_LIBRARY)
  set(check_library "${N3MAPPING_CORE_LIBRARY}")
else()
  message(FATAL_ERROR "N3MAPPING_CHECK_LIBRARY is required")
endif()

if(NOT DEFINED N3MAPPING_CHECK_TARGET_NAME)
  set(N3MAPPING_CHECK_TARGET_NAME "n3mapping_core")
endif()

execute_process(
  COMMAND ldd "${check_library}"
  RESULT_VARIABLE ldd_result
  OUTPUT_VARIABLE ldd_output
  ERROR_VARIABLE ldd_error
)

if(NOT ldd_result EQUAL 0)
  message(FATAL_ERROR "ldd failed for ${check_library}: ${ldd_error}")
endif()

set(forbidden_patterns
  "rclcpp"
  "roscpp"
  "std_msgs"
  "sensor_msgs"
  "nav_msgs"
  "geometry_msgs"
  "visualization_msgs"
  "tf2"
  "message_filters"
  "cv_bridge"
  "pcl_conversions"
  "livox_ros_driver2"
)

foreach(pattern IN LISTS forbidden_patterns)
  if(ldd_output MATCHES "${pattern}")
    message(FATAL_ERROR
      "${N3MAPPING_CHECK_TARGET_NAME} must remain ROS-free, but ldd output contains '${pattern}':\n${ldd_output}")
  endif()
endforeach()

message(STATUS "${N3MAPPING_CHECK_TARGET_NAME} has no direct ROS/Livox runtime dependencies")
