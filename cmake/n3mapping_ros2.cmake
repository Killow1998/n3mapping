set(N3MAPPING_ROS2_DEPENDENCIES
  rclcpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
)

function(n3mapping_configure_ros2_wrapper_target target_name)
  target_include_directories(${target_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROTO_GEN_DIR}>
    $<INSTALL_INTERFACE:include>
  )

  target_link_libraries(${target_name}
    n3mapping_core
    ${PCL_LIBRARIES}
  )

  ament_target_dependencies(${target_name}
    ${N3MAPPING_ROS2_DEPENDENCIES}
  )
endfunction()

function(n3mapping_configure_ros2_node_target target_name)
  target_link_libraries(${target_name}
    n3mapping_ros2_wrapper
  )

  ament_target_dependencies(${target_name}
    ${N3MAPPING_ROS2_DEPENDENCIES}
  )
endfunction()
