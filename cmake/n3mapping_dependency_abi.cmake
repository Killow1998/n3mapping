function(n3mapping_verify_dependency_abi)
  if(N3MAPPING_BUILD_RESEARCH_TOOLS)
    return()
  endif()

  foreach(gtsam_target IN ITEMS gtsam GTSAM::gtsam)
    if(NOT TARGET ${gtsam_target})
      continue()
    endif()

    get_target_property(gtsam_compile_options
      ${gtsam_target} INTERFACE_COMPILE_OPTIONS)
    if(gtsam_compile_options AND
       gtsam_compile_options MATCHES "(^|;)-march=native($|;)")
      message(FATAL_ERROR
        "Unsafe product dependency ABI: ${gtsam_target} exports "
        "-march=native. This can change Eigen's allocation ABI relative to "
        "the system PCL build and has caused point-cloud heap corruption on "
        "ROS 1 Noetic/PCL 1.10. Rebuild GTSAM with "
        "-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF.")
    endif()
  endforeach()
endfunction()
