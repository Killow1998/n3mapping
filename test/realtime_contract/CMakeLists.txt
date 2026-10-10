cmake_minimum_required(VERSION 3.10)
project(n3mapping_realtime_contract LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
enable_testing()
get_filename_component(N3MAPPING_ROOT ../.. ABSOLUTE BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Build the production output implementation directly, without a mapping backend
# or a previously built core that could hide stale symbols.
add_library(n3mapping_output_contract STATIC
  ${N3MAPPING_ROOT}/src/core/realtime_output.cpp
  ${N3MAPPING_ROOT}/src/core/run_mode.cpp
  ${N3MAPPING_ROOT}/src/ros_output_contract.cpp)
target_include_directories(n3mapping_output_contract PUBLIC
  ${N3MAPPING_ROOT}/include ${PCL_INCLUDE_DIRS} ${JSONCPP_INCLUDE_DIRS})
target_link_libraries(n3mapping_output_contract PUBLIC
  Eigen3::Eigen Threads::Threads ${JSONCPP_LIBRARIES})

foreach(target IN ITEMS test_realtime_odometry test_realtime_projection_contract test_realtime_output)
  add_executable(${target} ../${target}.cpp)
  target_link_libraries(${target} n3mapping_output_contract GTest::GTest GTest::Main)
  add_test(NAME ${target} COMMAND ${target})
endforeach()

# Message conversion and callback-queue checks need ROS, not the SLAM backend.
find_package(roscpp QUIET)
if(roscpp_FOUND)
  foreach(target IN ITEMS test_ros_output_contract test_noetic_live_output_queues)
    add_executable(${target} ../${target}.cpp)
    target_include_directories(${target} PRIVATE ${roscpp_INCLUDE_DIRS})
    target_link_libraries(${target}
      n3mapping_output_contract GTest::GTest GTest::Main ${roscpp_LIBRARIES})
    add_test(NAME ${target} COMMAND ${target})
  endforeach()
endif()

foreach(check IN ITEMS core_no_ros_includes noetic_wrapper)
  add_test(NAME ${check} COMMAND ${CMAKE_COMMAND} -DROOT_DIR=${N3MAPPING_ROOT}
    -P ${N3MAPPING_ROOT}/test/check_${check}.cmake)
endforeach()
