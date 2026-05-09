cmake_minimum_required(VERSION 3.10)
project(n3mapping_ros1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  tf2_ros
  pcl_ros
)

# The ROS 1 wrapper is intentionally kept as a separate skeleton until this
# repository is split into explicit core and wrapper packages. It is expected to
# link against the installed/exported n3mapping_core CMake target.
find_package(n3mapping REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(n3mapping_ros1_node
  src/n3mapping_ros1_node.cpp
)

target_link_libraries(n3mapping_ros1_node
  n3mapping::n3mapping_core
  ${catkin_LIBRARIES}
)

install(TARGETS n3mapping_ros1_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
