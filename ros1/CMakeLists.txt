cmake_minimum_required(VERSION 3.10)
project(n3mapping_ros1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  message_filters
  pcl_ros
  pcl_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

catkin_package(
  INCLUDE_DIRS include ../include
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs geometry_msgs tf2_ros message_filters pcl_ros pcl_conversions
)

include_directories(
  include
  ../include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_library(n3mapping_ros1_wrapper
  src/config_ros1.cpp
  src/conversions.cpp
)

target_link_libraries(n3mapping_ros1_wrapper
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

add_executable(n3mapping_ros1_node
  src/n3mapping_ros1_node.cpp
)

target_link_libraries(n3mapping_ros1_node
  n3mapping_ros1_wrapper
  ${catkin_LIBRARIES}
)

install(TARGETS n3mapping_ros1_wrapper n3mapping_ros1_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
