#include <ros/ros.h>

#include "n3mapping/core/n3mapping_session.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "n3mapping_ros1_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    n3mapping::Config config;
    pnh.param<std::string>("mode", config.mode, config.mode);
    pnh.param<std::string>("cloud_topic", config.cloud_topic, config.cloud_topic);
    pnh.param<std::string>("odom_topic", config.odom_topic, config.odom_topic);
    pnh.param<std::string>("frontend_mode", config.frontend_mode, config.frontend_mode);

    n3mapping::core::N3MappingSession session(config);
    ROS_INFO_STREAM("n3mapping ROS 1 wrapper skeleton initialized with frontend_mode="
                    << session.config().frontend_mode);

    ros::spin();
    return 0;
}
