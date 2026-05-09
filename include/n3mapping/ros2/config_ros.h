// ROS parameter and logging helpers for the otherwise ROS-independent Config.
#pragma once

#include <rclcpp/logger.hpp>
#include <rclcpp/node.hpp>

#include "n3mapping/config.h"

namespace n3mapping {

void loadConfigFromROS(rclcpp::Node* node, Config& config);
void printConfigToROS(const Config& config, const rclcpp::Logger& logger);

}  // namespace n3mapping

