#pragma once

#include <rclcpp/rclcpp.hpp>

#include "n3mapping/config.h"

namespace n3mapping {

void loadConfigFromRos2(rclcpp::Node* node, Config* config);

}  // namespace n3mapping
