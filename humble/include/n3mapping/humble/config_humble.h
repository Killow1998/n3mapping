#pragma once

#include <rclcpp/rclcpp.hpp>

#include "n3mapping/config.h"

namespace n3mapping {

void loadConfigFromHumble(rclcpp::Node* node, Config* config);

}  // namespace n3mapping
