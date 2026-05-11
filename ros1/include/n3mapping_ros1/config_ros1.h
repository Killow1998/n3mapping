#pragma once

#include <ros/ros.h>

#include "n3mapping/config.h"

namespace n3mapping {

void loadConfigFromRos1(ros::NodeHandle& node_handle, Config* config);

}  // namespace n3mapping
