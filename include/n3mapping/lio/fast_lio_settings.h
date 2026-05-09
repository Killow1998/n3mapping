// ROS-free FAST-LIO2-specific settings parsed from the common frontend config.
#pragma once

#include <string>

#include "n3mapping/lio/fast_lio_cloud_adapter.h"
#include "n3mapping/lio/frontend_config.h"

namespace n3mapping {
namespace lio {
namespace fast_lio {

enum class LidarType {
    Avia,
    Velodyne,
    Ouster,
    Marsim,
    Generic,
};

struct Settings {
    LidarType lidar_type = LidarType::Generic;
    size_t point_filter_num = 1;
    size_t scan_lines = 128;
    double blind = 0.0;
    double max_abs_coordinate = 1.0e8;
    double alignment_max_correspondence_distance = 1.0;
};

LidarType parseLidarType(std::string value);
Settings makeSettings(const LioFrontendConfig& config);
CloudAdapterOptions makeCloudAdapterOptions(const Settings& settings);

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
