// ROS-free DLIO-specific settings parsed from the common frontend config.
#pragma once

#include <string>

#include "n3mapping/lio/dlio_cloud_adapter.h"
#include "n3mapping/lio/frontend_config.h"

namespace n3mapping {
namespace lio {
namespace dlio {

enum class SensorType {
    Ouster,
    Velodyne,
    Hesai,
    Livox,
    Unknown,
};

struct Settings {
    SensorType sensor = SensorType::Unknown;
    TimeEncoding time_encoding = TimeEncoding::Auto;
    double blind = 0.0;
    double max_abs_coordinate = 1.0e8;
    double alignment_max_correspondence_distance = 1.0;
};

SensorType parseSensorType(std::string value);
TimeEncoding parseTimeEncoding(std::string value);
Settings makeSettings(const LioFrontendConfig& config);
CloudAdapterOptions makeCloudAdapterOptions(const Settings& settings);

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
