#include "n3mapping/lio/fast_lio_settings.h"

#include <algorithm>
#include <cctype>

namespace n3mapping {
namespace lio {
namespace fast_lio {
namespace {

std::string normalized(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

}  // namespace

LidarType parseLidarType(std::string value) {
    value = normalized(std::move(value));
    if (value == "avia" || value == "livox" || value == "mid360") {
        return LidarType::Avia;
    }
    if (value == "velodyne" || value == "velo16" || value == "vlp16") {
        return LidarType::Velodyne;
    }
    if (value == "ouster" || value == "oust64" || value == "os1") {
        return LidarType::Ouster;
    }
    if (value == "marsim" || value == "simulation") {
        return LidarType::Marsim;
    }
    return LidarType::Generic;
}

Settings makeSettings(const LioFrontendConfig& config) {
    Settings settings;
    settings.lidar_type = parseLidarType(config.lidar_type);
    settings.point_filter_num = config.point_filter_num;
    settings.scan_lines = config.scan_lines;
    settings.blind = config.blind;
    settings.max_abs_coordinate = config.max_abs_coordinate;
    settings.alignment_max_correspondence_distance =
        config.alignment_max_correspondence_distance;
    return settings;
}

CloudAdapterOptions makeCloudAdapterOptions(const Settings& settings) {
    CloudAdapterOptions options;
    options.point_filter_num = settings.point_filter_num;
    options.scan_lines = settings.scan_lines;
    options.blind = settings.blind;
    options.max_abs_coordinate = settings.max_abs_coordinate;
    return options;
}

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
