#include "n3mapping/lio/dlio_settings.h"

#include <algorithm>
#include <cctype>

namespace n3mapping {
namespace lio {
namespace dlio {
namespace {

std::string normalized(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

}  // namespace

SensorType parseSensorType(std::string value) {
    value = normalized(std::move(value));
    if (value == "ouster") return SensorType::Ouster;
    if (value == "velodyne") return SensorType::Velodyne;
    if (value == "hesai") return SensorType::Hesai;
    if (value == "livox" || value == "avia" || value == "mid360") {
        return SensorType::Livox;
    }
    return SensorType::Unknown;
}

TimeEncoding parseTimeEncoding(std::string value) {
    value = normalized(std::move(value));
    if (value == "velodyne" || value == "velodyne_seconds" ||
        value == "velodyne_offset_seconds") {
        return TimeEncoding::VelodyneOffsetSeconds;
    }
    if (value == "livox" || value == "livox_ns" || value == "livox_offset_ns") {
        return TimeEncoding::LivoxOffsetNs;
    }
    if (value == "ouster" || value == "ouster_ns" || value == "ouster_offset_ns") {
        return TimeEncoding::OusterOffsetNs;
    }
    return TimeEncoding::Auto;
}

Settings makeSettings(const LioFrontendConfig& config) {
    Settings settings;
    settings.sensor = parseSensorType(config.lidar_type);
    settings.time_encoding = parseTimeEncoding(config.dlio_time_encoding);
    settings.blind = config.blind;
    settings.max_abs_coordinate = config.max_abs_coordinate;
    settings.alignment_max_correspondence_distance =
        config.alignment_max_correspondence_distance;
    return settings;
}

CloudAdapterOptions makeCloudAdapterOptions(const Settings& settings) {
    CloudAdapterOptions options;
    options.time_encoding = settings.time_encoding;
    options.blind = settings.blind;
    options.max_abs_coordinate = settings.max_abs_coordinate;
    return options;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
