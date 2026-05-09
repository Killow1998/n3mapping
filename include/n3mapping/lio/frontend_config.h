// ROS-independent configuration passed to in-process LIO frontend adapters.
#pragma once

#include <string>

#include <Eigen/Geometry>

#include "n3mapping/config.h"

namespace n3mapping {
namespace lio {

struct LioFrontendConfig {
    std::string lidar_type = "generic";
    double time_offset = 0.0;
    bool publish_debug = false;
    bool debug_publish_odom = false;
    bool debug_publish_deskewed_cloud = false;
    bool debug_publish_local_map = false;
    bool debug_publish_timing = false;
    size_t imu_buffer_max_samples = 2000;
    size_t point_filter_num = 1;
    size_t scan_lines = 128;
    double blind = 0.0;
    double max_abs_coordinate = 1.0e8;
    bool prediction_only_output = false;
    std::string dlio_time_encoding = "auto";
    Eigen::Isometry3d T_body_lidar = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_body_imu = Eigen::Isometry3d::Identity();
};

Eigen::Isometry3d makeIsometryFromXyzRpy(double tx,
                                         double ty,
                                         double tz,
                                         double roll,
                                         double pitch,
                                         double yaw);
LioFrontendConfig makeLioFrontendConfig(const Config& config);

}  // namespace lio
}  // namespace n3mapping
