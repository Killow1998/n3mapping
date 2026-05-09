#include "n3mapping/lio/frontend_config.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace n3mapping {
namespace lio {
namespace {

int64_t offsetNsec(double time_offset_sec) {
    if (!std::isfinite(time_offset_sec)) {
        return 0;
    }
    const double offset = std::round(time_offset_sec * 1.0e9);
    if (offset > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    if (offset < static_cast<double>(std::numeric_limits<int64_t>::min())) {
        return std::numeric_limits<int64_t>::min();
    }
    return static_cast<int64_t>(offset);
}

int64_t addClamped(int64_t value, int64_t delta) {
    if (delta > 0 &&
        value > std::numeric_limits<int64_t>::max() - delta) {
        return std::numeric_limits<int64_t>::max();
    }
    if (delta < 0 &&
        value < std::numeric_limits<int64_t>::min() - delta) {
        return std::numeric_limits<int64_t>::min();
    }
    return value + delta;
}

}  // namespace

Eigen::Isometry3d makeIsometryFromXyzRpy(double tx,
                                         double ty,
                                         double tz,
                                         double roll,
                                         double pitch,
                                         double yaw) {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() << tx, ty, tz;
    transform.linear() =
        (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
    return transform;
}

core::TimeStamp applyTimeOffset(const core::TimeStamp& stamp,
                                double time_offset_sec) {
    core::TimeStamp shifted = stamp;
    shifted.nsec = addClamped(stamp.nsec, offsetNsec(time_offset_sec));
    return shifted;
}

core::ImuSample applyTimeOffset(const core::ImuSample& sample,
                                double time_offset_sec) {
    core::ImuSample shifted = sample;
    shifted.stamp = applyTimeOffset(sample.stamp, time_offset_sec);
    return shifted;
}

core::RawLidarFrame applyTimeOffset(const core::RawLidarFrame& frame,
                                    double time_offset_sec) {
    core::RawLidarFrame shifted = frame;
    shifted.stamp_begin = applyTimeOffset(frame.stamp_begin, time_offset_sec);
    shifted.stamp_end = applyTimeOffset(frame.stamp_end, time_offset_sec);
    return shifted;
}

LioFrontendConfig makeLioFrontendConfig(const Config& config) {
    LioFrontendConfig frontend_config;
    frontend_config.lidar_type = config.lidar_type;
    frontend_config.time_offset = config.frontend_time_offset;
    frontend_config.publish_debug = config.frontend_publish_debug;
    frontend_config.debug_publish_odom =
        config.frontend_publish_debug && config.frontend_debug_publish_odom;
    frontend_config.debug_publish_deskewed_cloud =
        config.frontend_publish_debug && config.frontend_debug_publish_deskewed_cloud;
    frontend_config.debug_publish_local_map =
        config.frontend_publish_debug && config.frontend_debug_publish_local_map;
    frontend_config.debug_publish_timing =
        config.frontend_publish_debug && config.frontend_debug_publish_timing;
    frontend_config.imu_buffer_max_samples =
        static_cast<size_t>(std::max(1, config.frontend_imu_buffer_max_samples));
    frontend_config.point_filter_num =
        static_cast<size_t>(std::max(1, config.frontend_point_filter_num));
    frontend_config.scan_lines =
        static_cast<size_t>(std::max(1, config.frontend_scan_lines));
    frontend_config.blind = std::max(0.0, config.frontend_blind);
    frontend_config.max_abs_coordinate =
        std::max(1.0, config.frontend_max_abs_coordinate);
    frontend_config.alignment_max_correspondence_distance =
        std::max(0.0, config.frontend_alignment_max_correspondence_distance);
    frontend_config.prediction_only_output = config.frontend_prediction_only_output;
    frontend_config.dlio_time_encoding = config.dlio_time_encoding;
    frontend_config.dlio_gravity = std::max(0.0, config.dlio_gravity);
    frontend_config.dlio_dense_map_leaf_size =
        std::max(0.0, config.dlio_dense_map_leaf_size);
    frontend_config.dlio_dense_input_skip =
        static_cast<size_t>(std::max(1, config.dlio_dense_input_skip));
    frontend_config.T_body_lidar = makeIsometryFromXyzRpy(
        config.frontend_lidar_to_body_tx,
        config.frontend_lidar_to_body_ty,
        config.frontend_lidar_to_body_tz,
        config.frontend_lidar_to_body_roll,
        config.frontend_lidar_to_body_pitch,
        config.frontend_lidar_to_body_yaw);
    frontend_config.T_body_imu = makeIsometryFromXyzRpy(
        config.frontend_imu_to_body_tx,
        config.frontend_imu_to_body_ty,
        config.frontend_imu_to_body_tz,
        config.frontend_imu_to_body_roll,
        config.frontend_imu_to_body_pitch,
        config.frontend_imu_to_body_yaw);
    return frontend_config;
}

}  // namespace lio
}  // namespace n3mapping
