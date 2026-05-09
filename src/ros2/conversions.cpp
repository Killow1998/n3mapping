#include "n3mapping/ros2/conversions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_field.hpp>

namespace n3mapping {
namespace ros2 {
namespace {

const sensor_msgs::msg::PointField* findField(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const std::vector<std::string>& names) {
    for (const auto& name : names) {
        const auto it = std::find_if(
            cloud_msg.fields.begin(), cloud_msg.fields.end(),
            [&name](const sensor_msgs::msg::PointField& field) {
                return field.name == name;
            });
        if (it != cloud_msg.fields.end()) {
            return &(*it);
        }
    }
    return nullptr;
}

template <typename T>
T readScalar(const uint8_t* ptr) {
    T value{};
    std::memcpy(&value, ptr, sizeof(T));
    return value;
}

bool readFieldAsDouble(const sensor_msgs::msg::PointField& field,
                       const uint8_t* ptr,
                       double& value) {
    using sensor_msgs::msg::PointField;
    switch (field.datatype) {
        case PointField::INT8:
            value = static_cast<double>(readScalar<int8_t>(ptr));
            return true;
        case PointField::UINT8:
            value = static_cast<double>(readScalar<uint8_t>(ptr));
            return true;
        case PointField::INT16:
            value = static_cast<double>(readScalar<int16_t>(ptr));
            return true;
        case PointField::UINT16:
            value = static_cast<double>(readScalar<uint16_t>(ptr));
            return true;
        case PointField::INT32:
            value = static_cast<double>(readScalar<int32_t>(ptr));
            return true;
        case PointField::UINT32:
            value = static_cast<double>(readScalar<uint32_t>(ptr));
            return true;
        case PointField::FLOAT32:
            value = static_cast<double>(readScalar<float>(ptr));
            return true;
        case PointField::FLOAT64:
            value = readScalar<double>(ptr);
            return true;
        default:
            return false;
    }
}

size_t fieldScalarSize(const sensor_msgs::msg::PointField& field) {
    using sensor_msgs::msg::PointField;
    switch (field.datatype) {
        case PointField::INT8:
        case PointField::UINT8:
            return 1;
        case PointField::INT16:
        case PointField::UINT16:
            return 2;
        case PointField::INT32:
        case PointField::UINT32:
        case PointField::FLOAT32:
            return 4;
        case PointField::FLOAT64:
            return 8;
        default:
            return 0;
    }
}

std::vector<double> readPointFieldValues(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const sensor_msgs::msg::PointField& field) {
    std::vector<double> values;
    const size_t scalar_size = fieldScalarSize(field);
    if (scalar_size == 0 || field.count != 1) {
        return values;
    }
    const size_t point_count =
        static_cast<size_t>(cloud_msg.width) * static_cast<size_t>(cloud_msg.height);
    values.reserve(point_count);
    for (uint32_t row = 0; row < cloud_msg.height; ++row) {
        for (uint32_t col = 0; col < cloud_msg.width; ++col) {
            const size_t offset = static_cast<size_t>(row) * cloud_msg.row_step +
                                  static_cast<size_t>(col) * cloud_msg.point_step +
                                  field.offset;
            if (offset + scalar_size > cloud_msg.data.size()) {
                values.clear();
                return values;
            }
            double value = 0.0;
            if (!readFieldAsDouble(field, cloud_msg.data.data() + offset, value)) {
                values.clear();
                return values;
            }
            values.push_back(value);
        }
    }
    return values;
}

uint32_t clampToUint32Ns(double value) {
    if (!std::isfinite(value) || value <= 0.0) {
        return 0u;
    }
    const double max_u32 = static_cast<double>(std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(std::llround(std::min(value, max_u32)));
}

std::vector<uint32_t> pointTimeOffsetsFromField(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const sensor_msgs::msg::PointField& field) {
    auto values = readPointFieldValues(cloud_msg, field);
    if (values.empty()) {
        return {};
    }

    using sensor_msgs::msg::PointField;
    const bool floating =
        field.datatype == PointField::FLOAT32 || field.datatype == PointField::FLOAT64;
    const bool absolute_seconds =
        field.name == "timestamp" && floating && values.front() > 100000.0;
    const double base = absolute_seconds ? values.front() : 0.0;
    std::vector<uint32_t> offsets;
    offsets.reserve(values.size());
    for (double value : values) {
        double ns = 0.0;
        if (floating) {
            ns = (value - base) * 1e9;
        } else {
            ns = value;
        }
        offsets.push_back(clampToUint32Ns(ns));
    }
    return offsets;
}

std::vector<uint8_t> pointLinesFromField(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const sensor_msgs::msg::PointField& field) {
    auto values = readPointFieldValues(cloud_msg, field);
    if (values.empty()) {
        return {};
    }
    std::vector<uint8_t> lines;
    lines.reserve(values.size());
    for (double value : values) {
        if (!std::isfinite(value) || value <= 0.0) {
            lines.push_back(0u);
        } else {
            lines.push_back(static_cast<uint8_t>(
                std::min(value, static_cast<double>(std::numeric_limits<uint8_t>::max()))));
        }
    }
    return lines;
}

}  // namespace

core::TimeStamp toCoreStamp(const builtin_interfaces::msg::Time& stamp) {
    return core::TimeStamp{
        static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec
    };
}

Eigen::Isometry3d poseFromOdom(const nav_msgs::msg::Odometry& odom_msg) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() << odom_msg.pose.pose.position.x,
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.z;
    const Eigen::Quaterniond q(odom_msg.pose.pose.orientation.w,
                               odom_msg.pose.pose.orientation.x,
                               odom_msg.pose.pose.orientation.y,
                               odom_msg.pose.pose.orientation.z);
    pose.linear() = q.toRotationMatrix();
    return pose;
}

Eigen::Matrix<double, 6, 6> poseCovarianceFromOdom(
    const nav_msgs::msg::Odometry& odom_msg) {
    Eigen::Matrix<double, 6, 6> covariance =
        Eigen::Matrix<double, 6, 6>::Identity();
    bool has_nonzero_value = false;
    bool all_finite = true;
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
            const double value =
                odom_msg.pose.covariance[static_cast<size_t>(r * 6 + c)];
            covariance(r, c) = value;
            has_nonzero_value = has_nonzero_value || std::abs(value) > 1e-12;
            all_finite = all_finite && std::isfinite(value);
        }
    }
    if (!has_nonzero_value || !all_finite) {
        covariance.setIdentity();
    }
    return covariance;
}

ExternalLioRosFrame externalLioFrameFromRos(
    const sensor_msgs::msg::PointCloud2& cloud_msg,
    const nav_msgs::msg::Odometry& odom_msg) {
    ExternalLioRosFrame frame;
    frame.stamp = toCoreStamp(cloud_msg.header.stamp);
    frame.T_world_lidar = poseFromOdom(odom_msg);
    frame.covariance = poseCovarianceFromOdom(odom_msg);
    frame.cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(cloud_msg, *frame.cloud);
    return frame;
}

core::ImuSample imuSampleFromRos(const sensor_msgs::msg::Imu& imu_msg) {
    core::ImuSample sample;
    sample.stamp = toCoreStamp(imu_msg.header.stamp);
    sample.linear_accel << imu_msg.linear_acceleration.x,
        imu_msg.linear_acceleration.y,
        imu_msg.linear_acceleration.z;
    sample.angular_velocity << imu_msg.angular_velocity.x,
        imu_msg.angular_velocity.y,
        imu_msg.angular_velocity.z;
    sample.orientation = Eigen::Quaterniond(imu_msg.orientation.w,
                                            imu_msg.orientation.x,
                                            imu_msg.orientation.y,
                                            imu_msg.orientation.z);
    sample.has_orientation =
        std::abs(imu_msg.orientation.w) > 1e-12 ||
        std::abs(imu_msg.orientation.x) > 1e-12 ||
        std::abs(imu_msg.orientation.y) > 1e-12 ||
        std::abs(imu_msg.orientation.z) > 1e-12;
    return sample;
}

core::RawLidarFrame rawLidarFrameFromRos(
    const sensor_msgs::msg::PointCloud2& cloud_msg) {
    core::RawLidarFrame frame;
    frame.stamp_begin = toCoreStamp(cloud_msg.header.stamp);
    frame.stamp_end = frame.stamp_begin;
    frame.frame_id = cloud_msg.header.frame_id;
    frame.source_format = "pointcloud2";
    frame.points = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(cloud_msg, *frame.points);
    if (const auto* time_field =
            findField(cloud_msg, {"offset_time", "time", "t", "timestamp"})) {
        frame.point_time_offsets_ns = pointTimeOffsetsFromField(cloud_msg, *time_field);
        if (!frame.point_time_offsets_ns.empty()) {
            const auto max_offset = *std::max_element(
                frame.point_time_offsets_ns.begin(), frame.point_time_offsets_ns.end());
            frame.stamp_end.nsec = frame.stamp_begin.nsec + max_offset;
        }
    }
    if (const auto* line_field = findField(cloud_msg, {"line", "ring"})) {
        frame.point_lines = pointLinesFromField(cloud_msg, *line_field);
    }
    return frame;
}

#ifdef N3MAPPING_HAS_LIVOX_ROS_DRIVER2
core::RawLidarFrame rawLidarFrameFromLivoxCustom(
    const livox_ros_driver2::msg::CustomMsg& msg) {
    core::RawLidarFrame frame;
    const int64_t header_stamp = toCoreStamp(msg.header.stamp).nsec;
    frame.stamp_begin.nsec =
        msg.timebase > 0 ? static_cast<int64_t>(msg.timebase) : header_stamp;
    frame.stamp_end = frame.stamp_begin;
    frame.frame_id = msg.header.frame_id;
    frame.source_format = "livox_custom";
    frame.points = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    frame.points->reserve(msg.points.size());
    frame.point_time_offsets_ns.reserve(msg.points.size());
    frame.point_lines.reserve(msg.points.size());

    uint32_t max_offset_ns = 0;
    for (const auto& src : msg.points) {
        pcl::PointXYZI dst;
        dst.x = src.x;
        dst.y = src.y;
        dst.z = src.z;
        dst.intensity = static_cast<float>(src.reflectivity);
        frame.points->push_back(dst);
        frame.point_time_offsets_ns.push_back(src.offset_time);
        frame.point_lines.push_back(src.line);
        max_offset_ns = std::max(max_offset_ns, src.offset_time);
    }

    frame.points->width = static_cast<uint32_t>(frame.points->size());
    frame.points->height = 1;
    frame.points->is_dense = false;
    frame.stamp_end.nsec = frame.stamp_begin.nsec + max_offset_ns;
    return frame;
}
#endif

}  // namespace ros2
}  // namespace n3mapping
