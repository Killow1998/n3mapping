// ROS-free cloud adapter for the DLIO extraction boundary.
#pragma once

#include <cstddef>
#include <cstdint>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace lio {
namespace dlio {

enum class TimeEncoding {
    Auto,
    OusterOffsetNs,
    VelodyneOffsetSeconds,
    LivoxOffsetNs,
};

struct Point {
    Point() : data{0.0f, 0.0f, 0.0f, 1.0f}, intensity(0.0f), t(0u) {}

    PCL_ADD_POINT4D;
    float intensity;
    union {
        uint32_t t;
        float time;
        double timestamp;
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

using PointCloud = pcl::PointCloud<Point>;

struct CloudAdapterOptions {
    TimeEncoding time_encoding = TimeEncoding::Auto;
    double blind = 0.0;
    double max_abs_coordinate = 1.0e8;
};

struct CloudAdapterStats {
    size_t input_points = 0;
    size_t output_points = 0;
    size_t skipped_non_finite = 0;
    size_t skipped_blind = 0;
};

struct CloudAdapterResult {
    PointCloud::Ptr cloud;
    CloudAdapterStats stats;
    TimeEncoding resolved_time_encoding = TimeEncoding::OusterOffsetNs;
};

CloudAdapterResult cloudFromRawLidar(const core::RawLidarFrame& frame,
                                     const CloudAdapterOptions& options = {});

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping

POINT_CLOUD_REGISTER_POINT_STRUCT(n3mapping::lio::dlio::Point,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (std::uint32_t, t, t)
                                  (float, time, time)
                                  (double, timestamp, timestamp))
