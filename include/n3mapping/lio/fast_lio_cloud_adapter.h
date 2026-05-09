// ROS-free cloud adapter for the FAST-LIO2 extraction boundary.
#pragma once

#include <cstddef>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/types.h"

namespace n3mapping {
namespace lio {
namespace fast_lio {

using PointType = pcl::PointXYZINormal;
using PointCloud = pcl::PointCloud<PointType>;

struct CloudAdapterOptions {
    size_t point_filter_num = 1;
    size_t scan_lines = 128;
    double blind = 0.0;
    double max_abs_coordinate = 1.0e8;
};

struct CloudAdapterStats {
    size_t input_points = 0;
    size_t output_points = 0;
    size_t skipped_non_finite = 0;
    size_t skipped_invalid_line = 0;
    size_t skipped_blind = 0;
    size_t skipped_by_filter = 0;
};

struct CloudAdapterResult {
    PointCloud::Ptr cloud;
    CloudAdapterStats stats;
};

CloudAdapterResult cloudFromRawLidar(const core::RawLidarFrame& frame,
                                     const CloudAdapterOptions& options = {});

}  // namespace fast_lio
}  // namespace lio
}  // namespace n3mapping
