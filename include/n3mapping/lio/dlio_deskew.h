// ROS-free DLIO pointcloud deskew primitive.
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/types.h"
#include "n3mapping/lio/dlio_imu_integration.h"
#include "n3mapping/lio/dlio_scan_timing.h"

namespace n3mapping {
namespace lio {
namespace dlio {

struct DeskewResult {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    size_t transformed_points = 0;
    bool valid = false;
};

DeskewResult deskewToWorld(
    const core::RawLidarFrame& frame,
    const ScanTiming& timing,
    const std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>& poses);

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
