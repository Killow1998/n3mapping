#pragma once

#include <cstddef>
#include <limits>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct HeightmapConsistencyDiagnostics {
    int overlap_cell_count = 0;
    double overlap_ratio = 0.0;
    double ground_dz_median = std::numeric_limits<double>::quiet_NaN();
    double ground_dz_p90 = std::numeric_limits<double>::quiet_NaN();
    double ground_dz_max = std::numeric_limits<double>::quiet_NaN();
    double ground_support_ratio = 0.0;
    double vertical_consistency_score = 0.0;
};

HeightmapConsistencyDiagnostics computeHeightmapConsistency(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const Eigen::Isometry3d& T_target_source,
    double cell_size = 0.5,
    std::size_t min_points_per_cell = 3,
    double low_quantile = 0.10);

}  // namespace n3mapping
