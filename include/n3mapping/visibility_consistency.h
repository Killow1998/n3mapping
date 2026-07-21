#pragma once

#include <cstddef>
#include <limits>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct VisibilityConsistencyOptions {
    double range_min_m = 0.5;
    double range_max_m = 30.0;
    double range_tolerance_m = 0.3;
    double min_angular_resolution_deg = 1.0;
    double max_angular_resolution_deg = 4.0;
};

struct VisibilityConsistencyResult {
    bool valid = false;
    std::size_t observed_bins = 0;
    std::size_t predicted_bins = 0;
    std::size_t common_bins = 0;
    std::size_t consistent_bins = 0;
    std::size_t foreground_conflict_bins = 0;
    double angular_resolution_deg = std::numeric_limits<double>::quiet_NaN();
    double observed_coverage = 0.0;
    double consistency_ratio = 0.0;
    double foreground_conflict_ratio = 0.0;
    double evidence_log_odds = std::numeric_limits<double>::quiet_NaN();
    double median_abs_range_error_m = std::numeric_limits<double>::quiet_NaN();
    double p90_abs_range_error_m = std::numeric_limits<double>::quiet_NaN();
};

VisibilityConsistencyResult evaluateVisibilityConsistency(
    const pcl::PointCloud<pcl::PointXYZI>& map_cloud,
    const pcl::PointCloud<pcl::PointXYZI>& query_cloud,
    const Eigen::Isometry3d& T_map_lidar,
    const VisibilityConsistencyOptions& options = {});

}  // namespace n3mapping
