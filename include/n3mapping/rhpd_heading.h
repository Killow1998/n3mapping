#pragma once

#include <array>
#include <memory>
#include <vector>
#include <utility>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {
// Direction-preserving channels of RHPD; derived from keyframe/query clouds.
struct RHPDHeadingParams {
    static constexpr int channels = 7;
    int rings = 20;
    int sectors = 60;
    double max_radius = 80.0;
    double min_radius = 0.5;
    double lidar_height = 2.0;
    bool log_polar = true;
    double z_min = -2.0;
    double z_max = 6.0;
    double search_ratio = 0.1;
    std::array<double, channels> weights{{1.0, 0.8, 0.6, 0.7, 0.7, 0.7, 0.7}};
    int rows() const { return channels * rings; }
    double sectorAngleDeg() const { return 360.0 / sectors; }
};

// Immutable after construction. Sparse ring columns retain full angular and
// radial resolution; keys and column norms are computed once, then shared.
struct RHPDHeading {
    int rings = 0, sectors = 0;
    std::vector<int> offsets, ring_indices;
    std::vector<double> values;
    Eigen::MatrixXd column_norms, sector_key;
    std::vector<float> ring_key;
};
using RHPDHeadingPtr = std::shared_ptr<const RHPDHeading>;
RHPDHeadingPtr makeRHPDHeading(const Eigen::MatrixXd& values);
RHPDHeadingPtr computeRHPDHeading(const pcl::PointCloud<pcl::PointXYZI>& cloud,
                                 const RHPDHeadingParams& params);
Eigen::MatrixXd rhpdHeadingDenseValues(const RHPDHeading& heading);
std::pair<double, int> matchRHPDHeading(const RHPDHeading& query,
                                      const RHPDHeading& reference,
                                      const RHPDHeadingParams& params);
} // namespace n3mapping
