#include "n3mapping/loop_heightmap_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "n3mapping/cloud_utils.h"

namespace n3mapping {
namespace {

using CellKey = std::pair<int64_t, int64_t>;
using VoxelKey = std::tuple<int64_t, int64_t, int64_t>;
using Heightmap = std::map<CellKey, double>;

double percentile(std::vector<double> values, double q)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double index = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(index));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(index));
    if (lo == hi) {
        return values[lo];
    }
    const double t = index - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

Heightmap buildHeightmap(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
                         const Eigen::Isometry3d& transform,
                         double cell_size,
                         std::size_t min_points_per_cell,
                         double low_quantile)
{
    std::map<CellKey, std::vector<double>> cells;
    if (!cloud || cell_size <= 0.0 || !std::isfinite(cell_size) || min_points_per_cell == 0) {
        return {};
    }

    for (const auto& point : cloud->points) {
        if (!isFinitePoint(point)) {
            continue;
        }
        const Eigen::Vector3d p =
            transform * Eigen::Vector3d(static_cast<double>(point.x),
                                        static_cast<double>(point.y),
                                        static_cast<double>(point.z));
        if (!p.allFinite()) {
            continue;
        }
        const int64_t ix = static_cast<int64_t>(std::floor(p.x() / cell_size));
        const int64_t iy = static_cast<int64_t>(std::floor(p.y() / cell_size));
        cells[{ix, iy}].push_back(p.z());
    }

    Heightmap heightmap;
    for (auto& [key, z_values] : cells) {
        if (z_values.size() < min_points_per_cell) {
            continue;
        }
        heightmap[key] = percentile(std::move(z_values), low_quantile);
    }
    return heightmap;
}

std::set<VoxelKey> buildVoxelSet(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud,
                                 const Eigen::Isometry3d& transform,
                                 double voxel_size)
{
    std::set<VoxelKey> voxels;
    if (!cloud || voxel_size <= 0.0 || !std::isfinite(voxel_size)) {
        return voxels;
    }

    for (const auto& point : cloud->points) {
        if (!isFinitePoint(point)) {
            continue;
        }
        const Eigen::Vector3d p =
            transform * Eigen::Vector3d(static_cast<double>(point.x),
                                        static_cast<double>(point.y),
                                        static_cast<double>(point.z));
        if (!p.allFinite()) {
            continue;
        }
        voxels.emplace(
            static_cast<int64_t>(std::floor(p.x() / voxel_size)),
            static_cast<int64_t>(std::floor(p.y() / voxel_size)),
            static_cast<int64_t>(std::floor(p.z() / voxel_size)));
    }
    return voxels;
}

}  // namespace

HeightmapConsistencyDiagnostics computeHeightmapConsistency(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const Eigen::Isometry3d& T_target_source,
    double cell_size,
    std::size_t min_points_per_cell,
    double low_quantile)
{
    HeightmapConsistencyDiagnostics diagnostics;
    const Heightmap target_map = buildHeightmap(
        target, Eigen::Isometry3d::Identity(), cell_size, min_points_per_cell, low_quantile);
    const Heightmap source_map = buildHeightmap(
        source, T_target_source, cell_size, min_points_per_cell, low_quantile);
    if (target_map.empty() || source_map.empty()) {
        return diagnostics;
    }

    std::vector<double> abs_dz;
    abs_dz.reserve(std::min(target_map.size(), source_map.size()));
    for (const auto& [key, source_z] : source_map) {
        const auto target_it = target_map.find(key);
        if (target_it == target_map.end()) {
            continue;
        }
        const double dz = std::abs(source_z - target_it->second);
        if (std::isfinite(dz)) {
            abs_dz.push_back(dz);
        }
    }

    diagnostics.overlap_cell_count = static_cast<int>(abs_dz.size());
    const auto smaller = static_cast<double>(std::min(target_map.size(), source_map.size()));
    const auto larger = static_cast<double>(std::max(target_map.size(), source_map.size()));
    diagnostics.overlap_ratio = smaller > 0.0 ? static_cast<double>(abs_dz.size()) / smaller : 0.0;
    diagnostics.ground_support_ratio = larger > 0.0 ? static_cast<double>(abs_dz.size()) / larger : 0.0;
    if (abs_dz.empty()) {
        return diagnostics;
    }

    diagnostics.ground_dz_median = percentile(abs_dz, 0.50);
    diagnostics.ground_dz_p90 = percentile(abs_dz, 0.90);
    diagnostics.ground_dz_max = *std::max_element(abs_dz.begin(), abs_dz.end());
    diagnostics.vertical_consistency_score =
        diagnostics.ground_support_ratio / (1.0 + diagnostics.ground_dz_p90);
    return diagnostics;
}

SubmapOverlapDiagnostics computeSubmapOverlapConsistency(
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& target,
    const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& source,
    const Eigen::Isometry3d& T_target_source,
    double voxel_size)
{
    SubmapOverlapDiagnostics diagnostics;
    const auto target_voxels = buildVoxelSet(target, Eigen::Isometry3d::Identity(), voxel_size);
    const auto source_voxels = buildVoxelSet(source, T_target_source, voxel_size);
    diagnostics.target_cell_count = static_cast<int>(target_voxels.size());
    diagnostics.source_cell_count = static_cast<int>(source_voxels.size());
    if (target_voxels.empty() || source_voxels.empty()) {
        return diagnostics;
    }

    std::size_t overlap = 0;
    const auto& smaller = target_voxels.size() <= source_voxels.size() ? target_voxels : source_voxels;
    const auto& larger = target_voxels.size() <= source_voxels.size() ? source_voxels : target_voxels;
    for (const auto& voxel : smaller) {
        if (larger.find(voxel) != larger.end()) {
            ++overlap;
        }
    }

    diagnostics.overlap_cell_count = static_cast<int>(overlap);
    const double smaller_count = static_cast<double>(smaller.size());
    const double larger_count = static_cast<double>(larger.size());
    diagnostics.overlap_ratio = smaller_count > 0.0 ? static_cast<double>(overlap) / smaller_count : 0.0;
    diagnostics.support_ratio = larger_count > 0.0 ? static_cast<double>(overlap) / larger_count : 0.0;
    diagnostics.consistency_score = 0.5 * (diagnostics.overlap_ratio + diagnostics.support_ratio);
    return diagnostics;
}

}  // namespace n3mapping
