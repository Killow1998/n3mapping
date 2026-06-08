// cloud_utils.h — Backward sector cutting for directional ambiguity breaking on 360-degree LiDARs.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include "n3mapping/pcl_compat.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr
cutBackwardSector(const typename pcl::PointCloud<PointT>::Ptr& cloud_in, double cut_angle_deg)
{
    if (!cloud_in || cloud_in->empty() || cut_angle_deg <= 0.0)
        return cloud_in;

    const double threshold = M_PI - (cut_angle_deg / 2.0) * M_PI / 180.0;
    auto out = pcl::make_shared<pcl::PointCloud<PointT>>();
    out->header = cloud_in->header;
    out->reserve(cloud_in->size());

    for (const auto& pt : cloud_in->points) {
        if (std::abs(std::atan2(double(pt.y), double(pt.x))) < threshold)
            out->push_back(pt);
    }

    out->width = out->size();
    out->height = 1;
    out->is_dense = cloud_in->is_dense;
    return out;
}

template <typename PointT>
bool isFinitePoint(const PointT& pt)
{
    return std::isfinite(static_cast<double>(pt.x)) &&
           std::isfinite(static_cast<double>(pt.y)) &&
           std::isfinite(static_cast<double>(pt.z));
}

template <typename PointT>
bool voxelGridIndexingSafe(const typename pcl::PointCloud<PointT>::Ptr& cloud, double leaf_size)
{
    if (!cloud || cloud->empty() || leaf_size <= 0.0 || !std::isfinite(leaf_size)) {
        return false;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    bool has_finite = false;

    for (const auto& pt : cloud->points) {
        if (!isFinitePoint(pt)) {
            continue;
        }
        has_finite = true;
        min_x = std::min(min_x, static_cast<double>(pt.x));
        min_y = std::min(min_y, static_cast<double>(pt.y));
        min_z = std::min(min_z, static_cast<double>(pt.z));
        max_x = std::max(max_x, static_cast<double>(pt.x));
        max_y = std::max(max_y, static_cast<double>(pt.y));
        max_z = std::max(max_z, static_cast<double>(pt.z));
    }

    if (!has_finite) {
        return false;
    }

    const auto bins = [leaf_size](double min_v, double max_v) -> long double {
        return std::floor((max_v - min_v) / leaf_size) + 1.0L;
    };
    const long double bx = bins(min_x, max_x);
    const long double by = bins(min_y, max_y);
    const long double bz = bins(min_z, max_z);
    const long double int_max = static_cast<long double>(std::numeric_limits<int>::max());
    return bx > 0.0L && by > 0.0L && bz > 0.0L &&
           bx <= int_max && by <= int_max && bz <= int_max &&
           bx * by * bz <= int_max;
}

template <typename PointT>
bool safeVoxelGridFilter(const typename pcl::PointCloud<PointT>::Ptr& input,
                         double leaf_size,
                         typename pcl::PointCloud<PointT>::Ptr* output)
{
    if (!output || !input || input->empty() || leaf_size <= 0.0 || !std::isfinite(leaf_size)) {
        return false;
    }
    if (!voxelGridIndexingSafe<PointT>(input, leaf_size)) {
        return false;
    }

    pcl::VoxelGrid<PointT> voxel;
    voxel.setLeafSize(static_cast<float>(leaf_size),
                      static_cast<float>(leaf_size),
                      static_cast<float>(leaf_size));
    voxel.setInputCloud(input);
    auto filtered = pcl::make_shared<pcl::PointCloud<PointT>>();
    voxel.filter(*filtered);
    *output = filtered;
    return true;
}

} // namespace n3mapping
