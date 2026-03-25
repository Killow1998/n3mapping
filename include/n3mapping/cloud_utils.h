// cloud_utils.h — Backward sector cutting for directional ambiguity breaking on 360-degree LiDARs.
#pragma once

#include <cmath>
#include <memory>
#include <pcl/memory.h>
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

} // namespace n3mapping
