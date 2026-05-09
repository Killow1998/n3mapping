#include "n3mapping/lio/dlio_deskew.h"

#include <algorithm>
#include <cmath>

#include <pcl/common/transforms.h>

namespace n3mapping {
namespace lio {
namespace dlio {
namespace {

double pointStampSeconds(const core::RawLidarFrame& frame, size_t index) {
    const uint32_t offset_ns =
        index < frame.point_time_offsets_ns.size()
            ? frame.point_time_offsets_ns[index]
            : 0u;
    return static_cast<double>(frame.stamp_begin.nsec) * 1.0e-9 +
           static_cast<double>(offset_ns) * 1.0e-9;
}

size_t nearestPoseIndex(
    const std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>& poses,
    double stamp) {
    auto it = std::lower_bound(
        poses.begin(), poses.end(), stamp,
        [](const IntegratedPose& pose, double value) {
            return pose.stamp < value;
        });
    if (it == poses.begin()) {
        return 0;
    }
    if (it == poses.end()) {
        return poses.size() - 1;
    }
    const auto prev = std::prev(it);
    return std::abs(prev->stamp - stamp) <= std::abs(it->stamp - stamp)
               ? static_cast<size_t>(std::distance(poses.begin(), prev))
               : static_cast<size_t>(std::distance(poses.begin(), it));
}

}  // namespace

DeskewResult deskewToWorld(
    const core::RawLidarFrame& frame,
    const ScanTiming& timing,
    const std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>& poses) {
    return deskewToReference(frame, timing, poses, Eigen::Matrix4f::Identity());
}

DeskewResult deskewToReference(
    const core::RawLidarFrame& frame,
    const ScanTiming& timing,
    const std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>& poses,
    const Eigen::Matrix4f& T_world_reference) {
    DeskewResult result;
    result.cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (!frame.points || frame.points->empty() || poses.empty() || !timing.valid) {
        return result;
    }

    const Eigen::Matrix4f T_reference_world = T_world_reference.inverse();
    result.cloud->reserve(frame.points->size());
    if (!timing.has_point_timing || frame.point_time_offsets_ns.empty()) {
        const auto pose_index = nearestPoseIndex(poses, timing.stamp_median);
        pcl::transformPointCloud(*frame.points, *result.cloud,
                                 T_reference_world * poses[pose_index].T);
        result.transformed_points = result.cloud->size();
        result.valid = true;
        return result;
    }

    for (size_t i = 0; i < frame.points->size(); ++i) {
        const auto pose_index = nearestPoseIndex(poses, pointStampSeconds(frame, i));
        const Eigen::Vector4f point(frame.points->at(i).x,
                                    frame.points->at(i).y,
                                    frame.points->at(i).z,
                                    1.0f);
        const Eigen::Vector4f transformed =
            T_reference_world * poses[pose_index].T * point;
        pcl::PointXYZI out = frame.points->at(i);
        out.x = transformed.x();
        out.y = transformed.y();
        out.z = transformed.z();
        result.cloud->push_back(out);
    }

    result.cloud->width = static_cast<uint32_t>(result.cloud->size());
    result.cloud->height = 1;
    result.cloud->is_dense = frame.points->is_dense;
    result.transformed_points = result.cloud->size();
    result.valid = true;
    return result;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
