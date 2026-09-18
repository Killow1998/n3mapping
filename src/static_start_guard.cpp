// static_start_guard.cpp
#include "n3mapping/static_start_guard.h"

#include "n3mapping/config.h"

#include <cmath>

namespace n3mapping {
namespace {

bool isFinitePoint(const pcl::PointXYZI& p)
{
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

}  // namespace

std::unordered_set<StaticStartGuard::VoxelKey>
StaticStartGuard::voxelize(const pcl::PointCloud<pcl::PointXYZI>& cloud) const
{
    std::unordered_set<VoxelKey> voxels;
    voxels.reserve(cloud.size() / 4 + 16);
    const double inv = 1.0 / options_.voxel_m;
    for (const auto& point : cloud.points) {
        if (!isFinitePoint(point)) {
            continue;
        }
        // Twenty-one bits per axis covers +-300 m at 0.3 m, which is more than
        // any single sweep reaches.
        const auto ix = static_cast<std::int64_t>(std::floor(point.x * inv));
        const auto iy = static_cast<std::int64_t>(std::floor(point.y * inv));
        const auto iz = static_cast<std::int64_t>(std::floor(point.z * inv));
        const VoxelKey key = (static_cast<VoxelKey>(ix & 0x1FFFFF) << 42) |
                             (static_cast<VoxelKey>(iy & 0x1FFFFF) << 21) |
                             static_cast<VoxelKey>(iz & 0x1FFFFF);
        voxels.insert(key);
    }
    return voxels;
}

bool StaticStartGuard::update(double timestamp,
                              const pcl::PointCloud<pcl::PointXYZI>& cloud,
                              const Eigen::Isometry3d& odom_pose)
{
    if (released_) {
        return true;
    }
    if (options_.max_wait_s <= 0.0) {
        released_ = true;
        release_timestamp_ = timestamp;
        return true;
    }
    if (!std::isfinite(timestamp) || !odom_pose.matrix().allFinite() ||
        static_cast<int>(cloud.size()) < options_.min_points) {
        stable_since_ = -1.0;
        return false;
    }

    const bool continuous = previous_timestamp_ >= 0.0 &&
        timestamp > previous_timestamp_ && timestamp - previous_timestamp_ <= 0.5;
    previous_timestamp_ = timestamp;
    if (!continuous) stable_since_ = -1.0;

    if (!has_reference_) {
        reference_ = voxelize(cloud);
        if (reference_.empty()) {
            return false;
        }
        has_reference_ = true;
        first_timestamp_ = timestamp;
        stable_since_ = timestamp;
        stable_pose_ = odom_pose;
        return false;
    }

    const auto current = voxelize(cloud);
    std::size_t shared = 0;
    for (const auto key : current) {
        if (reference_.count(key) != 0) {
            ++shared;
        }
    }
    last_overlap_ = static_cast<double>(shared) /
                    static_cast<double>(reference_.size());

    // Three seconds within 2 cm / 0.01 rad accepts settled stationary input,
    // not the metres of false motion that motivated the original guard.
    const auto delta = stable_pose_.inverse() * odom_pose;
    const bool stable_view = last_overlap_ >= options_.moved_overlap;
    if (!stable_view) {
        stable_since_ = -1.0;
    } else if (stable_since_ < 0.0 || delta.translation().norm() > 0.02 ||
               Eigen::AngleAxisd(delta.rotation()).angle() > 0.01) {
        stable_since_ = timestamp;
        stable_pose_ = odom_pose;
    } else if (continuous && timestamp - stable_since_ >= 3.0) {
        released_ = true;
        release_timestamp_ = timestamp;
        return true;
    }

    if (last_overlap_ < options_.moved_overlap) {
        ++consecutive_below_;
    } else {
        consecutive_below_ = 0;
    }

    if (consecutive_below_ >= options_.moved_consecutive) {
        released_ = true;
        release_timestamp_ = timestamp;
        return true;
    }
    // The wait is bounded so a test that fails to notice motion cannot suppress
    // the map for the whole session.
    if (timestamp - first_timestamp_ >= options_.max_wait_s) {
        released_ = true;
        released_by_timeout_ = true;
        release_timestamp_ = timestamp;
        return true;
    }
    return false;
}

StaticStartGuard::Options staticStartGuardOptionsFromConfig(const Config& config) {
  StaticStartGuard::Options options;
  options.voxel_m = config.mapping_static_voxel_m;
  options.moved_overlap = config.mapping_static_moved_overlap;
  options.moved_consecutive = config.mapping_static_moved_consecutive;
  options.max_wait_s = config.mapping_static_max_wait_s;
  return options;
}

}  // namespace n3mapping
