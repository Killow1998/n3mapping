// static_start_guard.h - rejects unsettled odometry at mapping startup.
#pragma once

#include <cstdint>
#include <unordered_set>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct Config;

// A stable view plus a bounded, continuous pose window permits stationary start.
// Otherwise require scan-observed motion or timeout; odometry drift alone does
// not count as physical motion.
class StaticStartGuard {
public:
    struct Options {
        // Coarse enough to be insensitive to range noise, fine enough that a
        // metre of travel empties most of them.
        double voxel_m = 0.3;
        // Compare with the opening view so slow displacement accumulates.
        double moved_overlap = 0.6;
        // One frame below the threshold can be a passer-by; a run of them is
        // the platform leaving.
        int moved_consecutive = 3;
        // Minimum scan support for an overlap estimate.
        int min_points = 200;
        // Maximum startup delay; non-positive bypasses the guard.
        double max_wait_s = 120.0;
    };

    StaticStartGuard() = default;
    explicit StaticStartGuard(const Options& options) : options_(options) {}

    bool update(double timestamp, const pcl::PointCloud<pcl::PointXYZI>& cloud,
                const Eigen::Isometry3d& odom_pose);

    bool released() const { return released_; }
    // Diagnostics for the frame that released the guard.
    double releaseTimestamp() const { return release_timestamp_; }
    double lastOverlap() const { return last_overlap_; }
    bool releasedByTimeout() const { return released_by_timeout_; }

private:
    using VoxelKey = std::uint64_t;
    std::unordered_set<VoxelKey> voxelize(
        const pcl::PointCloud<pcl::PointXYZI>& cloud) const;

    Options options_;
    bool released_ = false;
    bool released_by_timeout_ = false;
    bool has_reference_ = false;
    double first_timestamp_ = 0.0;
    double release_timestamp_ = 0.0;
    double last_overlap_ = 1.0;
    int consecutive_below_ = 0;
    double stable_since_ = -1.0;
    double previous_timestamp_ = -1.0;
    Eigen::Isometry3d stable_pose_ = Eigen::Isometry3d::Identity();
    std::unordered_set<VoxelKey> reference_;
};

StaticStartGuard::Options staticStartGuardOptionsFromConfig(const Config& config);

}  // namespace n3mapping
