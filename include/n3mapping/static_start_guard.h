#pragma once

#include <cstdint>
#include <unordered_set>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct Config;

class StaticStartGuard {
public:
    struct Options {

        double voxel_m = 0.3;

        double moved_overlap = 0.6;

        int moved_consecutive = 3;

        int min_points = 200;

        double max_wait_s = 120.0;
    };

    StaticStartGuard() = default;
    explicit StaticStartGuard(const Options& options) : options_(options) {}

    bool update(double timestamp, const pcl::PointCloud<pcl::PointXYZI>& cloud,
                const Eigen::Isometry3d& odom_pose);

    bool released() const { return released_; }

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

}
