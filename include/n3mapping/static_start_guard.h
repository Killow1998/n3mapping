// static_start_guard.h - rejects unsettled odometry at mapping startup.
#pragma once

#include <cstdint>
#include <unordered_set>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct Config;

// The estimator is at its worst while the platform stands still: with no motion
// there is nothing to separate a tilted body from a gravity vector pointing
// elsewhere, and it slides along that direction. On the 0723 recording it spends
// 33 s reporting 9.37 m of travel and 8.24 degrees of attitude swing that the
// accelerometer says did not happen. Mapping those frames writes the moving part
// of that error into the odometry edges, where nothing downstream can undo it.
//
// A stable scan alone cannot distinguish that drift from a healthy stationary
// estimator. Require both a stable view and a continuous, bounded pose window
// to start without motion. Otherwise retain the scan-based motion test and the
// existing timeout. Odometry alone is never evidence of physical movement.
class StaticStartGuard {
public:
    struct Options {
        // Coarse enough to be insensitive to range noise, fine enough that a
        // metre of travel empties most of them.
        double voxel_m = 0.3;
        // Overlap with the opening view, below which the platform has moved.
        // Comparing against the first scan rather than the previous one matters:
        // at 0.17 m/s and 10 Hz a frame advances 17 mm, which barely disturbs
        // consecutive-frame overlap, while overlap against a fixed reference
        // falls away as displacement accumulates.
        double moved_overlap = 0.6;
        // One frame below the threshold can be a passer-by; a run of them is
        // the platform leaving.
        int moved_consecutive = 3;
        // Two things this test does not see, both acceptable. A surface lying
        // exactly on a voxel boundary flips between two cells under millimetre
        // noise, so a floor at an exact multiple of voxel_m reads as less
        // stable than it is; real floors do not sit on the grid. And turning on
        // the spot over a floor-dominated scene barely changes which cells are
        // occupied, so pure rotation may not release the guard -- which is
        // fine, because rotation is exactly the excitation the estimator was
        // missing, and any translation afterwards releases it.
        // Too few points to judge anything by.
        int min_points = 200;
        // Start mapping regardless after this long. A session that never moves
        // has nothing to map, so this only guards against the overlap test
        // failing to notice motion it should have seen. Non-positive disables
        // the guard entirely.
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

// The guard's settings as configuration sets them. A function rather than a
// block inside the mapping loop, so a test can assert that each key arrives
// where it is meant to.
StaticStartGuard::Options staticStartGuardOptionsFromConfig(const Config& config);

}  // namespace n3mapping
