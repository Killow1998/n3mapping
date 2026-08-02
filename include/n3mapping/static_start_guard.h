// static_start_guard.h - waits for the platform to actually move before mapping.
#pragma once

#include <cstdint>
#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

// The estimator is at its worst while the platform stands still: with no motion
// there is nothing to separate a tilted body from a gravity vector pointing
// elsewhere, and it slides along that direction. On the 0723 recording it spends
// 33 s reporting 9.37 m of travel and 8.24 degrees of attitude swing that the
// accelerometer says did not happen. Mapping those frames writes the moving part
// of that error into the odometry edges, where nothing downstream can undo it.
//
// Dropping them is not a workaround: a stationary opening carries no mapping
// information to lose. Doing exactly that by hand -- truncating the bag to the
// moment the robot moved -- took the worst revisit pair from 2.3275 m to 0.5865
// and the floor residual span from 0.434 to 0.255, after thirteen stages in
// which nothing else had moved either number.
//
// The odometry cannot be asked whether the platform moved, because during that
// same stationary opening it claims 9.37 m and speeds up to 0.94 m/s. The scan
// can: standing still, what the sensor sees does not change.
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

    // Feeds one frame's cloud in the sensor frame. Returns true once the
    // platform has been seen to move; latches, because a platform that has
    // moved does not become un-moved.
    bool update(double timestamp, const pcl::PointCloud<pcl::PointXYZI>& cloud);

    bool moved() const { return moved_; }
    // Diagnostics for the frame that released the guard.
    double releaseTimestamp() const { return release_timestamp_; }
    double lastOverlap() const { return last_overlap_; }
    bool releasedByTimeout() const { return released_by_timeout_; }

private:
    using VoxelKey = std::uint64_t;
    std::unordered_set<VoxelKey> voxelize(
        const pcl::PointCloud<pcl::PointXYZI>& cloud) const;

    Options options_;
    bool moved_ = false;
    bool released_by_timeout_ = false;
    bool has_reference_ = false;
    double first_timestamp_ = 0.0;
    double release_timestamp_ = 0.0;
    double last_overlap_ = 1.0;
    int consecutive_below_ = 0;
    std::unordered_set<VoxelKey> reference_;
};

}  // namespace n3mapping
