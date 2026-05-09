// ROS-free local map container shared by in-process LIO cores.
#pragma once

#include <cstddef>
#include <deque>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/lio/core_state.h"

namespace n3mapping {
namespace lio {

class LioLocalMap {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

    explicit LioLocalMap(size_t max_keyframes = 20);

    struct AlignmentStats {
        size_t source_points = 0;
        size_t matched_points = 0;
        double mean_squared_error = 0.0;
        Eigen::Vector3d centroid_correction_world = Eigen::Vector3d::Zero();
        bool valid = false;
    };

    void clear();
    bool addFrame(const LioCoreState& state,
                  const PointCloud::ConstPtr& lidar_cloud);
    AlignmentStats estimateAlignmentCorrection(
        const LioCoreState& predicted_state,
        const PointCloud::ConstPtr& lidar_cloud,
        double max_correspondence_distance) const;

    size_t size() const { return keyframes_.size(); }
    size_t maxKeyframes() const { return max_keyframes_; }
    PointCloud::ConstPtr cloud() const { return aggregate_; }

private:
    struct Keyframe {
        LioCoreState state;
        PointCloud::Ptr cloud_world;
    };

    void rebuildAggregate();

    size_t max_keyframes_;
    std::deque<Keyframe> keyframes_;
    PointCloud::Ptr aggregate_;
};

}  // namespace lio
}  // namespace n3mapping
