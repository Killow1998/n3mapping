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

    void clear();
    bool addFrame(const LioCoreState& state,
                  const PointCloud::ConstPtr& lidar_cloud);

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
