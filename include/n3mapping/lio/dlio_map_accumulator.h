// ROS-free DLIO dense-map accumulator extracted from the DLIO map node shape.
#pragma once

#include <cstddef>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {
namespace lio {
namespace dlio {

class MapAccumulator {
public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZI>;

    struct Options {
        double leaf_size = 0.5;
        size_t dense_input_skip = 1;
    };

    struct AddResult {
        bool accepted = false;
        size_t input_points = 0;
        size_t filtered_points = 0;
        size_t map_points = 0;
    };

    MapAccumulator();
    explicit MapAccumulator(const Options& options);

    AddResult addKeyframe(const PointCloud::ConstPtr& keyframe);
    void clear();

    size_t inputsSeen() const { return inputs_seen_; }
    size_t acceptedKeyframes() const { return accepted_keyframes_; }
    const Options& options() const { return options_; }
    PointCloud::ConstPtr map() const { return map_; }

private:
    PointCloud::Ptr filteredCopy(const PointCloud::ConstPtr& keyframe) const;

    Options options_;
    size_t inputs_seen_ = 0;
    size_t accepted_keyframes_ = 0;
    size_t skip_counter_ = 0;
    PointCloud::Ptr map_;
};

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
