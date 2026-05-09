#include <gtest/gtest.h>

#include "n3mapping/core/mapping_mode_processor.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (float x = -2.0f; x <= 2.0f; x += 0.2f) {
        pcl::PointXYZI point;
        point.x = x;
        point.y = 1.0f;
        point.z = 0.5f;
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(MappingModeProcessorTest, AddsKeyframesAndDescriptors) {
    Config config;
    config.keyframe_distance_threshold = 0.5;
    config.rhpd_submap_voxel_size = 0.0;

    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    GraphOptimizer optimizer(config);
    core::MappingModeProcessor processor(config, keyframes, loop_detector, optimizer);

    Eigen::Isometry3d pose0 = Eigen::Isometry3d::Identity();
    auto out0 = processor.process(1.0, pose0, makeCloud());
    ASSERT_TRUE(out0.accepted_keyframe);
    EXPECT_EQ(out0.keyframe_id, 0);
    EXPECT_TRUE(optimizer.hasNode(0));
    auto kf0 = keyframes.getKeyframe(0);
    ASSERT_TRUE(kf0);
    EXPECT_EQ(kf0->rhpd_descriptor.size(), RHPD_DIM);

    auto duplicate = processor.process(1.1, pose0, makeCloud());
    EXPECT_FALSE(duplicate.accepted_keyframe);
    EXPECT_EQ(keyframes.size(), 1u);

    Eigen::Isometry3d pose1 = Eigen::Isometry3d::Identity();
    pose1.translation().x() = 1.0;
    auto out1 = processor.process(2.0, pose1, makeCloud());
    ASSERT_TRUE(out1.accepted_keyframe);
    EXPECT_EQ(out1.keyframe_id, 1);
    EXPECT_TRUE(optimizer.hasNode(1));
    EXPECT_EQ(keyframes.size(), 2u);
}

}  // namespace test
}  // namespace n3mapping
