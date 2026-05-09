#include <gtest/gtest.h>

#include "n3mapping/core/localization_mode_processor.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::PointXYZI point;
    point.x = 1.0f;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 1.0f;
    cloud->push_back(point);
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(LocalizationModeProcessorTest, NoMapPublishesOdomPose) {
    Config config;
    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    PointCloudMatcher matcher(config);
    WorldLocalizing world_localizing(config, keyframes, loop_detector, matcher);
    core::LocalizationModeProcessor processor(world_localizing);

    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    odom_pose.translation().x() = 3.0;
    auto result = processor.process(false, odom_pose, makeCloud());

    EXPECT_FALSE(result.map_loaded);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.relocalization_locked);
    EXPECT_NEAR(result.publish_pose.translation().x(), 3.0, 1e-12);
}

TEST(LocalizationModeProcessorTest, RelocalizedWithoutNearestKeyframePublishesPredictedPose) {
    Config config;
    KeyframeManager keyframes(config);
    LoopDetector loop_detector(config);
    PointCloudMatcher matcher(config);
    WorldLocalizing world_localizing(config, keyframes, loop_detector, matcher);
    core::LocalizationModeProcessor processor(world_localizing);

    Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
    T_map_odom.translation().x() = 10.0;
    world_localizing.setMapToOdomTransform(T_map_odom);

    Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
    odom_pose.translation().x() = 3.0;
    auto result = processor.process(true, odom_pose, makeCloud());

    EXPECT_TRUE(result.map_loaded);
    EXPECT_TRUE(result.success);
    EXPECT_FALSE(result.relocalization_locked);
    EXPECT_NEAR(result.publish_pose.translation().x(), 13.0, 1e-12);
}

}  // namespace test
}  // namespace n3mapping
