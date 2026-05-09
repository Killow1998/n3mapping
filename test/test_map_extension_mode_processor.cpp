#include <gtest/gtest.h>

#include "n3mapping/core/map_extension_mode_processor.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
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

struct Fixture {
    Config config;
    KeyframeManager keyframes{config};
    LoopDetector loop_detector{config};
    PointCloudMatcher matcher{config};
    GraphOptimizer optimizer{config};
    MapSerializer serializer{config};
    WorldLocalizing world_localizing{config, keyframes, loop_detector, matcher};
    MappingResuming mapping_resuming{
        config, keyframes, loop_detector, matcher, optimizer, serializer, world_localizing};
    core::MapExtensionModeProcessor processor{
        keyframes, optimizer, world_localizing, mapping_resuming};
};

}  // namespace

TEST(MapExtensionModeProcessorTest, NoMapDoesNotPublish) {
    Fixture f;
    auto result = f.processor.process(false, 1.0, Eigen::Isometry3d::Identity(), makeCloud());
    EXPECT_FALSE(result.map_loaded);
    EXPECT_FALSE(result.should_publish);
    EXPECT_FALSE(result.accepted_keyframe);
}

TEST(MapExtensionModeProcessorTest, NotInitializedMapLoadedDoesNotPublish) {
    Fixture f;
    auto result = f.processor.process(true, 1.0, Eigen::Isometry3d::Identity(), makeCloud());
    EXPECT_TRUE(result.map_loaded);
    EXPECT_FALSE(result.should_publish);
    EXPECT_FALSE(result.initial_relocalization_attempted);
    EXPECT_FALSE(result.accepted_keyframe);
}

}  // namespace test
}  // namespace n3mapping
