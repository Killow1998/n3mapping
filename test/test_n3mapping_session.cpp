#include <gtest/gtest.h>

#include "n3mapping/core/n3mapping_session.h"

namespace n3mapping {
namespace test {
namespace {

pcl::PointCloud<pcl::PointXYZI>::Ptr makeCloud() {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (float x = -1.0f; x <= 1.0f; x += 0.1f) {
        pcl::PointXYZI point;
        point.x = x;
        point.y = 0.5f;
        point.z = 0.25f;
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(N3MappingSessionTest, OwnsCoreComponentGraph) {
    Config config;
    config.keyframe_distance_threshold = 0.5;
    config.rhpd_submap_voxel_size = 0.0;
    core::N3MappingSession session(config);

    EXPECT_EQ(session.keyframes().size(), 0u);
    auto result = session.mappingModeProcessor().process(
        1.0, Eigen::Isometry3d::Identity(), makeCloud());
    ASSERT_TRUE(result.accepted_keyframe);
    EXPECT_EQ(result.keyframe_id, 0);
    EXPECT_TRUE(session.graphOptimizer().hasNode(0));
    EXPECT_EQ(session.keyframes().size(), 1u);

    auto missing_loop = session.processLoopClosureForKeyframe(999);
    EXPECT_FALSE(missing_loop.query_exists);
    EXPECT_FALSE(missing_loop.candidates_found);
    EXPECT_TRUE(missing_loop.selected_loops.empty());
}

}  // namespace test
}  // namespace n3mapping
