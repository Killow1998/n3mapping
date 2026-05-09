#include <gtest/gtest.h>

#include "n3mapping/lio/local_map.h"

namespace n3mapping {
namespace test {
namespace {

lio::LioLocalMap::PointCloud::Ptr makeCloud(float x) {
    auto cloud = pcl::make_shared<lio::LioLocalMap::PointCloud>();
    pcl::PointXYZI point;
    point.x = x;
    point.y = 0.0f;
    point.z = 0.0f;
    point.intensity = 1.0f;
    cloud->push_back(point);
    cloud->width = 1;
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

lio::LioCoreState makeState(double tx) {
    lio::LioCoreState state;
    state.initialized = true;
    state.T_world_lidar.translation().x() = tx;
    return state;
}

}  // namespace

TEST(LioLocalMapTest, RejectsInvalidFrames) {
    lio::LioLocalMap map;
    EXPECT_FALSE(map.addFrame(lio::LioCoreState{}, makeCloud(0.0f)));
    EXPECT_FALSE(map.addFrame(makeState(0.0), nullptr));
    EXPECT_EQ(map.size(), 0u);
    ASSERT_TRUE(map.cloud());
    EXPECT_TRUE(map.cloud()->empty());
}

TEST(LioLocalMapTest, TransformsCloudIntoWorldFrame) {
    lio::LioLocalMap map;
    ASSERT_TRUE(map.addFrame(makeState(2.0), makeCloud(1.0f)));

    ASSERT_TRUE(map.cloud());
    ASSERT_EQ(map.cloud()->size(), 1u);
    EXPECT_NEAR(map.cloud()->at(0).x, 3.0f, 1e-6f);
}

TEST(LioLocalMapTest, KeepsBoundedKeyframeWindow) {
    lio::LioLocalMap map(2);
    ASSERT_TRUE(map.addFrame(makeState(0.0), makeCloud(1.0f)));
    ASSERT_TRUE(map.addFrame(makeState(1.0), makeCloud(1.0f)));
    ASSERT_TRUE(map.addFrame(makeState(2.0), makeCloud(1.0f)));

    EXPECT_EQ(map.size(), 2u);
    ASSERT_TRUE(map.cloud());
    ASSERT_EQ(map.cloud()->size(), 2u);
    EXPECT_NEAR(map.cloud()->at(0).x, 2.0f, 1e-6f);
    EXPECT_NEAR(map.cloud()->at(1).x, 3.0f, 1e-6f);
}

TEST(LioLocalMapTest, ClearResetsAggregate) {
    lio::LioLocalMap map;
    ASSERT_TRUE(map.addFrame(makeState(1.0), makeCloud(1.0f)));
    map.clear();
    EXPECT_EQ(map.size(), 0u);
    ASSERT_TRUE(map.cloud());
    EXPECT_TRUE(map.cloud()->empty());
}

TEST(LioLocalMapTest, EstimatesCentroidCorrectionFromNearestNeighbors) {
    lio::LioLocalMap map;
    ASSERT_TRUE(map.addFrame(makeState(0.0), makeCloud(0.0f)));

    auto source = makeCloud(0.0f);
    auto predicted = makeState(0.25);
    const auto stats = map.estimateAlignmentCorrection(predicted, source, 1.0);

    EXPECT_TRUE(stats.valid);
    EXPECT_EQ(stats.source_points, 1u);
    EXPECT_EQ(stats.matched_points, 1u);
    EXPECT_NEAR(stats.centroid_correction_world.x(), -0.25, 1e-6);
    EXPECT_NEAR(stats.mean_squared_error, 0.0625, 1e-6);
}

TEST(LioLocalMapTest, RejectsAlignmentWhenNoCorrespondences) {
    lio::LioLocalMap map;
    ASSERT_TRUE(map.addFrame(makeState(0.0), makeCloud(0.0f)));

    auto source = makeCloud(10.0f);
    const auto stats = map.estimateAlignmentCorrection(makeState(0.0), source, 0.5);

    EXPECT_FALSE(stats.valid);
    EXPECT_EQ(stats.matched_points, 0u);
}

}  // namespace test
}  // namespace n3mapping
