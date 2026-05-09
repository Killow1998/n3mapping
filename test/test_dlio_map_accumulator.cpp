#include <gtest/gtest.h>

#include "n3mapping/lio/dlio_map_accumulator.h"

namespace n3mapping {
namespace test {
namespace {

lio::dlio::MapAccumulator::PointCloud::Ptr makeCloud(
    const std::vector<float>& xs) {
    auto cloud = pcl::make_shared<lio::dlio::MapAccumulator::PointCloud>();
    for (float x : xs) {
        pcl::PointXYZI point;
        point.x = x;
        point.y = 0.0f;
        point.z = 0.0f;
        point.intensity = 1.0f;
        cloud->push_back(point);
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

}  // namespace

TEST(DlioMapAccumulatorTest, SkipsDenseInputsLikeDlioMapNode) {
    lio::dlio::MapAccumulator::Options options;
    options.dense_input_skip = 3;
    options.leaf_size = 0.0;
    lio::dlio::MapAccumulator accumulator(options);

    EXPECT_FALSE(accumulator.addKeyframe(makeCloud({1.0f})).accepted);
    EXPECT_FALSE(accumulator.addKeyframe(makeCloud({2.0f})).accepted);
    const auto accepted = accumulator.addKeyframe(makeCloud({3.0f}));

    EXPECT_TRUE(accepted.accepted);
    EXPECT_EQ(accumulator.inputsSeen(), 3u);
    EXPECT_EQ(accumulator.acceptedKeyframes(), 1u);
    ASSERT_TRUE(accumulator.map());
    ASSERT_EQ(accumulator.map()->size(), 1u);
    EXPECT_NEAR(accumulator.map()->at(0).x, 3.0f, 1e-6f);
}

TEST(DlioMapAccumulatorTest, VoxelFiltersAcceptedKeyframesBeforeAccumulating) {
    lio::dlio::MapAccumulator::Options options;
    options.dense_input_skip = 1;
    options.leaf_size = 0.5;
    lio::dlio::MapAccumulator accumulator(options);

    const auto result = accumulator.addKeyframe(makeCloud({0.0f, 0.1f, 2.0f}));

    EXPECT_TRUE(result.accepted);
    EXPECT_EQ(result.input_points, 3u);
    EXPECT_EQ(result.filtered_points, 2u);
    EXPECT_EQ(result.map_points, 2u);
    ASSERT_TRUE(accumulator.map());
    EXPECT_EQ(accumulator.map()->size(), 2u);
}

TEST(DlioMapAccumulatorTest, ClearResetsCountersAndMap) {
    lio::dlio::MapAccumulator accumulator;
    ASSERT_TRUE(accumulator.addKeyframe(makeCloud({1.0f})).accepted);

    accumulator.clear();

    EXPECT_EQ(accumulator.inputsSeen(), 0u);
    EXPECT_EQ(accumulator.acceptedKeyframes(), 0u);
    ASSERT_TRUE(accumulator.map());
    EXPECT_TRUE(accumulator.map()->empty());
}

TEST(DlioMapAccumulatorTest, RejectsEmptyAcceptedInputsWithoutGrowingMap) {
    lio::dlio::MapAccumulator accumulator;
    const auto result = accumulator.addKeyframe(makeCloud({}));

    EXPECT_FALSE(result.accepted);
    EXPECT_EQ(result.input_points, 0u);
    EXPECT_EQ(result.map_points, 0u);
    EXPECT_EQ(accumulator.inputsSeen(), 1u);
    EXPECT_EQ(accumulator.acceptedKeyframes(), 0u);
}

}  // namespace test
}  // namespace n3mapping
