#include <gtest/gtest.h>

#include <cstdint>

#include <Eigen/Geometry>

#include "n3mapping/global_map_cache.h"

namespace n3mapping {
namespace {

using Cloud = pcl::PointCloud<pcl::PointXYZI>;

Cloud::Ptr makeCloud(const std::vector<Eigen::Vector3f>& points)
{
    auto cloud = pcl::make_shared<Cloud>();
    cloud->points.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        pcl::PointXYZI point;
        point.x = points[i].x();
        point.y = points[i].y();
        point.z = points[i].z();
        point.intensity = static_cast<float>(i + 1);
        cloud->points.push_back(point);
    }
    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

Keyframe::Ptr makeKeyframe(int64_t id, const Eigen::Vector3d& translation, const Cloud::Ptr& cloud)
{
    auto pose = Eigen::Isometry3d::Identity();
    pose.translation() = translation;
    auto keyframe = Keyframe::create(id, static_cast<double>(id), pose, cloud);
    keyframe->pose_optimized = pose;
    return keyframe;
}

TEST(GlobalMapCacheTest, RepeatedUpdateWithoutNewKeyframesKeepsRevision)
{
    GlobalMapCache cache(0.0);
    auto keyframe = makeKeyframe(0, Eigen::Vector3d(1.0, 2.0, 3.0), makeCloud({Eigen::Vector3f(0.0f, 0.0f, 0.0f)}));

    auto cloud = cache.update({keyframe});
    ASSERT_NE(cloud, nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FLOAT_EQ(cloud->points.front().x, 1.0f);
    EXPECT_FLOAT_EQ(cloud->points.front().y, 2.0f);
    EXPECT_FLOAT_EQ(cloud->points.front().z, 3.0f);
    const auto revision = cache.revision();

    cloud = cache.update({keyframe});
    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->size(), 1u);
    EXPECT_EQ(cache.revision(), revision);
}

TEST(GlobalMapCacheTest, NewKeyframeAppendsIncrementally)
{
    GlobalMapCache cache(0.0);
    auto kf0 = makeKeyframe(0, Eigen::Vector3d(0.0, 0.0, 0.0), makeCloud({Eigen::Vector3f(0.0f, 0.0f, 0.0f)}));
    auto kf1 = makeKeyframe(1, Eigen::Vector3d(10.0, 0.0, 0.0), makeCloud({Eigen::Vector3f(1.0f, 0.0f, 0.0f)}));

    auto cloud = cache.update({kf0});
    ASSERT_NE(cloud, nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    const auto revision0 = cache.revision();

    cloud = cache.update({kf0, kf1});
    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->size(), 2u);
    EXPECT_GT(cache.revision(), revision0);
    EXPECT_EQ(cache.cachedKeyframeCount(), 2u);
}

TEST(GlobalMapCacheTest, FullRebuildUsesUpdatedOptimizedPoses)
{
    GlobalMapCache cache(0.0);
    auto keyframe = makeKeyframe(0, Eigen::Vector3d(0.0, 0.0, 0.0), makeCloud({Eigen::Vector3f(1.0f, 0.0f, 0.0f)}));

    auto cloud = cache.update({keyframe});
    ASSERT_NE(cloud, nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FLOAT_EQ(cloud->points.front().x, 1.0f);
    const auto revision0 = cache.revision();

    keyframe->pose_optimized.translation() = Eigen::Vector3d(5.0, 0.0, 0.0);
    cloud = cache.update({keyframe});
    ASSERT_NE(cloud, nullptr);
    EXPECT_FLOAT_EQ(cloud->points.front().x, 1.0f);
    EXPECT_EQ(cache.revision(), revision0);

    cache.markFullRebuildRequired();
    cloud = cache.update({keyframe});
    ASSERT_NE(cloud, nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_FLOAT_EQ(cloud->points.front().x, 6.0f);
    EXPECT_GT(cache.revision(), revision0);
}

TEST(GlobalMapCacheTest, VoxelModeMergesPointsInSameVoxel)
{
    GlobalMapCache cache(1.0);
    auto keyframe = makeKeyframe(
        0,
        Eigen::Vector3d::Zero(),
        makeCloud({Eigen::Vector3f(0.1f, 0.1f, 0.1f), Eigen::Vector3f(0.2f, 0.2f, 0.2f)}));

    auto cloud = cache.update({keyframe});
    ASSERT_NE(cloud, nullptr);
    ASSERT_EQ(cloud->size(), 1u);
    EXPECT_NEAR(cloud->points.front().x, 0.15f, 1e-5f);
    EXPECT_NEAR(cloud->points.front().y, 0.15f, 1e-5f);
    EXPECT_NEAR(cloud->points.front().z, 0.15f, 1e-5f);
}

}  // namespace
}  // namespace n3mapping
