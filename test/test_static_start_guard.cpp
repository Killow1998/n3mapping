#include <gtest/gtest.h>

#include <cmath>

#include "n3mapping/static_start_guard.h"

namespace {

using n3mapping::StaticStartGuard;

pcl::PointCloud<pcl::PointXYZI> roomSeenFrom(double x, double y, double yaw = 0.0)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    const double c = std::cos(-yaw), s = std::sin(-yaw);
    for (int i = -60; i <= 60; ++i) {
        for (int j = -60; j <= 60; ++j) {

            const double wx = i * 0.1 - x;
            const double wy = j * 0.1 - y;
            pcl::PointXYZI p;
            p.x = static_cast<float>(wx * c - wy * s);
            p.y = static_cast<float>(wx * s + wy * c);
            p.z = -0.65f;
            p.intensity = 0.0f;
            cloud.points.push_back(p);
        }
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    return cloud;
}

Eigen::Isometry3d driftingPose(double time)
{
    auto pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = time * 0.3;
    return pose;
}

TEST(StaticStartGuard, ReleasesStableStationaryInputAfterThreeSeconds)
{
    StaticStartGuard guard;
    const auto still = roomSeenFrom(0.0, 0.0);
    for (int i = 0; i < 30; ++i) {
        EXPECT_FALSE(guard.update(i * 0.1, still, Eigen::Isometry3d::Identity()));
    }
    EXPECT_TRUE(guard.update(3.0, still, Eigen::Isometry3d::Identity()));
    EXPECT_FALSE(guard.releasedByTimeout());
}

TEST(StaticStartGuard, RejectsDriftingOdometryDespiteStableNoisyScans)
{
    StaticStartGuard guard;
    for (int i = 0; i < 300; ++i) {
        auto cloud = roomSeenFrom(0.0, 0.0);
        for (auto& p : cloud.points) {
            p.z += static_cast<float>(0.01 * std::sin(i + p.x));
        }
        guard.update(i * 0.1, cloud, driftingPose(i * 0.1));
    }
    EXPECT_FALSE(guard.released());
}

TEST(StaticStartGuard, ReleasesOnceSlowMotionAccumulates)
{
    StaticStartGuard guard;
    for (int i = 0; i < 40; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0), driftingPose(i * 0.1));
    }
    ASSERT_FALSE(guard.released());

    bool released = false;
    for (int i = 40; i < 400 && !released; ++i) {
        released = guard.update(i * 0.1, roomSeenFrom((i - 39) * 0.017, 0.0), driftingPose(i * 0.1));
    }
    EXPECT_TRUE(released);
    EXPECT_FALSE(guard.releasedByTimeout());
}

TEST(StaticStartGuard, MayNotSeeRotationAlone_ButSeesWhatFollows)
{
    StaticStartGuard guard;
    for (int i = 0; i < 20; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0), driftingPose(i * 0.1));
    }
    for (int i = 20; i < 200; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0, (i - 19) * 0.02), driftingPose(i * 0.1));
    }
    bool released = false;
    for (int i = 200; i < 500 && !released; ++i) {
        released = guard.update(i * 0.1, roomSeenFrom((i - 199) * 0.05, 0.0), driftingPose(i * 0.1));
    }
    EXPECT_TRUE(released);
}

TEST(StaticStartGuard, Latches)
{
    StaticStartGuard guard;
    for (int i = 0; i < 20; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0), driftingPose(i * 0.1));
    }
    for (int i = 20; i < 400; ++i) {
        if (guard.update(i * 0.1, roomSeenFrom((i - 19) * 0.05, 0.0), driftingPose(i * 0.1))) {
            break;
        }
    }
    ASSERT_TRUE(guard.released());
    const double released_at = guard.releaseTimestamp();
    for (int i = 0; i < 50; ++i) {
        EXPECT_TRUE(guard.update(100.0 + i * 0.1, roomSeenFrom(0.0, 0.0), Eigen::Isometry3d::Identity()));
    }
    EXPECT_DOUBLE_EQ(guard.releaseTimestamp(), released_at);
}

TEST(StaticStartGuard, GivesUpWaitingAfterTheCap)
{
    StaticStartGuard::Options options;
    options.max_wait_s = 5.0;
    StaticStartGuard guard(options);
    const auto still = roomSeenFrom(0.0, 0.0);
    bool released = false;
    for (int i = 0; i < 200 && !released; ++i) {
        released = guard.update(i * 0.1, still, driftingPose(i * 0.1));
    }
    EXPECT_TRUE(released);
    EXPECT_TRUE(guard.releasedByTimeout());
}

TEST(StaticStartGuard, NonPositiveCapDisablesTheWait)
{
    StaticStartGuard::Options options;
    options.max_wait_s = 0.0;
    StaticStartGuard guard(options);
    EXPECT_TRUE(guard.update(0.0, roomSeenFrom(0.0, 0.0), Eigen::Isometry3d::Identity()));
}

TEST(StaticStartGuard, IgnoresCloudsTooSmallToJudge)
{
    StaticStartGuard guard;
    pcl::PointCloud<pcl::PointXYZI> sparse;
    for (int i = 0; i < 10; ++i) {
        pcl::PointXYZI p;
        p.x = static_cast<float>(i);
        p.y = p.z = 0.0f;
        p.intensity = 0.0f;
        sparse.points.push_back(p);
    }
    sparse.width = sparse.points.size();
    sparse.height = 1;
    for (int i = 0; i < 100; ++i) {
        EXPECT_FALSE(guard.update(i * 0.1, sparse, Eigen::Isometry3d::Identity()));
    }
}

TEST(StaticStartGuard, SparseOrDiscontinuousInputCannotAccumulateStableTime)
{
    StaticStartGuard guard;
    const auto still = roomSeenFrom(0.0, 0.0);
    const auto pose = Eigen::Isometry3d::Identity();
    for (int i = 0; i < 10; ++i) EXPECT_FALSE(guard.update(i, still, pose));
    for (int i = 0; i < 25; ++i) EXPECT_FALSE(guard.update(10.0 + i * 0.1, still, pose));
    EXPECT_FALSE(guard.update(12.5, {}, pose));
    for (int i = 0; i < 30; ++i) EXPECT_FALSE(guard.update(12.6 + i * 0.1, still, pose));
    EXPECT_TRUE(guard.update(15.7, still, pose));
}

TEST(StaticStartGuard, RejectsRotationalDriftAndRepeatedTimestamps)
{
    StaticStartGuard guard;
    const auto still = roomSeenFrom(0.0, 0.0);
    for (int i = 0; i < 300; ++i) {
        auto pose = Eigen::Isometry3d::Identity();
        pose.linear() = Eigen::AngleAxisd(i * 0.001, Eigen::Vector3d::UnitY()).toRotationMatrix();
        EXPECT_FALSE(guard.update(i * 0.1, still, pose));
    }
    StaticStartGuard repeated;
    for (int i = 0; i < 100; ++i) EXPECT_FALSE(repeated.update(1.0, still, Eigen::Isometry3d::Identity()));
}

}
