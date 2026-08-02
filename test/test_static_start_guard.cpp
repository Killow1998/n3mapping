#include <gtest/gtest.h>

#include <cmath>

#include "n3mapping/static_start_guard.h"

namespace {

using n3mapping::StaticStartGuard;

// A room the sensor sits in: floor, and four walls a few metres out.
pcl::PointCloud<pcl::PointXYZI> roomSeenFrom(double x, double y, double yaw = 0.0)
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    const double c = std::cos(-yaw), s = std::sin(-yaw);
    for (int i = -60; i <= 60; ++i) {
        for (int j = -60; j <= 60; ++j) {
            // Points are in the sensor frame, so moving the sensor moves them
            // the other way.
            const double wx = i * 0.1 - x;
            const double wy = j * 0.1 - y;
            pcl::PointXYZI p;
            p.x = static_cast<float>(wx * c - wy * s);
            p.y = static_cast<float>(wx * s + wy * c);
            p.z = -0.65f;  // off the 0.3 m voxel boundary, as a real floor is
            p.intensity = 0.0f;
            cloud.points.push_back(p);
        }
    }
    cloud.width = cloud.points.size();
    cloud.height = 1;
    return cloud;
}

TEST(StaticStartGuard, HoldsWhileTheViewDoesNotChange)
{
    StaticStartGuard guard;
    const auto still = roomSeenFrom(0.0, 0.0);
    for (int i = 0; i < 300; ++i) {
        EXPECT_FALSE(guard.update(i * 0.1, still)) << "released at frame " << i;
    }
    EXPECT_FALSE(guard.moved());
}

// The odometry claims 9.37 m across the stationary opening it should claim none
// of, which is why the guard is not allowed to look at it. Sensor noise is the
// only thing that moves here.
TEST(StaticStartGuard, IsNotFooledByNoise)
{
    StaticStartGuard guard;
    for (int i = 0; i < 300; ++i) {
        auto cloud = roomSeenFrom(0.0, 0.0);
        for (auto& p : cloud.points) {
            p.z += static_cast<float>(0.01 * std::sin(i + p.x));
        }
        guard.update(i * 0.1, cloud);
    }
    EXPECT_FALSE(guard.moved());
}

// 0.17 m/s at 10 Hz advances 17 mm a frame, which is why the reference is the
// opening view and not the previous one: consecutive frames barely differ.
TEST(StaticStartGuard, ReleasesOnceSlowMotionAccumulates)
{
    StaticStartGuard guard;
    for (int i = 0; i < 40; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0));
    }
    ASSERT_FALSE(guard.moved());

    bool released = false;
    for (int i = 40; i < 400 && !released; ++i) {
        released = guard.update(i * 0.1, roomSeenFrom((i - 39) * 0.017, 0.0));
    }
    EXPECT_TRUE(released);
    EXPECT_FALSE(guard.releasedByTimeout());
}

// Turning on the spot over a floor-dominated scene barely changes which cells
// are occupied, so the guard need not release on it. That is acceptable: the
// rotation is itself the excitation the estimator was short of, and the
// translation that follows releases the guard anyway.
TEST(StaticStartGuard, MayNotSeeRotationAlone_ButSeesWhatFollows)
{
    StaticStartGuard guard;
    for (int i = 0; i < 20; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0));
    }
    for (int i = 20; i < 200; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0, (i - 19) * 0.02));
    }
    bool released = false;
    for (int i = 200; i < 500 && !released; ++i) {
        released = guard.update(i * 0.1, roomSeenFrom((i - 199) * 0.05, 0.0));
    }
    EXPECT_TRUE(released);
}

// Having moved once, a platform does not become un-moved.
TEST(StaticStartGuard, Latches)
{
    StaticStartGuard guard;
    for (int i = 0; i < 20; ++i) {
        guard.update(i * 0.1, roomSeenFrom(0.0, 0.0));
    }
    for (int i = 20; i < 400; ++i) {
        if (guard.update(i * 0.1, roomSeenFrom((i - 19) * 0.05, 0.0))) {
            break;
        }
    }
    ASSERT_TRUE(guard.moved());
    const double released_at = guard.releaseTimestamp();
    for (int i = 0; i < 50; ++i) {
        EXPECT_TRUE(guard.update(100.0 + i * 0.1, roomSeenFrom(0.0, 0.0)));
    }
    EXPECT_DOUBLE_EQ(guard.releaseTimestamp(), released_at);
}

// A test that fails to notice motion must not suppress the map for a whole
// session, so the wait is bounded.
TEST(StaticStartGuard, GivesUpWaitingAfterTheCap)
{
    StaticStartGuard::Options options;
    options.max_wait_s = 5.0;
    StaticStartGuard guard(options);
    const auto still = roomSeenFrom(0.0, 0.0);
    bool released = false;
    for (int i = 0; i < 200 && !released; ++i) {
        released = guard.update(i * 0.1, still);
    }
    EXPECT_TRUE(released);
    EXPECT_TRUE(guard.releasedByTimeout());
}

TEST(StaticStartGuard, NonPositiveCapDisablesTheWait)
{
    StaticStartGuard::Options options;
    options.max_wait_s = 0.0;
    StaticStartGuard guard(options);
    EXPECT_TRUE(guard.update(0.0, roomSeenFrom(0.0, 0.0)));
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
        EXPECT_FALSE(guard.update(i * 0.1, sparse));
    }
}

}  // namespace
