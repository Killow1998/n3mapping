#include <limits>
#include <gtest/gtest.h>

#include "n3mapping/core/latest_frame_slot.h"
#include "n3mapping/core/realtime_odometry.h"
#include "n3mapping/core/types.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping::test {
namespace {

core::LioFrame makeFrame(int64_t stamp_nsec, const Eigen::Isometry3d& pose) {
    core::LioFrame frame;
    frame.stamp.nsec = stamp_nsec;
    frame.T_world_lidar = pose;
    frame.undistorted_cloud = pcl::make_shared<core::LioFrame::PointCloud>();
    frame.pose_valid = true;
    return frame;
}

}  // namespace

TEST(LatestFrameSlot, CoalescesBacklogWithoutRestampingOrMixingPoseAndCloud)
{
    core::LatestFrameSlot<core::LioFrame> pending;
    EXPECT_FALSE(pending.take().has_value());
    core::LioFrame newest;
    for (int64_t index = 1; index <= 100; ++index) {
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = static_cast<double>(index);
        auto frame = makeFrame(index * 100000000, pose);
        if (index == 100) {
            newest = frame;
        }
        pending.submit(frame.stamp.nsec, std::move(frame));
    }
    auto older = makeFrame(9900000000, Eigen::Isometry3d::Identity());
    pending.submit(older.stamp.nsec, std::move(older));
    const auto selected = pending.take();
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->stamp.nsec, newest.stamp.nsec);
    EXPECT_EQ(selected->T_world_lidar.translation().x(), 100.0);
    EXPECT_EQ(selected->undistorted_cloud, newest.undistorted_cloud);
    EXPECT_FALSE(pending.take().has_value());

    // A new burst starts afresh, including after a source-clock reset.
    auto restarted = makeFrame(100000000, Eigen::Isometry3d::Identity());
    pending.submit(restarted.stamp.nsec, std::move(restarted));
    ASSERT_TRUE(pending.take().has_value());
}

TEST(RealtimeOdometry, ComposesLatestRawPoseWithPairedMapCorrection)
{
    core::RealtimeOdometry live;
    Eigen::Isometry3d raw = Eigen::Isometry3d::Identity();
    raw.translation().x() = 2.0;
    Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
    correction.linear() = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    correction.translation() << 5.0, -3.0, 1.0;
    EXPECT_FALSE(live.project(900000000, 900000000, "odom", "body", raw));
    live.updateCorrection(1000000000, "odom", "body", raw, correction * raw, true);
    raw.translation().x() = 3.0;
    const auto projected = live.project(1100000000, 1110000000, "odom", "body", raw);
    ASSERT_TRUE(projected);
    EXPECT_EQ(projected->stamp_nsec, 1100000000);
    EXPECT_EQ(projected->correction_stamp_nsec, 1000000000);
    EXPECT_TRUE(projected->pose.matrix().isApprox((correction * raw).matrix()));
    EXPECT_FALSE(live.project(1100000000, 1110000000, "odom", "body", raw));
}

TEST(RealtimeOdometry, RetainsTrustedAnchorForOneMinuteWithoutRefreshingItsStamp)
{
    core::RealtimeOdometry live;
    const Eigen::Isometry3d raw = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d map_pose = raw;
    map_pose.translation() << 4.0, -2.0, 0.3;
    core::RealtimeOdometryDiagnostic quality;
    EXPECT_FALSE(live.project(900000000, 910000000, "odom", "body", raw, true, &quality));
    live.updateCorrection(1000000000, "odom", "body", raw, map_pose, true);
    for (int second = 2; second <= 62; ++second) {
        const int64_t stamp = int64_t(second) * 1000000000;
        const auto projected = live.project(stamp, stamp + 12000000, "odom", "body", raw, true, &quality);
        ASSERT_TRUE(projected);
        EXPECT_EQ(projected->stamp_nsec, stamp);
        EXPECT_EQ(projected->correction_stamp_nsec, 1000000000);
        EXPECT_TRUE(projected->pose.matrix().isApprox(map_pose.matrix()));
        EXPECT_EQ(quality.reason, core::RealtimeOdometryReason::Ready);
    }
}

TEST(RealtimeOdometry, ScanProjectionDoesNotRegressTheLiveInputStream)
{
    core::RealtimeOdometry live;
    const Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
    live.updateCorrection(1000000000, "odom", "body", origin, origin, true);
    Eigen::Isometry3d latest = origin;
    latest.translation().x() = 2.0;
    ASSERT_TRUE(live.project(1400000000, 1400000000, "odom", "body", latest));
    Eigen::Isometry3d scan_pose = origin;
    scan_pose.translation().x() = 1.0;
    const auto scan = live.projectScan(1200000000, 1400000000, "odom", "body", scan_pose);
    ASSERT_TRUE(scan);
    EXPECT_DOUBLE_EQ(scan->pose.translation().x(), 1.0);
    EXPECT_FALSE(live.projectScan(1200000000, 1400000000, "wrong", "body", scan_pose));
    EXPECT_TRUE(live.project(1500000000, 1500000000, "odom", "body", latest));
}

TEST(RealtimeOdometry, NewerCorrectionSurvivesNormallyQueuedOlderInput)
{
    core::RealtimeOdometry live;
    const auto pose = Eigen::Isometry3d::Identity();
    live.updateCorrection(1000000000, "odom", "body", pose, pose, true);
    ASSERT_TRUE(live.project(1100000000, 1100000000, "odom", "body", pose));
    // The backend queue wins the race; the live queue still contains 1.2 s.
    live.updateCorrection(1300000000, "odom", "body", pose, pose, true);
    core::RealtimeOdometryDiagnostic diagnostic;
    EXPECT_FALSE(live.project(1200000000, 1350000000, "odom", "body", pose, true, &diagnostic));
    EXPECT_EQ(diagnostic.reason, core::RealtimeOdometryReason::InputBeforeCorrection);
    EXPECT_EQ(diagnostic.input_stamp_nsec, 1200000000);
    EXPECT_EQ(diagnostic.correction_stamp_nsec, 1300000000);
    const auto fresh = live.project(1400000000, 1400000000, "odom", "body", pose, true, &diagnostic);
    ASSERT_TRUE(fresh);
    EXPECT_EQ(diagnostic.reason, core::RealtimeOdometryReason::Ready);
    EXPECT_EQ(fresh->correction_stamp_nsec, 1300000000);
    EXPECT_EQ(fresh->stamp_nsec, 1400000000);
}

TEST(RealtimeOdometry, RejectedObservationKeepsTheLastTrustedCorrection)
{
    const auto pose = Eigen::Isometry3d::Identity();
    core::RealtimeOdometry live;
    live.updateCorrection(1000000000, "odom", "body", pose, pose, true);
    EXPECT_FALSE(live.updateCorrection(1100000000, "odom", "body", pose, pose, false));
    auto projected = live.project(1200000000, 1200000000, "odom", "body", pose);
    ASSERT_TRUE(projected);
    EXPECT_EQ(projected->correction_stamp_nsec, 1000000000);
    EXPECT_FALSE(live.updateCorrection(1000000000, "odom", "body", pose, pose, true));
    // An asynchronous observation may finish behind the current raw input.
    ASSERT_TRUE(live.updateCorrection(1150000000, "odom", "body", pose, pose, true));
    projected = live.project(1300000000, 1300000000, "odom", "body", pose);
    ASSERT_TRUE(projected);
    EXPECT_EQ(projected->correction_stamp_nsec, 1150000000);
}

TEST(RealtimeOdometry, LatchedRawDivergenceStopsPoseAndScanProjection)
{
    const auto pose = Eigen::Isometry3d::Identity();
    core::RealtimeOdometry live;
    ASSERT_TRUE(live.updateCorrection(
        1000000000, "odom", "body", pose, pose, true));
    ASSERT_TRUE(live.project(1100000000, 1100000000,
                             "odom", "body", pose));

    live.invalidateForOdometryDivergence(1200000000);
    core::RealtimeOdometryDiagnostic diagnostic;
    EXPECT_FALSE(live.project(1300000000, 1300000000,
                              "odom", "body", pose, true, &diagnostic));
    EXPECT_EQ(diagnostic.reason, core::RealtimeOdometryReason::OdometryDiverged);
    EXPECT_FALSE(live.projectScan(1300000000, 1300000000,
                                  "odom", "body", pose));
    // A later correction cannot silently re-enable the latched raw input.
    EXPECT_FALSE(live.updateCorrection(
        1400000000, "odom", "body", pose, pose, true));
    EXPECT_FALSE(live.project(1500000000, 1500000000,
                              "odom", "body", pose, true, &diagnostic));
    EXPECT_EQ(diagnostic.reason, core::RealtimeOdometryReason::OdometryDiverged);
}

TEST(RealtimeOdometry, ClockAndFrameDiscontinuitiesCannotReviveOldCorrection)
{
    const auto pose = Eigen::Isometry3d::Identity();
    for (int failure = 0; failure < 5; ++failure) {
        SCOPED_TRACE(failure);
        core::RealtimeOdometry live;
        live.updateCorrection(1000000000, "odom", "body", pose, pose, true);
        ASSERT_TRUE(live.project(1100000000, 1100000000, "odom", "body", pose));
        if (failure == 0) {
            EXPECT_FALSE(live.project(1000000000, 1200000000, "odom", "body", pose));
        }
        if (failure == 1) {
            EXPECT_FALSE(live.project(1300000000, 1200000000, "odom", "body", pose));
        }
        if (failure == 2) {
            EXPECT_FALSE(live.project(1200000000, 1200000000, "other", "body", pose));
        }
        if (failure == 3) {
            EXPECT_FALSE(live.project(1200000000, 1200000000, "odom", "other", pose));
        }
        if (failure == 4) {
            EXPECT_FALSE(live.project(1200000000, 1200000000, "odom", "body", pose, false));
        }
        live.updateCorrection(1000000000, "odom", "body", pose, pose, true);
        EXPECT_FALSE(live.project(1400000000, 1400000000, "odom", "body", pose));
        live.updateCorrection(1500000000, "odom", "body", pose, pose, true);
        EXPECT_TRUE(live.project(1600000000, 1600000000, "odom", "body", pose));
    }
}

TEST(RealtimeOdometry, RejectsOverflowInCorrectionAndProjectedPose)
{
    const auto identity = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d positive = identity;
    positive.translation().x() = std::numeric_limits<double>::max();
    Eigen::Isometry3d negative = identity;
    negative.translation().x() = -std::numeric_limits<double>::max();
    core::RealtimeOdometry invalid_correction;
    invalid_correction.updateCorrection(1000000000, "odom", "body", negative, positive, true);
    EXPECT_FALSE(invalid_correction.project(1100000000, 1100000000, "odom", "body", identity));
    core::RealtimeOdometry invalid_projection;
    invalid_projection.updateCorrection(1000000000, "odom", "body", identity, positive, true);
    EXPECT_FALSE(invalid_projection.project(1100000000, 1100000000, "odom", "body", positive));
    EXPECT_FALSE(invalid_projection.project(1200000000, 1200000000, "odom", "body", identity));
}

}  // namespace n3mapping::test
