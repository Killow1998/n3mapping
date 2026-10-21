#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <thread>

#include "n3mapping/core/realtime_odometry.h"
#include "../src/tracking_pose_correction.h"

namespace n3mapping::core {
namespace {

Eigen::Isometry3d translated(double x) {
  auto pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = x;
  return pose;
}

TEST(RealtimeProjectionContract, TrackingCorrectionIsIndependentOfOdometryOrigin) {
  Eigen::Isometry3d odom = translated(20), current = translated(3);
  const Eigen::Isometry3d prediction = current * odom;
  auto endpoint = prediction;
  endpoint.linear() = Eigen::AngleAxisd(.05, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Isometry3d actual = trackingPoseCorrection(current, odom, endpoint, .9, false) * odom;
  EXPECT_LT((actual.translation() - prediction.translation()).norm(), 1e-12);
  auto origin = translated(500);
  origin.translation().y() = -200;
  origin.linear() = Eigen::AngleAxisd(.6, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Isometry3d moved_odom = origin * odom;
  const Eigen::Isometry3d changed = trackingPoseCorrection(
      current * origin.inverse(), moved_odom, endpoint, .9, false) * moved_odom;
  EXPECT_LT((actual.matrix() - changed.matrix()).norm(), 1e-10);
}

TEST(RealtimeProjectionContract, OdomThenDelayedCorrectionThenCloudSharesGeometry) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto odom = live.project(100, 100, "odom", "body", raw);
  ASSERT_TRUE(odom);
  ASSERT_TRUE(live.updateCorrection(99, "odom", "body", raw, translated(22.0), true));
  const auto cloud = live.projectScan(100, 101, "odom", "body", raw);
  ASSERT_TRUE(cloud);
  const Eigen::Vector3d body_point(1.0, 2.0, 3.0);
  EXPECT_LT((cloud->pose * body_point - odom->pose * body_point).norm(), 1e-12);
  EXPECT_EQ(cloud->stamp_nsec, 100);
  EXPECT_EQ(cloud->correction_stamp_nsec, 90);
  EXPECT_TRUE(live.isCurrent(*cloud));
  const auto next = live.project(101, 101, "odom", "body", raw);
  ASSERT_TRUE(next);
  EXPECT_DOUBLE_EQ(next->pose.translation().x(), 22.0);
  EXPECT_EQ(next->correction_stamp_nsec, 99);
}

TEST(RealtimeProjectionContract, CloudFirstFreezesPoseAndReportedCorrectionStamp) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto cloud = live.projectScan(100, 100, "odom", "body", raw);
  ASSERT_TRUE(cloud);
  ASSERT_TRUE(live.updateCorrection(99, "odom", "body", raw, translated(22.0), true));
  RealtimeOdometryDiagnostic diagnostic;
  const auto odom = live.project(100, 101, "odom", "body", raw, true, &diagnostic);
  ASSERT_TRUE(odom);
  EXPECT_TRUE(odom->pose.matrix().isApprox(cloud->pose.matrix(), 1e-12));
  EXPECT_EQ(odom->correction_stamp_nsec, 90);
  EXPECT_EQ(diagnostic.correction_stamp_nsec, 90);
  EXPECT_EQ(diagnostic.reason, RealtimeOdometryReason::Ready);
  EXPECT_FALSE(live.project(100, 101, "odom", "body", raw));
}

TEST(RealtimeProjectionContract, RetainedOldBasisSurvivesButUnreservedPreCorrectionScanRejects) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto first = live.project(100, 100, "odom", "body", raw);
  ASSERT_TRUE(first);
  ASSERT_TRUE(live.updateCorrection(110, "odom", "body", raw, translated(22.0), true));
  ASSERT_TRUE(live.project(120, 120, "odom", "body", raw));
  const auto delayed = live.projectScan(100, 120, "odom", "body", raw);
  ASSERT_TRUE(delayed);
  EXPECT_TRUE(delayed->pose.matrix().isApprox(first->pose.matrix()));
  EXPECT_FALSE(live.projectScan(105, 120, "odom", "body", raw));
}

TEST(RealtimeProjectionContract, EvictionNeverRecreatesAnObservationWithNewCorrection) {
  RealtimeOdometry live(2);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto first = live.project(100, 100, "odom", "body", raw);
  ASSERT_TRUE(first);
  ASSERT_TRUE(live.project(101, 101, "odom", "body", raw));
  ASSERT_TRUE(live.project(102, 102, "odom", "body", raw));
  ASSERT_TRUE(live.updateCorrection(99, "odom", "body", raw, translated(22.0), true));
  // An already-reserved single in-flight cloud keeps its immutable value;
  // eviction forbids a new lookup, not publication using the copied old basis.
  EXPECT_TRUE(live.isCurrent(*first));
  EXPECT_FALSE(live.projectScan(100, 103, "odom", "body", raw));
  const auto retained = live.projectScan(101, 103, "odom", "body", raw);
  ASSERT_TRUE(retained);
  EXPECT_DOUBLE_EQ(retained->pose.translation().x(), 12.0);
  live.setSynchronizationRetention(1);
  EXPECT_FALSE(live.projectScan(101, 103, "odom", "body", raw));
  EXPECT_TRUE(live.projectScan(102, 103, "odom", "body", raw));
}

TEST(RealtimeProjectionContract, EvictedCloudFirstOdomIsDroppedNotReprojected) {
  RealtimeOdometry live(1);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  ASSERT_TRUE(live.projectScan(100, 100, "odom", "body", raw));
  ASSERT_TRUE(live.projectScan(101, 101, "odom", "body", raw));
  ASSERT_TRUE(live.updateCorrection(99, "odom", "body", raw, translated(22.0), true));
  RealtimeOdometryDiagnostic diagnostic;
  EXPECT_FALSE(live.project(100, 102, "odom", "body", raw, true, &diagnostic));
  EXPECT_EQ(diagnostic.reason, RealtimeOdometryReason::ProjectionExpired);
  const auto odom = live.project(101, 102, "odom", "body", raw);
  ASSERT_TRUE(odom);
  EXPECT_DOUBLE_EQ(odom->pose.translation().x(), 12.0);
}

TEST(RealtimeProjectionContract, SameStampMustHaveSameRigidPoseAndFrames) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto reserved = live.projectScan(100, 100, "odom", "body", raw);
  ASSERT_TRUE(reserved);
  EXPECT_FALSE(live.projectScan(100, 101, "other", "body", raw));
  EXPECT_FALSE(live.projectScan(100, 101, "odom", "other", raw));
  EXPECT_FALSE(live.projectScan(100, 101, "odom", "body", translated(3.0)));
  RealtimeOdometryDiagnostic diagnostic;
  EXPECT_FALSE(live.project(100, 101, "odom", "body", translated(3.0), true, &diagnostic));
  EXPECT_EQ(diagnostic.reason, RealtimeOdometryReason::SourceObservationMismatch);
  EXPECT_FALSE(live.isCurrent(*reserved));
}

TEST(RealtimeProjectionContract, InconsistentDuplicateLiveInputCancelsFrozenBasis) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto odom = live.project(100, 100, "odom", "body", raw);
  ASSERT_TRUE(odom);
  RealtimeOdometryDiagnostic diagnostic;
  EXPECT_FALSE(live.project(100, 101, "odom", "body", translated(3.0), true, &diagnostic));
  EXPECT_EQ(diagnostic.reason, RealtimeOdometryReason::SourceObservationMismatch);
  EXPECT_FALSE(live.isCurrent(*odom));
  EXPECT_FALSE(live.projectScan(100, 101, "odom", "body", raw));
}

TEST(RealtimeProjectionContract, CorrectionSourceChangeCancelsOldInFlightBasis) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto odom = live.project(100, 100, "odom", "body", raw);
  ASSERT_TRUE(odom);
  ASSERT_TRUE(live.updateCorrection(110, "other", "body", raw, translated(22.0), true));
  EXPECT_FALSE(live.isCurrent(*odom));
  EXPECT_FALSE(live.projectScan(100, 120, "odom", "body", raw));
  const auto next = live.project(120, 120, "other", "body", raw);
  ASSERT_TRUE(next);
  EXPECT_EQ(next->correction_stamp_nsec, 110);
}

TEST(RealtimeProjectionContract, RejectedObservationDoesNotCancelFrozenBasis) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  const auto odom = live.project(100, 100, "odom", "body", raw);
  ASSERT_TRUE(odom);
  EXPECT_FALSE(live.updateCorrection(99, "odom", "body", raw, translated(22.0), false));
  EXPECT_TRUE(live.isCurrent(*odom));
  const auto cloud = live.projectScan(100, 110, "odom", "body", raw);
  ASSERT_TRUE(cloud);
  EXPECT_TRUE(cloud->pose.matrix().isApprox(odom->pose.matrix()));
  EXPECT_EQ(cloud->correction_stamp_nsec, 90);
}

TEST(RealtimeProjectionContract, InvalidationsCancelBothStreamsAndInFlightCloudWork) {
  const auto raw = translated(2.0);
  for (int failure = 0; failure < 6; ++failure) {
    SCOPED_TRACE(failure);
    RealtimeOdometry live(4);
    ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
    const auto odom = live.project(100, 100, "odom", "body", raw);
    ASSERT_TRUE(odom);
    const auto cloud = live.projectScan(105, 105, "odom", "body", raw);
    ASSERT_TRUE(cloud);
    if (failure == 0) EXPECT_FALSE(live.project(99, 110, "odom", "body", raw));
    if (failure == 1) EXPECT_FALSE(live.project(120, 110, "odom", "body", raw));
    if (failure == 2) EXPECT_FALSE(live.project(110, 110, "other", "body", raw));
    if (failure == 3) EXPECT_FALSE(live.project(110, 110, "odom", "other", raw));
    if (failure == 4) EXPECT_FALSE(live.project(110, 110, "odom", "body", raw, false));
    if (failure == 5) live.invalidateForOdometryDivergence(110);
    EXPECT_FALSE(live.isCurrent(*odom));
    EXPECT_FALSE(live.isCurrent(*cloud));
    EXPECT_FALSE(live.projectScan(100, 120, "odom", "body", raw));
    EXPECT_FALSE(live.projectScan(105, 120, "odom", "body", raw));
    if (failure != 5) {
      ASSERT_TRUE(live.updateCorrection(115, "odom", "body", raw, translated(22.0), true));
      EXPECT_FALSE(live.isCurrent(*cloud));
      EXPECT_FALSE(live.projectScan(105, 120, "odom", "body", raw));
      EXPECT_TRUE(live.project(120, 120, "odom", "body", raw));
    }
  }
}

TEST(RealtimeProjectionContract, StalledCloudWorkDoesNotHoldProjectionOwner) {
  RealtimeOdometry live(4);
  const auto raw = translated(2.0);
  ASSERT_TRUE(live.updateCorrection(90, "odom", "body", raw, translated(12.0), true));
  std::promise<void> cloud_started;
  std::promise<void> release_cloud;
  auto release = release_cloud.get_future();
  auto cloud = std::async(std::launch::async, [&] {
    const auto projection = live.projectScan(100, 100, "odom", "body", raw);
    cloud_started.set_value();
    release.wait();  // Stand-in for an arbitrarily long conversion/publication.
    return projection;
  });
  cloud_started.get_future().wait();
  auto odom = std::async(std::launch::async, [&] {
    for (int64_t stamp = 101; stamp < 1001; ++stamp) {
      if (!live.project(stamp, stamp, "odom", "body", raw)) return false;
    }
    return true;
  });
  const auto completion = odom.wait_for(std::chrono::seconds(2));
  release_cloud.set_value();
  EXPECT_EQ(completion, std::future_status::ready);
  EXPECT_TRUE(odom.get());
  EXPECT_TRUE(cloud.get());
}

}  // namespace
}  // namespace n3mapping::core
