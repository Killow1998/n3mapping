#include <gtest/gtest.h>
#include "n3mapping/relocalization_order.h"

#include <limits>

#include "n3mapping/core/types.h"
#include "n3mapping/odometry_pose_validation.h"
#include "n3mapping/relocalization_output_authority.h"

namespace n3mapping {
namespace test {

TEST(CoreTypesTest, SharedOdometryBoundaryRejectsNonRigidInput) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  EXPECT_FALSE(tryMakeRigidOdometryPose(
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, &pose));
  EXPECT_TRUE(pose.isApprox(Eigen::Isometry3d::Identity()));

  EXPECT_FALSE(tryMakeRigidOdometryPose(
      std::numeric_limits<double>::infinity(), 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0, &pose));
  EXPECT_TRUE(pose.isApprox(Eigen::Isometry3d::Identity()));

  EXPECT_TRUE(tryMakeRigidOdometryPose(
      1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0, &pose));
  EXPECT_TRUE(pose.matrix().allFinite());
  EXPECT_TRUE(pose.linear().isUnitary(1e-12));
}

TEST(CoreTypesTest, RelocalizationScoreOrderingIsStrict) {
  const std::vector<double> values = {
      0.0, 0.75e-9, 1.5e-9, -1.0, 1.0,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity()};
  for (const double a : values) {
    EXPECT_EQ(compareRelocalizationScore(a, a), 0);
    for (const double b : values) {
      EXPECT_EQ(compareRelocalizationScore(a, b), -compareRelocalizationScore(b, a));
      for (const double c : values) {
        if (compareRelocalizationScore(a, b) < 0 && compareRelocalizationScore(b, c) < 0) {
          EXPECT_LT(compareRelocalizationScore(a, c), 0);
        }
        if (compareRelocalizationScore(a, b) == 0 && compareRelocalizationScore(b, c) == 0) {
          EXPECT_EQ(compareRelocalizationScore(a, c), 0);
        }
      }
    }
  }
}

TEST(CoreTypesTest, DefaultValuesAreRosFreeAndStable) {
  core::TimeStamp stamp;
  EXPECT_EQ(stamp.nsec, 0);

  core::ImuSample imu;
  EXPECT_EQ(imu.stamp.nsec, 0);
  EXPECT_TRUE(imu.linear_accel.isZero());
  EXPECT_TRUE(imu.angular_velocity.isZero());
  EXPECT_TRUE(imu.orientation.isApprox(Eigen::Quaterniond::Identity()));
  EXPECT_FALSE(imu.has_orientation);

  core::RawLidarFrame raw_frame;
  EXPECT_EQ(raw_frame.frame_id, "");
  EXPECT_EQ(raw_frame.source_format, "pointcloud2");
  EXPECT_EQ(raw_frame.points, nullptr);
  EXPECT_TRUE(raw_frame.point_time_offsets_ns.empty());
  EXPECT_TRUE(raw_frame.point_lines.empty());

  core::LioFrame lio_frame;
  EXPECT_EQ(lio_frame.stamp.nsec, 0);
  EXPECT_TRUE(lio_frame.source_frame_id.empty());
  EXPECT_TRUE(lio_frame.T_world_lidar.isApprox(Eigen::Isometry3d::Identity()));
  EXPECT_EQ(lio_frame.undistorted_cloud, nullptr);
  EXPECT_TRUE(
      lio_frame.covariance.isApprox(Eigen::Matrix<double, 6, 6>::Identity()));
  EXPECT_FALSE(lio_frame.covariance_valid);
  EXPECT_FALSE(lio_frame.pose_valid);

  core::BackendOutput output;
  EXPECT_FALSE(output.success);
  EXPECT_FALSE(output.accepted_keyframe);
  EXPECT_FALSE(output.relocalization_locked);
  EXPECT_EQ(output.relocalization_state, RelocalizationState::SEARCHING);
  EXPECT_EQ(output.pose_source, PoseSource::NONE);
  EXPECT_EQ(output.relocalization_decision, "not_attempted");
  EXPECT_EQ(output.keyframe_id, -1);
  EXPECT_EQ(output.relocalization_seed_keyframe_id, -1);
  EXPECT_EQ(output.relocalization_support_keyframe_id, -1);
  EXPECT_EQ(output.matched_keyframe_id, -1);
  EXPECT_TRUE(output.T_world_lidar.isApprox(Eigen::Isometry3d::Identity()));
  EXPECT_EQ(output.cloud_body, nullptr);
  EXPECT_EQ(output.cloud_world, nullptr);
}

TEST(CoreTypesTest, RelocalizationPoseAuthorityIsFailClosed) {
  EXPECT_EQ(static_cast<std::uint8_t>(RelocalizationState::SEARCHING), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(RelocalizationState::PROVISIONAL),
            1U);
  EXPECT_EQ(static_cast<std::uint8_t>(
                RelocalizationState::FULL_6DOF_LOCKED),
            2U);
  EXPECT_EQ(static_cast<std::uint8_t>(RelocalizationState::RECENTLY_LOST),
            3U);
  EXPECT_EQ(
      static_cast<std::uint8_t>(RelocalizationState::DEGRADED_TRACKING), 4U);
  EXPECT_EQ(static_cast<std::uint8_t>(RelocalizationState::LOST), 5U);
  EXPECT_EQ(static_cast<std::uint8_t>(PoseSource::NONE), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(PoseSource::ODOM_PREDICTED), 1U);
  EXPECT_EQ(
      static_cast<std::uint8_t>(PoseSource::GEOMETRICALLY_CORRECTED), 2U);

  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::SEARCHING, PoseSource::NONE));
  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::PROVISIONAL, PoseSource::NONE));

  EXPECT_TRUE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::FULL_6DOF_LOCKED,
      PoseSource::GEOMETRICALLY_CORRECTED));
  EXPECT_TRUE(hasAuthoritativeRelocalizationInitializationPose(
      RelocalizationState::FULL_6DOF_LOCKED,
      PoseSource::GEOMETRICALLY_CORRECTED));

  EXPECT_TRUE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::RECENTLY_LOST, PoseSource::ODOM_PREDICTED));
  EXPECT_FALSE(hasAuthoritativeRelocalizationInitializationPose(
      RelocalizationState::RECENTLY_LOST, PoseSource::ODOM_PREDICTED));

  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::FULL_6DOF_LOCKED, PoseSource::ODOM_PREDICTED));
  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::DEGRADED_TRACKING,
      PoseSource::ODOM_PREDICTED));
  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::LOST, PoseSource::NONE));
  EXPECT_FALSE(hasAuthoritativeRelocalizationInitializationPose(
      RelocalizationState::SEARCHING,
      PoseSource::GEOMETRICALLY_CORRECTED));
}

TEST(CoreTypesTest, PoseAuthorityRejectsNonFiniteAndNonRigidTransforms) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  EXPECT_TRUE(isFiniteRigidPose(pose));

  pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(isFiniteRigidPose(pose));

  pose = Eigen::Isometry3d::Identity();
  pose.linear()(0, 0) = 2.0;
  EXPECT_FALSE(isFiniteRigidPose(pose));
}

TEST(CoreTypesTest, StatefulOutputAuthorityUsesLocalizationStateEdges) {
  RelocalizationOutputAuthority authority;
  core::BackendOutput output;

  auto searching =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(searching.publish_status);
  EXPECT_FALSE(searching.publish_global_pose);
  EXPECT_FALSE(searching.publish_world_cloud);
  EXPECT_FALSE(searching.publish_authoritative_pose);
  EXPECT_FALSE(searching.publish_legacy_lock);
  EXPECT_EQ(searching.lock_epoch, 0U);

  output.relocalization_state = RelocalizationState::PROVISIONAL;
  auto provisional =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(provisional.publish_status);
  EXPECT_FALSE(provisional.publish_global_pose);
  EXPECT_FALSE(provisional.publish_world_cloud);
  EXPECT_FALSE(provisional.publish_authoritative_pose);
  EXPECT_FALSE(provisional.publish_legacy_lock);
  EXPECT_EQ(provisional.lock_epoch, 0U);

  output.success = true;
  output.relocalization_locked = true;
  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
  auto first_full =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(first_full.publish_status);
  EXPECT_TRUE(first_full.publish_global_pose);
  EXPECT_TRUE(first_full.publish_world_cloud);
  EXPECT_TRUE(first_full.publish_authoritative_pose);
  EXPECT_TRUE(first_full.publish_legacy_lock);
  EXPECT_EQ(first_full.lock_epoch, 1U);

  output.relocalization_locked = false;
  auto continued_full =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(continued_full.publish_status);
  EXPECT_TRUE(continued_full.publish_global_pose);
  EXPECT_TRUE(continued_full.publish_world_cloud);
  EXPECT_FALSE(continued_full.publish_authoritative_pose);
  EXPECT_FALSE(continued_full.publish_legacy_lock);
  EXPECT_EQ(continued_full.lock_epoch, 1U);

  output.relocalization_state = RelocalizationState::RECENTLY_LOST;
  output.pose_source = PoseSource::ODOM_PREDICTED;
  auto recently_lost =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(recently_lost.publish_status);
  EXPECT_TRUE(recently_lost.publish_global_pose);
  EXPECT_TRUE(recently_lost.publish_world_cloud);
  EXPECT_FALSE(recently_lost.publish_authoritative_pose);
  EXPECT_FALSE(recently_lost.publish_legacy_lock);
  EXPECT_EQ(recently_lost.lock_epoch, 1U);

  output.success = false;
  output.relocalization_state = RelocalizationState::LOST;
  output.pose_source = PoseSource::NONE;
  auto lost = authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(lost.publish_status);
  EXPECT_FALSE(lost.publish_global_pose);
  EXPECT_FALSE(lost.publish_world_cloud);
  EXPECT_FALSE(lost.publish_authoritative_pose);
  EXPECT_FALSE(lost.publish_legacy_lock);
  EXPECT_EQ(lost.lock_epoch, 1U);

  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
  output.success = true;
  ASSERT_FALSE(output.relocalization_locked);
  auto recovered_full =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(recovered_full.publish_authoritative_pose);
  EXPECT_TRUE(recovered_full.publish_legacy_lock);
  EXPECT_EQ(recovered_full.lock_epoch, 2U);
  EXPECT_EQ(authority.lockEpoch(), 2U);
}

TEST(CoreTypesTest, StatefulOutputAuthoritySuppressesInvalidUsablePose) {
  RelocalizationOutputAuthority authority;
  core::BackendOutput output;
  output.success = true;
  output.relocalization_locked = true;
  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
  output.relocalization_decision = "accepted";
  output.T_world_lidar.linear()(0, 0) = 2.0;

  auto invalid_full =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(invalid_full.publish_status);
  EXPECT_FALSE(invalid_full.publish_global_pose);
  EXPECT_FALSE(invalid_full.publish_world_cloud);
  EXPECT_FALSE(invalid_full.publish_authoritative_pose);
  EXPECT_FALSE(invalid_full.publish_legacy_lock);
  EXPECT_TRUE(invalid_full.inconsistent_lock_event);
  EXPECT_TRUE(invalid_full.invalid_usable_pose_suppressed);
  EXPECT_EQ(invalid_full.lock_epoch, 0U);
  EXPECT_FALSE(output.success);
  EXPECT_FALSE(output.relocalization_locked);
  EXPECT_EQ(output.relocalization_state, RelocalizationState::SEARCHING);
  EXPECT_EQ(output.pose_source, PoseSource::NONE);
  EXPECT_EQ(output.relocalization_decision, "invalid_nonrigid_pose");
}

TEST(CoreTypesTest, StatefulOutputAuthorityPreservesMapExtensionLegacyEvent) {
  RelocalizationOutputAuthority authority;
  core::BackendOutput output;
  output.success = true;
  output.relocalization_locked = true;
  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;

  auto extension =
      authority.process(RelocalizationOutputMode::MAP_EXTENSION, output);
  EXPECT_FALSE(extension.publish_status);
  EXPECT_TRUE(extension.publish_global_pose);
  EXPECT_TRUE(extension.publish_world_cloud);
  EXPECT_FALSE(extension.publish_authoritative_pose);
  EXPECT_TRUE(extension.publish_legacy_lock);
  EXPECT_FALSE(extension.invalid_usable_pose_suppressed);
  EXPECT_EQ(extension.lock_epoch, 1U);

  output.T_world_lidar.linear()(0, 0) = 2.0;
  auto invalid_extension =
      authority.process(RelocalizationOutputMode::MAP_EXTENSION, output);
  EXPECT_FALSE(invalid_extension.publish_status);
  EXPECT_TRUE(invalid_extension.publish_global_pose);
  EXPECT_TRUE(invalid_extension.publish_world_cloud);
  EXPECT_FALSE(invalid_extension.publish_authoritative_pose);
  EXPECT_FALSE(invalid_extension.publish_legacy_lock);
  EXPECT_FALSE(invalid_extension.invalid_usable_pose_suppressed);
  EXPECT_EQ(invalid_extension.lock_epoch, 1U);
  EXPECT_EQ(output.relocalization_state,
            RelocalizationState::FULL_6DOF_LOCKED);
  EXPECT_EQ(output.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
}

} // namespace test
} // namespace n3mapping
