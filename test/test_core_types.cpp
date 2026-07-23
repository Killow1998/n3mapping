#include <gtest/gtest.h>

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

pcl::PointXYZI makePointXYZI(float x, float y, float z, float intensity) {
  pcl::PointXYZI point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = intensity;
  return point;
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

TEST(CoreTypesTest, PointCloudBackedFramesCanBePopulated) {
  core::RawLidarFrame::PointCloud::Ptr cloud(
      new core::RawLidarFrame::PointCloud);
  cloud->push_back(makePointXYZI(1.0f, 2.0f, 3.0f, 4.0f));

  core::RawLidarFrame raw_frame;
  raw_frame.stamp_begin.nsec = 100;
  raw_frame.stamp_end.nsec = 200;
  raw_frame.frame_id = "lidar";
  raw_frame.source_format = "livox_custom";
  raw_frame.points = cloud;
  raw_frame.point_time_offsets_ns = {10U};
  raw_frame.point_lines = {3U};

  ASSERT_NE(raw_frame.points, nullptr);
  ASSERT_EQ(raw_frame.points->size(), 1U);
  EXPECT_EQ(raw_frame.frame_id, "lidar");
  EXPECT_EQ(raw_frame.source_format, "livox_custom");
  EXPECT_EQ(raw_frame.point_time_offsets_ns.front(), 10U);
  EXPECT_EQ(raw_frame.point_lines.front(), 3U);

  core::LioFrame lio_frame;
  lio_frame.stamp.nsec = 1234;
  lio_frame.pose_valid = true;
  lio_frame.covariance_valid = true;
  lio_frame.undistorted_cloud = cloud;
  lio_frame.T_world_lidar.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);

  ASSERT_NE(lio_frame.undistorted_cloud, nullptr);
  EXPECT_EQ(lio_frame.undistorted_cloud->size(), 1U);
  EXPECT_TRUE(lio_frame.pose_valid);
  EXPECT_TRUE(lio_frame.covariance_valid);
  EXPECT_DOUBLE_EQ(lio_frame.T_world_lidar.translation().x(), 1.0);
  EXPECT_DOUBLE_EQ(lio_frame.T_world_lidar.translation().y(), 2.0);
  EXPECT_DOUBLE_EQ(lio_frame.T_world_lidar.translation().z(), 3.0);
}

TEST(CoreTypesTest, BackendOutputCarriesCoreResults) {
  core::LioFrame::PointCloud::Ptr body_cloud(
      new core::LioFrame::PointCloud);
  core::LioFrame::PointCloud::Ptr world_cloud(
      new core::LioFrame::PointCloud);
  body_cloud->push_back(makePointXYZI(0.0f, 0.0f, 0.0f, 1.0f));
  world_cloud->push_back(makePointXYZI(5.0f, 0.0f, 0.0f, 1.0f));

  core::BackendOutput output;
  output.success = true;
  output.accepted_keyframe = true;
  output.relocalization_locked = true;
  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
  output.relocalization_decision = "accepted";
  output.keyframe_id = 42;
  output.relocalization_seed_keyframe_id = 12;
  output.relocalization_support_keyframe_id = 24;
  output.matched_keyframe_id = 24;
  output.cloud_body = body_cloud;
  output.cloud_world = world_cloud;
  output.T_world_lidar.translation() = Eigen::Vector3d(5.0, 6.0, 7.0);

  EXPECT_TRUE(output.success);
  EXPECT_TRUE(output.accepted_keyframe);
  EXPECT_TRUE(output.relocalization_locked);
  EXPECT_EQ(output.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
  EXPECT_EQ(output.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  EXPECT_EQ(output.relocalization_decision, "accepted");
  EXPECT_EQ(output.keyframe_id, 42);
  EXPECT_EQ(output.relocalization_seed_keyframe_id, 12);
  EXPECT_EQ(output.relocalization_support_keyframe_id, 24);
  EXPECT_EQ(output.matched_keyframe_id, 24);
  ASSERT_NE(output.cloud_body, nullptr);
  ASSERT_NE(output.cloud_world, nullptr);
  EXPECT_EQ(output.cloud_body->size(), 1U);
  EXPECT_EQ(output.cloud_world->size(), 1U);
  EXPECT_DOUBLE_EQ(output.T_world_lidar.translation().x(), 5.0);
  EXPECT_DOUBLE_EQ(output.T_world_lidar.translation().y(), 6.0);
  EXPECT_DOUBLE_EQ(output.T_world_lidar.translation().z(), 7.0);
}

TEST(CoreTypesTest, RelocalizationPoseAuthorityIsFailClosed) {
  EXPECT_EQ(static_cast<std::uint8_t>(RelocalizationState::SEARCHING), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(RelocalizationState::REGION_HYPOTHESIS),
            1U);
  EXPECT_EQ(static_cast<std::uint8_t>(
                RelocalizationState::FULL_6DOF_LOCKED),
            2U);
  EXPECT_EQ(
      static_cast<std::uint8_t>(RelocalizationState::DEGRADED_TRACKING), 3U);
  EXPECT_EQ(static_cast<std::uint8_t>(PoseSource::NONE), 0U);
  EXPECT_EQ(static_cast<std::uint8_t>(PoseSource::ODOM_PREDICTED), 1U);
  EXPECT_EQ(
      static_cast<std::uint8_t>(PoseSource::GEOMETRICALLY_CORRECTED), 2U);

  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::SEARCHING, PoseSource::NONE));
  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::REGION_HYPOTHESIS, PoseSource::NONE));

  EXPECT_TRUE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::FULL_6DOF_LOCKED,
      PoseSource::GEOMETRICALLY_CORRECTED));
  EXPECT_TRUE(hasAuthoritativeRelocalizationInitializationPose(
      RelocalizationState::FULL_6DOF_LOCKED,
      PoseSource::GEOMETRICALLY_CORRECTED));

  EXPECT_TRUE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::DEGRADED_TRACKING, PoseSource::ODOM_PREDICTED));
  EXPECT_FALSE(hasAuthoritativeRelocalizationInitializationPose(
      RelocalizationState::DEGRADED_TRACKING, PoseSource::ODOM_PREDICTED));

  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::FULL_6DOF_LOCKED, PoseSource::ODOM_PREDICTED));
  EXPECT_FALSE(hasUsableGlobalRelocalizationPose(
      RelocalizationState::DEGRADED_TRACKING,
      PoseSource::GEOMETRICALLY_CORRECTED));
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

  output.relocalization_state = RelocalizationState::REGION_HYPOTHESIS;
  auto region =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(region.publish_status);
  EXPECT_FALSE(region.publish_global_pose);
  EXPECT_FALSE(region.publish_world_cloud);
  EXPECT_FALSE(region.publish_authoritative_pose);
  EXPECT_FALSE(region.publish_legacy_lock);
  EXPECT_EQ(region.lock_epoch, 0U);

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

  output.relocalization_state = RelocalizationState::DEGRADED_TRACKING;
  output.pose_source = PoseSource::ODOM_PREDICTED;
  auto degraded =
      authority.process(RelocalizationOutputMode::LOCALIZATION, output);
  EXPECT_TRUE(degraded.publish_status);
  EXPECT_TRUE(degraded.publish_global_pose);
  EXPECT_TRUE(degraded.publish_world_cloud);
  EXPECT_FALSE(degraded.publish_authoritative_pose);
  EXPECT_FALSE(degraded.publish_legacy_lock);
  EXPECT_EQ(degraded.lock_epoch, 1U);

  output.relocalization_state = RelocalizationState::FULL_6DOF_LOCKED;
  output.pose_source = PoseSource::GEOMETRICALLY_CORRECTED;
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
