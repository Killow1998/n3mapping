#include "n3mapping/submap_builder.h"
#include "n3mapping/submap_graph_factor.h"
#include "n3mapping/submap_graph_projection.h"

#include <gtsam/slam/BetweenFactor.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <Eigen/Cholesky>

namespace n3mapping {
namespace {

Keyframe::Ptr makeKeyframe(int64_t id, MapSessionId session_id, double x,
                           std::size_t point_count = 2) {
  auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
  for (std::size_t i = 0; i < point_count; ++i) {
    pcl::PointXYZI point;
    point.x = static_cast<float>(i);
    point.y = static_cast<float>(id);
    point.z = 0.0f;
    point.intensity = static_cast<float>(10 + i);
    cloud->push_back(point);
  }
  Eigen::Isometry3d session_pose = Eigen::Isometry3d::Identity();
  session_pose.translation().x() = x;
  Eigen::Isometry3d map_pose = session_pose;
  map_pose.translation().y() = static_cast<double>(session_id) * 10.0;
  return Keyframe::create(id, static_cast<double>(id), session_pose, map_pose,
                          cloud, session_id);
}

std::vector<MapSessionInfo> makeSessions() {
  MapSessionInfo session0;
  session0.id = 0;
  MapSessionInfo session1;
  session1.id = 1;
  return {session0, session1};
}

Eigen::Isometry3d makePose(double x, double y, double yaw) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(x, y, 0.0);
  pose.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
  return pose;
}

Eigen::Isometry3d makePose6(const Eigen::Vector3d& translation,
                            double roll, double pitch, double yaw) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = translation;
  pose.linear() =
      (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  return pose;
}

gtsam::Pose3 toGtsam(const Eigen::Isometry3d& pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.rotation()),
                      gtsam::Point3(pose.translation()));
}

double normalizeAngle(double angle) {
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

TEST(SubmapBuilderTest, DisabledBuilderDoesNotChangeState) {
  SubmapBuilder builder(SubmapBuilderOptions{});
  EXPECT_FALSE(builder.appendKeyframe(makeKeyframe(0, 0, 0.0)));
  EXPECT_TRUE(builder.getSubmaps().empty());
}

TEST(SubmapBuilderTest, RejectsEmptyAndNonFiniteClouds) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.cloud_max_bytes = 4 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  auto empty = makeKeyframe(0, 0, 0.0);
  empty->cloud->clear();
  EXPECT_FALSE(builder.appendKeyframe(empty));

  auto non_finite = makeKeyframe(1, 0, 0.0);
  non_finite->cloud->front().x =
      std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(builder.appendKeyframe(non_finite));
  EXPECT_TRUE(builder.getSubmaps().empty());
}

TEST(SubmapBuilderTest, GroupsDeterministicallyAndKeepsOneBoundedCloud) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 3 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(0, 0, 5.0)));
  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(1, 0, 7.0)));
  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(2, 0, 9.0)));

  const auto submaps = builder.getSubmaps();
  ASSERT_EQ(submaps.size(), 2u);
  EXPECT_EQ(submaps[0].keyframe_ids, (std::vector<int64_t>{0, 1}));
  EXPECT_TRUE(submaps[0].closed);
  EXPECT_EQ(submaps[0].cloud_point_count, 3u);
  EXPECT_TRUE(submaps[0].cloud_truncated);
  EXPECT_EQ(submaps[0].materializedCloudBytes(),
            3u * sizeof(pcl::PointXYZI));
  EXPECT_EQ(submaps[0].registration_cloud,
            submaps[0].visualization_cloud);
  ASSERT_EQ(submaps[0].registration_cloud->size(), 3u);
  EXPECT_FLOAT_EQ(submaps[0].registration_cloud->points[2].x, 2.0f);
  EXPECT_EQ(submaps[0].descriptor_keyframe_id, 0);
  EXPECT_EQ(submaps[0].content_revision, 3u);

  EXPECT_EQ(submaps[1].keyframe_ids, (std::vector<int64_t>{2}));
  EXPECT_FALSE(submaps[1].closed);
  EXPECT_EQ(submaps[1].content_revision, 1u);
}

TEST(SubmapBuilderTest, SessionBoundaryClosesWithoutCrossSessionMembership) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 10;
  options.cloud_max_bytes = 32 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(0, 0, 0.0)));
  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(1, 1, 0.0)));
  const auto before_more_input = builder.getSubmaps();
  ASSERT_EQ(before_more_input.size(), 2u);
  EXPECT_TRUE(before_more_input[0].closed);
  EXPECT_EQ(before_more_input[0].session_id, 0u);
  EXPECT_EQ(before_more_input[1].session_id, 1u);

  ASSERT_TRUE(builder.appendKeyframe(makeKeyframe(2, 1, 1.0)));
  const auto after_more_input = builder.getSubmaps();
  EXPECT_EQ(after_more_input[0].keyframe_ids,
            before_more_input[0].keyframe_ids);
  EXPECT_EQ(after_more_input[0].content_revision,
            before_more_input[0].content_revision);
  EXPECT_EQ(after_more_input[1].keyframe_ids,
            (std::vector<int64_t>{1, 2}));
}

TEST(SubmapBuilderTest, LoadMaterializesAndFailureIsTransactional) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 4 * sizeof(pcl::PointXYZI);
  SubmapBuilder source(options);
  const std::vector<Keyframe::Ptr> keyframes = {
      makeKeyframe(0, 0, 0.0), makeKeyframe(1, 0, 1.0)};
  ASSERT_TRUE(source.appendKeyframe(keyframes[0]));
  ASSERT_TRUE(source.appendKeyframe(keyframes[1]));

  const auto serialized = source.getSubmaps();
  SubmapBuilder loaded(options);
  ASSERT_TRUE(loaded.loadSubmaps(serialized, keyframes, makeSessions()));
  const auto round_trip = loaded.getSubmaps();
  ASSERT_EQ(round_trip.size(), 1u);
  EXPECT_EQ(round_trip[0].keyframe_ids,
            serialized[0].keyframe_ids);
  EXPECT_EQ(round_trip[0].cloud_point_count,
            serialized[0].cloud_point_count);
  EXPECT_EQ(round_trip[0].registration_cloud,
            round_trip[0].visualization_cloud);

  auto invalid = serialized;
  invalid[0].id = kInvalidSubmapId - 1u;
  EXPECT_FALSE(loaded.loadSubmaps(invalid, keyframes, makeSessions()));
  const auto after_failed_load = loaded.getSubmaps();
  ASSERT_EQ(after_failed_load.size(), 1u);
  EXPECT_EQ(after_failed_load[0].id, serialized[0].id);
  EXPECT_EQ(after_failed_load[0].cloud_point_count,
            serialized[0].cloud_point_count);
}

TEST(SubmapBuilderTest,
     RefreshesOriginFromAnchorAndMeasuresRigidProjectionResidual) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 3;
  options.cloud_max_bytes = 16 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  auto anchor = makeKeyframe(0, 0, 5.0);
  auto member = makeKeyframe(1, 0, 7.0);
  auto unassigned = makeKeyframe(2, 0, 9.0);
  ASSERT_TRUE(builder.appendKeyframe(anchor));
  ASSERT_TRUE(builder.appendKeyframe(member));
  const std::uint64_t content_revision =
      builder.getSubmaps().front().content_revision;

  anchor->pose_optimized = Eigen::Isometry3d::Identity();
  anchor->pose_optimized.translation() = Eigen::Vector3d(50.0, 10.0, 0.0);
  member->pose_optimized = Eigen::Isometry3d::Identity();
  member->pose_optimized.translation() = Eigen::Vector3d(52.0, 10.25, 0.0);
  member->pose_optimized.rotate(
      Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()));

  const auto stale = builder.evaluatePoseProjection(
      {anchor, member, unassigned});
  ASSERT_TRUE(stale.valid) << stale.failure_reason;
  EXPECT_GT(stale.max_translation_residual_m, 40.0);

  const auto refreshed = builder.refreshMapPoses(
      {anchor, member, unassigned});
  ASSERT_TRUE(refreshed.valid) << refreshed.failure_reason;
  EXPECT_EQ(refreshed.submap_count, 1u);
  EXPECT_EQ(refreshed.projected_keyframe_count, 2u);
  EXPECT_EQ(refreshed.unassigned_keyframe_count, 1u);
  EXPECT_EQ(refreshed.refreshed_submap_count, 1u);
  EXPECT_NEAR(refreshed.mean_translation_residual_m, 0.125, 1e-9);
  EXPECT_NEAR(refreshed.max_translation_residual_m, 0.25, 1e-9);
  EXPECT_NEAR(refreshed.mean_rotation_residual_rad, 0.05, 1e-9);
  EXPECT_NEAR(refreshed.max_rotation_residual_rad, 0.1, 1e-9);

  const auto submaps = builder.getSubmaps();
  ASSERT_EQ(submaps.size(), 1u);
  EXPECT_TRUE(submaps.front().T_map_submap.isApprox(
      anchor->pose_optimized, 1e-12));
  // Origin refresh is pose state, not a logical content mutation.
  EXPECT_EQ(submaps.front().content_revision, content_revision);

  const auto evaluated = builder.evaluatePoseProjection(
      {anchor, member, unassigned});
  ASSERT_TRUE(evaluated.valid) << evaluated.failure_reason;
  EXPECT_NEAR(evaluated.max_translation_residual_m, 0.25, 1e-9);
  EXPECT_NEAR(evaluated.max_rotation_residual_rad, 0.1, 1e-9);
}

TEST(SubmapBuilderTest, PoseRefreshFailureDoesNotPartiallyMutateOrigins) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 1;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  auto first = makeKeyframe(0, 0, 0.0);
  auto second = makeKeyframe(1, 0, 2.0);
  ASSERT_TRUE(builder.appendKeyframe(first));
  ASSERT_TRUE(builder.appendKeyframe(second));
  const auto before = builder.getSubmaps();
  ASSERT_EQ(before.size(), 2u);

  first->pose_optimized.translation().y() = 10.0;
  second->pose_optimized.linear()(0, 0) =
      std::numeric_limits<double>::quiet_NaN();
  const auto failed = builder.refreshMapPoses({first, second});
  EXPECT_FALSE(failed.valid);
  EXPECT_EQ(failed.failure_reason, "invalid_keyframe_pose");

  const auto after = builder.getSubmaps();
  ASSERT_EQ(after.size(), before.size());
  for (std::size_t index = 0; index < after.size(); ++index) {
    EXPECT_TRUE(after[index].T_map_submap.isApprox(
        before[index].T_map_submap, 1e-12));
  }
}

TEST(SubmapBuilderTest, EnabledLoadReanchorsLegacyDescriptorFallback) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder source(options);
  auto anchor = makeKeyframe(0, 0, 3.0);
  auto member = makeKeyframe(1, 0, 4.0);
  ASSERT_TRUE(source.appendKeyframe(anchor));
  ASSERT_TRUE(source.appendKeyframe(member));

  auto serialized = source.getSubmaps();
  ASSERT_EQ(serialized.size(), 1u);
  serialized.front().descriptor_keyframe_id = -1;
  serialized.front().T_map_submap.translation().x() = -100.0;
  anchor->pose_optimized = Eigen::Isometry3d::Identity();
  anchor->pose_optimized.translation() = Eigen::Vector3d(20.0, 30.0, 0.0);
  member->pose_optimized = anchor->pose_optimized *
      (serialized.front().T_session_submap.inverse() * member->pose_odom);

  SubmapBuilder loaded(options);
  ASSERT_TRUE(loaded.loadSubmaps(
      serialized, {anchor, member}, makeSessions()));
  const auto reanchored = loaded.getSubmaps();
  ASSERT_EQ(reanchored.size(), 1u);
  EXPECT_TRUE(reanchored.front().T_map_submap.isApprox(
      anchor->pose_optimized, 1e-12));
  EXPECT_EQ(reanchored.front().descriptor_keyframe_id, -1);
  EXPECT_EQ(reanchored.front().content_revision,
            serialized.front().content_revision);

  const auto diagnostics = loaded.evaluatePoseProjection({anchor, member});
  ASSERT_TRUE(diagnostics.valid) << diagnostics.failure_reason;
  EXPECT_EQ(diagnostics.projected_keyframe_count, 2u);
  EXPECT_EQ(diagnostics.unassigned_keyframe_count, 0u);
  EXPECT_NEAR(diagnostics.max_translation_residual_m, 0.0, 1e-9);
  EXPECT_NEAR(diagnostics.max_rotation_residual_rad, 0.0, 1e-9);
}

TEST(SubmapGraphProjectionTest,
     ProjectsEverySourceConstraintWithExplicitOwnership) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 32 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);

  const Eigen::Isometry3d T_map_a = makePose(10.0, 5.0, 0.3);
  const Eigen::Isometry3d T_a_a1 = makePose(1.0, 0.2, 0.15);
  const Eigen::Isometry3d T_session_b = makePose(4.0, -1.0, 0.4);
  const Eigen::Isometry3d T_map_b = makePose(14.0, 8.0, -0.2);
  const Eigen::Isometry3d T_b_b1 = makePose(0.5, -0.3, -0.1);

  auto a0 = makeKeyframe(0, 0, 0.0);
  auto a1 = makeKeyframe(1, 0, 0.0);
  auto b0 = makeKeyframe(2, 0, 0.0);
  auto b1 = makeKeyframe(3, 0, 0.0);
  auto unassigned = makeKeyframe(4, 0, 0.0);
  a0->pose_odom = Eigen::Isometry3d::Identity();
  a0->pose_optimized = T_map_a;
  a1->pose_odom = T_a_a1;
  a1->pose_optimized = T_map_a * T_a_a1;
  b0->pose_odom = T_session_b;
  b0->pose_optimized = T_map_b;
  b1->pose_odom = T_session_b * T_b_b1;
  b1->pose_optimized = T_map_b * T_b_b1;
  unassigned->pose_odom = makePose(8.0, 1.0, 0.2);
  unassigned->pose_optimized = makePose(18.0, 9.0, -0.1);

  ASSERT_TRUE(builder.appendKeyframe(a0));
  ASSERT_TRUE(builder.appendKeyframe(a1));
  ASSERT_TRUE(builder.appendKeyframe(b0));
  ASSERT_TRUE(builder.appendKeyframe(b1));
  const auto submaps_before = builder.getSubmaps();
  ASSERT_EQ(submaps_before.size(), 2u);

  Eigen::Matrix<double, 6, 6> loop_information =
      Eigen::Matrix<double, 6, 6>::Identity() * 7.0;
  loop_information(1, 5) = 0.25;
  loop_information(5, 1) = 0.25;

  EdgeInfo intra;
  intra.from_id = 0;
  intra.to_id = 1;
  intra.measurement = a0->pose_optimized.inverse() * a1->pose_optimized;
  intra.information = Eigen::Matrix<double, 6, 6>::Identity() * 2.0;
  intra.type = EdgeType::ODOMETRY;

  EdgeInfo cross_odom;
  cross_odom.from_id = 1;
  cross_odom.to_id = 2;
  cross_odom.measurement =
      a1->pose_optimized.inverse() * b0->pose_optimized;
  cross_odom.information =
      Eigen::Matrix<double, 6, 6>::Identity() * 3.0;
  cross_odom.type = EdgeType::ODOMETRY;

  EdgeInfo cross_loop;
  cross_loop.from_id = 0;
  cross_loop.to_id = 3;
  cross_loop.measurement =
      a0->pose_optimized.inverse() * b1->pose_optimized;
  cross_loop.information = loop_information;
  cross_loop.type = EdgeType::LOOP;
  cross_loop.constraint_mode = EdgeConstraintMode::XY_YAW;

  EdgeInfo unassigned_edge;
  unassigned_edge.from_id = 3;
  unassigned_edge.to_id = 4;
  unassigned_edge.measurement =
      b1->pose_optimized.inverse() * unassigned->pose_optimized;
  unassigned_edge.information =
      Eigen::Matrix<double, 6, 6>::Identity() * 5.0;
  unassigned_edge.type = EdgeType::SESSION_ANCHOR;

  FloorAttitudeConstraint assigned_floor;
  assigned_floor.node_id = 1;
  assigned_floor.normal_body = Eigen::Vector3d(0.1, 0.0, 0.995);
  assigned_floor.sigma_rad = 0.02;
  FloorAttitudeConstraint unassigned_floor;
  unassigned_floor.node_id = 4;
  unassigned_floor.normal_body = Eigen::Vector3d::UnitZ();
  unassigned_floor.sigma_rad = 0.03;

  const auto snapshot = buildSubmapGraphSnapshot(
      submaps_before, {a0, a1, b0, b1, unassigned},
      {intra, cross_odom, cross_loop, unassigned_edge},
      {assigned_floor, unassigned_floor});
  ASSERT_TRUE(snapshot.valid) << snapshot.failure_reason;
  ASSERT_EQ(snapshot.nodes.size(), 2u);
  EXPECT_EQ(snapshot.nodes[0].submap_id, 0u);
  EXPECT_EQ(snapshot.nodes[1].submap_id, 1u);
  EXPECT_TRUE(snapshot.nodes[0].T_map_submap.isApprox(T_map_a, 1e-12));
  EXPECT_TRUE(snapshot.nodes[1].T_map_submap.isApprox(T_map_b, 1e-12));
  EXPECT_EQ(snapshot.keyframe_ownership.size(), 4u);
  EXPECT_EQ(snapshot.keyframe_ownership.at(0), 0u);
  EXPECT_EQ(snapshot.keyframe_ownership.at(3), 1u);
  EXPECT_EQ(snapshot.unassigned_keyframe_ids,
            (std::vector<int64_t>{4}));
  ASSERT_EQ(snapshot.keyframe_projections.size(), 4u);
  EXPECT_EQ(snapshot.keyframe_projections[0].keyframe_id, 0);
  EXPECT_EQ(snapshot.keyframe_projections[1].keyframe_id, 1);
  EXPECT_EQ(snapshot.keyframe_projections[2].keyframe_id, 2);
  EXPECT_EQ(snapshot.keyframe_projections[3].keyframe_id, 3);
  EXPECT_TRUE(snapshot.keyframe_projections[1].T_submap_keyframe.isApprox(
      T_a_a1, 1e-12));
  EXPECT_TRUE(snapshot.keyframe_projections[1]
                  .T_map_keyframe_reference.isApprox(
                      a1->pose_optimized, 1e-12));
  EXPECT_TRUE(snapshot.keyframe_projections[3].T_submap_keyframe.isApprox(
      T_b_b1, 1e-12));
  EXPECT_TRUE(snapshot.keyframe_projections[3]
                  .T_map_keyframe_reference.isApprox(
                      b1->pose_optimized, 1e-12));

  ASSERT_EQ(snapshot.edge_projections.size(), 4u);
  EXPECT_EQ(snapshot.source_edge_count, 4u);
  EXPECT_EQ(snapshot.intra_submap_edge_count, 1u);
  EXPECT_EQ(snapshot.cross_submap_edge_count, 2u);
  EXPECT_EQ(snapshot.unassigned_endpoint_edge_count, 1u);
  EXPECT_EQ(snapshot.cross_submap_odometry_edge_count, 1u);
  EXPECT_EQ(snapshot.cross_submap_loop_edge_count, 1u);
  EXPECT_EQ(snapshot.cross_submap_session_anchor_edge_count, 0u);
  EXPECT_EQ(snapshot.edge_projections[0].classification,
            SubmapGraphEdgeClass::INTRA_SUBMAP);
  EXPECT_FALSE(snapshot.edge_projections[0].has_projected_measurement);
  EXPECT_EQ(snapshot.edge_projections[1].classification,
            SubmapGraphEdgeClass::CROSS_SUBMAP);
  EXPECT_EQ(snapshot.edge_projections[2].classification,
            SubmapGraphEdgeClass::CROSS_SUBMAP);
  EXPECT_EQ(snapshot.edge_projections[3].classification,
            SubmapGraphEdgeClass::UNASSIGNED_ENDPOINT);
  EXPECT_EQ(snapshot.edge_projections[3].from_submap_id, 1u);
  EXPECT_EQ(snapshot.edge_projections[3].to_submap_id,
            kInvalidSubmapId);

  const Eigen::Isometry3d expected_submap_measurement =
      T_map_a.inverse() * T_map_b;
  EXPECT_TRUE(snapshot.edge_projections[1]
                  .T_from_submap_from_keyframe.isApprox(T_a_a1, 1e-12));
  EXPECT_TRUE(snapshot.edge_projections[1]
                  .T_to_submap_to_keyframe.isApprox(
                      Eigen::Isometry3d::Identity(), 1e-12));
  EXPECT_TRUE(snapshot.edge_projections[1]
                  .T_from_submap_to_submap_measurement.isApprox(
                      expected_submap_measurement, 1e-12));
  EXPECT_TRUE(snapshot.edge_projections[2]
                  .T_to_submap_to_keyframe.isApprox(T_b_b1, 1e-12));
  EXPECT_TRUE(snapshot.edge_projections[2]
                  .T_from_submap_to_submap_measurement.isApprox(
                      expected_submap_measurement, 1e-12));
  EXPECT_TRUE(snapshot.edge_projections[2]
                  .source_edge.information.isApprox(loop_information, 0.0));
  EXPECT_EQ(snapshot.edge_projections[2].source_edge.constraint_mode,
            EdgeConstraintMode::XY_YAW);
  EXPECT_LT(snapshot.max_cross_edge_translation_residual_m, 1e-9);
  EXPECT_LT(snapshot.max_cross_edge_rotation_residual_rad, 1e-9);

  ASSERT_EQ(snapshot.floor_projections.size(), 2u);
  EXPECT_EQ(snapshot.source_floor_constraint_count, 2u);
  EXPECT_EQ(snapshot.assigned_floor_constraint_count, 1u);
  EXPECT_EQ(snapshot.unassigned_floor_constraint_count, 1u);
  EXPECT_TRUE(snapshot.floor_projections[0].assigned);
  EXPECT_EQ(snapshot.floor_projections[0].submap_id, 0u);
  EXPECT_TRUE(snapshot.floor_projections[0].T_submap_keyframe.isApprox(
      T_a_a1, 1e-12));
  EXPECT_FALSE(snapshot.floor_projections[1].assigned);

  const auto topology = evaluateSubmapGraphTopology(snapshot);
  ASSERT_TRUE(topology.valid) << topology.failure_reason;
  EXPECT_TRUE(topology.connected);
  EXPECT_FALSE(topology.keyframe_ownership_complete);
  EXPECT_FALSE(topology.constraint_coverage_complete);
  EXPECT_FALSE(topology.shadow_graph_ready);
  EXPECT_EQ(topology.component_count, 1u);
  EXPECT_EQ(topology.cross_edge_count, 2u);
  EXPECT_EQ(topology.cross_session_edge_count, 0u);
  ASSERT_EQ(topology.components.size(), 1u);
  EXPECT_EQ(topology.components[0].cross_edge_projection_indices,
            (std::vector<std::size_t>{1, 2}));

  const auto factors = evaluateSubmapGraphFactors(snapshot);
  ASSERT_TRUE(factors.valid) << factors.failure_reason;
  EXPECT_EQ(factors.cross_edge_count, 2u);
  EXPECT_EQ(factors.full_6d_edge_count, 1u);
  EXPECT_EQ(factors.xy_yaw_exact_lifted_only_count, 1u);
  EXPECT_EQ(factors.direct_between_residual_equivalent_count, 1u);
  EXPECT_EQ(factors.direct_between_information_transportable_count, 1u);
  EXPECT_EQ(factors.information_fallback_required_count, 0u);
  EXPECT_EQ(factors.assigned_floor_factor_count, 1u);
  EXPECT_EQ(factors.unassigned_floor_factor_count, 1u);
  EXPECT_LT(factors.max_residual_transport_error_norm, 1e-9);
  EXPECT_LT(factors.max_mahalanobis_squared_delta, 1e-9);
  EXPECT_LT(factors.max_floor_residual_error_norm, 1e-9);

  const auto submaps_after = builder.getSubmaps();
  ASSERT_EQ(submaps_after.size(), submaps_before.size());
  for (std::size_t index = 0; index < submaps_after.size(); ++index) {
    EXPECT_TRUE(submaps_after[index].T_map_submap.isApprox(
        submaps_before[index].T_map_submap, 1e-12));
    EXPECT_EQ(submaps_after[index].content_revision,
              submaps_before[index].content_revision);
  }
}

TEST(SubmapGraphFactorTest,
     Full6DAdjointTransportPreservesResidualAndMahalanobisCost) {
  const Eigen::Isometry3d T_map_a = makePose6(
      Eigen::Vector3d(3.0, -2.0, 0.5), 0.1, -0.2, 0.4);
  const Eigen::Isometry3d T_a_keyframe = makePose6(
      Eigen::Vector3d(1.2, -0.7, 0.4), -0.15, 0.08, 0.3);
  const Eigen::Isometry3d T_b_keyframe = makePose6(
      Eigen::Vector3d(-0.8, 1.1, 0.6), 0.2, -0.1, -0.35);
  const Eigen::Isometry3d source_measurement = makePose6(
      Eigen::Vector3d(4.0, 2.0, -0.3), 0.12, -0.18, 0.5);
  const Eigen::Isometry3d projected_measurement =
      T_a_keyframe * source_measurement * T_b_keyframe.inverse();
  const Eigen::Isometry3d nominal_T_map_b =
      T_map_a * projected_measurement;
  const Eigen::Isometry3d T_map_b = nominal_T_map_b * makePose6(
      Eigen::Vector3d(0.25, -0.15, 0.12), 0.04, -0.03, 0.07);

  SubmapGraphEdgeProjection projection;
  projection.source_edge_index = 4;
  projection.source_edge.from_id = 10;
  projection.source_edge.to_id = 20;
  projection.source_edge.measurement = source_measurement;
  projection.source_edge.information =
      Eigen::Matrix<double, 6, 6>::Zero();
  projection.source_edge.information.diagonal() <<
      4.0, 9.0, 16.0, 25.0, 36.0, 49.0;
  projection.source_edge.information(0, 5) = 0.7;
  projection.source_edge.information(5, 0) = 0.7;
  projection.source_edge.information(1, 3) = -0.4;
  projection.source_edge.information(3, 1) = -0.4;
  projection.source_edge.type = EdgeType::LOOP;
  projection.source_edge.constraint_mode = EdgeConstraintMode::FULL_6DOF;
  projection.classification = SubmapGraphEdgeClass::CROSS_SUBMAP;
  projection.from_submap_id = 1;
  projection.to_submap_id = 2;
  projection.T_from_submap_from_keyframe = T_a_keyframe;
  projection.T_to_submap_to_keyframe = T_b_keyframe;
  projection.has_projected_measurement = true;
  projection.T_from_submap_to_submap_measurement = projected_measurement;

  const auto evaluation = evaluateSubmapGraphEdgeFactor(
      projection, T_map_a, T_map_b, 8);
  ASSERT_TRUE(evaluation.valid) << evaluation.failure_reason;
  EXPECT_EQ(evaluation.edge_projection_index, 8u);
  EXPECT_EQ(evaluation.source_edge_index, 4u);
  EXPECT_EQ(evaluation.exact_lifted_residual.size(), 6);
  EXPECT_TRUE(evaluation.explicit_source_information_usable);
  EXPECT_TRUE(evaluation.direct_between_residual_equivalent);
  EXPECT_TRUE(evaluation.direct_between_information_transportable);

  const gtsam::BetweenFactor<gtsam::Pose3> source_factor(
      0, 1, toGtsam(source_measurement));
  const gtsam::Vector expected_source_residual =
      source_factor.evaluateError(
          toGtsam(T_map_a * T_a_keyframe),
          toGtsam(T_map_b * T_b_keyframe));
  EXPECT_TRUE(evaluation.exact_lifted_residual.isApprox(
      expected_source_residual, 1e-12));
  EXPECT_LT(evaluation.residual_transport_error_norm, 1e-10);
  EXPECT_LT(evaluation.mahalanobis_squared_delta, 1e-10);
  EXPECT_TRUE(evaluation.direct_between_information.isApprox(
      evaluation.direct_between_information.transpose(), 1e-12));
  const Eigen::LLT<Eigen::Matrix<double, 6, 6>> projected_information_llt(
      evaluation.direct_between_information);
  EXPECT_EQ(projected_information_llt.info(), Eigen::Success);

  const double naive_cost = evaluation.direct_between_residual.dot(
      evaluation.source_information_factor_order *
      evaluation.direct_between_residual);
  EXPECT_GT(std::abs(naive_cost -
                     evaluation.exact_lifted_mahalanobis_squared),
            1e-3);
}

TEST(SubmapGraphFactorTest, XYYawRequiresExactLiftedFactor) {
  const Eigen::Isometry3d T_map_a = makePose6(
      Eigen::Vector3d(2.0, 1.0, 0.3), 0.08, -0.1, 0.6);
  const Eigen::Isometry3d T_a_keyframe = makePose6(
      Eigen::Vector3d(1.0, -0.5, 0.2), 0.1, 0.05, -0.4);
  const Eigen::Isometry3d T_b_keyframe = makePose6(
      Eigen::Vector3d(-0.4, 1.3, -0.1), -0.06, 0.09, 0.35);
  const Eigen::Isometry3d source_measurement = makePose6(
      Eigen::Vector3d(3.5, -1.2, 0.4), 0.02, -0.03, 0.45);
  const Eigen::Isometry3d projected_measurement =
      T_a_keyframe * source_measurement * T_b_keyframe.inverse();
  const Eigen::Isometry3d T_map_b =
      T_map_a * projected_measurement * makePose6(
          Eigen::Vector3d(0.2, 0.1, 0.0), 0.01, -0.02, 0.08);

  SubmapGraphEdgeProjection projection;
  projection.source_edge.from_id = 3;
  projection.source_edge.to_id = 7;
  projection.source_edge.measurement = source_measurement;
  projection.source_edge.information =
      Eigen::Matrix<double, 6, 6>::Identity();
  projection.source_edge.information(0, 0) = 4.0;
  projection.source_edge.information(1, 1) = 9.0;
  projection.source_edge.information(5, 5) = 25.0;
  projection.source_edge.type = EdgeType::LOOP;
  projection.source_edge.constraint_mode = EdgeConstraintMode::XY_YAW;
  projection.classification = SubmapGraphEdgeClass::CROSS_SUBMAP;
  projection.from_submap_id = 1;
  projection.to_submap_id = 2;
  projection.T_from_submap_from_keyframe = T_a_keyframe;
  projection.T_to_submap_to_keyframe = T_b_keyframe;
  projection.has_projected_measurement = true;
  projection.T_from_submap_to_submap_measurement = projected_measurement;

  const auto evaluation = evaluateSubmapGraphEdgeFactor(
      projection, T_map_a, T_map_b);
  ASSERT_TRUE(evaluation.valid) << evaluation.failure_reason;
  EXPECT_EQ(evaluation.exact_lifted_residual.size(), 3);
  EXPECT_TRUE(evaluation.explicit_source_information_usable);
  EXPECT_FALSE(evaluation.direct_between_residual_equivalent);
  EXPECT_FALSE(evaluation.direct_between_information_transportable);
  EXPECT_EQ(evaluation.direct_between_limitation,
            "xy_yaw_requires_exact_lifted_factor");

  const gtsam::Pose3 source_prediction =
      toGtsam(T_map_a * T_a_keyframe).between(
          toGtsam(T_map_b * T_b_keyframe));
  const gtsam::Pose3 measured = toGtsam(source_measurement);
  Eigen::Vector3d expected;
  expected << source_prediction.translation().x() -
                  measured.translation().x(),
              source_prediction.translation().y() -
                  measured.translation().y(),
              normalizeAngle(source_prediction.rotation().yaw() -
                             measured.rotation().yaw());
  EXPECT_TRUE(evaluation.exact_lifted_residual.isApprox(expected, 1e-12));

  const gtsam::Pose3 naive_prediction =
      toGtsam(T_map_a).between(toGtsam(T_map_b));
  const gtsam::Pose3 naive_measurement = toGtsam(projected_measurement);
  Eigen::Vector3d naive;
  naive << naive_prediction.translation().x() -
               naive_measurement.translation().x(),
           naive_prediction.translation().y() -
               naive_measurement.translation().y(),
           normalizeAngle(naive_prediction.rotation().yaw() -
                          naive_measurement.rotation().yaw());
  EXPECT_GT((naive - expected).norm(), 1e-3);
}

TEST(SubmapGraphFactorTest,
     ConfigFallbackInformationIsNotAdvertisedAsTransportable) {
  SubmapGraphEdgeProjection projection;
  projection.source_edge.from_id = 0;
  projection.source_edge.to_id = 1;
  projection.source_edge.measurement = makePose(1.0, 0.0, 0.1);
  projection.source_edge.information.setZero();
  projection.source_edge.type = EdgeType::ODOMETRY;
  projection.source_edge.constraint_mode = EdgeConstraintMode::FULL_6DOF;
  projection.classification = SubmapGraphEdgeClass::CROSS_SUBMAP;
  projection.from_submap_id = 0;
  projection.to_submap_id = 1;
  projection.T_from_submap_from_keyframe = Eigen::Isometry3d::Identity();
  projection.T_to_submap_to_keyframe = Eigen::Isometry3d::Identity();
  projection.has_projected_measurement = true;
  projection.T_from_submap_to_submap_measurement =
      projection.source_edge.measurement;

  const Eigen::Isometry3d T_map_from = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d T_map_to = projection.source_edge.measurement;
  T_map_to.translation().y() += 0.2;
  const auto evaluation = evaluateSubmapGraphEdgeFactor(
      projection, T_map_from, T_map_to);
  ASSERT_TRUE(evaluation.valid) << evaluation.failure_reason;
  EXPECT_EQ(evaluation.exact_lifted_residual.size(), 6);
  EXPECT_TRUE(evaluation.direct_between_residual_equivalent);
  EXPECT_FALSE(evaluation.explicit_source_information_usable);
  EXPECT_FALSE(evaluation.direct_between_information_transportable);
  EXPECT_EQ(evaluation.information_limitation,
            "full_6d_config_fallback_or_invalid_information");
  EXPECT_EQ(evaluation.direct_between_limitation,
            "explicit_source_information_unavailable");
  EXPECT_TRUE(std::isnan(evaluation.exact_lifted_mahalanobis_squared));
  EXPECT_TRUE(std::isnan(evaluation.direct_between_mahalanobis_squared));
}

TEST(SubmapGraphFactorTest, FloorNormalProjectionPreservesAttitudeFactor) {
  SubmapGraphFloorProjection projection;
  projection.source_constraint_index = 2;
  projection.source_constraint.node_id = 11;
  projection.source_constraint.normal_body =
      Eigen::Vector3d(0.15, -0.2, 0.95).normalized();
  projection.source_constraint.sigma_rad = 0.03;
  projection.assigned = true;
  projection.submap_id = 4;
  projection.T_submap_keyframe = makePose6(
      Eigen::Vector3d(1.0, 2.0, -0.5), 0.2, -0.15, 0.4);
  const Eigen::Isometry3d T_map_submap = makePose6(
      Eigen::Vector3d(5.0, -3.0, 1.0), -0.1, 0.25, -0.3);

  const auto evaluation = evaluateSubmapGraphFloorFactor(
      projection, T_map_submap, 6);
  ASSERT_TRUE(evaluation.valid) << evaluation.failure_reason;
  EXPECT_EQ(evaluation.floor_projection_index, 6u);
  EXPECT_EQ(evaluation.source_constraint_index, 2u);
  EXPECT_TRUE(evaluation.normal_submap.isApprox(
      projection.T_submap_keyframe.rotation() *
          projection.source_constraint.normal_body,
      1e-12));
  EXPECT_LT(evaluation.residual_error_norm, 1e-12);
  EXPECT_NEAR(evaluation.exact_lifted_mahalanobis_squared,
              evaluation.projected_submap_mahalanobis_squared, 1e-10);
}

TEST(SubmapGraphTopologyTest,
     ConnectedMultiSessionGraphPreservesParallelConstraints) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 1;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);
  auto first = makeKeyframe(0, 0, 0.0);
  auto second = makeKeyframe(1, 1, 0.0);
  second->pose_optimized.translation().x() = 5.0;
  ASSERT_TRUE(builder.appendKeyframe(first));
  ASSERT_TRUE(builder.appendKeyframe(second));

  EdgeInfo anchor;
  anchor.from_id = 0;
  anchor.to_id = 1;
  anchor.measurement =
      first->pose_optimized.inverse() * second->pose_optimized;
  anchor.information = Eigen::Matrix<double, 6, 6>::Identity() * 4.0;
  anchor.type = EdgeType::SESSION_ANCHOR;
  EdgeInfo loop = anchor;
  loop.information = Eigen::Matrix<double, 6, 6>::Identity() * 9.0;
  loop.type = EdgeType::LOOP;

  const auto snapshot = buildSubmapGraphSnapshot(
      builder.getSubmaps(), {first, second}, {anchor, loop});
  ASSERT_TRUE(snapshot.valid) << snapshot.failure_reason;
  const auto topology = evaluateSubmapGraphTopology(snapshot);
  ASSERT_TRUE(topology.valid) << topology.failure_reason;
  EXPECT_TRUE(topology.keyframe_ownership_complete);
  EXPECT_TRUE(topology.constraint_coverage_complete);
  EXPECT_TRUE(topology.connected);
  EXPECT_TRUE(topology.shadow_graph_ready);
  EXPECT_EQ(topology.node_count, 2u);
  EXPECT_EQ(topology.component_count, 1u);
  EXPECT_EQ(topology.isolated_submap_count, 0u);
  EXPECT_EQ(topology.cross_edge_count, 2u);
  EXPECT_EQ(topology.cross_session_edge_count, 2u);
  ASSERT_EQ(topology.components.size(), 1u);
  EXPECT_EQ(topology.components[0].anchor_submap_id, 0u);
  EXPECT_EQ(topology.components[0].submap_ids,
            (std::vector<SubmapId>{0, 1}));
  EXPECT_EQ(topology.components[0].session_ids,
            (std::vector<MapSessionId>{0, 1}));
  EXPECT_EQ(topology.components[0].cross_edge_projection_indices,
            (std::vector<std::size_t>{0, 1}));
}

TEST(SubmapGraphTopologyTest,
     DisconnectedGraphReportsDeterministicComponentsAndIsolation) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 1;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);
  auto first = makeKeyframe(0, 0, 0.0);
  auto second = makeKeyframe(1, 0, 1.0);
  auto isolated = makeKeyframe(2, 0, 2.0);
  ASSERT_TRUE(builder.appendKeyframe(first));
  ASSERT_TRUE(builder.appendKeyframe(second));
  ASSERT_TRUE(builder.appendKeyframe(isolated));

  EdgeInfo edge;
  edge.from_id = 0;
  edge.to_id = 1;
  edge.measurement =
      first->pose_optimized.inverse() * second->pose_optimized;
  edge.information = Eigen::Matrix<double, 6, 6>::Identity();
  edge.type = EdgeType::ODOMETRY;
  const auto snapshot = buildSubmapGraphSnapshot(
      builder.getSubmaps(), {first, second, isolated}, {edge});
  ASSERT_TRUE(snapshot.valid) << snapshot.failure_reason;

  const auto topology = evaluateSubmapGraphTopology(snapshot);
  ASSERT_TRUE(topology.valid) << topology.failure_reason;
  EXPECT_TRUE(topology.keyframe_ownership_complete);
  EXPECT_TRUE(topology.constraint_coverage_complete);
  EXPECT_FALSE(topology.connected);
  EXPECT_FALSE(topology.shadow_graph_ready);
  EXPECT_EQ(topology.component_count, 2u);
  EXPECT_EQ(topology.isolated_submap_ids,
            (std::vector<SubmapId>{2}));
  ASSERT_EQ(topology.components.size(), 2u);
  EXPECT_EQ(topology.components[0].anchor_submap_id, 0u);
  EXPECT_EQ(topology.components[0].submap_ids,
            (std::vector<SubmapId>{0, 1}));
  EXPECT_EQ(topology.components[0].cross_edge_projection_indices,
            (std::vector<std::size_t>{0}));
  EXPECT_EQ(topology.components[1].anchor_submap_id, 2u);
  EXPECT_EQ(topology.components[1].submap_ids,
            (std::vector<SubmapId>{2}));
  EXPECT_TRUE(topology.components[1].cross_edge_projection_indices.empty());
}

TEST(SubmapGraphTopologyTest, CompleteSingleNodeGraphIsStructurallyReady) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 2;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);
  auto keyframe = makeKeyframe(0, 0, 0.0);
  ASSERT_TRUE(builder.appendKeyframe(keyframe));
  const auto snapshot = buildSubmapGraphSnapshot(
      builder.getSubmaps(), {keyframe}, {});
  ASSERT_TRUE(snapshot.valid) << snapshot.failure_reason;

  const auto topology = evaluateSubmapGraphTopology(snapshot);
  ASSERT_TRUE(topology.valid) << topology.failure_reason;
  EXPECT_TRUE(topology.connected);
  EXPECT_TRUE(topology.shadow_graph_ready);
  EXPECT_EQ(topology.component_count, 1u);
  EXPECT_EQ(topology.isolated_submap_ids,
            (std::vector<SubmapId>{0}));
}

TEST(SubmapGraphTopologyTest, MalformedSnapshotReturnsNoPartialTopology) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 1;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);
  auto first = makeKeyframe(0, 0, 0.0);
  auto second = makeKeyframe(1, 0, 1.0);
  ASSERT_TRUE(builder.appendKeyframe(first));
  ASSERT_TRUE(builder.appendKeyframe(second));
  EdgeInfo edge;
  edge.from_id = 0;
  edge.to_id = 1;
  edge.measurement =
      first->pose_optimized.inverse() * second->pose_optimized;
  edge.information = Eigen::Matrix<double, 6, 6>::Identity();
  edge.type = EdgeType::ODOMETRY;
  auto snapshot = buildSubmapGraphSnapshot(
      builder.getSubmaps(), {first, second}, {edge});
  ASSERT_TRUE(snapshot.valid) << snapshot.failure_reason;
  snapshot.edge_projections[0].to_submap_id = 99;

  const auto topology = evaluateSubmapGraphTopology(snapshot);
  EXPECT_FALSE(topology.valid);
  EXPECT_EQ(topology.failure_reason, "invalid_cross_submap_projection");
  EXPECT_TRUE(topology.components.empty());
  EXPECT_TRUE(topology.component_by_submap.empty());
  EXPECT_TRUE(topology.isolated_submap_ids.empty());

  const auto factors = evaluateSubmapGraphFactors(snapshot);
  EXPECT_FALSE(factors.valid);
  EXPECT_EQ(factors.failure_reason, "missing_factor_edge_node");
  EXPECT_TRUE(factors.edge_evaluations.empty());
  EXPECT_TRUE(factors.floor_evaluations.empty());
}

TEST(SubmapGraphProjectionTest, InvalidInputsReturnNoPartialSnapshot) {
  SubmapBuilderOptions options;
  options.enable = true;
  options.max_keyframes = 1;
  options.cloud_max_bytes = 8 * sizeof(pcl::PointXYZI);
  SubmapBuilder builder(options);
  auto first = makeKeyframe(0, 0, 0.0);
  auto second = makeKeyframe(1, 0, 1.0);
  ASSERT_TRUE(builder.appendKeyframe(first));
  ASSERT_TRUE(builder.appendKeyframe(second));

  EdgeInfo edge;
  edge.from_id = 0;
  edge.to_id = 99;
  edge.measurement = Eigen::Isometry3d::Identity();
  edge.information = Eigen::Matrix<double, 6, 6>::Identity();
  edge.type = EdgeType::ODOMETRY;
  const auto missing_edge_keyframe = buildSubmapGraphSnapshot(
      builder.getSubmaps(), {first, second}, {edge});
  EXPECT_FALSE(missing_edge_keyframe.valid);
  EXPECT_EQ(missing_edge_keyframe.failure_reason,
            "missing_edge_keyframe");
  EXPECT_TRUE(missing_edge_keyframe.nodes.empty());
  EXPECT_TRUE(missing_edge_keyframe.keyframe_ownership.empty());
  EXPECT_TRUE(missing_edge_keyframe.edge_projections.empty());

  edge.to_id = 1;
  edge.measurement =
      first->pose_optimized.inverse() * second->pose_optimized;
  FloorAttitudeConstraint missing_floor;
  missing_floor.node_id = 99;
  missing_floor.normal_body = Eigen::Vector3d::UnitZ();
  missing_floor.sigma_rad = 0.02;
  const auto missing_floor_keyframe = buildSubmapGraphSnapshot(
      builder.getSubmaps(), {first, second}, {edge}, {missing_floor});
  EXPECT_FALSE(missing_floor_keyframe.valid);
  EXPECT_EQ(missing_floor_keyframe.failure_reason,
            "missing_floor_keyframe");
  EXPECT_TRUE(missing_floor_keyframe.nodes.empty());
  EXPECT_TRUE(missing_floor_keyframe.edge_projections.empty());
  EXPECT_TRUE(missing_floor_keyframe.floor_projections.empty());
}

}  // namespace
}  // namespace n3mapping
