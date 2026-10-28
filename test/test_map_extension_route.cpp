// End-to-end extension through a 1.2 m doorway into an unrecorded room.
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/pcl_compat.h"
#include "n3map.pb.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace n3mapping {
namespace {
// Ray intersections provide occlusion, floor and ceiling observations. The
// loaded map uses acquisitions inside the first room, including returns
// visible through the doorway. Unobserved surfaces are never added.
core::LioFrame roomFrame(int64_t stamp, double x, double y) {
  struct Face { int axis; double coordinate, low_a, high_a, low_b, high_b; };
  const Face faces[] = {
    {0,-6,-4,4,0,3.5}, {0,8,-4,4,0,3.5},
    {1,-4,0,3.5,-6,8}, {1,4,0,3.5,-6,8},
    {2,0,-6,8,-4,4}, {2,3.5,-6,8,-4,4},
    {0,0,-4,-0.6,0,3.5}, {0,0,0.6,4,0,3.5},
    {0,0,-0.6,0.6,2.3,3.5},
    {0,-4,-2.5,-1.5,0,2.1}, {1,-1.5,0,2.1,-5,-4},
    {0,5,1,3,0,2.7}, {1,1,0,2.7,5,7}
  };
  core::LioFrame frame;
  frame.stamp.nsec = stamp;
  frame.source_frame_id = "lio_odom";
  frame.pose_valid = true;
  frame.T_world_lidar.translation() = Eigen::Vector3d(x, y, 1.0);
  auto cloud = pcl::make_shared<core::LioFrame::PointCloud>();
  for (int v = -35; v <= 55; v += 5) {
    const double pitch = v * M_PI / 180.0;
    for (int h = 0; h < 360; h += 2) {
      const double yaw = h * M_PI / 180.0;
      const Eigen::Vector3d ray(std::cos(pitch)*std::cos(yaw),
                                std::cos(pitch)*std::sin(yaw), std::sin(pitch));
      double nearest = 30.0;
      for (const auto& face : faces) {
        const int a = (face.axis + 1) % 3, b = (face.axis + 2) % 3;
        if (std::abs(ray[face.axis]) < 1e-8) continue;
        const double t = (face.coordinate-frame.T_world_lidar.translation()[face.axis])/ray[face.axis];
        if (t <= 0.01 || t >= nearest) continue;
        const auto hit = (frame.T_world_lidar.translation() + t*ray).eval();
        if (hit[a] >= face.low_a && hit[a] <= face.high_a &&
            hit[b] >= face.low_b && hit[b] <= face.high_b) nearest = t;
      }
      if (nearest >= 30.0) continue;
      pcl::PointXYZI point;
      point.x = nearest*ray.x(); point.y = nearest*ray.y(); point.z = nearest*ray.z();
      point.intensity = 1.0;
      cloud->push_back(point);
    }
  }
  frame.undistorted_cloud = cloud;
  return frame;
}

class MapExtensionRoute : public ::testing::TestWithParam<bool> {};

TEST_P(MapExtensionRoute, ExtendsBeyondLoadedGeometryAndRecoversWithoutAnotherInitialization) {
  Config config;
  config.mapping_static_start_guard_enable = false;  // The authored poses are settled.
  config.num_threads = 4;
  const auto dir = std::filesystem::temp_directory_path() / "n3mapping_extension_door_route";
  std::filesystem::create_directories(dir);
  const auto original_path = (dir / "original.pbstream").string();
  const auto extended_path = (dir / "extended.pbstream").string();
  N3MappingCore mapping(config, N3MappingCore::WorldCloudOutput::Disabled);
  for (int i = 0; i < 3; ++i)
    ASSERT_TRUE(mapping.processMappingFrame(roomFrame((i+1)*1000000000LL, -4+i, 0)).accepted_keyframe);
  ASSERT_TRUE(mapping.saveMap(original_path));

  N3MappingCore core(config, N3MappingCore::WorldCloudOutput::Disabled);
  std::string switch_error;
  EXPECT_FALSE(core.startMapExtension(&switch_error));
  ASSERT_TRUE(core.loadMap(original_path));
  EXPECT_FALSE(core.startMapExtension(&switch_error));
  Eigen::Isometry3d map_from_odom = Eigen::Isometry3d::Identity();
  if (GetParam()) {
    map_from_odom.linear() = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    map_from_odom.translation() = Eigen::Vector3d(12, -7, 0.3);
  }
  const auto sensor_frame = [&](int64_t timestamp, double x, double y) {
    auto frame = roomFrame(timestamp, x, y);
    frame.T_world_lidar = map_from_odom.inverse() * frame.T_world_lidar;
    return frame;
  };
  const auto original_poses = core.getOptimizedPoses();
  int64_t stamp = 10000000000LL;
  core::BackendOutput output;
  int locks = 0;
  for (int i = 0; i < 24 && !output.relocalization_locked; ++i) {
    output = GetParam() ? core.processLocalizationFrame(sensor_frame(stamp, -3, 0))
                        : core.processMapExtensionFrame(sensor_frame(stamp, -3, 0));
    stamp += 100000000LL;
    locks += output.relocalization_locked;
  }
  ASSERT_EQ(locks, 1) << output.relocalization_decision;
  if (GetParam()) {
    auto sparse = sensor_frame(stamp, -3, 0);
    sparse.undistorted_cloud->resize(1);
    output = core.processLocalizationFrame(sparse);
    EXPECT_NE(output.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
    EXPECT_FALSE(core.startMapExtension(&switch_error));
    stamp += 100000000LL;
    output = core.processLocalizationFrame(sensor_frame(stamp, -3, 0));
    ASSERT_EQ(output.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
    const auto before_switch = core.mapRevision();
    const auto started = std::chrono::steady_clock::now();
    ASSERT_TRUE(core.startMapExtension(&switch_error)) << switch_error;
    RecordProperty("switch_ms", std::to_string(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count()));
    EXPECT_EQ(core.mapRevision(), before_switch);
    EXPECT_TRUE(core.startMapExtension(&switch_error));
    stamp += 100000000LL;
  }
  ASSERT_TRUE(core.processMapExtensionFrame(sensor_frame(stamp, -3, 0)).accepted_keyframe);
  stamp += 1000000000LL;
  int prediction_frames = 0, prediction_keyframes = 0, lost_frames = 0;
  std::vector<int64_t> predicted_keyframes;
  double maximum_error = 0.0;
  const double route[][2] = {
    {-2,0},{-1,0},{0,0},{1,0},{2,0},{3,0},{4,0},
    {4,-1},{4,-2},{5,-2},{6,-2},{6,-1},{6,0},{5,0},{4,0},
    {3,0},{2,0},{1,0},{0,0},{-1,0},{-2,0},{-3,0}
  };
  for (const auto& point : route) {
    auto frame = sensor_frame(stamp, point[0], point[1]);
    output = core.processMapExtensionFrame(frame);
    stamp += 1000000000LL;
    EXPECT_TRUE(output.success) << point[0] << ',' << point[1] << ' ' << output.relocalization_decision;
    locks += output.relocalization_locked;
    EXPECT_FALSE(output.relocalization_performance.available);
    maximum_error = std::max(maximum_error,
        (output.T_world_lidar.translation()-Eigen::Vector3d(point[0], point[1], 1.0)).norm());
    if (output.pose_source == PoseSource::ODOM_PREDICTED) {
      ++prediction_frames;
      prediction_keyframes += output.accepted_keyframe;
      if (output.accepted_keyframe) predicted_keyframes.push_back(output.keyframe_id);
      EXPECT_NE(output.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
    }
    lost_frames += output.relocalization_state == RelocalizationState::LOST;
    frame.stamp.nsec += 100000000LL;
    const auto intermediate = core.processMapExtensionFrame(frame);
    EXPECT_TRUE(intermediate.success);
    EXPECT_FALSE(intermediate.accepted_keyframe);
    EXPECT_NEAR(core.getDenseOptimizedTrajectory().back().timestamp,
                frame.stamp.nsec*1e-9, 1e-9);
    core.processPendingLoopClosuresAsync();
  }
  EXPECT_GT(prediction_keyframes, 0);
  EXPECT_GT(lost_frames, 0);
  EXPECT_EQ(locks, 1);
  EXPECT_EQ(output.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
  EXPECT_EQ(output.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  const auto returned_keyframe = output.keyframe_id;
  EXPECT_LT(maximum_error, 0.1);
  RecordProperty("prediction_frames", prediction_frames);
  RecordProperty("prediction_keyframes", prediction_keyframes);
  RecordProperty("lost_frames", lost_frames);
  RecordProperty("maximum_error_mm", std::to_string(maximum_error * 1000.0));

  // Neither a changed coordinate frame nor an invalid pose may extend this session.
  const auto before_invalid = core.getAllKeyframes().size();
  auto invalid = sensor_frame(stamp, -3, 0);
  invalid.source_frame_id = "restarted_lio";
  output = core.processMapExtensionFrame(invalid);
  EXPECT_FALSE(output.success);
  EXPECT_EQ(output.mapping_block_reason, core::MappingBlockReason::InvalidInput);
  invalid.source_frame_id = "lio_odom";
  invalid.pose_valid = false;
  EXPECT_FALSE(core.processMapExtensionFrame(invalid).success);
  EXPECT_EQ(core.getAllKeyframes().size(), before_invalid);

  ASSERT_TRUE(core.saveMap(extended_path));  // Drains the async geometry job.
  for (const auto& [id, pose] : original_poses)
    EXPECT_LT((core.getOptimizedPoses().at(id).translation()-pose.translation()).norm(), 0.02);
  N3Map serialized;
  std::ifstream input(extended_path, std::ios::binary);
  ASSERT_TRUE(serialized.ParseFromIstream(&input));
  int anchors = 0, loops = 0;
  bool return_constraint = false;
  for (const auto& edge : serialized.edges()) {
    anchors += edge.type() == EdgeProto::SESSION_ANCHOR;
    loops += edge.type() == EdgeProto::LOOP;
    if (edge.type() == EdgeProto::LOOP &&
        original_poses.count(edge.from_id())) {
      EXPECT_EQ(std::find(predicted_keyframes.begin(), predicted_keyframes.end(), edge.to_id()),
                predicted_keyframes.end());
      return_constraint |= edge.to_id() == returned_keyframe;
    }
  }
  EXPECT_EQ(anchors, 1);
  EXPECT_GT(loops, 0);
  EXPECT_TRUE(return_constraint);
  for (const auto& frame : core.getAllKeyframes()) {
    const auto reference = frame->is_from_loaded_map ? frame->pose_odom : map_from_odom * frame->pose_odom;
    EXPECT_LT((frame->pose_optimized.translation()-reference.translation()).norm(), 0.1);
  }
  N3MappingCore reloaded(config, N3MappingCore::WorldCloudOutput::Disabled);
  ASSERT_TRUE(reloaded.loadMap(extended_path));
  ASSERT_EQ(reloaded.getAllKeyframes().size(), core.getAllKeyframes().size());
  for (const auto& [id, pose] : core.getOptimizedPoses())
    EXPECT_TRUE(reloaded.getOptimizedPoses().at(id).isApprox(pose, 1e-6));
  EXPECT_EQ(reloaded.buildGlobalMap()->size(), core.buildGlobalMap()->size());
  EXPECT_EQ(reloaded.getDenseOptimizedTrajectory().size(), core.getDenseOptimizedTrajectory().size());
  // Reject every impossible jump, including the samples before the existing
  // consecutive-violation threshold latches a front-end runaway.
  const auto before_runaway = core.getAllKeyframes().size();
  for (int i = 1; i <= config.odom_sanity_max_consecutive; ++i) {
    auto runaway = sensor_frame(stamp + i*100000000LL, -3, 0);
    runaway.T_world_lidar.translation().x() += i*1000.0;
    output = core.processMapExtensionFrame(runaway);
    EXPECT_FALSE(output.success);
    EXPECT_FALSE(output.accepted_keyframe);
    EXPECT_EQ(core.getAllKeyframes().size(), before_runaway);
  }
  EXPECT_EQ(output.mapping_block_reason, core::MappingBlockReason::OdometryDiverged);
  if (GetParam()) {
    for (int i = 0; i < 24; ++i) {
      output = reloaded.processLocalizationFrame(sensor_frame(stamp, -3, 0));
      stamp += 100000000LL;
      if (output.relocalization_locked) break;
    }
    ASSERT_TRUE(reloaded.startMapExtension(&switch_error)) << switch_error;
    auto reset = sensor_frame(stamp, -3, 0);
    reset.T_world_lidar.translation().x() += 1000.;
    const auto before_reset = reloaded.getAllKeyframes().size();
    output = reloaded.processMapExtensionFrame(reset);
    EXPECT_FALSE(output.success);
    EXPECT_EQ(output.mapping_block_reason, core::MappingBlockReason::InvalidInput);
    EXPECT_EQ(reloaded.getAllKeyframes().size(), before_reset);
  }
  if (!HasFailure()) std::filesystem::remove_all(dir);
}
INSTANTIATE_TEST_SUITE_P(Startup, MapExtensionRoute, ::testing::Values(false, true));
} // namespace
} // namespace n3mapping
