#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/pcl_compat.h"
#include "n3map.pb.h"
#include "n3mapping/world_localizing.h"
#include "../src/core/loop_closure_async.h"
#include "../src/continuation_cleaning.h"
#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace n3mapping {
namespace {

core::LioFrame roomFrame(int64_t stamp, double x, double y,
                         bool partition = false, bool dense = false) {
  struct Face { int axis; double coordinate, low_a, high_a, low_b, high_b; };
  std::vector<Face> faces = {
    {0,-6,-4,4,0,3.5}, {0,8,-4,4,0,3.5},
    {1,-4,0,3.5,-6,8}, {1,4,0,3.5,-6,8},
    {2,0,-6,8,-4,4}, {2,3.5,-6,8,-4,4},
    {0,0,-4,-0.6,0,3.5}, {0,0,0.6,4,0,3.5},
    {0,0,-0.6,0.6,2.3,3.5},
    {0,-4,-2.5,-1.5,0,2.1}, {1,-1.5,0,2.1,-5,-4},
    {0,5,1,3,0,2.7}, {1,1,0,2.7,5,7}
  };
  if (partition) faces.push_back({0,-1.8,-0.7,0.7,0.55,1.55});
  core::LioFrame frame;
  frame.stamp.nsec = stamp;
  frame.source_frame_id = "lio_odom";
  frame.pose_valid = true;
  frame.T_world_lidar.translation() = Eigen::Vector3d(x, y, 1.0);
  auto cloud = pcl::make_shared<core::LioFrame::PointCloud>();
  for (double v = -35; v <= 55; v += dense ? .75 : 5.) {
    const double pitch = v * M_PI / 180.0;
    for (double h = 0; h < 360; h += dense ? .75 : 2.) {
      const double yaw = h * M_PI / 180.0;
      const Eigen::Vector3d ray(std::cos(pitch)*std::cos(yaw),
                                std::cos(pitch)*std::sin(yaw), std::sin(pitch));
      double nearest = 30.0;
      float intensity = 1.0f;
      for (const auto& face : faces) {
        const int a = (face.axis + 1) % 3, b = (face.axis + 2) % 3;
        if (std::abs(ray[face.axis]) < 1e-8) continue;
        const double t = (face.coordinate-frame.T_world_lidar.translation()[face.axis])/ray[face.axis];
        if (t <= 0.01 || t >= nearest) continue;
        const auto hit = (frame.T_world_lidar.translation() + t*ray).eval();
        if (hit[a] >= face.low_a && hit[a] <= face.high_a &&
            hit[b] >= face.low_b && hit[b] <= face.high_b) {
          nearest = t;
          intensity = partition && &face == &faces.back() ? 10.0f : 1.0f;
        }
      }
      if (nearest >= 30.0) continue;
      pcl::PointXYZI point;
      point.x = nearest*ray.x(); point.y = nearest*ray.y(); point.z = nearest*ray.z();
      point.intensity = intensity;
      cloud->push_back(point);
    }
  }
  frame.undistorted_cloud = cloud;
  return frame;
}

class MapExtensionRoute : public ::testing::TestWithParam<bool> {};

TEST_P(MapExtensionRoute, ExtendsBeyondLoadedGeometryAndRecoversWithoutAnotherInitialization) {
  Config config;
  config.mapping_static_start_guard_enable = false;
  config.num_threads = 4;
  config.loop_kf_gap = 1;
  config.loop_min_path_length_m = 5.0;
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
  std::map<int64_t, Eigen::Isometry3d> original_keyframes;
  std::map<int64_t, Keyframe::PointCloudT::Ptr> original_clouds;
  for (const auto& frame : core.getAllKeyframes()) {
    original_keyframes.emplace(frame->id, frame->pose_optimized);
    original_clouds.emplace(frame->id, frame->cloud);
  }
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
  const auto anchor = core.processMapExtensionFrame(sensor_frame(stamp, -3, 0));
  ASSERT_TRUE(anchor.accepted_keyframe);
  ASSERT_EQ(anchor.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  const auto overlap = core.processPendingLoopClosuresAsync();
  EXPECT_EQ(overlap.queued_keyframe_count, 1u);
  EXPECT_EQ(overlap.detected_candidate_count, 0u);
  auto geometry = core.processPendingLoopClosures();
  EXPECT_EQ(geometry.atlas_prepare_ms, 0.0);
  stamp += 1000000000LL;
  int prediction_frames = 0, prediction_keyframes = 0, lost_frames = 0;
  std::set<int64_t> tracked_keyframes{anchor.keyframe_id};
  std::vector<VerifiedLoop> accepted_loops;
  bool geometry_optimized = geometry.geometry.optimized;
  const auto record = [&](const CoreLoopClosureResult& result) {
    geometry_optimized |= result.geometry.optimized;
    EXPECT_EQ(result.atlas_prepare_ms, 0.0);
    accepted_loops.insert(accepted_loops.end(),
                          result.accepted_loops.begin(), result.accepted_loops.end());
  };
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
      EXPECT_NE(output.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
    } else if (output.accepted_keyframe && output.pose_source == PoseSource::GEOMETRICALLY_CORRECTED) {
      tracked_keyframes.insert(output.keyframe_id);
    }
    lost_frames += output.relocalization_state == RelocalizationState::LOST;
    frame.stamp.nsec += 100000000LL;
    const auto intermediate = core.processMapExtensionFrame(frame);
    EXPECT_TRUE(intermediate.success);
    EXPECT_FALSE(intermediate.accepted_keyframe);
    EXPECT_NEAR(core.getDenseOptimizedTrajectory().back().timestamp,
                frame.stamp.nsec*1e-9, 1e-9);
    record(core.processPendingLoopClosuresAsync());
  }
  const auto completed = core.processPendingLoopClosures();
  ASSERT_TRUE(completed.optimized);
  record(completed);
  EXPECT_TRUE(geometry_optimized);
  EXPECT_TRUE(std::any_of(accepted_loops.begin(), accepted_loops.end(), [&](const auto& loop) {
    return !original_poses.count(loop.match_id) && !original_poses.count(loop.query_id);
  }));
  for (const auto& loop : accepted_loops) {
    if (original_poses.count(loop.match_id)) {
      EXPECT_EQ(tracked_keyframes.count(loop.query_id), 0u);
    }
  }
  const auto latest = core.getAllKeyframes().back();
  auto prediction = sensor_frame(stamp, -3, 0);
  prediction.undistorted_cloud->resize(1);
  const auto predicted = core.processMapExtensionFrame(prediction);
  ASSERT_EQ(predicted.pose_source, PoseSource::ODOM_PREDICTED);
  EXPECT_TRUE(predicted.T_world_lidar.isApprox(
      latest->pose_optimized * latest->pose_odom.inverse() * prediction.T_world_lidar, 1e-10));
  stamp += 1000000000LL;
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

  ASSERT_TRUE(core.saveMap(extended_path));
  const auto final_poses = core.getOptimizedPoses();
  for (const auto& [id, pose] : original_poses) {
    EXPECT_EQ(final_poses.at(id).matrix(), pose.matrix());
    EXPECT_EQ(core.getKeyframe(id)->pose_optimized.matrix(), original_keyframes.at(id).matrix());
    EXPECT_EQ(core.getKeyframe(id)->cloud, original_clouds.at(id));
  }
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
TEST(MapContinuation, DescriptorReturnRepairsDriftBeyondSpatialGate) {
  Config config;
  config.num_threads = 2;
  config.loop_min_path_length_m = 5.0;
  config.rhpd_num_candidates = 4;
  config.keyframe_distance_threshold = 0.25;
  constexpr int64_t first_new = 15;
  constexpr int64_t query_id = first_new + 240;
  constexpr double drift = 9.6;
  constexpr double return_x = -3.4;

  GraphOptimizer graph(config);
  KeyframeManager frames(config);
  LoopDetector detector(config);
  PointCloudMatcher matcher(config);
  std::vector<Keyframe::Ptr> keyframes;
  std::vector<Keyframe::Ptr> loaded;
  std::vector<std::pair<int64_t, Eigen::Isometry3d>> poses;
  std::vector<EdgeInfo> edges;
  std::set<int64_t> fixed;
  const Eigen::Matrix<double, 6, 6> information =
      Eigen::Matrix<double, 6, 6>::Identity() * 100;
  const auto stationary = roomFrame(0, return_x + 1.2, 0);
  const auto missing_scan = pcl::make_shared<core::LioFrame::PointCloud>();
  for (int64_t id = 0; id <= query_id; ++id) {
    const bool reference = id < first_new;
    auto frame = reference ? roomFrame(0, -4.8 + 0.2 * id, 0) : stationary;
    if (!reference) {
      const auto steps_to_return = query_id - id;
      if (steps_to_return < 6)
        frame = roomFrame(0, return_x + 0.2 * steps_to_return, 0);
      else if (id > first_new + 16)
        frame.undistorted_cloud = missing_scan;
      frame.T_world_lidar.translation().y() = drift * (id - first_new) / (query_id - first_new);
    }
    auto keyframe = Keyframe::create(id, id, frame.T_world_lidar, frame.undistorted_cloud);
    keyframe->session_id = reference ? 0 : 1;
    keyframes.push_back(keyframe);
    poses.emplace_back(id, frame.T_world_lidar);
    if (reference) {
      loaded.push_back(keyframe);
      fixed.insert(id);
    }
    if (id > 0) {
      const Eigen::Isometry3d delta = poses[id - 1].second.inverse() * poses[id].second;
      const auto type = id == first_new ? EdgeType::SESSION_ANCHOR : EdgeType::ODOMETRY;
      edges.push_back({id - 1, id, delta, information, type});
    }
    if (reference || id == query_id) {
      detector.addDescriptor(id, frame.undistorted_cloud);
      detector.addRHPD(id, frame.undistorted_cloud);
    }
  }
  MapSessionInfo old_session, new_session;
  old_session.source_frame_id = "reference";
  new_session.id = 1;
  new_session.source_frame_id = "continuation";
  new_session.start_timestamp = first_new;
  ASSERT_TRUE(frames.loadKeyframes(keyframes, {old_session, new_session}));
  std::map<int64_t, Keyframe::Ptr> by_id;
  for (const auto& frame : frames.getAllKeyframes()) {
    frame->is_from_loaded_map = fixed.count(frame->id) != 0;
    by_id.emplace(frame->id, frame);
  }
  ASSERT_TRUE(graph.loadGraph(poses, edges, {}, {}, true));
  graph.setFixedNodes(fixed);

  LocalizationAtlas atlas(config, matcher);
  ASSERT_TRUE(atlas.prepare(loaded));
  WorldLocalizing localizer(config, frames, detector, matcher);
  localizer.installOptimizedReference(atlas, Eigen::Isometry3d::Identity());
  DetectedLoopQuery query{query_id, detector.detectLoopCandidates(
      query_id, by_id, [&](int64_t id) { return fixed.count(id) != 0; })};
  ASSERT_FALSE(query.candidates.empty());
  for (const auto& candidate : query.candidates) {
    ASSERT_GT((poses.at(candidate.match_id).second.translation() -
               poses.back().second.translation()).norm(), 8.0);
    EXPECT_TRUE(candidate.fromRHPD() || candidate.fromHeading());
    EXPECT_DOUBLE_EQ(candidate.path_length_m, 0.0);
  }

  Config no_atlas_config = config;
  no_atlas_config.reloc_atlas_enable = false;
  no_atlas_config.loop_debug_enable = true;
  WorldLocalizing no_atlas(no_atlas_config, frames, detector, matcher);
  auto missing_visibility = no_atlas.snapshotLoadedMapVisibility();
  ASSERT_FALSE(missing_visibility);
  LoopCandidate same_session;
  same_session.query_id = first_new + 10;
  same_session.match_id = first_new + 1;
  same_session.source_flags = LoopCandidate::SOURCE_SPATIAL;
  auto no_atlas_graph = graph.cloneCommitted();
  LoopClosureManager no_atlas_manager(no_atlas_config);
  const auto no_atlas_result = processLoopCandidates(
      no_atlas_config, frames, matcher, *no_atlas_graph, no_atlas_manager,
      {query, {same_session.query_id, {same_session}}}, missing_visibility);
  EXPECT_TRUE(std::any_of(no_atlas_result.candidates.begin(), no_atlas_result.candidates.end(),
      [&](const auto& event) {
        return event.candidate.query_id == query_id &&
               event.reject_reason == "loaded_map_visibility_nonpositive";
      }));
  ASSERT_FALSE(no_atlas_result.summary.accepted_loops.empty());
  for (const auto& loop : no_atlas_result.summary.accepted_loops) {
    EXPECT_EQ(loop.query_id, same_session.query_id);
    EXPECT_EQ(loop.match_id, same_session.match_id);
  }

  AsyncLoopJob worker(config, frames, graph.cloneCommitted(), {query},
                      localizer.snapshotLoadedMapVisibility());
  worker.run();
  ASSERT_TRUE(worker.outcome.summary.optimized);
  EXPECT_TRUE(worker.outcome.summary.geometry.optimized);
  ASSERT_FALSE(worker.outcome.summary.accepted_loops.empty());
  EXPECT_EQ(worker.outcome.summary.atlas_prepare_ms, 0.0);
  for (const auto& loop : worker.outcome.summary.accepted_loops) {
    EXPECT_EQ(loop.query_id, query_id);
    EXPECT_EQ(fixed.count(loop.match_id), 1u);
    EXPECT_EQ(loop.consensus_shadow_decision, "commit");
  }
  for (int64_t id : fixed)
    EXPECT_EQ(worker.graph->getOptimizedPose(id).matrix(), graph.getOptimizedPose(id).matrix());
  EXPECT_NEAR(worker.graph->getOptimizedPose(query_id).translation().x(), return_x, 0.15);
  const auto corrected = worker.graph->getOptimizedPose(query_id).translation().eval();
  EXPECT_NEAR(corrected.y(), 0.0, 0.15);
  RecordProperty("return_lateral_error_m", std::to_string(std::abs(corrected.y())));
  RecordProperty("inserted_cross_session_edges", worker.outcome.summary.edge_count);

  // The live graph has more frames and new plane associations by the time the job returns.
  for (int64_t id = query_id + 1; id <= query_id + 3; ++id) {
    auto frame = roomFrame(0, return_x - 0.2 * (id - query_id), 0);
    frame.T_world_lidar.translation().y() = drift * (id - first_new) / (query_id - first_new);
    auto keyframe = Keyframe::create(id, id, frame.T_world_lidar, frame.undistorted_cloud);
    keyframe->session_id = new_session.id;
    keyframes.push_back(keyframe);
    const Eigen::Isometry3d delta = poses.back().second.inverse() * frame.T_world_lidar;
    poses.emplace_back(id, frame.T_world_lidar);
    ASSERT_TRUE(graph.addSessionOdometryEdge({id - 1, id, delta, information, EdgeType::ODOMETRY},
                                            frame.T_world_lidar));
    ASSERT_TRUE(graph.incrementalOptimize());
  }
  ASSERT_TRUE(frames.loadKeyframes(keyframes, {old_session, new_session}));
  for (const auto& frame : frames.getAllKeyframes())
    frame->is_from_loaded_map = fixed.count(frame->id) != 0;
  GeometryOptimizationStats stats;
  const auto stale_planes = buildSharedPlanes(frames.getAllKeyframes(), stats);
  ASSERT_TRUE(std::any_of(stale_planes.begin(), stale_planes.end(), [&](const auto& plane) {
    return std::any_of(plane.observations.begin(), plane.observations.end(),
        [&](const auto& obs) { return obs.node_id > query_id; });
  }));
  ASSERT_TRUE(graph.replaceSharedPlanes(stale_planes, false, stats));
  AsyncLoopJob retry(config, frames, graph.cloneCommitted(), {query},
                     localizer.snapshotLoadedMapVisibility());
  retry.outcome = std::move(worker.outcome);
  retry.rebase();
  ASSERT_FALSE(retry.outcome.summary.accepted_loops.empty());
  EXPECT_TRUE(retry.outcome.summary.geometry.optimized);
  EXPECT_EQ(retry.outcome.summary.atlas_prepare_ms, 0.0);
  EXPECT_NEAR(retry.graph->getOptimizedPose(query_id).translation().y(), 0.0, 0.15);
  double max_tail_error = 0.;
  for (int64_t id = query_id + 1; id <= poses.back().first; ++id) {
    const Eigen::Isometry3d actual =
        retry.graph->getOptimizedPose(id - 1).inverse() * retry.graph->getOptimizedPose(id);
    const Eigen::Isometry3d measured = poses[id - 1].second.inverse() * poses[id].second;
    max_tail_error = std::max(max_tail_error, (actual.translation() - measured.translation()).norm());
  }
  EXPECT_LT(max_tail_error, 0.05);
  for (int64_t id : fixed)
    EXPECT_EQ(retry.graph->getOptimizedPose(id).matrix(), graph.getOptimizedPose(id).matrix());
  RecordProperty("rebase_tail_translation_error_m", std::to_string(max_tail_error));
  RecordProperty("rebased_cross_session_edges", retry.outcome.summary.edge_count);
}

TEST(ContinuationCleaning, NeedsRepeatedBackgroundAndKeepsOcclusionEdgesAndDistantFrames) {
  const Eigen::Vector3d removed(2., .02, .01), missing(0., 3., 0.);
  const Eigen::Vector3d seam(-3., .01, .02), transient(0., -3., 0.), foreground(0., 0., 3.);
  const Eigen::Vector3d observed(0., -3., 3.);
  const auto make_frame = [](int64_t id, bool loaded, const std::vector<Eigen::Vector3d>& points) {
    auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
    for (const auto& xyz : points) {
      pcl::PointXYZI point;
      point.getVector3fMap() = xyz.cast<float>();
      point.intensity = cloud->size() + 1;
      cloud->push_back(point);
    }
    auto frame = Keyframe::create(id, id, Eigen::Isometry3d::Identity(), cloud);
    frame->is_from_loaded_map = loaded;
    return frame;
  };
  const auto old = make_frame(0, true, {removed, missing, seam, transient, foreground, observed});
  const auto distant = make_frame(1, true, {removed});
  distant->pose_optimized.translation().x() = 80.;
  const auto extreme = make_frame(2, true, {removed});
  std::vector<Keyframe::Ptr> frames{old, distant, extreme};
  for (int64_t id = 3; id <= 6; ++id) {
    std::vector<Eigen::Vector3d> rays{2. * removed, 2. * seam, {-3., -.01, .02},
                                     2. * foreground, foreground};
    if (id < 5) rays.push_back(2. * transient);
    rays.push_back(id < 6 ? Eigen::Vector3d(2. * observed) : observed);
    frames.push_back(make_frame(id, false, rays));
  }
  Config config;
  const auto cleaning = computeContinuationCleaning(config, frames);
  EXPECT_EQ(cleaning.inspected_keyframes, 2u);
  EXPECT_EQ(cleaning.skipped_empty_keyframes, 1u);
  EXPECT_EQ(cleaning.removed_points, 1u);
  ASSERT_EQ(cleaning.clouds.size(), 1u);
  ASSERT_EQ(cleaning.clouds.at(old->id)->size(), 5u);
  for (int i = 0; i < 5; ++i) {
    const auto& point = cleaning.clouds.at(old->id)->at(i);
    EXPECT_EQ(point.getVector3fMap(), old->cloud->at(i + 1).getVector3fMap());
    EXPECT_EQ(point.intensity, old->cloud->at(i + 1).intensity);
  }
  frames.resize(5);
  EXPECT_EQ(computeContinuationCleaning(config, frames).removed_points, 0u);
}

class SaveContinuationCleaning : public ::testing::TestWithParam<bool> {};

TEST_P(SaveContinuationCleaning, SavesAndReloadsOneConsistentMap) {
  struct TestDirectory {
    char path[sizeof("/tmp/n3m-cleaning-XXXXXX")] = "/tmp/n3m-cleaning-XXXXXX";
    ~TestDirectory() { std::filesystem::remove_all(path); }
  } directory;
  ASSERT_NE(mkdtemp(directory.path), nullptr);
  const std::filesystem::path root(directory.path);
  const auto original_file = (root / "original.pbstream").string();
  const auto clean_file = (root / "clean.pbstream").string();
  const auto pcd_file = (root / "global_map.pcd").string();

  Config config;
  config.num_threads = 2;
  config.mapping_static_start_guard_enable = false;
  config.keyframe_distance_threshold = .18;
  config.submap_shadow_enable = true;
  config.submap_max_keyframes = 3;
  const auto partition_points = [](const Keyframe::PointCloudT& cloud) {
    return std::count_if(cloud.begin(), cloud.end(),
        [](const auto& p) { return p.intensity == 10.f; });
  };
  N3MappingCore mapping(config, N3MappingCore::WorldCloudOutput::Disabled);
  int64_t stamp = 1000000000LL;
  for (int i = 0; i < 5; ++i) {
    auto frame = roomFrame(stamp, -4.8 + .28 * i, 0, true);
    if (GetParam() && i == 0) {
      auto& cloud = *frame.undistorted_cloud;
      cloud.erase(std::remove_if(cloud.begin(), cloud.end(),
          [](const auto& point) { return point.intensity != 10.f; }), cloud.end());
    }
    ASSERT_TRUE(mapping.processMappingFrame(frame).accepted_keyframe);
    stamp += 1000000000LL;
  }
  ASSERT_TRUE(mapping.saveMap(original_file));
  N3MappingCore extension(config, N3MappingCore::WorldCloudOutput::Disabled);
  ASSERT_TRUE(extension.loadMap(original_file));
  const auto original_poses = extension.getOptimizedPoses();
  std::map<int64_t, Keyframe::PointCloudT::Ptr> original_clouds;
  for (const auto& frame : extension.getAllKeyframes())
    original_clouds.emplace(frame->id, frame->cloud);
  ASSERT_GT(partition_points(*original_clouds.at(0)), 0);

  for (int i = 0; i < 24 && !extension.startMapExtension(nullptr); ++i) {
    extension.processLocalizationFrame(roomFrame(stamp, -4.5, 0, false, true));
    stamp += 1000000000LL;
  }
  ASSERT_TRUE(extension.startMapExtension(nullptr));
  for (int i = 0; i < 7; ++i) {
    auto output = extension.processMapExtensionFrame(
        roomFrame(stamp, -4.5 + .23 * i, -.22 + .07 * i, false, true));
    ASSERT_TRUE(output.accepted_keyframe) << i << " " << output.relocalization_decision;
    stamp += 1000000000LL;
  }
  for (const auto& [id, cloud] : original_clouds)
    EXPECT_EQ(extension.getKeyframe(id)->cloud, cloud);

  extension.processPendingLoopClosures();
  const auto before_save = extension.getOptimizedPoses();
  EXPECT_FALSE(extension.saveMap((root / "unavailable" / "map.pbstream").string()));
  for (const auto& [id, cloud] : original_clouds)
    EXPECT_EQ(extension.getKeyframe(id)->cloud, cloud);
  const auto before_revision = extension.mapRevision();
  ASSERT_TRUE(extension.saveMap(clean_file));
  ASSERT_TRUE(extension.saveGlobalMap(pcd_file));
  EXPECT_EQ(extension.mapRevision().structure_revision, before_revision.structure_revision + 1);
  const auto after_save = extension.getOptimizedPoses();
  for (const auto& [id, value] : before_save)
    EXPECT_EQ(after_save.at(id).matrix(), value.matrix());
  for (const auto& [id, value] : original_poses)
    EXPECT_EQ(after_save.at(id).matrix(), value.matrix());

  std::size_t removed = 0;
  for (const auto& [id, cloud] : original_clouds) {
    const auto kept = extension.getKeyframe(id)->cloud;
    if (GetParam() && id == 0) {
      EXPECT_EQ(kept, cloud);
      continue;
    }
    EXPECT_EQ(partition_points(*kept), 0);
    EXPECT_EQ(kept->size(), cloud->size() - partition_points(*cloud));
    auto actual = kept->begin();
    for (const auto& point : *cloud) {
      if (point.intensity == 10.f) continue;
      ASSERT_NE(actual, kept->end());
      EXPECT_EQ(actual->getVector3fMap(), point.getVector3fMap());
      EXPECT_EQ(actual->intensity, point.intensity);
      ++actual;
    }
    removed += cloud->size() - kept->size();
  }
  ASSERT_GT(removed, 0u);

  Keyframe::PointCloudT pcd;
  ASSERT_EQ(pcl::io::loadPCDFile(pcd_file, pcd), 0);
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  ASSERT_TRUE(atlas.load(clean_file, LocalizationAtlas::defaultAtlasPath(clean_file)));
  const auto contains_partition = [](const Keyframe::PointCloudT& cloud) {
    return std::any_of(cloud.begin(), cloud.end(), [](const auto& point) {
      return std::abs(point.x + 1.8) < .04 && std::abs(point.y) < .73 &&
             point.z > .52 && point.z < 1.58;
    });
  };
  EXPECT_EQ(contains_partition(pcd), GetParam());
  EXPECT_EQ(contains_partition(*atlas.globalMap()), GetParam());
  N3MappingCore reopened(config, N3MappingCore::WorldCloudOutput::Disabled);
  ASSERT_TRUE(reopened.loadMap(clean_file));
  ASSERT_EQ(reopened.getAllKeyframes().size(), extension.getAllKeyframes().size());
  for (const auto& frame : reopened.getAllKeyframes()) {
    EXPECT_EQ(frame->cloud->size(), extension.getKeyframe(frame->id)->cloud->size());
    EXPECT_TRUE(frame->pose_optimized.matrix().isApprox(after_save.at(frame->id).matrix(), 1e-12));
  }
  core::BackendOutput localized;
  for (int i = 0; i < 24 && !localized.relocalization_locked; ++i) {
    localized = reopened.processLocalizationFrame(roomFrame(stamp, -4.5, 0));
    stamp += 1000000000LL;
  }
  ASSERT_TRUE(localized.relocalization_locked);
  const auto tracked = reopened.processLocalizationFrame(roomFrame(stamp, -4.25, 0));
  EXPECT_EQ(tracked.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  EXPECT_LT((tracked.T_world_lidar.translation() - Eigen::Vector3d(-4.25, 0., 1.)).norm(), .15);
  RecordProperty("removed_obstacle_points", removed);
}

INSTANTIATE_TEST_SUITE_P(AssetCompatibility, SaveContinuationCleaning, ::testing::Values(false, true));
INSTANTIATE_TEST_SUITE_P(Startup, MapExtensionRoute, ::testing::Values(false, true));
}
}
