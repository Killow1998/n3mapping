#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/pcl_compat.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/world_localizing.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <pcl/common/transforms.h>
#include <random>
#include <string>
#include <vector>

namespace n3mapping {
namespace test {

class WorldLocalizingTest : public ::testing::Test {
protected:
  void SetUp() override {
    config_.keyframe_distance_threshold = 1.0;
    config_.keyframe_angle_threshold = 0.5;
    config_.gicp_downsampling_resolution = 0.5;
    config_.gicp_max_correspondence_distance = 2.0;
    config_.gicp_max_iterations = 30;
    config_.gicp_fitness_threshold = 0.5;
    config_.sc_dist_threshold = 0.3;
    config_.sc_num_exclude_recent = 5;
    config_.sc_num_candidates = 3;
    config_.reloc_num_candidates = 5;
    config_.reloc_sc_dist_threshold = 0.4;
    config_.reloc_search_radius = 15.0;
    config_.reloc_max_track_failures = 5;
    config_.reloc_temporal_window_size = 3;
    config_.reloc_lock_log_likelihood_threshold = -100.0;
    config_.reloc_min_confidence = 0.05;
    config_.reloc_min_inlier_ratio = 0.0;
    config_.reloc_ambiguity_min_basin_separation = 100.0;
    config_.reloc_static_agg_enable = false;
    config_.rhpd_enabled = true;
    config_.rhpd_dist_threshold = 100.0;
    config_.rhpd_num_candidates = 5;
    config_.rhpd_yaw_hypotheses = 4;
    config_.num_threads = 2;

    keyframe_manager_ = std::make_unique<KeyframeManager>(config_);
    loop_detector_ = std::make_unique<LoopDetector>(config_);
    matcher_ = std::make_unique<PointCloudMatcher>(config_);
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr
  generateCorridorCloud(const Eigen::Isometry3d &pose,
                        double corridor_width = 3.0) {
    auto world_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
    const float center_x = static_cast<float>(pose.translation().x());

    for (float x = center_x - 5.0f; x <= center_x + 5.0f; x += 0.2f) {
      for (float z = 0.0f; z <= 2.5f; z += 0.3f) {
        pcl::PointXYZI pt_left;
        pt_left.x = x + noise(rng);
        pt_left.y = -corridor_width / 2.0f + noise(rng);
        pt_left.z = z + noise(rng);
        pt_left.intensity = 50.0f;
        world_cloud->push_back(pt_left);
        pcl::PointXYZI pt_right;
        pt_right.x = x + noise(rng);
        pt_right.y = corridor_width / 2.0f + noise(rng);
        pt_right.z = z + noise(rng);
        pt_right.intensity = 50.0f;
        world_cloud->push_back(pt_right);
      }
    }
    for (float x = center_x - 5.0f; x <= center_x + 5.0f; x += 0.3f) {
      for (float y = -corridor_width / 2.0f; y <= corridor_width / 2.0f;
           y += 0.3f) {
        pcl::PointXYZI pt;
        pt.x = x + noise(rng);
        pt.y = y + noise(rng);
        pt.z = noise(rng);
        pt.intensity = 30.0f;
        world_cloud->push_back(pt);
      }
    }

    // A fixed vertical plate makes the success fixture observable in both
    // position and heading. Without it, an ideal corridor has a genuine
    // 180-degree symmetry.
    constexpr float kLandmarkX = 8.35f;
    if (std::abs(kLandmarkX - center_x) <= 5.0f) {
      for (float y = -0.9f; y <= -0.1f; y += 0.12f) {
        for (float z = 0.2f; z <= 2.3f; z += 0.12f) {
          pcl::PointXYZI point;
          point.x = kLandmarkX + noise(rng) * 0.2f;
          point.y = y + noise(rng) * 0.2f;
          point.z = z + noise(rng) * 0.2f;
          point.intensity = 90.0f;
          world_cloud->push_back(point);
        }
      }
    }

    auto body_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::transformPointCloud(*world_cloud, *body_cloud,
                             pose.inverse().matrix().cast<float>());

    body_cloud->width = body_cloud->size();
    body_cloud->height = 1;
    body_cloud->is_dense = true;
    return body_cloud;
  }

  void buildTestMap(int num_keyframes = 10, double spacing = 2.0,
                    double origin_x = 0.0) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation().x() = origin_x;
    for (int i = 0; i < num_keyframes; ++i) {
      auto cloud = generateCorridorCloud(pose);
      int64_t kf_id = keyframe_manager_->addKeyframe(i * 0.1, pose, cloud);
      loop_detector_->addDescriptor(kf_id, cloud);
      auto rhpd = loop_detector_->addRHPD(kf_id, cloud);
      auto kf = keyframe_manager_->getKeyframe(kf_id);
      ASSERT_NE(kf, nullptr);
      kf->rhpd_descriptor = rhpd;
      pose.translation().x() += spacing;
    }
  }

  Config config_;
  std::unique_ptr<KeyframeManager> keyframe_manager_;
  std::unique_ptr<LoopDetector> loop_detector_;
  std::unique_ptr<PointCloudMatcher> matcher_;
};

std::vector<std::string> readDebugLines(const std::filesystem::path &path) {
  std::vector<std::string> lines;
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }
  return lines;
}

TEST_F(WorldLocalizingTest, BasicConstruction) {
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  EXPECT_FALSE(reloc.isRelocalized());
  EXPECT_EQ(reloc.getLastMatchedKeyframeId(), -1);
  EXPECT_TRUE(
      reloc.getMapToOdomTransform().isApprox(Eigen::Isometry3d::Identity()));
}

TEST_F(WorldLocalizingTest, RelocalizationEmptyMap) {
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  auto cloud = generateCorridorCloud(pose);

  RelocResult result = reloc.relocalize(cloud, pose);

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.decision, "missing_keyframes");
  EXPECT_EQ(result.state, RelocalizationState::SEARCHING);
  EXPECT_EQ(result.pose_source, PoseSource::NONE);
  EXPECT_FALSE(reloc.isRelocalized());
}

TEST_F(WorldLocalizingTest, RegistrationProbeRequiresLoadedAtlas) {
  buildTestMap(3);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  const Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  const auto result =
      reloc.probeRegistrationSeeds(generateCorridorCloud(pose), pose, pose);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.error, "atlas_required");
  EXPECT_TRUE(result.attempts.empty());
}

TEST_F(WorldLocalizingTest, RelocalizationDebugWritesRejectPath) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "n3mapping_reloc_debug_reject";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path debug_path = dir / "relocalization_debug.jsonl";
  config_.reloc_debug_enable = true;
  config_.reloc_debug_path = debug_path.string();

  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  auto cloud = generateCorridorCloud(pose);

  RelocResult result = reloc.relocalize(cloud, pose);

  EXPECT_FALSE(result.success);
  const auto lines = readDebugLines(debug_path);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("\"record_type\":\"relocalize\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"lock_result\":\"rejected\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"reject_reason\":\"missing_keyframes\""),
            std::string::npos);

  std::filesystem::remove_all(dir);
}

TEST_F(WorldLocalizingTest, RelocalizationDebugWritesTrackingFailurePath) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "n3mapping_reloc_debug_tracking";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path debug_path = dir / "relocalization_debug.jsonl";
  config_.reloc_debug_enable = true;
  config_.reloc_debug_path = debug_path.string();
  config_.reloc_search_radius = 1.0;

  buildTestMap(3, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  reloc.setMapToOdomTransform(Eigen::Isometry3d::Identity());

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = 100.0;
  auto cloud = generateCorridorCloud(pose);

  RelocResult result = reloc.trackLocalization(cloud, pose);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.state, RelocalizationState::DEGRADED_TRACKING);
  EXPECT_EQ(result.pose_source, PoseSource::ODOM_PREDICTED);
  EXPECT_EQ(result.decision, "nearest_keyframe_missing");
  const auto lines = readDebugLines(debug_path);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("\"record_type\":\"tracking\""), std::string::npos);
  EXPECT_NE(lines[0].find("\"nearest_kf_id\":-1"), std::string::npos);
  EXPECT_NE(lines[0].find("\"reject_reason\":\"nearest_keyframe_missing\""),
            std::string::npos);

  std::filesystem::remove_all(dir);
}

TEST_F(WorldLocalizingTest, RelocalizationDebugWritesQueryCloudDiagnostics) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                    "n3mapping_reloc_debug_query_cloud";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path debug_path = dir / "relocalization_debug.jsonl";
  config_.reloc_debug_enable = true;
  config_.reloc_debug_path = debug_path.string();
  config_.reloc_static_agg_enable = true;
  config_.reloc_static_agg_max_frames = 3;
  config_.reloc_static_agg_min_frames = 1;
  config_.reloc_static_agg_max_translation = 0.01;
  config_.reloc_lock_min_margin = 0.1;

  buildTestMap(6, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = 4.0;
  auto cloud = generateCorridorCloud(pose);
  (void)reloc.relocalize(cloud, pose);

  Eigen::Isometry3d moved_pose = pose;
  moved_pose.translation().x() += 0.5;
  auto moved_cloud = generateCorridorCloud(moved_pose);
  (void)reloc.relocalize(moved_cloud, moved_pose);

  const auto lines = readDebugLines(debug_path);
  ASSERT_GE(lines.size(), 2u);
  const std::string &latest = lines.back();
  EXPECT_NE(latest.find("\"record_type\":\"relocalize\""), std::string::npos);
  EXPECT_NE(latest.find("\"query_mode\":\"stationary\""), std::string::npos);
  EXPECT_NE(latest.find("\"query_frame_count\":1"), std::string::npos);
  EXPECT_NE(latest.find("\"motion_query_mode\":\"motion_submap\""),
            std::string::npos);
  EXPECT_NE(latest.find("\"motion_query_frame_count\":2"), std::string::npos);
  EXPECT_NE(latest.find("\"motion_query_candidate_count\":"),
            std::string::npos);
  EXPECT_EQ(latest.find("\"query_candidate_count\":0"), std::string::npos);

  std::filesystem::remove_all(dir);
}

TEST_F(WorldLocalizingTest, GlobalRelocalizationSuccess) {
  config_.reloc_lock_min_margin = 0.1;
  buildTestMap(10, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d query_pose = Eigen::Isometry3d::Identity();
  query_pose.translation().x() = 8.0;
  auto cloud = generateCorridorCloud(query_pose);

  RelocResult result;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    result = reloc.relocalize(cloud, query_pose);
    if (i + 1 < config_.reloc_temporal_window_size) {
      EXPECT_FALSE(result.success);
      EXPECT_EQ(result.state, RelocalizationState::REGION_HYPOTHESIS);
      EXPECT_EQ(result.pose_source, PoseSource::NONE);
    }
  }

  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.state, RelocalizationState::FULL_6DOF_LOCKED);
  EXPECT_EQ(result.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  EXPECT_TRUE(reloc.isRelocalized());
  EXPECT_GE(result.seed_keyframe_id, 0);
  EXPECT_GE(result.support_keyframe_id, 0);
  EXPECT_EQ(result.matched_keyframe_id, result.support_keyframe_id);
  EXPECT_GE(result.matched_keyframe_id, 0);
  EXPECT_GT(result.confidence, 0.0);
  double position_error =
      (result.pose_in_map.translation() - query_pose.translation()).norm();
  EXPECT_LT(position_error, 3.0);
}

TEST_F(WorldLocalizingTest, MovingAcrossKeyframesKeepsPhysicalWinnerStreak) {
  config_.reloc_lock_min_margin = 0.1;
  config_.reloc_lock_min_winner_streak = 3;
  config_.reloc_lock_min_converged_updates = 3;
  constexpr double kMapOriginX = -2800.0;
  buildTestMap(10, 2.0, kMapOriginX);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d fake_map_odom = Eigen::Isometry3d::Identity();
  fake_map_odom.translation() = Eigen::Vector3d(20.0, -15.0, 0.0);
  fake_map_odom.linear() =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  RelocResult result;
  for (double x : {7.4, 8.4, 9.4}) {
    Eigen::Isometry3d query_pose = Eigen::Isometry3d::Identity();
    query_pose.translation().x() = kMapOriginX + x;
    const Eigen::Isometry3d odom_pose = fake_map_odom.inverse() * query_pose;
    result = reloc.relocalize(generateCorridorCloud(query_pose), odom_pose);
  }

  ASSERT_TRUE(result.success);
  EXPECT_EQ(result.decision, "accepted");
  EXPECT_LT((result.pose_in_map.translation() -
             Eigen::Vector3d(kMapOriginX + 9.4, 0.0, 0.0))
                .norm(),
            3.0);
}

TEST_F(WorldLocalizingTest, DuplicatedGeometryDoesNotClaimAUniquePlace) {
  config_.reloc_lock_min_margin = 0.1;
  config_.reloc_ambiguity_min_basin_separation = 3.0;

  Eigen::Isometry3d observation_pose = Eigen::Isometry3d::Identity();
  observation_pose.translation().x() = 8.0;
  auto repeated_observation = generateCorridorCloud(observation_pose);
  for (double x : {0.0, 20.0}) {
    Eigen::Isometry3d map_pose = Eigen::Isometry3d::Identity();
    map_pose.translation().x() = x;
    const int64_t kf_id =
        keyframe_manager_->addKeyframe(0.0, map_pose, repeated_observation);
    loop_detector_->addDescriptor(kf_id, repeated_observation);
    auto rhpd = loop_detector_->addRHPD(kf_id, repeated_observation);
    auto keyframe = keyframe_manager_->getKeyframe(kf_id);
    ASSERT_NE(keyframe, nullptr);
    keyframe->rhpd_descriptor = rhpd;
  }
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  const Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();

  RelocResult result;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    result = reloc.relocalize(repeated_observation, odom_pose);
  }

  EXPECT_FALSE(result.success);
  EXPECT_FALSE(reloc.isRelocalized());
}

TEST_F(WorldLocalizingTest, RelocalizationWindowOneCanLock) {
  config_.reloc_temporal_window_size = 1;
  config_.reloc_lock_min_winner_streak = 3;
  config_.reloc_lock_min_converged_updates = 3;
  config_.reloc_lock_min_margin = 0.1;
  buildTestMap(10, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d query_pose = Eigen::Isometry3d::Identity();
  query_pose.translation().x() = 8.0;
  auto cloud = generateCorridorCloud(query_pose);

  RelocResult result = reloc.relocalize(cloud, query_pose);

  ASSERT_TRUE(result.success);
  EXPECT_TRUE(reloc.isRelocalized());
  EXPECT_GE(result.seed_keyframe_id, 0);
  EXPECT_GE(result.support_keyframe_id, 0);
  EXPECT_EQ(result.matched_keyframe_id, result.support_keyframe_id);
  EXPECT_GE(result.matched_keyframe_id, 0);
}

TEST_F(WorldLocalizingTest, TrackLocalization) {
  buildTestMap(10, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d initial_pose = Eigen::Isometry3d::Identity();
  initial_pose.translation().x() = 4.0;
  auto initial_cloud = generateCorridorCloud(initial_pose);

  RelocResult initial_result = reloc.relocalize(initial_cloud, initial_pose);

  if (!initial_result.success) {
    reloc.setMapToOdomTransform(Eigen::Isometry3d::Identity());
  }

  Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
  odom_pose.translation().x() = 6.0;
  auto track_cloud = generateCorridorCloud(odom_pose);

  RelocResult track_result = reloc.trackLocalization(track_cloud, odom_pose);
  EXPECT_TRUE(track_result.success);
  EXPECT_EQ(track_result.state, RelocalizationState::FULL_6DOF_LOCKED);
  EXPECT_EQ(track_result.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  EXPECT_EQ(track_result.decision, "tracking_geometric");
}

TEST_F(WorldLocalizingTest, LoadedMapTrackingIgnoresExtensionKeyframes) {
  buildTestMap(5, 2.0);
  for (const auto &keyframe : keyframe_manager_->getAllKeyframes()) {
    ASSERT_NE(keyframe, nullptr);
    keyframe->is_from_loaded_map = true;
  }

  Eigen::Isometry3d extension_pose = Eigen::Isometry3d::Identity();
  extension_pose.translation().x() = 40.0;
  const auto extension_cloud = generateCorridorCloud(extension_pose);
  const int64_t extension_id = keyframe_manager_->addKeyframe(
      10.0, extension_pose, extension_cloud);
  ASSERT_FALSE(keyframe_manager_->getKeyframe(extension_id)->is_from_loaded_map);

  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  reloc.setMapToOdomTransform(Eigen::Isometry3d::Identity());
  const RelocResult result = reloc.trackLoadedMap(
      extension_cloud, extension_pose);

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.decision, "nearest_keyframe_missing");
  EXPECT_NE(result.matched_keyframe_id, extension_id);
}

TEST_F(WorldLocalizingTest, LoadedMapTrackingAcceptsLocalGeometricEvidence) {
  buildTestMap(6, 2.0);
  for (const auto &keyframe : keyframe_manager_->getAllKeyframes()) {
    ASSERT_NE(keyframe, nullptr);
    keyframe->is_from_loaded_map = true;
  }

  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  reloc.setMapToOdomTransform(Eigen::Isometry3d::Identity());
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().x() = 8.0;
  const RelocResult result = reloc.trackLoadedMap(
      generateCorridorCloud(pose), pose);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.state, RelocalizationState::FULL_6DOF_LOCKED);
  EXPECT_EQ(result.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
  EXPECT_EQ(result.decision, "loaded_map_tracking_geometric");
  EXPECT_LT((result.pose_in_map.translation() - pose.translation()).norm(),
            0.25);
}

TEST_F(WorldLocalizingTest, Reset) {
  buildTestMap(5, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
  T_map_odom.translation().x() = 1.0;
  reloc.setMapToOdomTransform(T_map_odom);
  EXPECT_TRUE(reloc.isRelocalized());
  reloc.reset();
  EXPECT_FALSE(reloc.isRelocalized());
  EXPECT_EQ(reloc.getLastMatchedKeyframeId(), -1);
  EXPECT_TRUE(
      reloc.getMapToOdomTransform().isApprox(Eigen::Isometry3d::Identity()));
}

TEST_F(WorldLocalizingTest, SetMapToOdomTransform) {
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
  T_map_odom.translation() = Eigen::Vector3d(1.0, 2.0, 0.0);
  T_map_odom.rotate(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
  reloc.setMapToOdomTransform(T_map_odom);
  EXPECT_TRUE(reloc.isRelocalized());
  EXPECT_TRUE(reloc.getMapToOdomTransform().isApprox(T_map_odom, 1e-6));
}

TEST_F(WorldLocalizingTest, EmptyCloudInput) {
  buildTestMap(5, 2.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  auto empty_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

  RelocResult result =
      reloc.relocalize(empty_cloud, Eigen::Isometry3d::Identity());
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.decision, "empty_cloud");

  RelocResult result2 =
      reloc.relocalize(nullptr, Eigen::Isometry3d::Identity());
  EXPECT_FALSE(result2.success);
  EXPECT_EQ(result2.decision, "empty_cloud");
}

TEST_F(WorldLocalizingTest, PoseTransformConsistency) {
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  Eigen::Isometry3d T_map_odom = Eigen::Isometry3d::Identity();
  T_map_odom.translation() = Eigen::Vector3d(5.0, 3.0, 0.0);
  T_map_odom.rotate(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitZ()));
  reloc.setMapToOdomTransform(T_map_odom);

  Eigen::Isometry3d odom_pose = Eigen::Isometry3d::Identity();
  odom_pose.translation() = Eigen::Vector3d(1.0, 0.0, 0.0);
  Eigen::Isometry3d expected_map_pose = T_map_odom * odom_pose;
  Eigen::Isometry3d actual_map_pose = reloc.getMapToOdomTransform() * odom_pose;
  EXPECT_TRUE(expected_map_pose.isApprox(actual_map_pose, 1e-9));
}

TEST_F(WorldLocalizingTest, FreeSpaceDisabledKeepsLockPathUnchanged) {
  // reloc_free_space_enable=false must skip the free-space evidence path
  // entirely. On this corridor fixture the default veto does not fire anyway
  // (the free-space best agrees with the ranked top1), so the observable
  // contract is: disabling the switch reproduces the enabled lock exactly.
  config_.reloc_lock_min_margin = 0.1;
  buildTestMap(10, 2.0);

  Config config_off = config_;
  config_off.reloc_free_space_enable = false;

  WorldLocalizing reloc_on(config_, *keyframe_manager_, *loop_detector_,
                           *matcher_);
  WorldLocalizing reloc_off(config_off, *keyframe_manager_, *loop_detector_,
                            *matcher_);

  Eigen::Isometry3d query_pose = Eigen::Isometry3d::Identity();
  query_pose.translation().x() = 8.0;
  auto cloud = generateCorridorCloud(query_pose);

  RelocResult result_on;
  RelocResult result_off;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    result_on = reloc_on.relocalize(cloud, query_pose);
    result_off = reloc_off.relocalize(cloud, query_pose);
  }

  ASSERT_TRUE(result_on.success);
  ASSERT_TRUE(result_off.success);
  EXPECT_EQ(result_on.state, result_off.state);
  EXPECT_EQ(result_on.matched_keyframe_id, result_off.matched_keyframe_id);
  EXPECT_TRUE(result_on.pose_in_map.isApprox(result_off.pose_in_map, 1e-6));
  EXPECT_TRUE(reloc_on.isRelocalized());
  EXPECT_TRUE(reloc_off.isRelocalized());
}

TEST_F(WorldLocalizingTest, MapReplacementSameKeyframeCountDoesNotReuseOldIndex) {
  // Map A along x in [0, 18]; the frame-RHPD index, reloc map cache and
  // free-space grid are built by relocalizing inside it.
  buildTestMap(10, 2.0, 0.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d query_a = Eigen::Isometry3d::Identity();
  query_a.translation().x() = 8.0;
  auto cloud_a = generateCorridorCloud(query_a);
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    reloc.relocalize(cloud_a, query_a);
  }
  WorldLocalizing::WorldLocalizingCacheDiagnostics diag =
      reloc.cacheDiagnostics();
  ASSERT_GT(diag.frame_rhpd_indexed_keyframes, 0u);
  ASSERT_GT(diag.reloc_map_cached_keyframes, 0u);

  // Reset only runtime state, deliberately retaining map-derived caches. Then
  // replace with map B at the same keyframe count. Revision-keyed lazy rebuild
  // must supersede the old count-based behavior without a manual cache notice.
  reloc.resetLocalizationState();
  keyframe_manager_->clear();
  loop_detector_->clear();
  buildTestMap(10, 2.0, 100.0);

  diag = reloc.cacheDiagnostics();
  EXPECT_GT(diag.frame_rhpd_indexed_keyframes, 0u);
  EXPECT_GT(diag.reloc_map_cached_keyframes, 0u);

  // A query inside B must lock to B (x > 90); a leaked A index would answer
  // with map A's geometry around x = 8.
  Eigen::Isometry3d query_b = Eigen::Isometry3d::Identity();
  query_b.translation().x() = 108.0;
  auto cloud_b = generateCorridorCloud(query_b);
  bool locked = false;
  RelocResult result;
  for (int i = 0; i < config_.reloc_temporal_window_size * 2; ++i) {
    result = reloc.relocalize(cloud_b, query_b);
    if (result.success) {
      locked = true;
      break;
    }
  }
  ASSERT_TRUE(locked) << "expected revision-keyed caches to rebuild on map B";
  EXPECT_GT(result.pose_in_map.translation().x(), 90.0);
}

TEST_F(WorldLocalizingTest, FreeSpaceStateClearedOnMapReplacement) {
  buildTestMap(10, 2.0, 0.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d query_a = Eigen::Isometry3d::Identity();
  query_a.translation().x() = 8.0;
  auto cloud_a = generateCorridorCloud(query_a);
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    reloc.relocalize(cloud_a, query_a);
  }
  ASSERT_TRUE(reloc.cacheDiagnostics().free_space_grid_valid);

  keyframe_manager_->clear();
  loop_detector_->clear();
  buildTestMap(10, 2.0, 100.0);
  reloc.notifyMapReplaced();

  WorldLocalizing::WorldLocalizingCacheDiagnostics diag =
      reloc.cacheDiagnostics();
  EXPECT_FALSE(diag.free_space_grid_valid);
  EXPECT_EQ(diag.free_space_grid_keyframes, 0u);
  EXPECT_FALSE(diag.free_space_grid_failed);

  // The new map must be able to rebuild the grid: neither the old keyframe
  // count nor a latched failure state may survive the replacement.
  Eigen::Isometry3d query_b = Eigen::Isometry3d::Identity();
  query_b.translation().x() = 108.0;
  auto cloud_b = generateCorridorCloud(query_b);
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    reloc.relocalize(cloud_b, query_b);
  }
  diag = reloc.cacheDiagnostics();
  EXPECT_TRUE(diag.free_space_grid_valid);
  EXPECT_EQ(diag.free_space_grid_keyframes, 10u);
}

TEST_F(WorldLocalizingTest, AtlasClearedOnMapReplacement) {
  // Build a real small atlas file so the load path is exercised.
  buildTestMap(2, 2.0, 0.0);
  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("n3mapping_wl_atlas_" + std::to_string(nonce));
  std::filesystem::create_directories(dir);
  const std::filesystem::path map_file = dir / "map.pbstream";
  const std::filesystem::path atlas_file = dir / "map.localization_atlas.pb";
  {
    std::ofstream stream(map_file, std::ios::binary | std::ios::trunc);
    stream << "map-v1";
  }
  std::vector<Keyframe::Ptr> keyframes;
  for (const auto &kf : keyframe_manager_->getAllKeyframes()) {
    if (kf) {
      keyframes.push_back(kf);
    }
  }
  LocalizationAtlas compiled(config_, *matcher_);
  LocalizationAtlasStats stats;
  std::string error;
  ASSERT_TRUE(compiled.compileAndSave(map_file.string(), keyframes,
                                      atlas_file.string(), false, &stats,
                                      &error))
      << error;

  config_.reloc_atlas_enable = true;
  config_.reloc_atlas_path = atlas_file.string();
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);
  ASSERT_TRUE(reloc.loadLocalizationAtlas(map_file.string(), &error)) << error;
  ASSERT_TRUE(reloc.localizationAtlasLoaded());

  reloc.notifyMapReplaced();
  EXPECT_FALSE(reloc.localizationAtlasLoaded());
  EXPECT_FALSE(reloc.cacheDiagnostics().atlas_loaded);

  ASSERT_TRUE(reloc.loadLocalizationAtlas(map_file.string(), &error)) << error;
  EXPECT_TRUE(reloc.localizationAtlasLoaded());

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST_F(WorldLocalizingTest, StateOnlyResetKeepsMapCache) {
  config_.reloc_lock_min_margin = 0.1;
  buildTestMap(10, 2.0, 0.0);
  WorldLocalizing reloc(config_, *keyframe_manager_, *loop_detector_,
                        *matcher_);

  Eigen::Isometry3d query_pose = Eigen::Isometry3d::Identity();
  query_pose.translation().x() = 8.0;
  auto cloud = generateCorridorCloud(query_pose);
  RelocResult result;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    result = reloc.relocalize(cloud, query_pose);
  }
  ASSERT_TRUE(result.success);
  ASSERT_TRUE(reloc.isRelocalized());
  WorldLocalizing::WorldLocalizingCacheDiagnostics diag =
      reloc.cacheDiagnostics();
  ASSERT_GT(diag.reloc_map_cached_keyframes, 0u);
  ASSERT_GT(diag.frame_rhpd_indexed_keyframes, 0u);

  reloc.resetLocalizationState();
  EXPECT_FALSE(reloc.isRelocalized());
  EXPECT_EQ(reloc.getLastMatchedKeyframeId(), -1);
  WorldLocalizing::WorldLocalizingCacheDiagnostics diag2 =
      reloc.cacheDiagnostics();
  EXPECT_EQ(diag2.reloc_map_cached_keyframes, diag.reloc_map_cached_keyframes);
  EXPECT_EQ(diag2.frame_rhpd_indexed_keyframes,
            diag.frame_rhpd_indexed_keyframes);
  EXPECT_TRUE(diag2.free_space_grid_valid);

  // The same map must still be localizable right after the state-only reset.
  bool locked = false;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    result = reloc.relocalize(cloud, query_pose);
    if (result.success) {
      locked = true;
      break;
    }
  }
  ASSERT_TRUE(locked);
}

} // namespace test
} // namespace n3mapping
