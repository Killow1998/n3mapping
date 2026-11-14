#include "support/localization_scene.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/world_localizing.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace n3mapping::test {

TEST(DirectionalReference, PartialMapRequiresConsistentReciprocalGeometry) {
  auto config = synthetic_scene::makeSyntheticRelocConfig();
  config.num_threads = 4;
  config.reloc_temporal_window_size = 3;
  config.reloc_lock_min_winner_streak = 3;
  config.reloc_lock_min_converged_updates = 3;
  config.reloc_free_space_enable = false;
  auto scene = synthetic_scene::makeWorldScene();
  const auto pose = synthetic_scene::makePose(4, 0, 0);
  auto query = synthetic_scene::synthesizeBodyCloud(scene, pose, 12, 0, 0, 42);
  auto partial = pcl::make_shared<synthetic_scene::Cloud>();
  for (const auto& point : *query)
    if (point.y > 0) partial->push_back(point);
  KeyframeManager keyframes(config);
  LoopDetector detector(config);
  PointCloudMatcher matcher(config);
  const auto id = keyframes.addKeyframe(0, pose, partial);
  detector.addDescriptor(id, partial);
  detector.addRHPD(id, partial);
  keyframes.getKeyframe(id)->is_from_loaded_map = true;
  WorldLocalizing localizer(config, keyframes, detector, matcher);
  bool localized = false;
  uint64_t invalidated = 0;
  uint64_t alignments = 0;
  for (int i = 0; i < 12; ++i) {
    const auto result = localizer.relocalize(query, Eigen::Isometry3d::Identity());
    if (i + 1 < config.reloc_temporal_window_size) {
      EXPECT_FALSE(result.success);
    }
    if (result.success) {
      localized = true;
      EXPECT_LT((result.pose_in_map.translation()-pose.translation()).norm(), 0.1);
    }
    invalidated += result.performance.invalidated_seed_poses;
    alignments += result.performance.alignments;
    if (result.decision == "directional_visibility_insufficient_coverage") {
      EXPECT_EQ(result.performance.invalidated_seed_poses, 0u);
    }
  }
  EXPECT_TRUE(localized);
  EXPECT_EQ(invalidated, 0u);
  RecordProperty("invalidated_seed_poses", std::to_string(invalidated));
  RecordProperty("alignments", std::to_string(alignments));
  RelocResult recovered;
  for (int i = 0; i < 12 && !recovered.success; ++i)
    recovered = localizer.relocalize(partial, Eigen::Isometry3d::Identity());
  EXPECT_TRUE(recovered.success) << recovered.decision;
  if (recovered.success) {
    EXPECT_LT((recovered.pose_in_map.translation()-pose.translation()).norm(),0.1);
  }
}

TEST(DirectionalReference, StoredObservationMustIndependentlySupportTheCandidate) {
  for (const std::string reference_mode : {"conflict", "complete", "partial"}) {
    SCOPED_TRACE(reference_mode);
    auto config = synthetic_scene::makeSyntheticRelocConfig();
    config.num_threads = 4;
    config.global_map_voxel_size = 0.05;
    config.reloc_temporal_window_size = 3;
    config.reloc_lock_min_winner_streak = 3;
    config.reloc_lock_min_converged_updates = 3;
    config.reloc_free_space_enable = false;
    const auto pose = synthetic_scene::makePose(-2, 0, 0);
    auto scene = synthetic_scene::makeWorldScene();
    for (float y = -3; y <= 3; y += 0.2F) {
      for (float z = -0.2F; z <= 2.2F; z += 0.2F) {
        pcl::PointXYZI point;
        point.x = -4; point.y = y; point.z = z; point.intensity = 200;
        scene->push_back(point);
      }
    }
    const auto query = synthetic_scene::synthesizeBodyCloud(
        scene, pose, 12, 0, 0, 42);
    auto foreground = pcl::make_shared<synthetic_scene::Cloud>();
    for (auto point : *query) {
      if (point.x > -1.99F) continue;
      point.x *= 0.5F; point.y *= 0.5F; point.z *= 0.5F;
      foreground->push_back(point);
    }
    auto reference = pcl::make_shared<synthetic_scene::Cloud>(*query);
    if (reference_mode == "conflict") *reference += *foreground;
    if (reference_mode == "partial") {
      reference->clear();
      for (const auto& point : *query)
        if (point.y > 0) reference->push_back(point);
    }
    KeyframeManager keyframes(config);
    LoopDetector detector(config);
    PointCloudMatcher matcher(config);
    for (const auto& cloud : {reference, foreground}) {
      const auto id = keyframes.addKeyframe(keyframes.size(), pose, cloud);
      detector.addDescriptor(id, cloud);
      detector.addRHPD(id, cloud);
      keyframes.getKeyframe(id)->is_from_loaded_map = true;
    }
    const auto directory = std::filesystem::temp_directory_path() /
        ("n3mapping_directional_reference_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    const auto map = directory / "map.pbstream";
    { std::ofstream file(map); file << "directional-reference"; }
    config.reloc_atlas_enable = true;
    config.reloc_atlas_path = (directory / "map.atlas.pb").string();
    LocalizationAtlas atlas(config, matcher);
    std::string error;
    ASSERT_TRUE(atlas.compileAndSave(map.string(), keyframes.getAllKeyframes(),
        config.reloc_atlas_path, false, nullptr, &error)) << error;
    WorldLocalizing localizer(config, keyframes, detector, matcher);
    ASSERT_TRUE(localizer.loadLocalizationAtlas(map.string(), &error)) << error;
    RelocResult result;
    for (int i = 0; i < config.reloc_temporal_window_size; ++i) {
      result = localizer.relocalize(query, Eigen::Isometry3d::Identity());
      if (i + 1 < config.reloc_temporal_window_size) {
        EXPECT_FALSE(result.success);
      }
    }
    EXPECT_EQ(result.success, reference_mode != "conflict") << result.decision;
    if (reference_mode == "partial") {
      EXPECT_EQ(result.decision, "accepted");
      EXPECT_LE(result.performance.directional_min_support, 0.5);
      EXPECT_EQ(result.performance.invalidated_seed_poses, 0u);
      RecordProperty("partial_reference_invalidated_seeds",
                     std::to_string(result.performance.invalidated_seed_poses));
    }
    if (reference_mode == "conflict") {
      EXPECT_GT(result.performance.invalidated_seed_poses, 0u);
    }
    if (result.success) {
      EXPECT_LT((result.pose_in_map.translation()-pose.translation()).norm(), 0.1);
      if (reference_mode == "complete") {
        EXPECT_GT(result.performance.directional_min_support, 0.5);
      }
    }
    std::filesystem::remove_all(directory);
  }
}

} // namespace n3mapping::test
