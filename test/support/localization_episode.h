#pragma once

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>
#include <json/json.h>

#include "n3mapping/core/realtime_output.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/point_cloud_matcher.h"
#include "localization_scene.h"

namespace n3mapping::test {
using namespace synthetic_scene;

inline Config episodeConfig() {
  auto config = makeSyntheticRelocConfig();
  config.reloc_temporal_window_size = 3;
  config.reloc_lock_min_winner_streak = 3;
  config.reloc_lock_min_converged_updates = 3;
  config.reloc_lock_min_margin = 0.1;
  config.reloc_min_confidence = 0.3;
  config.reloc_min_inlier_ratio = 0.03;
  config.reloc_ambiguity_min_basin_separation = 3.0;
  config.num_threads = 1;
  if (const auto* mode = std::getenv("N3MAPPING_EPISODE_FREE_SPACE_MODE"))
    config.reloc_free_space_mode = mode;
  config.reloc_atlas_enable = std::getenv("N3MAPPING_EPISODE_ATLAS") != nullptr;
  return config;
}

inline Cloud::Ptr unrelatedCloud() {
  auto cloud = pcl::make_shared<Cloud>();
  for (int x = 0; x < 12; ++x) {
    for (int y = 0; y < 12; ++y) {
      for (int z = 0; z < 3; ++z) {
        pcl::PointXYZI point;
        point.x = 25.0F + x * 0.2F;
        point.y = 25.0F + y * 0.2F;
        point.z = 1.0F + z * 0.2F;
        point.intensity = 10.0F;
        cloud->push_back(point);
      }
    }
  }
  return cloud;
}

struct Step {
  core::OdometrySource source;
  core::BackendOutput backend;
  core::BackendPublication publication;
  core::OdometryEmission live;
};

inline Json::Value status(const core::OdometryEmission& live) {
  Json::Value result;
  Json::CharReaderBuilder builder;
  std::istringstream input(live.status_json);
  std::string error;
  EXPECT_TRUE(Json::parseFromStream(builder, input, &result, &error)) << error;
  return result;
}

// Optional observations, never a machine-dependent test threshold. Each process
// chooses a fresh trace path so baseline/candidate runs remain independent.
inline void trace(const char* stage, const Step& step, double elapsed_ms) {
  const char* filename = std::getenv("N3MAPPING_EPISODE_TRACE");
  if (!filename || !*filename) return;
  const bool header = !std::filesystem::exists(filename);
  std::ofstream stream(filename, std::ios::app);
  ASSERT_TRUE(stream.good());
  if (header) {
    stream << "test,stage,stamp_nsec,state,pose_source,success,lock,live_pose,"
              "correction_stamp_nsec,seed_id,support_id,matched_id,decision,"
              "status_state,status_reason,status_available,observation_stamp,"
              "source_preparations,visibility_preparations,visibility_evaluations,"
              "target_builds,target_preparations,cache_hits,cache_misses,"
              "candidates,hypotheses,alignments,total_ms,tracking_available,"
              "tracking_strict,tracking_total_ms,tracking_submap_build_ms,"
              "tracking_target_prepare_ms,tracking_cache_hit,tracking_cache_miss,"
              "tracking_loaded_cache_ms,tracking_source_prepare_ms,tracking_visibility_ms";
    for (const auto* prefix : {"backend", "live", "raw"}) {
      for (const auto* component : {"x", "y", "z", "qx", "qy", "qz", "qw"}) {
        stream << ',' << prefix << '_' << component;
      }
    }
    stream << '\n';
  }
  const auto public_status = status(step.live);
  const auto& cost = step.backend.relocalization_performance;
  stream << std::setprecision(17)
         << ::testing::UnitTest::GetInstance()->current_test_info()->name() << ','
         << stage << ',' << step.source.stamp_nsec << ','
         << static_cast<int>(step.backend.relocalization_state) << ','
         << static_cast<int>(step.backend.pose_source) << ',' << step.backend.success << ','
         << step.backend.relocalization_locked << ',' << step.live.pose.has_value() << ','
         << step.live.diagnostic.correction_stamp_nsec << ','
         << step.backend.relocalization_seed_keyframe_id << ','
         << step.backend.relocalization_support_keyframe_id << ','
         << step.backend.matched_keyframe_id << ',' << step.backend.relocalization_decision << ','
         << public_status["state"].asString() << ',' << public_status["reason"].asString() << ','
         << public_status["projection"]["available"].asBool() << ','
         << public_status["observation_stamp"].asDouble() << ','
         << cost.source_preparations << ',' << cost.visibility_preparations << ','
         << cost.visibility_evaluations << ',' << cost.target_builds << ','
         << cost.target_preparations << ',' << cost.cache_hits << ',' << cost.cache_misses << ','
         << cost.candidates << ',' << cost.hypotheses << ',' << cost.alignments << ',' << elapsed_ms;
  const auto& tracking = step.backend.tracking_performance;
  stream << ',' << tracking.available << ',' << tracking.strict_loaded_map << ','
         << tracking.tracking_total_ms << ',' << tracking.submap_build_ms << ','
         << tracking.target_prepare_ms << ',' << tracking.loaded_map_target_cache_hit << ','
         << tracking.loaded_map_target_cache_miss << ',' << tracking.loaded_map_cache_ms << ','
         << tracking.source_prepare_ms << ',' << tracking.visibility_ms;
  const auto pose = [&](const Eigen::Isometry3d* transform) {
    if (!transform) {
      for (int i = 0; i < 7; ++i) stream << ',';
      return;
    }
    for (int i = 0; i < 3; ++i) stream << ',' << transform->translation()[i];
    const Eigen::Quaterniond orientation(transform->linear());
    for (int i = 0; i < 4; ++i) stream << ',' << orientation.coeffs()[i];
  };
  pose(&step.backend.T_world_lidar);
  pose(step.live.pose ? &step.live.pose->pose : nullptr);
  pose(&step.source.pose);
  stream << '\n';
}

class LocalizationEpisode : public ::testing::Test {
 protected:
  void SetUp() override {
    char pattern[] = "/tmp/n3mapping_localization_episode_XXXXXX";
    const auto directory = mkdtemp(pattern);
    ASSERT_NE(directory, nullptr);
    directory_ = directory;
    scene_ = makeWorldScene();
    map_path_ = directory_ / "scene.pbstream";
    buildMap(map_path_, 0.0);
    localizer_ = std::make_unique<N3MappingCore>(
        config_, N3MappingCore::WorldCloudOutput::Disabled);
    ASSERT_TRUE(localizer_->loadMap(map_path_.string()));
    startOutput();
  }

  void TearDown() override {
    live_.reset();
    localizer_.reset();
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }

  void buildMap(const std::filesystem::path& path, double offset) {
    N3MappingCore mapper(config_, N3MappingCore::WorldCloudOutput::Disabled);
    for (int i = 0; i < 7; ++i) {
      const auto local_pose = makePose(i * 1.5, 0.0, 0.0);
      auto pose = local_pose;
      pose.translation().x() += offset;
      auto cloud = synthesizeBodyCloud(scene_, local_pose, 12.0, 0.0, 0.0, 100U + i);
      const auto output = mapper.processMappingFrame(
          makeFrame((i + 1) * 1000000000LL, pose, cloud));
      ASSERT_TRUE(output.success);
      ASSERT_TRUE(output.accepted_keyframe);
    }
    ASSERT_TRUE(mapper.saveMap(path.string()));
    compileAtlas(mapper, path);
  }

  void compileAtlas(N3MappingCore& mapper, const std::filesystem::path& path) {
    if (!config_.reloc_atlas_enable) return;
    PointCloudMatcher matcher(config_);
    LocalizationAtlas atlas(config_, matcher);
    LocalizationAtlasStats stats;
    std::string error;
    ASSERT_TRUE(atlas.compileAndSave(path.string(), mapper.getAllKeyframes(),
        LocalizationAtlas::defaultAtlasPath(path.string()), false, &stats, &error)) << error;
  }

  void startOutput() {
    live_ = std::make_unique<core::RealtimeOutput>(mode_, "map", 32);
  }

  Step run(const char* stage, const Eigen::Isometry3d& map_pose, const Cloud::Ptr& cloud) {
    Step step;
    step.source.stamp_nsec = ++tick_ * 1000000000LL;
    step.source.source_frame = odom_frame_;
    step.source.body_frame = "lidar";
    step.source.pose = map_from_odom_.inverse() * map_pose;
    const auto started = std::chrono::steady_clock::now();
    auto frame = makeFrame(step.source.stamp_nsec, step.source.pose, cloud);
    frame.source_frame_id = step.source.source_frame;
    step.backend = localizer_->processFrame(mode_, frame);
    step.publication = live_->acceptBackend(step.source, static_cast<double>(tick_), step.backend);
    step.live = live_->observeOdometry(step.source, step.source.stamp_nsec, true);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    trace(stage, step, elapsed);
    return step;
  }

  Step lock(double map_offset = 0.0, bool retained_pose = false) {
    Step step;
    for (int i = 0; i < 8; ++i) {
      auto pose = makePose(6.0 + map_offset, 0.0, 0.0);
      const auto cloud = synthesizeBodyCloud(scene_, makePose(6.0, 0.0, 0.0),
                                            12.0, 0.15, 0.005, 900U + i);
      step = run("mapped", pose, cloud);
      if (step.backend.relocalization_locked) break;
      EXPECT_EQ(step.live.pose.has_value(), retained_pose);
      EXPECT_FALSE(step.publication.plan.publish_authoritative_pose);
    }
    EXPECT_TRUE(step.backend.relocalization_locked) << step.backend.relocalization_decision;
    EXPECT_EQ(step.backend.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
    EXPECT_TRUE(step.live.pose);
    EXPECT_LT((step.backend.T_world_lidar.translation() -
               Eigen::Vector3d(6.0 + map_offset, 0.0, 0.0)).norm(), 0.5);
    return step;
  }

  std::string odom_frame_ = "odom";
  Config config_ = episodeConfig();
  Cloud::Ptr scene_;
  std::filesystem::path directory_, map_path_;
  std::unique_ptr<N3MappingCore> localizer_;
  std::unique_ptr<core::RealtimeOutput> live_;
  CoreRunMode mode_ = CoreRunMode::LOCALIZATION;
  Eigen::Isometry3d map_from_odom_ = makePose(20.0, -8.0, M_PI_2);
  int64_t tick_ = 20;
};

} // namespace n3mapping::test
