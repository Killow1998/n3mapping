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
#include "support/localization_scene.h"

namespace n3mapping::test {
namespace {
using namespace synthetic_scene;

Config episodeConfig() {
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

Cloud::Ptr unrelatedCloud() {
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

Json::Value status(const core::OdometryEmission& live) {
  Json::Value result;
  Json::CharReaderBuilder builder;
  std::istringstream input(live.status_json);
  std::string error;
  EXPECT_TRUE(Json::parseFromStream(builder, input, &result, &error)) << error;
  return result;
}

// Optional observations, never a machine-dependent test threshold. Each process
// chooses a fresh trace path so baseline/candidate runs remain independent.
void trace(const char* stage, const Step& step, double elapsed_ms) {
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
    step.source.source_frame = "odom";
    step.source.body_frame = "lidar";
    step.source.pose = map_from_odom_.inverse() * map_pose;
    const auto started = std::chrono::steady_clock::now();
    step.backend = localizer_->processFrame(mode_,
        makeFrame(step.source.stamp_nsec, step.source.pose, cloud));
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

  Config config_ = episodeConfig();
  Cloud::Ptr scene_;
  std::filesystem::path directory_, map_path_;
  std::unique_ptr<N3MappingCore> localizer_;
  std::unique_ptr<core::RealtimeOutput> live_;
  CoreRunMode mode_ = CoreRunMode::LOCALIZATION;
  Eigen::Isometry3d map_from_odom_ = makePose(20.0, -8.0, M_PI_2);
  int64_t tick_ = 20;
};

TEST_F(LocalizationEpisode, UnknownInitialYawLocksToKnownMapPose) {
  for (const double degrees : {-137.0, -47.0, 73.0, 173.0}) {
    ASSERT_TRUE(localizer_->loadMap(map_path_.string()));
    startOutput();
    const auto pose = makePose(6.0, 0.0, degrees * M_PI / 180.0);
    Step step;
    for (int i = 0; i < 12; ++i) {
      step = run("rotated", pose,
          synthesizeBodyCloud(scene_, pose, 12.0, 0.15, 0.005, 1900U + i));
      if (step.backend.relocalization_locked) break;
      EXPECT_FALSE(step.live.pose);
    }
    ASSERT_TRUE(step.backend.relocalization_locked) << degrees << ": "
        << step.backend.relocalization_decision;
    EXPECT_LT((step.backend.T_world_lidar.translation() - pose.translation()).norm(), 0.5);
    EXPECT_LT(Eigen::AngleAxisd(pose.linear().transpose() *
        step.backend.T_world_lidar.linear()).angle(), 5.0 * M_PI / 180.0);
  }
}

TEST_F(LocalizationEpisode, OutOfMapMotionKeepsTrustedCorrectionAndReentryRelocks) {
  const auto mismatch = unrelatedCloud();
  ASSERT_GT(mismatch->size(), 200U);
  const auto searching = run("initial_unmapped", makePose(6.0, 0.0, 0.0), mismatch);
  EXPECT_EQ(searching.backend.relocalization_state, RelocalizationState::SEARCHING);
  EXPECT_FALSE(searching.live.pose);
  EXPECT_FALSE(searching.publication.plan.publish_authoritative_pose);

  const auto locked = lock();
  ASSERT_TRUE(locked.live.pose);
  const auto anchor = locked.live.pose->map_from_source;
  const auto correction_stamp = locked.live.pose->correction_stamp_nsec;
  EXPECT_EQ(locked.publication.plan.lock_epoch, 1U);
  Step outside;
  for (int i = 0; i < config_.reloc_max_track_failures + 4; ++i) {
    outside = run("outside", makePose(18.0 + 2.0 * i, 0.0, 0.03 * i), mismatch);
    EXPECT_FALSE(outside.backend.relocalization_locked);
    EXPECT_FALSE(outside.publication.plan.publish_authoritative_pose);
    ASSERT_TRUE(outside.live.pose);
    EXPECT_EQ(outside.live.pose->stamp_nsec, outside.source.stamp_nsec);
    EXPECT_EQ(outside.live.pose->correction_stamp_nsec, correction_stamp);
    EXPECT_TRUE(outside.live.pose->pose.matrix().isApprox(
        (anchor * outside.source.pose).matrix(), 1e-10));
    EXPECT_DOUBLE_EQ(status(outside.live)["observation_stamp"].asDouble(),
                     static_cast<double>(outside.source.stamp_nsec) * 1e-9);
    EXPECT_DOUBLE_EQ(status(outside.live)["correction_stamp"].asDouble(),
                     static_cast<double>(correction_stamp) * 1e-9);
  }
  EXPECT_EQ(outside.backend.relocalization_state, RelocalizationState::LOST);
  const auto recovered = lock(0.0, true);
  ASSERT_TRUE(recovered.live.pose);
  EXPECT_TRUE(recovered.publication.plan.publish_authoritative_pose);
  EXPECT_EQ(recovered.publication.plan.lock_epoch, 2U);
  EXPECT_GT(recovered.live.pose->correction_stamp_nsec, correction_stamp);
}

TEST_F(LocalizationEpisode, ConfirmationCorrectsBiasedOdometryBeforePublishing) {
  // Geometry stays in the same map while odometry develops a translation bias
  // during confirmation. Reusing a proposal must not publish that bias.
  const auto original_map_from_odom = map_from_odom_;
  Step step;
  Eigen::Isometry3d true_pose;
  for (int i = 0; i < 9; ++i) {
    true_pose = makePose(6.0 + 0.1 * i, 0.0, 0.0);
    map_from_odom_ = original_map_from_odom;
    map_from_odom_.translation().x() -= 0.3 * i;
    step = run("biased_confirmation", true_pose,
        synthesizeBodyCloud(scene_, true_pose, 12.0, 0.05, 0.002, 2940U + i));
    if (step.backend.relocalization_locked) break;
    EXPECT_FALSE(step.live.pose);
    EXPECT_FALSE(step.publication.plan.publish_authoritative_pose);
  }
  ASSERT_TRUE(step.backend.relocalization_locked);
  ASSERT_TRUE(step.live.pose);
  EXPECT_LT((step.live.pose->pose.translation() - true_pose.translation()).norm(), 0.1);
  EXPECT_LT(Eigen::AngleAxisd(true_pose.linear().transpose() *
      step.live.pose->pose.linear()).angle(), M_PI / 180.0);
}

TEST_F(LocalizationEpisode, StationaryTrackingKeepsRigidOutputWithoutRelocking) {
  // Repeated matrix/quaternion conversions at a large world-frame yaw used
  // to amplify roundoff until publication rejected an otherwise tracked pose.
  map_from_odom_ = makePose(20.0, -8.0, -110.0 * M_PI / 180.0);
  const auto locked = lock();
  const auto pose = makePose(6.0, 0.0, 0.0);
  for (int i = 0; i < 200; ++i) {
    const auto step = run("stationary_tracking", pose,
        synthesizeBodyCloud(scene_, pose, 12.0, 0.05, 0.002, 3900U + i));
    ASSERT_TRUE(step.backend.success) << i << ": " << step.backend.relocalization_decision;
    ASSERT_TRUE(step.live.pose) << i;
    EXPECT_EQ(step.backend.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
    EXPECT_FALSE(step.backend.relocalization_locked);
    EXPECT_EQ(step.publication.plan.lock_epoch, locked.publication.plan.lock_epoch);
    EXPECT_TRUE(isFiniteRigidPose(step.backend.T_world_lidar, 1e-10));
    EXPECT_TRUE(isFiniteRigidPose(step.live.pose->pose, 1e-10));
    EXPECT_LT((step.live.pose->pose.translation() - pose.translation()).norm(), 0.1);
    EXPECT_LT(Eigen::AngleAxisd(pose.linear().transpose() *
        step.live.pose->pose.linear()).angle(), M_PI / 180.0);
  }
}

TEST_F(LocalizationEpisode, InvalidRawCancelsProjectionUntilARealGeometricCorrection) {
  const auto locked = lock();
  ASSERT_TRUE(locked.live.pose);
  auto raw = locked.source;
  raw.stamp_nsec = ++tick_ * 1000000000LL;
  const auto frozen = live_->observeCloud(raw, raw.stamp_nsec,
                                         raw.stamp_nsec, raw.body_frame);
  ASSERT_TRUE(frozen);
  raw.stamp_nsec = ++tick_ * 1000000000LL;
  raw.pose_valid = false;
  raw.pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
  const auto invalid = live_->observeOdometry(raw, raw.stamp_nsec, true);
  EXPECT_FALSE(invalid.pose);
  EXPECT_FALSE(live_->isCurrent(*frozen));
  EXPECT_FALSE(status(invalid)["projection"]["available"].asBool());
  EXPECT_EQ(status(invalid)["state"].asString(), "error");

  const auto rejected = run("invalid_then_mismatch", makePose(7.0, 0.0, 0.0), unrelatedCloud());
  EXPECT_FALSE(rejected.live.pose);
  const auto pose = makePose(6.5, 0.0, 0.0);
  const auto recovered = run("geometric_restore", pose,
      synthesizeBodyCloud(scene_, pose, 12.0, 0.0, 0.0, 1042));
  EXPECT_TRUE(recovered.publication.live_authorized);
  ASSERT_TRUE(recovered.live.pose);
  EXPECT_GT(recovered.live.pose->generation, locked.live.pose->generation);
  EXPECT_EQ(recovered.live.pose->correction_stamp_nsec, recovered.source.stamp_nsec);
}

TEST_F(LocalizationEpisode, FreshMapLifecycleCannotReuseOldAuthorityOrPose) {
  const auto old_lock = lock();
  ASSERT_TRUE(old_lock.live.pose);
  const auto replacement = directory_ / "replacement.pbstream";
  buildMap(replacement, 100.0);
  ASSERT_TRUE(localizer_->loadMap(replacement.string()));
  // Loading the backend and restarting its live-output owner form the same
  // fresh lifecycle used by a node restart; this is not a hot-swap feature.
  startOutput();
  const auto searching = run("replacement_unmapped", makePose(106.0, 0.0, 0.0), unrelatedCloud());
  EXPECT_FALSE(searching.live.pose);
  EXPECT_FALSE(searching.publication.plan.publish_authoritative_pose);
  EXPECT_EQ(searching.publication.plan.lock_epoch, 0U);
  const auto replacement_lock = lock(100.0);
  ASSERT_TRUE(replacement_lock.live.pose);
  EXPECT_EQ(replacement_lock.publication.plan.lock_epoch, 1U);
  EXPECT_LT((replacement_lock.live.pose->pose.translation() -
             Eigen::Vector3d(106.0, 0.0, 0.0)).norm(), 0.5);
  EXPECT_GT((replacement_lock.live.pose->pose.translation() -
             old_lock.live.pose->pose.translation()).norm(), 90.0);
}

TEST_F(LocalizationEpisode, RepeatedPlacesStayUnpublishedAndFreshGeometryCanLock) {
  const auto cloud = synthesizeBodyCloud(scene_, makePose(6.0, 0.0, 0.0),
                                        12.0, 0.0, 0.0, 1442);
  auto unique_cloud = pcl::make_shared<Cloud>(*cloud);
  for (auto& point : *unique_cloud) {
    point.x *= 0.6F;
    point.y *= 1.6F;
    point.z *= 1.3F;
  }
  const auto repeated_path = directory_ / "repeated.pbstream";
  {
    N3MappingCore mapper(config_, N3MappingCore::WorldCloudOutput::Disabled);
    for (int i = 0; i < 2; ++i) {
      const auto mapped = mapper.processMappingFrame(makeFrame(
          (i + 1) * 1000000000LL, makePose(i * 20.0, 0.0, 0.0), cloud));
      ASSERT_TRUE(mapped.accepted_keyframe);
    }
    ASSERT_TRUE(mapper.processMappingFrame(makeFrame(
        3000000000LL, makePose(40.0, 0.0, 0.0), unique_cloud)).accepted_keyframe);
    ASSERT_TRUE(mapper.saveMap(repeated_path.string()));
    compileAtlas(mapper, repeated_path);
  }
  ASSERT_TRUE(localizer_->loadMap(repeated_path.string()));
  startOutput();
  for (int i = 0; i < 5 * config_.reloc_temporal_window_size; ++i) {
    const auto query = run("ambiguous", makePose(0.0, 0.0, 0.0), cloud);
    EXPECT_FALSE(query.backend.relocalization_locked);
    EXPECT_NE(query.backend.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
    EXPECT_FALSE(query.publication.plan.publish_authoritative_pose);
    EXPECT_FALSE(query.live.pose);
  }
  Step identified;
  for (int i = 0; i < 3 * config_.reloc_temporal_window_size; ++i) {
    identified = run("distinct_after_ambiguity", makePose(40.0, 0.0, 0.0), unique_cloud);
    if (identified.backend.relocalization_locked) break;
    EXPECT_FALSE(identified.live.pose);
  }
  ASSERT_TRUE(identified.backend.relocalization_locked);
  ASSERT_TRUE(identified.live.pose);
  EXPECT_LT((identified.live.pose->pose.translation() - Eigen::Vector3d(40, 0, 0)).norm(), 0.5);
}

TEST_F(LocalizationEpisode, OppositeHeadingsInSymmetricGeometryRemainAmbiguous) {
  auto symmetric = synthesizeBodyCloud(scene_, makePose(6.0, 0.0, 0.0),
                                       12.0, 0.0, 0.0, 1442);
  const auto original = *symmetric;
  for (auto point : original) {
    point.x = -point.x;
    point.y = -point.y;
    symmetric->push_back(point);
  }
  const auto path = directory_ / "symmetric.pbstream";
  {
    N3MappingCore mapper(config_, N3MappingCore::WorldCloudOutput::Disabled);
    ASSERT_TRUE(mapper.processMappingFrame(makeFrame(
        1000000000LL, makePose(0.0, 0.0, 0.0), symmetric)).accepted_keyframe);
    ASSERT_TRUE(mapper.saveMap(path.string()));
    compileAtlas(mapper, path);
  }
  ASSERT_TRUE(localizer_->loadMap(path.string()));
  startOutput();
  const auto pose = makePose(0.0, 0.0, 42.0 * M_PI / 180.0);
  const auto query_cloud = synthesizeBodyCloud(symmetric, pose, 12.0, 0.0, 0.0, 19);
  for (int i = 0; i < 2 * config_.reloc_temporal_window_size; ++i) {
    const auto query = run("heading_ambiguous", pose, query_cloud);
    EXPECT_FALSE(query.backend.relocalization_locked);
    EXPECT_FALSE(query.publication.plan.publish_authoritative_pose);
    EXPECT_FALSE(query.live.pose);
  }
}

TEST_F(LocalizationEpisode, UnchangedSeedsCannotStrandSearchInOldPoseModes) {
  // All three places remain in retrieval with the same uniform yaw seeds.
  // The third mode has no geometric support in the initial observation.
  config_.rhpd_use_sc_yaw = false;
  const auto repeated = synthesizeBodyCloud(scene_, makePose(6, 0, 0), 12, 0, 0, 1442);
  auto distinct = pcl::make_shared<Cloud>(*repeated);
  for (auto& point : *distinct) point.z += 5.5F;
  const auto path = directory_ / "mode_transition.pbstream";
  {
    N3MappingCore mapper(config_, N3MappingCore::WorldCloudOutput::Disabled);
    ASSERT_TRUE(mapper.processMappingFrame(makeFrame(1000000000LL, makePose(0, 0, 0), repeated)).accepted_keyframe);
    ASSERT_TRUE(mapper.processMappingFrame(makeFrame(2000000000LL, makePose(20, 0, 0), repeated)).accepted_keyframe);
    ASSERT_TRUE(mapper.processMappingFrame(makeFrame(3000000000LL, makePose(140, 0, 0), distinct)).accepted_keyframe);
    ASSERT_TRUE(mapper.saveMap(path.string()));
    compileAtlas(mapper, path);
  }
  localizer_ = std::make_unique<N3MappingCore>(config_, N3MappingCore::WorldCloudOutput::Disabled);
  ASSERT_TRUE(localizer_->loadMap(path.string()));
  startOutput();
  for (int i = 0; i < 3 * config_.reloc_temporal_window_size; ++i) {
    const auto step = run("unmapped_alias", makePose(100, 0, 0), repeated);
    ASSERT_FALSE(step.backend.relocalization_locked);
    ASSERT_FALSE(step.live.pose);
  }
  Step identified;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    identified = run("new_supported_mode", makePose(140, 0, 0), distinct);
  }
  ASSERT_TRUE(identified.backend.relocalization_locked);
  ASSERT_TRUE(identified.live.pose);
  EXPECT_LT((identified.live.pose->pose.translation() - Eigen::Vector3d(140, 0, 0)).norm(), 0.5);
}

TEST_F(LocalizationEpisode, NoPlaceCandidateLeavesPoseUnavailable) {
  config_.rhpd_dist_threshold = 1e-9;
  // With a populated RHPD map this also disables building a ScanContext query,
  // so an unrelated descriptor cannot be rescued by SC fallback.
  config_.rhpd_use_sc_yaw = false;
  localizer_ = std::make_unique<N3MappingCore>(
      config_, N3MappingCore::WorldCloudOutput::Disabled);
  ASSERT_TRUE(localizer_->loadMap(map_path_.string()));
  startOutput();
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    const auto query = run("no_candidates", makePose(6.0, 0.0, 0.0), unrelatedCloud());
    EXPECT_EQ(query.backend.relocalization_decision, "no_candidates");
    EXPECT_FALSE(query.backend.relocalization_locked);
    EXPECT_FALSE(query.publication.plan.publish_authoritative_pose);
    EXPECT_FALSE(query.live.pose);
  }
}

TEST_F(LocalizationEpisode, MapExtensionTracksImmutableLoadedGeometryAcrossAnchors) {
  mode_ = CoreRunMode::MAP_EXTENSION;
  startOutput();
  const auto locked = lock();
  ASSERT_TRUE(locked.backend.relocalization_locked);
  for (const double x : {6.0, 6.1, 6.2, 6.1, 7.0, 7.1, 7.2, 7.1, 8.5, 8.6, 8.7, 8.6}) {
    const auto pose = makePose(x, 0.0, 0.0);
    const auto tracked = run("strict_tracking", pose,
        synthesizeBodyCloud(scene_, pose, 12.0, 0.05, 0.003, 2026));
    ASSERT_TRUE(tracked.backend.success) << tracked.backend.relocalization_decision;
    EXPECT_EQ(tracked.backend.relocalization_state, RelocalizationState::FULL_6DOF_LOCKED);
    EXPECT_EQ(tracked.backend.pose_source, PoseSource::GEOMETRICALLY_CORRECTED);
    EXPECT_TRUE(tracked.backend.tracking_performance.available);
    EXPECT_TRUE(tracked.backend.tracking_performance.strict_loaded_map);
    const auto anchor = localizer_->getKeyframe(tracked.backend.matched_keyframe_id);
    ASSERT_NE(anchor, nullptr);
    EXPECT_TRUE(anchor->is_from_loaded_map);
    ASSERT_TRUE(tracked.live.pose);
    EXPECT_TRUE(tracked.publication.live_authorized);
    EXPECT_EQ(tracked.live.pose->correction_stamp_nsec, tracked.source.stamp_nsec);
    EXPECT_LT((tracked.live.pose->pose.translation() - pose.translation()).norm(), 0.5);
  }
}

}  // namespace
}  // namespace n3mapping::test
