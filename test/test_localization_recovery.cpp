#include "support/localization_episode.h"

namespace n3mapping::test {
namespace {

TEST_F(LocalizationEpisode, ChangedFurnitureKeepsTheEstablishedMapLock) {
  localizer_.reset();
  config_.reloc_atlas_enable = true;
  config_.gicp_downsampling_resolution = 0.1;
  config_.gicp_fitness_threshold = 0.3;
  config_.loop_min_inlier_ratio = 0.7;
  config_.global_map_voxel_size = 0.1;
  config_.num_threads = 4;
  for (double y = -3; y <= 3; y += 0.25) {
    for (double z = -0.25; z <= 2.75; z += 0.25) {
      pcl::PointXYZI point;
      point.x = -3;
      point.y = y;
      point.z = z;
      point.intensity = 40;
      scene_->push_back(point);
    }
  }
  for (double x = -3; x <= 14; x += 0.35) {
    for (double y = -3; y <= 3; y += 0.35) {
      pcl::PointXYZI point;
      point.x = x;
      point.y = y;
      point.z = 2.75;
      point.intensity = 40;
      scene_->push_back(point);
    }
  }
  buildMap(map_path_, 0.0);
  localizer_ = std::make_unique<N3MappingCore>(
      config_, N3MappingCore::WorldCloudOutput::Disabled);
  ASSERT_TRUE(localizer_->loadMap(map_path_.string()));
  startOutput();
  const auto initial = lock();
  ASSERT_TRUE(initial.live.pose);
  auto changed = pcl::make_shared<Cloud>(*scene_);
  for (auto &point : *changed) {
    if (point.intensity >= 80 && point.intensity < 120) {
      point.x += 0.24F;
      point.y -= 0.16F;
    }
  }
  const auto pose = makePose(6, 0, 0);
  for (int i = 0; i < 50; ++i) {
    const auto step =
        run("changed_furniture", pose,
            synthesizeBodyCloud(changed, pose, 12, 0.05, 0.003, 5100U + i));
    ASSERT_TRUE(step.backend.success)
        << i << ": " << step.backend.relocalization_decision;
    ASSERT_TRUE(step.live.pose);
    EXPECT_EQ(step.backend.relocalization_state,
              RelocalizationState::FULL_6DOF_LOCKED);
    EXPECT_EQ(step.publication.plan.lock_epoch,
              initial.publication.plan.lock_epoch);
    EXPECT_FALSE(step.backend.relocalization_locked);
    EXPECT_LT((step.live.pose->pose.translation() - pose.translation()).norm(),
              0.1);
    EXPECT_LT(Eigen::AngleAxisd(step.live.pose->pose.linear()).angle(),
              M_PI / 180.0);
  }
}

TEST_F(LocalizationEpisode, SourceFrameReplacementStartsFreshConfirmation) {
  const auto initial = lock();
  ASSERT_TRUE(initial.live.pose);
  const auto frozen = live_->observeCloud(initial.source, initial.source.stamp_nsec,
      initial.source.stamp_nsec, initial.source.body_frame);
  ASSERT_TRUE(frozen);
  odom_frame_ = "restarted_odom";
  map_from_odom_ = makePose(70, 40, -M_PI_2);
  Step step;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    step = run("source_replaced", makePose(6, 0, 0),
        synthesizeBodyCloud(scene_, makePose(6, 0, 0), 12, 0.15, 0.005, 900U+i));
    if (i + 1 < config_.reloc_temporal_window_size) {
      EXPECT_FALSE(step.backend.relocalization_locked);
      EXPECT_FALSE(step.live.pose);
    }
  }
  EXPECT_TRUE(step.backend.relocalization_locked);
  ASSERT_TRUE(step.live.pose);
  EXPECT_LT((step.live.pose->pose.translation()-Eigen::Vector3d(6, 0, 0)).norm(), 0.5);
  EXPECT_FALSE(live_->isCurrent(*frozen));
  EXPECT_FALSE(live_->observeCloud(initial.source, step.source.stamp_nsec,
      initial.source.stamp_nsec, initial.source.body_frame));
}

TEST_F(LocalizationEpisode, InvalidBackendOdometryCannotCarryConfirmationAcrossRestart) {
  const auto initial = lock();
  ASSERT_TRUE(initial.live.pose);
  auto source = initial.source;
  source.stamp_nsec = ++tick_ * 1000000000LL;
  source.pose_valid = false;
  auto frame = makeFrame(source.stamp_nsec, source.pose, unrelatedCloud());
  frame.source_frame_id = source.source_frame;
  frame.pose_valid = false;
  auto invalid = localizer_->processFrame(mode_, frame);
  live_->acceptBackend(source, {tick_ * 1000000000LL}, invalid);
  EXPECT_FALSE(live_->observeOdometry(source, source.stamp_nsec, true).pose);

  map_from_odom_ = makePose(70, 40, -M_PI_2);
  Step step;
  for (int i = 0; i < config_.reloc_temporal_window_size; ++i) {
    step = run("same_frame_after_invalid", makePose(6, 0, 0),
        synthesizeBodyCloud(scene_, makePose(6, 0, 0), 12, 0.15, 0.005, 900U+i));
    if (i + 1 < config_.reloc_temporal_window_size) {
      EXPECT_FALSE(step.backend.relocalization_locked);
      EXPECT_FALSE(step.live.pose);
    }
  }
  EXPECT_TRUE(step.backend.relocalization_locked);
  ASSERT_TRUE(step.live.pose);
  EXPECT_GT(step.live.pose->correction_stamp_nsec, initial.source.stamp_nsec);
  EXPECT_LT((step.live.pose->pose.translation()-Eigen::Vector3d(6, 0, 0)).norm(), 0.5);
}

TEST_F(LocalizationEpisode, GradualYawDriftStillRecoversFromFreshGeometry) {
  ASSERT_TRUE(lock().live.pose);
  const auto original = map_from_odom_;
  const auto mismatch = unrelatedCloud();
  for (int i = 1; i <= 45; ++i) {
    map_from_odom_ = original;
    map_from_odom_.linear() = Eigen::AngleAxisd(
        i*M_PI/90.0, Eigen::Vector3d::UnitZ()).toRotationMatrix()*original.linear();
    run("gradual_odom_yaw_drift", makePose(6, 0, 0), mismatch);
  }
  const auto recovered = lock(0.0, true);
  ASSERT_TRUE(recovered.live.pose);
  EXPECT_LT((recovered.live.pose->pose.translation()-Eigen::Vector3d(6, 0, 0)).norm(), 0.5);
  EXPECT_LT(Eigen::AngleAxisd(recovered.live.pose->pose.linear()).angle(), 5.0*M_PI/180.0);
}

}
}
