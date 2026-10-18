// Diagnostic counterexamples; sparse-copy assertions currently fail in baseline too.
#include "../../../test/test_localization_episode.cpp"
namespace n3mapping::test { namespace {
TEST_F(LocalizationEpisode, SparseCopiesRemainAmbiguous) {
  config_.reloc_temporal_window_size = 5;
  config_.reloc_lock_min_margin = 0.35;
  config_.reloc_lock_log_likelihood_threshold = 2.0;
  localizer_ = std::make_unique<N3MappingCore>(
      config_, N3MappingCore::WorldCloudOutput::Disabled);
  const auto complete = synthesizeBodyCloud(scene_, makePose(6.0, 0.0, 0.0),
                                           12.0, 0.0, 0.0, 1442);
  for (const int stride : {2, 4, 10}) {
    auto partial = pcl::make_shared<Cloud>();
    for (size_t i = 0; i < complete->size(); ++i)
      if (i % stride != 0) partial->push_back((*complete)[i]);
    const auto path = directory_ / ("partial_" + std::to_string(stride) + ".pbstream");
    {
      N3MappingCore mapper(config_, N3MappingCore::WorldCloudOutput::Disabled);
      ASSERT_TRUE(mapper.processMappingFrame(makeFrame(
          1000000000LL, makePose(0.0, 0.0, 0.0), partial)).accepted_keyframe);
      ASSERT_TRUE(mapper.processMappingFrame(makeFrame(
          2000000000LL, makePose(20.0, 0.0, 0.0), complete)).accepted_keyframe);
      ASSERT_TRUE(mapper.saveMap(path.string()));
      compileAtlas(mapper, path);
    }
    ASSERT_TRUE(localizer_->loadMap(path.string()));
    startOutput();
    // Every observed return exists at both locations. More map samples at one
    // location do not identify where this partial scan was actually observed.
    for (int i = 0; i < 2 * config_.reloc_temporal_window_size; ++i) {
      const auto step = run("partial_alias", makePose(20.0, 0.0, 0.0), partial);
      EXPECT_FALSE(step.backend.relocalization_locked) << "stride=" << stride;
      EXPECT_FALSE(step.live.pose) << "stride=" << stride;
    }
  }
}

TEST_F(LocalizationEpisode, FractionalGridCopiesRemainAmbiguous) {
  config_.reloc_temporal_window_size = 5;
  config_.reloc_lock_min_margin = 0.35;
  config_.reloc_lock_log_likelihood_threshold = 2.0;
  localizer_ = std::make_unique<N3MappingCore>(
      config_, N3MappingCore::WorldCloudOutput::Disabled);
  const auto cloud = synthesizeBodyCloud(scene_, makePose(6.0, 0.0, 0.0),
                                        12.0, 0.0, 0.0, 1442);
  for (const double separation : {20.05, 20.10, 20.15, 20.20, 20.30}) {
    const auto path = directory_ / ("shift_" + std::to_string(separation) + ".pbstream");
    {
      N3MappingCore mapper(config_, N3MappingCore::WorldCloudOutput::Disabled);
      ASSERT_TRUE(mapper.processMappingFrame(makeFrame(
          1000000000LL, makePose(0.0, 0.0, 0.0), cloud)).accepted_keyframe);
      ASSERT_TRUE(mapper.processMappingFrame(makeFrame(
          2000000000LL, makePose(separation, 0.0, 0.0), cloud)).accepted_keyframe);
      ASSERT_TRUE(mapper.saveMap(path.string()));
      compileAtlas(mapper, path);
    }
    ASSERT_TRUE(localizer_->loadMap(path.string()));
    startOutput();
    for (int i = 0; i < 2 * config_.reloc_temporal_window_size; ++i) {
      const auto step = run("fractional_alias", makePose(0.0, 0.0, 0.0), cloud);
      EXPECT_FALSE(step.backend.relocalization_locked) << "separation=" << separation;
      EXPECT_FALSE(step.live.pose) << "separation=" << separation;
    }
  }
}
} }
