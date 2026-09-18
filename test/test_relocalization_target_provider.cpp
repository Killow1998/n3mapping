#include "n3mapping/relocalization_target_provider.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <vector>

#include <gtest/gtest.h>

namespace n3mapping {
namespace {

using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

PointCloudT::Ptr makeTargetCloud(float x_offset = 0.0f) {
  auto cloud = pcl::make_shared<PointCloudT>();
  for (int x = 0; x < 20; ++x) {
    for (int y = 0; y < 20; ++y) {
      pcl::PointXYZI point;
      point.x = x_offset + static_cast<float>(x) * 0.1f;
      point.y = static_cast<float>(y) * 0.1f;
      point.z = static_cast<float>((x + 2 * y) % 7) * 0.03f;
      point.intensity = static_cast<float>(x + y);
      cloud->push_back(point);
    }
  }
  return cloud;
}

RelocTargetRequest makeRequest(int64_t anchor_id,
                               const PointCloudT::Ptr &cloud) {
  RelocTargetRequest request;
  request.anchor_id = anchor_id;
  request.map_revision = {7, 11, 13};
  request.crop_center =
      Eigen::Vector3d(static_cast<double>(anchor_id), 2.0, 0.0);
  request.local_target = cloud;
  request.target_build_ms = 1.25;
  return request;
}

struct ProviderFixture {
  explicit ProviderFixture(Config config)
      : config(std::move(config)), matcher(this->config),
        atlas(this->config, matcher),
        provider(makeRelocTargetProvider(this->config, matcher, atlas)) {}

  Config config;
  PointCloudMatcher matcher;
  LocalizationAtlas atlas;
  std::unique_ptr<RelocTargetProvider> provider;
};

TEST(RelocTargetProviderTest, SmallCropDoesNotRetainGlobalMapCapacity) {
  Config config;
  config.rhpd_max_range = 2.0;
  config.gicp_max_correspondence_distance = 0.0;
  auto map = pcl::make_shared<PointCloudT>();
  map->resize(10000);
  for (auto &point : *map) point.x = 100.0f;
  map->front().x = 2.0f;
  map->front().intensity = 17.0f;
  const auto crop = LocalizationAtlas::cropGlobalMap(config, map, Eigen::Vector3d::Zero());
  ASSERT_EQ(crop->size(), 1u);
  EXPECT_EQ(crop->front().x, 2.0f);
  EXPECT_EQ(crop->front().intensity, 17.0f);
  EXPECT_LT(crop->points.capacity(), map->size());
}

TEST(RelocTargetProviderTest, LegacyWithoutAtlasUsesBoundedLocalTarget) {
  Config config;
  config.reloc_target_mode = "legacy_global_atlas";
  ProviderFixture fixture(config);
  const auto cloud = makeTargetCloud();

  const PreparedRelocTarget target =
      fixture.provider->getTarget(makeRequest(3, cloud));

  ASSERT_TRUE(target.valid());
  EXPECT_EQ(target.visibility_target, cloud);
  EXPECT_EQ(target.metrics.registration_source, "legacy_local_lru");
  EXPECT_EQ(target.metrics.anchor_id, 3);
  EXPECT_EQ(target.metrics.target_points, cloud->size());
  EXPECT_DOUBLE_EQ(target.metrics.target_build_ms, 1.25);
  EXPECT_FALSE(target.metrics.cache_hit);
  EXPECT_TRUE(target.metrics.cache_miss);
  EXPECT_EQ(fixture.provider->diagnostics().cache_entries, 1u);
  EXPECT_EQ(target.metrics.effective_max_bytes, 128u * 1024 * 1024);
  EXPECT_EQ(target.metrics.effective_max_entries, 8u);
}

TEST(RelocTargetProviderTest, LazyCropReuseIsEquivalentAndCenterInvalidates) {
  Config config;
  config.reloc_target_mode = "legacy_global_atlas";
  ProviderFixture cached(config);
  config.reloc_target_mode = "local_no_cache";
  ProviderFixture uncached(config);
  auto request = makeRequest(3, {});
  const auto map = makeTargetCloud();
  std::size_t builds = 0;
  request.build_local_target = [&]() {
    ++builds;
    return LocalizationAtlas::cropGlobalMap(config, map, request.crop_center);
  };
  const auto first = cached.provider->getTarget(request);
  const auto reused = cached.provider->getTarget(request);
  const auto reference = uncached.provider->getTarget(request);
  ASSERT_TRUE(first.valid());
  ASSERT_TRUE(reference.valid());
  EXPECT_EQ(builds, 2u);
  EXPECT_EQ(reused.metrics.target_builds, 0u);
  EXPECT_EQ(reused.metrics.target_preparations, 0u);
  EXPECT_EQ(reused.visibility_target, first.visibility_target);
  EXPECT_EQ(reused.registration_target, first.registration_target);
  ASSERT_EQ(reused.visibility_target->size(), reference.visibility_target->size());
  for (std::size_t i = 0; i < reused.visibility_target->size(); ++i) {
    EXPECT_EQ(reused.visibility_target->at(i).getVector4fMap(),
              reference.visibility_target->at(i).getVector4fMap());
    EXPECT_EQ(reused.visibility_target->at(i).intensity,
              reference.visibility_target->at(i).intensity);
  }
  const auto source = cached.matcher.prepareSourceCloud(map);
  const auto a = cached.matcher.alignPrepared(*reused.registration_target, source,
                                               Eigen::Isometry3d::Identity());
  const auto b = uncached.matcher.alignPrepared(*reference.registration_target, source,
                                                 Eigen::Isometry3d::Identity());
  EXPECT_EQ(a.converged, b.converged);
  EXPECT_DOUBLE_EQ(a.fitness_score, b.fitness_score);
  EXPECT_DOUBLE_EQ(a.inlier_ratio, b.inlier_ratio);
  EXPECT_TRUE(a.T_target_source.matrix().isApprox(b.T_target_source.matrix(), 1e-12));
  request.crop_center.x() += 0.1;
  EXPECT_TRUE(cached.provider->getTarget(request).metrics.cache_miss);
  EXPECT_EQ(builds, 3u);
}

TEST(RelocTargetProviderTest, ClearDuringBuildDoesNotRepopulateCache) {
  ProviderFixture fixture(Config{});
  auto request = makeRequest(3, {});
  std::promise<void> started;
  std::promise<void> resume;
  auto resumed = resume.get_future();
  request.build_local_target = [&]() {
    started.set_value();
    resumed.wait();
    return makeTargetCloud();
  };
  auto result = std::async(std::launch::async, [&]() {
    return fixture.provider->getTarget(request);
  });
  started.get_future().wait();
  fixture.provider->clear();
  resume.set_value();
  EXPECT_TRUE(result.get().valid());
  EXPECT_EQ(fixture.provider->diagnostics().cache_entries, 0u);
  EXPECT_EQ(fixture.provider->diagnostics().cache_total_bytes, 0u);
}

TEST(RelocTargetProviderTest, CandidateFallbackCannotPopulateHypothesisCrop) {
  ProviderFixture fixture(Config{});
  auto request = makeRequest(3, {});
  request.build_local_target = [] { return pcl::make_shared<PointCloudT>(); };
  request.build_fallback_target = [] { return makeTargetCloud(); };
  EXPECT_TRUE(fixture.provider->getTarget(request).valid());
  EXPECT_EQ(fixture.provider->diagnostics().cache_entries, 0u);
  request.build_fallback_target = {};
  EXPECT_FALSE(fixture.provider->getTarget(request).valid());
}

TEST(RelocTargetProviderTest, ReuseConstructionAblation) {
  const auto cloud = makeTargetCloud();
  constexpr int kRequests = 8;
  for (const std::string mode : {"local_no_cache", "legacy_global_atlas"}) {
    Config config;
    config.reloc_target_mode = mode;
    ProviderFixture fixture(config);
    auto request = makeRequest(3, {});
    request.build_local_target = [&] {
      return LocalizationAtlas::cropGlobalMap(config, cloud, request.crop_center);
    };
    std::size_t builds = 0;
    std::size_t preparations = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < kRequests; ++i) {
      const auto target = fixture.provider->getTarget(request);
      ASSERT_TRUE(target.valid());
      builds += target.metrics.target_builds;
      preparations += target.metrics.target_preparations;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    const std::size_t expected = mode == "local_no_cache" ? kRequests : 1;
    EXPECT_EQ(builds, expected);
    EXPECT_EQ(preparations, expected);
    // Timings are evidence, not a load-sensitive pass/fail threshold.
    RecordProperty(mode + "_milliseconds", std::to_string(ms));
    RecordProperty(mode + "_builds", static_cast<int>(builds));
    RecordProperty(mode + "_preparations", static_cast<int>(preparations));
  }
}

TEST(RelocTargetProviderTest, LegacyUsesLoadedGlobalAtlasForRegistration) {
  Config config;
  config.reloc_target_mode = "legacy_global_atlas";
  ProviderFixture fixture(config);
  const auto cloud = makeTargetCloud();
  std::vector<Keyframe::Ptr> keyframes = {
      Keyframe::create(0, 1.0, Eigen::Isometry3d::Identity(), cloud)};

  const auto nonce =
      std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("n3mapping_target_provider_atlas_" + std::to_string(nonce));
  std::filesystem::create_directories(directory);
  const std::filesystem::path map_path = directory / "map.pbstream";
  const std::filesystem::path atlas_path = directory / "map.atlas.pb";
  {
    std::ofstream map_file(map_path, std::ios::binary | std::ios::trunc);
    map_file << "target-provider-test-map";
  }

  LocalizationAtlasStats stats;
  std::string error;
  ASSERT_TRUE(fixture.atlas.compileAndSave(
      map_path.string(), keyframes, atlas_path.string(), false, &stats, &error))
      << error;
  ASSERT_TRUE(fixture.atlas.load(map_path.string(), atlas_path.string(), &stats,
                                 &error))
      << error;

  const PreparedRelocTarget target =
      fixture.provider->getTarget(makeRequest(0, cloud));

  ASSERT_TRUE(target.valid());
  EXPECT_EQ(target.visibility_target, cloud);
  EXPECT_EQ(target.metrics.registration_source, "legacy_global_atlas");
  EXPECT_EQ(target.metrics.target_points, fixture.atlas.globalMap()->size());
  EXPECT_FALSE(target.metrics.cache_hit);
  EXPECT_FALSE(target.metrics.cache_miss);

  std::error_code cleanup_error;
  std::filesystem::remove_all(directory, cleanup_error);
}

TEST(RelocTargetProviderTest, LocalLruHitsAndMapRevisionInvalidates) {
  Config config;
  config.reloc_target_mode = "local_lru";
  config.reloc_target_cache_max_bytes = 64 * 1024 * 1024;
  config.reloc_target_cache_max_entries = 4;
  ProviderFixture fixture(config);
  RelocTargetRequest request = makeRequest(5, makeTargetCloud());

  const PreparedRelocTarget first = fixture.provider->getTarget(request);
  const PreparedRelocTarget second = fixture.provider->getTarget(request);

  ASSERT_TRUE(first.valid());
  ASSERT_TRUE(second.valid());
  EXPECT_TRUE(first.metrics.cache_miss);
  EXPECT_TRUE(second.metrics.cache_hit);
  EXPECT_EQ(second.metrics.cache_entries, 1u);
  EXPECT_GT(second.metrics.cache_entry_bytes, 0u);

  ++request.map_revision.pose_revision;
  const PreparedRelocTarget after_revision =
      fixture.provider->getTarget(request);
  EXPECT_TRUE(after_revision.metrics.cache_miss);
  EXPECT_FALSE(after_revision.metrics.cache_hit);
  EXPECT_EQ(after_revision.metrics.cache_entries, 1u);
}

TEST(RelocTargetProviderTest, LocalLruHonorsEntryBudget) {
  Config config;
  config.reloc_target_mode = "local_lru";
  config.reloc_target_cache_max_bytes = 64 * 1024 * 1024;
  config.reloc_target_cache_max_entries = 1;
  ProviderFixture fixture(config);

  ASSERT_TRUE(
      fixture.provider->getTarget(makeRequest(1, makeTargetCloud())).valid());
  ASSERT_TRUE(fixture.provider->getTarget(makeRequest(2, makeTargetCloud(4.0f)))
                  .valid());

  const auto diagnostics = fixture.provider->diagnostics();
  EXPECT_EQ(diagnostics.cache_entries, 1u);
  EXPECT_LE(diagnostics.cache_total_bytes,
            static_cast<std::size_t>(config.reloc_target_cache_max_bytes));
}

TEST(RelocTargetProviderTest, ZeroBudgetNeverRetainsPreparedTargets) {
  Config config;
  config.reloc_target_mode = "local_lru";
  config.reloc_target_cache_max_bytes = 0;
  config.reloc_target_cache_max_entries = 0;
  ProviderFixture fixture(config);
  const RelocTargetRequest request = makeRequest(8, makeTargetCloud());

  const PreparedRelocTarget first = fixture.provider->getTarget(request);
  const PreparedRelocTarget second = fixture.provider->getTarget(request);

  EXPECT_TRUE(first.valid());
  EXPECT_TRUE(second.valid());
  EXPECT_TRUE(first.metrics.cache_miss);
  EXPECT_TRUE(second.metrics.cache_miss);
  EXPECT_EQ(fixture.provider->diagnostics().cache_entries, 0u);
  EXPECT_EQ(fixture.provider->diagnostics().cache_total_bytes, 0u);
}

TEST(RelocTargetProviderTest, OversizedPreparedTargetIsNotCached) {
  Config sizing_config;
  sizing_config.reloc_target_mode = "local_lru";
  sizing_config.reloc_target_cache_max_bytes = 64 * 1024 * 1024;
  sizing_config.reloc_target_cache_max_entries = 2;
  ProviderFixture sizing_fixture(sizing_config);
  const RelocTargetRequest request = makeRequest(10, makeTargetCloud());
  const PreparedRelocTarget sized = sizing_fixture.provider->getTarget(request);
  ASSERT_GT(sized.metrics.cache_entry_bytes, 1u);

  Config bounded_config = sizing_config;
  bounded_config.reloc_target_cache_max_bytes =
      static_cast<int>(sized.metrics.cache_entry_bytes - 1);
  ProviderFixture bounded_fixture(bounded_config);
  const PreparedRelocTarget bounded =
      bounded_fixture.provider->getTarget(request);

  EXPECT_TRUE(bounded.valid());
  EXPECT_TRUE(bounded.metrics.cache_miss);
  EXPECT_EQ(bounded_fixture.provider->diagnostics().cache_entries, 0u);
  EXPECT_EQ(bounded_fixture.provider->diagnostics().cache_total_bytes, 0u);
}

TEST(RelocTargetProviderTest, ConcurrentHitsKeepSharedTargetAlive) {
  Config config;
  config.reloc_target_mode = "local_lru";
  config.reloc_target_cache_max_bytes = 64 * 1024 * 1024;
  config.reloc_target_cache_max_entries = 2;
  ProviderFixture fixture(config);
  const RelocTargetRequest request = makeRequest(9, makeTargetCloud());
  ASSERT_TRUE(fixture.provider->getTarget(request).valid());

  std::vector<std::future<PreparedRelocTarget>> futures;
  for (int i = 0; i < 8; ++i) {
    futures.push_back(std::async(std::launch::async, [&fixture, request]() {
      return fixture.provider->getTarget(request);
    }));
  }
  for (auto &future : futures) {
    const PreparedRelocTarget target = future.get();
    EXPECT_TRUE(target.valid());
    EXPECT_TRUE(target.metrics.cache_hit);
  }

  const auto diagnostics = fixture.provider->diagnostics();
  EXPECT_EQ(diagnostics.cache_entries, 1u);
  EXPECT_GE(diagnostics.cache_hits, 8u);
}

} // namespace
} // namespace n3mapping
