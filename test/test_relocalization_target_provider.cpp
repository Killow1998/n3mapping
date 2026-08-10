#include "n3mapping/relocalization_target_provider.h"

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

TEST(RelocTargetProviderTest, LegacyWithoutAtlasUsesFreshLocalTarget) {
  Config config;
  config.reloc_target_mode = "legacy_global_atlas";
  ProviderFixture fixture(config);
  const auto cloud = makeTargetCloud();

  const PreparedRelocTarget target =
      fixture.provider->getTarget(makeRequest(3, cloud));

  ASSERT_TRUE(target.valid());
  EXPECT_EQ(target.visibility_target, cloud);
  EXPECT_EQ(target.metrics.registration_source, "legacy_local_no_cache");
  EXPECT_EQ(target.metrics.target_points, cloud->size());
  EXPECT_DOUBLE_EQ(target.metrics.target_build_ms, 1.25);
  EXPECT_FALSE(target.metrics.cache_hit);
  EXPECT_FALSE(target.metrics.cache_miss);
  EXPECT_EQ(fixture.provider->diagnostics().cache_entries, 0u);
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
