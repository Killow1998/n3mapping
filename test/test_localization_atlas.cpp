#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "n3mapping/localization_atlas.h"

namespace n3mapping {
namespace test {
namespace {

class TemporaryAtlasFiles {
public:
  TemporaryAtlasFiles() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    directory_ = std::filesystem::temp_directory_path() /
                 ("n3mapping_atlas_test_" + std::to_string(nonce));
    std::filesystem::create_directories(directory_);
    map_ = directory_ / "map.pbstream";
    atlas_ = directory_ / "map.localization_atlas.pb";
    writeMap("map-v1");
  }

  ~TemporaryAtlasFiles() {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  void writeMap(const std::string &contents) {
    std::ofstream stream(map_, std::ios::binary | std::ios::trunc);
    stream << contents;
  }

  const std::filesystem::path &map() const { return map_; }
  const std::filesystem::path &atlas() const { return atlas_; }

private:
  std::filesystem::path directory_;
  std::filesystem::path map_;
  std::filesystem::path atlas_;
};

Keyframe::PointCloudT::Ptr makeStructuredCloud() {
  auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
  for (int x = -10; x <= 10; ++x) {
    for (int y = -10; y <= 10; ++y) {
      pcl::PointXYZI floor;
      floor.x = 0.2f * static_cast<float>(x);
      floor.y = 0.2f * static_cast<float>(y);
      floor.z = 0.03f * static_cast<float>((x + 2 * y) % 3);
      floor.intensity = static_cast<float>(x + 10);
      cloud->push_back(floor);

      pcl::PointXYZI wall = floor;
      wall.x = -2.0f;
      wall.z = 0.15f * static_cast<float>(y + 10);
      wall.intensity += 30.0f;
      cloud->push_back(wall);
    }
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = true;
  return cloud;
}

std::vector<Keyframe::Ptr> makeKeyframes() {
  auto first = Keyframe::create(0, 0.0, Eigen::Isometry3d::Identity(),
                                makeStructuredCloud());
  Eigen::Isometry3d second_pose = Eigen::Isometry3d::Identity();
  second_pose.translation() = Eigen::Vector3d(3.0, 1.0, 0.2);
  auto second = Keyframe::create(1, 1.0, second_pose, makeStructuredCloud());
  return {first, second};
}

Config makeConfig() {
  Config config;
  config.num_threads = 1;
  config.global_map_voxel_size = 0.05;
  config.gicp_downsampling_resolution = 0.2;
  config.gicp_num_neighbors = 5;
  config.icp_refine_use_gicp = true;
  config.icp_refine_downsampling_resolution = 0.1;
  return config;
}

void expectPreparedLevelExact(
    const PointCloudMatcher::PreparedTargetLevel &expected,
    const PointCloudMatcher::PreparedTargetLevel &actual) {
  ASSERT_DOUBLE_EQ(expected.resolution, actual.resolution);
  ASSERT_NE(expected.cloud, nullptr);
  ASSERT_NE(actual.cloud, nullptr);
  ASSERT_NE(actual.kdtree, nullptr);
  ASSERT_EQ(expected.cloud->size(), actual.cloud->size());
  for (std::size_t i = 0; i < expected.cloud->size(); ++i) {
    EXPECT_EQ(expected.cloud->point(i), actual.cloud->point(i));
    EXPECT_EQ(expected.cloud->normal(i), actual.cloud->normal(i));
    EXPECT_EQ(expected.cloud->cov(i), actual.cloud->cov(i));
  }
}

} // namespace

TEST(LocalizationAtlasTest, RoundTripsPreparedTargetExactly) {
  TemporaryAtlasFiles files;
  const Config config = makeConfig();
  PointCloudMatcher compiler_matcher(config);
  LocalizationAtlas compiled(config, compiler_matcher);
  LocalizationAtlasStats compile_stats;
  std::string error;
  ASSERT_TRUE(compiled.compileAndSave(files.map().string(), makeKeyframes(),
                                      files.atlas().string(), false,
                                      &compile_stats, &error))
      << error;
  ASSERT_GT(compile_stats.global_point_count, 0u);
  ASSERT_GT(compile_stats.prepared_point_count, 0u);

  PointCloudMatcher loader_matcher(config);
  LocalizationAtlas loaded(config, loader_matcher);
  LocalizationAtlasStats load_stats;
  ASSERT_TRUE(loaded.load(files.map().string(), files.atlas().string(),
                          &load_stats, &error))
      << error;
  ASSERT_TRUE(loaded.loaded());
  ASSERT_EQ(compiled.globalMap()->size(), loaded.globalMap()->size());
  for (std::size_t i = 0; i < compiled.globalMap()->size(); ++i) {
    EXPECT_EQ(compiled.globalMap()->points[i].x, loaded.globalMap()->points[i].x);
    EXPECT_EQ(compiled.globalMap()->points[i].y, loaded.globalMap()->points[i].y);
    EXPECT_EQ(compiled.globalMap()->points[i].z, loaded.globalMap()->points[i].z);
    EXPECT_EQ(compiled.globalMap()->points[i].intensity,
              loaded.globalMap()->points[i].intensity);
  }
  ASSERT_EQ(compiled.preparedTarget().plane_levels.size(),
            loaded.preparedTarget().plane_levels.size());
  for (std::size_t i = 0;
       i < compiled.preparedTarget().plane_levels.size(); ++i) {
    expectPreparedLevelExact(compiled.preparedTarget().plane_levels[i],
                             loaded.preparedTarget().plane_levels[i]);
  }
  ASSERT_TRUE(compiled.preparedTarget().has_refine_level);
  ASSERT_TRUE(loaded.preparedTarget().has_refine_level);
  expectPreparedLevelExact(compiled.preparedTarget().refine_level,
                           loaded.preparedTarget().refine_level);
  EXPECT_EQ(compiled.mapSha256(), loaded.mapSha256());
  EXPECT_EQ(compiled.configSha256(), loaded.configSha256());
}

TEST(LocalizationAtlasTest, RejectsChangedMapAndChangedTargetConfig) {
  TemporaryAtlasFiles files;
  const Config config = makeConfig();
  PointCloudMatcher compiler_matcher(config);
  LocalizationAtlas compiled(config, compiler_matcher);
  std::string error;
  ASSERT_TRUE(compiled.compileAndSave(files.map().string(), makeKeyframes(),
                                      files.atlas().string(), false, nullptr,
                                      &error))
      << error;

  files.writeMap("map-v2");
  PointCloudMatcher changed_map_matcher(config);
  LocalizationAtlas changed_map(config, changed_map_matcher);
  EXPECT_FALSE(changed_map.load(files.map().string(), files.atlas().string(),
                                nullptr, &error));
  EXPECT_NE(error.find("map SHA-256 mismatch"), std::string::npos);

  files.writeMap("map-v1");
  Config changed_config = config;
  changed_config.gicp_downsampling_resolution += 0.01;
  PointCloudMatcher changed_config_matcher(changed_config);
  LocalizationAtlas mismatched(changed_config, changed_config_matcher);
  EXPECT_FALSE(mismatched.load(files.map().string(), files.atlas().string(),
                               nullptr, &error));
  EXPECT_NE(error.find("config SHA-256 mismatch"), std::string::npos);
}

} // namespace test
} // namespace n3mapping
