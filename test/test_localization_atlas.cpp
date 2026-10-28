#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <pcl/common/transforms.h>

#include "n3mapping/localization_atlas.h"
#include "n3mapping/cloud_utils.h"
#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/keyframe_manager.h"
#include "localization_atlas.pb.h"

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
  ASSERT_EQ(expected.cloud->covs.empty(), actual.cloud->covs.empty());
  // Prepared storage may be packed for tree locality on load. Compare the
  // complete records in canonical order, including duplicate coordinates and
  // their associated features; retain the bit-for-bit checks below.
  const auto ordered = [](const auto& cloud) {
    std::vector<std::size_t> indices(cloud.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
      int order = std::memcmp(cloud.point(a).data(), cloud.point(b).data(), 4 * sizeof(double));
      if (order == 0)
        order = std::memcmp(cloud.normal(a).data(), cloud.normal(b).data(), 4 * sizeof(double));
      if (order == 0 && !cloud.covs.empty())
        order = std::memcmp(cloud.cov(a).data(), cloud.cov(b).data(), 16 * sizeof(double));
      return order < 0;
    });
    return indices;
  };
  const auto expected_indices = ordered(*expected.cloud);
  const auto actual_indices = ordered(*actual.cloud);
  for (std::size_t i = 0; i < expected.cloud->size(); ++i) {
    const auto e = expected_indices[i], a = actual_indices[i];
    EXPECT_EQ(expected.cloud->point(e), actual.cloud->point(a));
    EXPECT_EQ(expected.cloud->normal(e), actual.cloud->normal(a));
    if (!expected.cloud->covs.empty())
      EXPECT_EQ(expected.cloud->cov(e), actual.cloud->cov(a));
    // Effective components must survive bit-for-bit, including signed zero.
    EXPECT_EQ(std::memcmp(expected.cloud->point(e).data(),
                          actual.cloud->point(a).data(), 3 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(expected.cloud->normal(e).data(),
                          actual.cloud->normal(a).data(), 3 * sizeof(double)), 0);
    for (int column = 0; !expected.cloud->covs.empty() && column < 3; ++column) {
      EXPECT_EQ(std::memcmp(expected.cloud->cov(e).col(column).data(),
                            actual.cloud->cov(a).col(column).data(),
                            3 * sizeof(double)), 0);
    }
  }
}

} // namespace

TEST(LocalizationAtlasTest, PackedAssemblyRetainsVoxelCentroidsAndUnsafeGridFallback) {
  auto frames = makeKeyframes();
  frames[1]->pose_optimized.linear() =
      Eigen::AngleAxisd(.37, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  frames[0]->cloud->is_dense = false;
  frames[0]->cloud->at(0).x = std::numeric_limits<float>::quiet_NaN();
  auto raw = pcl::make_shared<Keyframe::PointCloudT>();
  for (const auto& frame : frames) {
    Keyframe::PointCloudT transformed;
    pcl::transformPointCloud(*frame->cloud, transformed,
                             frame->pose_optimized.matrix().cast<float>());
    *raw += transformed;
  }
  for (const double leaf : {.08, .2, 1e-3, std::numeric_limits<double>::infinity()}) {
    auto config = makeConfig();
    config.global_map_voxel_size = leaf;
    Keyframe::PointCloudT::Ptr expected;
    if (!safeVoxelGridFilter<pcl::PointXYZI>(raw, leaf, &expected)) expected = raw;
    const auto actual = LocalizationAtlas::buildGlobalMap(config, frames);
    ASSERT_EQ(actual->size(), expected->size());
    for (size_t i = 0; i < actual->size(); ++i) {
      const auto& a = actual->at(i);
      const auto& b = expected->at(i);
      const float lhs[] = {a.x, a.y, a.z, a.intensity};
      const float rhs[] = {b.x, b.y, b.z, b.intensity};
      EXPECT_EQ(std::memcmp(lhs, rhs, sizeof(lhs)), 0) << i << " leaf=" << leaf;
    }
  }
  EXPECT_TRUE(LocalizationAtlas::buildGlobalMap(makeConfig(), {})->empty());
}

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
  ASSERT_NE(loaded.placeCatalog(), nullptr);
  ASSERT_EQ(loaded.placeCatalog()->size(), 2u);
  EXPECT_EQ(loaded.placeCatalog()->at(0), std::vector<int64_t>{0});
  EXPECT_EQ(loaded.placeCatalog()->at(1), std::vector<int64_t>{1});
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
  RHPDescriptor descriptor(rhpdParamsFromConfig(config));
  ASSERT_NE(compiled.placeCatalog(), nullptr);
  EXPECT_EQ(*compiled.placeCatalog(), *loaded.placeCatalog());
  ASSERT_EQ(compiled.frameDescriptors().size(), loaded.frameDescriptors().size());
  ASSERT_EQ(compiled.submapDescriptors().size(), loaded.submapDescriptors().size());
  ASSERT_EQ(compiled.headingDescriptors().size(), loaded.headingDescriptors().size());
  ASSERT_EQ(loaded.frameDescriptors().size(), makeKeyframes().size());
  for (const auto &keyframe : makeKeyframes()) {
    EXPECT_TRUE(rhpdDenseValues(*compiled.frameDescriptors().at(keyframe->id)).isApprox(
        rhpdDenseValues(*loaded.frameDescriptors().at(keyframe->id)), 0.0));
    EXPECT_TRUE(rhpdDenseValues(*compiled.submapDescriptors().at(keyframe->id)).isApprox(
        rhpdDenseValues(*loaded.submapDescriptors().at(keyframe->id)), 0.0));
    EXPECT_EQ(compiled.headingDescriptors().at(keyframe->id)->values,
              loaded.headingDescriptors().at(keyframe->id)->values);
    EXPECT_TRUE(rhpdDenseValues(*loaded.frameDescriptors().at(keyframe->id)).isApprox(
        rhpdDenseValues(descriptor.compute(keyframe->cloud)), 0.0));
  }
  KeyframeManager manager(config);
  ASSERT_TRUE(manager.loadKeyframes(makeKeyframes()));
  LoopDetector rebuilt(config), cached(config);
  rebuilt.restoreMapDescriptors(manager);
  cached.restoreMapDescriptors(manager, loaded.submapDescriptors(), loaded.headingDescriptors());
  ASSERT_EQ(loaded.submapDescriptors().size(), makeKeyframes().size());
  ASSERT_EQ(loaded.headingDescriptors().size(), makeKeyframes().size());
  for (const auto& [id, place] : rebuilt.getRHPDManager().snapshot()) {
    const auto other = cached.getRHPDManager().snapshot().at(id);
    EXPECT_TRUE(rhpdDenseValues(*place).isApprox(rhpdDenseValues(*other), 0.0));
  }
  const auto expected_headings = rebuilt.getDescriptors();
  const auto actual_headings = cached.getDescriptors();
  ASSERT_EQ(expected_headings.size(), actual_headings.size());
  for (size_t i = 0; i < expected_headings.size(); ++i) {
    const auto& a = *expected_headings[i].second;
    const auto& b = *actual_headings[i].second;
    EXPECT_EQ(a.offsets, b.offsets);
    EXPECT_EQ(a.ring_indices, b.ring_indices);
    EXPECT_EQ(a.values, b.values);
    EXPECT_EQ(a.ring_key, b.ring_key);
    EXPECT_EQ(a.column_norms, b.column_norms);
    EXPECT_EQ(a.sector_key, b.sector_key);
  }
  EXPECT_EQ(compiled.mapSha256(), loaded.mapSha256());
  EXPECT_EQ(compiled.configSha256(), loaded.configSha256());
  const auto built_grid = compiled.takeFreeSpaceGrid();
  const auto loaded_grid = loaded.takeFreeSpaceGrid();
  ASSERT_TRUE(built_grid.valid());
  ASSERT_TRUE(loaded_grid.valid());
  EXPECT_EQ(built_grid.occupiedCells(), loaded_grid.occupiedCells());
  EXPECT_EQ(built_grid.freeCells(), loaded_grid.freeCells());
  EXPECT_EQ(loaded_grid.buildSeconds(), 0.);
  const auto query = makeStructuredCloud();
  for (int i = -12; i <= 12; ++i) {
    auto pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(i * .17, i * .11, i * .07);
    const auto expected = built_grid.score(*query, pose);
    const auto actual = loaded_grid.score(*query, pose);
    EXPECT_EQ(expected.valid, actual.valid);
    EXPECT_EQ(expected.scored_points, actual.scored_points);
    EXPECT_EQ(expected.occupied_points, actual.occupied_points);
    EXPECT_EQ(expected.free_points, actual.free_points);
    EXPECT_DOUBLE_EQ(expected.value, actual.value);
    EXPECT_DOUBLE_EQ(expected.stddev, actual.stddev);
  }
  EXPECT_FALSE(loaded.takeFreeSpaceGrid().valid());
}

TEST(LocalizationAtlasTest, FreeSpaceCacheMissFallsBackAndMalformedMatchingCacheIsRejected) {
  TemporaryAtlasFiles files;
  const auto config = makeConfig();
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), makeKeyframes(),
                                  files.atlas().string(), false, nullptr, &error));
  for (int change = 0; change < 5; ++change) {
    auto changed = config;
    if (change == 0) changed.reloc_free_space_resolution *= 2;
    if (change == 1) changed.reloc_free_space_max_ray_length *= 2;
    if (change == 2) changed.reloc_free_space_occupied_min_points += 1;
    if (change == 3) changed.reloc_free_space_map_pcd = "external.pcd";
    if (change == 4) changed.reloc_free_space_enable = false;
    LocalizationAtlas other(changed, matcher);
    ASSERT_TRUE(other.load(files.map().string(), files.atlas().string(), nullptr, &error)) << error;
    EXPECT_FALSE(other.takeFreeSpaceGrid().valid());
  }
  LocalizationAtlasData data;
  { std::ifstream in(files.atlas(), std::ios::binary); ASSERT_TRUE(data.ParseFromIstream(&in)); }
  ASSERT_TRUE(data.has_free_space_grid());
  data.mutable_free_space_grid()->mutable_occupied()->pop_back();
  { std::ofstream out(files.atlas(), std::ios::binary); ASSERT_TRUE(data.SerializeToOstream(&out)); }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_NE(error.find("free-space"), std::string::npos);
  data.clear_free_space_grid();
  data.clear_free_space_config();
  { std::ofstream out(files.atlas(), std::ios::binary); ASSERT_TRUE(data.SerializeToOstream(&out)); }
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error)) << error;
  EXPECT_FALSE(atlas.takeFreeSpaceGrid().valid());
}

TEST(LocalizationAtlasTest, DescriptorConfigMissRebuildsButMalformedMatchingCacheIsRejected) {
  TemporaryAtlasFiles files;
  Config config = makeConfig();
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), makeKeyframes(),
                                  files.atlas().string(), false, nullptr, &error));
  Config changed = config;
  changed.rhpd_z_min -= .1;
  LocalizationAtlas changed_atlas(changed, matcher);
  ASSERT_TRUE(changed_atlas.load(files.map().string(), files.atlas().string(), nullptr, &error)) << error;
  EXPECT_TRUE(changed_atlas.frameDescriptors().empty());
  LocalizationAtlasData data;
  { std::ifstream in(files.atlas(), std::ios::binary); ASSERT_TRUE(data.ParseFromIstream(&in)); }
  ASSERT_GT(data.frame_descriptors_size(), 0);
  data.mutable_frame_descriptors(0)->mutable_ring_height()->add_indices(999);
  { std::ofstream out(files.atlas(), std::ios::binary); ASSERT_TRUE(data.SerializeToOstream(&out)); }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_NE(error.find("descriptor cache"), std::string::npos);
}

TEST(LocalizationAtlasTest, LoopDescriptorFallbackAndMalformedHeading) {
  TemporaryAtlasFiles files;
  const auto config = makeConfig();
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), makeKeyframes(),
                                  files.atlas().string(), false, nullptr, &error));
  Config changed = config;
  ++changed.rhpd_submap_kf_radius;
  LocalizationAtlas changed_atlas(changed, matcher);
  ASSERT_TRUE(changed_atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_TRUE(changed_atlas.submapDescriptors().empty());
  EXPECT_TRUE(changed_atlas.headingDescriptors().empty());
  LocalizationAtlasData data;
  { std::ifstream in(files.atlas(), std::ios::binary); ASSERT_TRUE(data.ParseFromIstream(&in)); }
  const auto original = data;
  data.clear_loop_descriptor_config();
  data.clear_loop_descriptors();
  { std::ofstream out(files.atlas(), std::ios::binary); ASSERT_TRUE(data.SerializeToOstream(&out)); }
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_TRUE(atlas.submapDescriptors().empty());
  data = original;
  data.mutable_loop_descriptors(0)->set_heading_offsets(1, 99999999);
  { std::ofstream out(files.atlas(), std::ios::binary); ASSERT_TRUE(data.SerializeToOstream(&out)); }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_NE(error.find("loop descriptor cache"), std::string::npos);
}

TEST(LocalizationAtlasTest, CompiledPlaceViewsMatchLoadedMapPreparation) {
  TemporaryAtlasFiles files;
  Config config = makeConfig();
  config.rhpd_submap_kf_radius = 0;
  config.rhpd_submap_voxel_size = 0.0;
  auto frames = makeKeyframes();
  for (int i = 2; i < 6; ++i) {
    auto repeat = Keyframe::create(i, i, Eigen::Isometry3d::Identity(), frames[0]->cloud);
    frames.push_back(repeat);
  }
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), frames, files.atlas().string(), false, nullptr, &error));
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error)) << error;
  ASSERT_NE(atlas.placeCatalog(), nullptr);
  ASSERT_EQ(atlas.placeCatalog()->size(), 2u);
  EXPECT_EQ(atlas.placeCatalog()->at(0), std::vector<int64_t>{0});
  EXPECT_EQ(atlas.placeCatalog()->at(1), std::vector<int64_t>{1});
  // Catalog construction is a map artifact operation; input observations stay intact.
  EXPECT_EQ(frames.size(), 6u);
  EXPECT_FALSE(frames[0]->is_from_loaded_map);
  LocalizationAtlasData data;
  { std::ifstream in(files.atlas(), std::ios::binary); ASSERT_TRUE(data.ParseFromIstream(&in)); }
  data.mutable_places(1)->add_view_ids(0);
  { std::ofstream out(files.atlas(), std::ios::binary); ASSERT_TRUE(data.SerializeToOstream(&out)); }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_NE(error.find("place catalog"), std::string::npos);
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

TEST(LocalizationAtlasTest, KeepsExplicitCovarianceWhenItIsNotTheFixedNormalModel) {
  TemporaryAtlasFiles files;
  const Config config = makeConfig();
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), makeKeyframes(),
                                  files.atlas().string(), false, nullptr, &error));
  ASSERT_TRUE(atlas.preparedTarget().refine_level.cloud->covs.empty());
  LocalizationAtlasData data;
  {
    std::ifstream stream(files.atlas(), std::ios::binary);
    ASSERT_TRUE(data.ParseFromIstream(&stream));
  }
  auto* level = data.mutable_refine_level();
  ASSERT_TRUE(level->covariance_from_normal());
  level->set_covariance_from_normal(false);
  for (uint64_t i = 0; i < level->num_points(); ++i) {
    Eigen::Vector3d normal;
    for (int axis = 0; axis < 3; ++axis) normal[axis] = level->normals(3 * i + axis);
    const Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity() - .999 * normal * normal.transpose();
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) level->add_covariances(covariance(row, col));
  }
  for (int i = level->covariances_size() - 9; i < level->covariances_size(); ++i)
    level->set_covariances(i, 2 * level->covariances(i));
  {
    std::ofstream stream(files.atlas(), std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(data.SerializeToOstream(&stream));
  }
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  const auto& loaded = *atlas.preparedTarget().refine_level.cloud;
  ASSERT_EQ(loaded.covs.size(), loaded.size());
  // A change in storage order cannot hide a modified covariance model.
  size_t changed = 0;
  for (size_t i = 0; i < loaded.size(); ++i) {
    if (std::abs(loaded.cov(i).trace() - 4.002) < 1e-12) ++changed;
    else EXPECT_NEAR(loaded.cov(i).trace(), 2.001, 1e-12);
  }
  EXPECT_EQ(changed, 1u);
}

TEST(LocalizationAtlasTest, RejectsOldFormatAndTruncatedPreparedArrays) {
  TemporaryAtlasFiles files;
  const Config config = makeConfig();
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), makeKeyframes(),
                                  files.atlas().string(), false, nullptr, &error))
      << error;
  LocalizationAtlasData data;
  {
    std::ifstream stream(files.atlas(), std::ios::binary);
    ASSERT_TRUE(data.ParseFromIstream(&stream));
  }
  const auto current_version = data.format_version();
  data.set_format_version("n3mapping_localization_atlas_v1");
  {
    std::ofstream stream(files.atlas(), std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(data.SerializeToOstream(&stream));
  }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_FALSE(atlas.loaded());
  EXPECT_NE(error.find("rebuild from the source map"), std::string::npos);

  data.set_format_version(current_version);
  data.mutable_plane_levels(0)->mutable_points()->RemoveLast();
  {
    std::ofstream stream(files.atlas(), std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(data.SerializeToOstream(&stream));
  }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_FALSE(atlas.loaded());
  EXPECT_NE(error.find("array size mismatch"), std::string::npos);
}

} // namespace test
} // namespace n3mapping
