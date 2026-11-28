#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "n3mapping/localization_atlas.h"
#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/keyframe_manager.h"
#include "atlas_test_file.h"
#include "../src/point_cloud_target_view.h"

namespace n3mapping {
namespace test {
namespace {

struct AtlasLevelFeatures {
  const PointCloudMatcher::PreparedTargetLevel& level;
  explicit AtlasLevelFeatures(const PointCloudMatcher::PreparedTargetLevel& source) : level(source) {}
  std::size_t size() const { return level.size(); }
  bool hasExplicitCovariance() const {
    return level.mapped ? !level.mapped->covariance_from_normal : !level.cloud->covs.empty();
  }
  Eigen::Vector4d point(std::size_t i) const { return level.mapped ? level.mapped->point(i) : level.cloud->point(i); }
  Eigen::Vector4d normal(std::size_t i) const { return level.mapped ? level.mapped->normal(i) : level.cloud->normal(i); }
  Eigen::Matrix4d covariance(std::size_t i) const {
    return level.mapped ? level.mapped->covariance(i) :
        registration_detail::TargetView{*level.cloud, level.lazy_normals.get()}.covariance(i);
  }
};

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
  ASSERT_TRUE(expected.ready());
  ASSERT_TRUE(actual.ready());
  const AtlasLevelFeatures e(expected), a(actual);
  ASSERT_EQ(e.size(), a.size());
  ASSERT_EQ(e.hasExplicitCovariance(), a.hasExplicitCovariance());
  const auto ordered = [](const auto& view) {
    std::vector<std::size_t> indices(view.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](auto i, auto j) {
      const Eigen::Vector4f pi = view.point(i).template cast<float>(),
                            pj = view.point(j).template cast<float>(),
                            ni = view.normal(i).template cast<float>(),
                            nj = view.normal(j).template cast<float>();
      int order = std::memcmp(pi.data(), pj.data(), 4 * sizeof(float));
      if (order == 0) order = std::memcmp(ni.data(), nj.data(), 3 * sizeof(float));
      if (order == 0 && view.hasExplicitCovariance()) {
        const auto ci = view.covariance(i), cj = view.covariance(j);
        order = std::memcmp(ci.data(), cj.data(), 16 * sizeof(double));
      }
      return order < 0;
    });
    return indices;
  };
  const auto ei = ordered(e), ai = ordered(a);
  for (std::size_t i = 0; i < e.size(); ++i) {
    const Eigen::Vector4d ep = e.point(ei[i]).cast<float>().cast<double>(), ap = a.point(ai[i]);
    const Eigen::Vector4d en = e.normal(ei[i]).cast<float>().cast<double>(), an = a.normal(ai[i]);
    EXPECT_EQ(ep, ap);
    EXPECT_EQ(en, an);
    EXPECT_EQ(std::memcmp(ep.data(), ap.data(), 3 * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(en.data(), an.data(), 3 * sizeof(double)), 0);
    if (e.hasExplicitCovariance()) {
      const auto ec = e.covariance(ei[i]), ac = a.covariance(ai[i]);
      EXPECT_EQ(ec, ac);
      for (int column = 0; column < 3; ++column)
        EXPECT_EQ(std::memcmp(ec.col(column).data(), ac.col(column).data(), 3 * sizeof(double)), 0);
    }
  }
}

}

TEST(FreeSpaceGridTest, SparseAndMappedScoresMatchDenseReferenceAcrossBlocks) {
  constexpr double resolution = 0.5;
  constexpr int side = 24;
  const Eigen::Vector3d origin(-1, -1, -1);
  const Eigen::Vector3d sensor(0.37, 0.63, 0.89);
  const auto cell = [&](const Eigen::Vector3d& p) {
    return Eigen::Vector3i(((p - origin) / resolution).array().floor().cast<int>());
  };
  const auto inside = [=](const Eigen::Vector3i& c) {
    return (c.array() >= 0).all() && (c.array() < side).all();
  };
  const auto index = [](const Eigen::Vector3i& c) {
    return (c.x() * side + c.y()) * side + c.z();
  };

  pcl::PointCloud<pcl::PointXYZI> cloud;
  std::vector<unsigned> hits(side * side * side);
  const auto add_point = [&](float x, float y, float z) {
    pcl::PointXYZI p;
    p.x = x; p.y = y; p.z = z;
    cloud.push_back(p);
    ++hits[index(cell(Eigen::Vector3d(x, y, z)))];
  };
  add_point(0, 0, 0);
  add_point(9.5, 9.5, 9.5);
  for (int i = 1; i < 9; ++i) {
    for (int j = 1; j < 9; ++j) {
      add_point(i + .125f, j + .125f, 4.125f);
      if ((i + j) % 3) add_point(i + .375f, j + .375f, 4.375f);
    }
  }
  std::vector<bool> free(hits.size());
  for (const auto& p : cloud) {
    const Eigen::Vector3d delta = Eigen::Vector3d(p.x, p.y, p.z) - sensor;
    const double length = delta.norm();
    const Eigen::Vector3d direction = delta / length;
    for (int step = 1; step < static_cast<int>(length / resolution); ++step) {
      const auto c = cell(sensor + direction * (static_cast<double>(step) * resolution));
      if (!inside(c) || hits[index(c)] >= 2) break;
      free[index(c)] = true;
    }
  }

  FreeSpaceGrid built, mapped;
  ASSERT_TRUE(built.build(cloud, {sensor}, resolution, 30, 2));
  EXPECT_EQ(built.occupiedCells(), std::count_if(hits.begin(), hits.end(), [](unsigned n) { return n >= 2; }));
  EXPECT_EQ(built.freeCells(), std::count(free.begin(), free.end(), true));
  std::ostringstream output;
  ASSERT_TRUE(built.write(output));
  auto bytes = std::make_shared<std::string>(output.str());
  ASSERT_TRUE(mapped.map(bytes->data(), bytes->size(), bytes));
  EXPECT_EQ(mapped.occupiedCells(), built.occupiedCells());
  EXPECT_EQ(mapped.freeCells(), built.freeCells());
  for (int x = -1; x < 40; ++x) {
    for (int y = 0; y < 19; ++y) {
      for (int z = 0; z < 19; ++z) {
        pcl::PointXYZI p;
        p.x = x * .25f - .81875f; p.y = y * .53125f; p.z = z * .28125f;
        const Eigen::Vector3d point(p.x, p.y, p.z);
        const Eigen::Vector3d local = (point - origin) / resolution - Eigen::Vector3d::Constant(0.5);
        const Eigen::Vector3i lower = local.array().floor().cast<int>();
        const Eigen::Vector3d fraction = local - lower.cast<double>();
        double occupied = 0, known_free = 0;
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
              const Eigen::Vector3i c = lower + Eigen::Vector3i(i, j, k);
              if (!inside(c)) continue;
              const double weight = (i ? fraction.x() : 1 - fraction.x()) *
                                    (j ? fraction.y() : 1 - fraction.y()) *
                                    (k ? fraction.z() : 1 - fraction.z());
              if (hits[index(c)] >= 2) occupied += weight;
              else if (free[index(c)]) known_free += weight;
            }
          }
        }
        pcl::PointCloud<pcl::PointXYZI> query;
        query.push_back(p);
        for (const auto* grid : {&built, &mapped}) {
          const auto score = grid->score(query, Eigen::Isometry3d::Identity());
          ASSERT_EQ(score.valid, inside(cell(point)));
          if (!score.valid) continue;
          ASSERT_EQ(score.occupied_points, occupied);
          ASSERT_EQ(score.free_points, known_free);
          ASSERT_EQ(score.value, occupied - known_free);
          ASSERT_EQ(score.known_points, occupied + known_free);
        }
      }
    }
  }
}

TEST(FreeSpaceGridTest, SurfaceBoundaryIsContinuousAndObservedFreeSpaceStillContradicts) {
  pcl::PointCloud<pcl::PointXYZI> floor;
  for (int x = -40; x <= 40; ++x) {
    for (int y = -40; y <= 40; ++y) {
      pcl::PointXYZI p;
      p.x = x * 0.05f; p.y = y * 0.05f; p.z = 0.0f;
      p.intensity = 1.0f;
      floor.push_back(p);
    }
  }
  FreeSpaceGrid grid;
  constexpr double resolution = 0.2, offset = 0.0001;
  ASSERT_TRUE(grid.build(floor, {Eigen::Vector3d(0, 0, 1)}, resolution, 30.0, 2));
  pcl::PointCloud<pcl::PointXYZI> query;
  for (const auto& p : floor) {
    if (std::abs(p.x) > 0.1f || std::abs(p.y) > 0.1f) continue;
    auto local = p; local.z -= 1.0f;
    query.push_back(local);
  }
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation().z() = 1.0 - offset;
  const auto below = grid.score(query, pose);
  pose.translation().z() = 1.0 + offset;
  const auto above = grid.score(query, pose);
  ASSERT_TRUE(below.valid && above.valid);
  EXPECT_GT(below.value, 0.0);
  EXPECT_GT(above.value, 0.0);
  EXPECT_LT(std::abs(above.value - below.value), 4.0 * offset / resolution);
  EXPECT_FALSE(above.significantlyBetterThan(below, 5.0));
  EXPECT_FALSE(below.significantlyBetterThan(above, 5.0));
  pose.translation().z() = 1.4;
  const auto contradicted = grid.score(query, pose);
  ASSERT_TRUE(contradicted.valid);
  EXPECT_LT(contradicted.value, 0.0);
  EXPECT_GT(contradicted.free_given_known, contradicted.occupied_given_known);
  EXPECT_TRUE(above.significantlyBetterThan(contradicted, 5.0));
  EXPECT_FALSE(contradicted.significantlyBetterThan(above, 5.0));
  pose.translation().z() = 4.0;
  EXPECT_FALSE(grid.score(query, pose).valid);
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

TEST(LocalizationAtlasTest, MappedGeometryKeepsOtherFloorsAndSurvivesReferenceReplacement) {
  TemporaryAtlasFiles files;
  auto config = makeConfig();
  config.rhpd_max_range = 4;
  config.gicp_max_correspondence_distance = .25;
  config.reloc_track_retry_corr_scale = 1;
  auto frames = makeKeyframes();
  for (const auto& translation : {Eigen::Vector3d(0, 0, 6), Eigen::Vector3d(75, -25, 6)}) {
    auto pose = Eigen::Isometry3d::Identity();
    pose.translation() = translation;
    frames.push_back(Keyframe::create(frames.size(), frames.size(), pose, makeStructuredCloud()));
  }
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  LocalizationAtlasStats stats;
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), frames, files.atlas().string(),
                                  false, &stats, &error)) << error;
  ASSERT_TRUE(stats.paged_geometry);
  const auto native = matcher.prepareTargetCloud(atlas.globalMap());
  const auto retained = atlas.preparedTarget();
  for (std::size_t i = 0; i < native.plane_levels.size(); ++i)
    expectPreparedLevelExact(native.plane_levels[i], retained.plane_levels[i]);
  expectPreparedLevelExact(native.refine_level, retained.refine_level);
  const auto& level = *retained.plane_levels.front().mapped;
  const auto check_tree = [&](auto&& self, std::uint32_t id) -> Eigen::AlignedBox3d {
    Eigen::AlignedBox3d bounds;
    const auto& node = level.node(id);
    if (node.left == UINT32_MAX) {
      for (auto i = node.node_type.lr.first; i < node.node_type.lr.last; ++i)
        bounds.extend(level.point(i).head<3>());
      return bounds;
    }
    const auto left = self(self, node.left);
    const auto right = self(self, node.right);
    const auto axis = node.node_type.sub.proj.axis;
    EXPECT_LE(left.max()[axis], node.node_type.sub.thresh);
    EXPECT_EQ(right.min()[axis], node.node_type.sub.thresh);
    return bounds.extend(left).extend(right);
  };
  check_tree(check_tree, level.header.root);
  const auto query = matcher.prepareSourceCloud(makeStructuredCloud());
  atlas.prefetchTarget(query, {Eigen::Isometry3d::Identity()});
  for (const auto& frame : frames) {
    const auto indexed = atlas.cropVisibilityTarget(frame->pose_optimized.translation());
    const auto original = LocalizationAtlas::cropGlobalMap(config, atlas.globalMap(),
                                                           frame->pose_optimized.translation());
    ASSERT_EQ(indexed->size(), original->size());
    for (std::size_t i = 0; i < original->size(); ++i)
      EXPECT_EQ(indexed->points[i].getVector4fMap(), original->points[i].getVector4fMap());
    auto initial = frame->pose_optimized;
    initial.translation() += Eigen::Vector3d(.02, -.01, .006);
    const auto expected = matcher.alignPrepared(native, query, initial);
    const auto actual = matcher.alignPrepared(retained, query, initial);
    ASSERT_TRUE(expected.success);
    ASSERT_TRUE(actual.success);
    EXPECT_EQ(expected.num_inliers, actual.num_inliers);
    EXPECT_TRUE(expected.T_target_source.matrix().isApprox(actual.T_target_source.matrix(), 1e-6));
    EXPECT_NEAR(expected.fitness_score, actual.fitness_score, 1e-8);
  }
  atlas.clear();
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), frames, files.atlas().string(),
                                  true, nullptr, &error)) << error;
  const auto expected = matcher.alignPrepared(native, query, frames.back()->pose_optimized);
  const auto actual = matcher.alignPrepared(retained, query, frames.back()->pose_optimized);
  ASSERT_TRUE(actual.success);
  EXPECT_TRUE(expected.T_target_source.matrix().isApprox(actual.T_target_source.matrix(), 1e-6));
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
  AtlasTestFile file(files.atlas());
  auto& section = file.section(atlas_codec::SectionTag::FreeSpace);
  --section.size;
  ASSERT_TRUE(file.write(files.atlas()));
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_NE(error.find("free-space"), std::string::npos);

  ++section.size;
  LocalizationAtlasData data;
  ASSERT_TRUE(file.readMetadata(data));
  data.clear_free_space_config();
  ASSERT_TRUE(file.writeMetadata(files.atlas(), data));
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
  AtlasTestFile file(files.atlas());
  ASSERT_TRUE(file.readMetadata(data));
  ASSERT_GT(data.frame_descriptors_size(), 0);
  data.mutable_frame_descriptors(0)->mutable_ring_height()->add_indices(999);
  ASSERT_TRUE(file.writeMetadata(files.atlas(), data));
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
  AtlasTestFile file(files.atlas());
  ASSERT_TRUE(file.readMetadata(data));
  const auto original = data;
  data.clear_loop_descriptor_config();
  data.clear_loop_descriptors();
  ASSERT_TRUE(file.writeMetadata(files.atlas(), data));
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_TRUE(atlas.submapDescriptors().empty());
  data = original;
  data.mutable_loop_descriptors(0)->set_heading_offsets(1, 99999999);
  ASSERT_TRUE(file.writeMetadata(files.atlas(), data));
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

  EXPECT_EQ(frames.size(), 6u);
  EXPECT_FALSE(frames[0]->is_from_loaded_map);
  LocalizationAtlasData data;
  AtlasTestFile file(files.atlas());
  ASSERT_TRUE(file.readMetadata(data));
  data.mutable_places(1)->add_view_ids(0);
  ASSERT_TRUE(file.writeMetadata(files.atlas(), data));
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
  ASSERT_FALSE(AtlasLevelFeatures(atlas.preparedTarget().refine_level).hasExplicitCovariance());
  using namespace atlas_codec;
  AtlasTestFile file(files.atlas());
  const auto section = file.section(SectionTag::RefineLevel);
  auto content = file.bytes.substr(section.offset, section.size);
  LevelHeader level;
  std::memcpy(&level, content.data(), sizeof(level));
  ASSERT_EQ(level.explicit_covariance, 0u);
  const auto stride = align8(12 * level.points);
  std::vector<double> covariance;
  for (std::uint64_t i = 0; i < level.points; ++i) {
    float normal_xyz[3];
    std::memcpy(normal_xyz, content.data() + sizeof(level) + stride + i * 12, 12);
    Eigen::Vector3d normal(normal_xyz[0], normal_xyz[1], normal_xyz[2]);
    normal.normalize();
    Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() - .999 * normal * normal.transpose();
    if (i + 1 == level.points) cov *= 2;
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) covariance.push_back(cov(row, col));
  }
  level.explicit_covariance = 1;
  std::memcpy(content.data(), &level, sizeof(level));
  content.insert(sizeof(level) + 2 * stride, reinterpret_cast<const char*>(covariance.data()),
                 covariance.size() * sizeof(double));
  file.replaceSection(SectionTag::RefineLevel, content);
  ASSERT_TRUE(file.write(files.atlas()));
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  const AtlasLevelFeatures loaded(atlas.preparedTarget().refine_level);
  ASSERT_TRUE(loaded.hasExplicitCovariance());

  size_t changed = 0;
  for (size_t i = 0; i < loaded.size(); ++i) {
    if (std::abs(loaded.covariance(i).trace() - 4.002) < 1e-12) ++changed;
    else EXPECT_NEAR(loaded.covariance(i).trace(), 2.001, 1e-12);
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
  using namespace atlas_codec;
  AtlasTestFile file(files.atlas());
  LocalizationAtlasData data;
  ASSERT_TRUE(file.readMetadata(data));
  data.set_format_version("n3mapping_localization_atlas_v4");
  {
    std::ofstream stream(files.atlas(), std::ios::binary);
    ASSERT_TRUE(data.SerializeToOstream(&stream));
  }
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_FALSE(atlas.loaded());
  EXPECT_NE(error.find("rebuild from the source map"), std::string::npos);
  ASSERT_TRUE(file.writeMetadata(files.atlas(), data));
  EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
  EXPECT_NE(error.find("rebuild from the source map"), std::string::npos);
}

TEST(LocalizationAtlasTest, ValidatesSectionDirectoryAndReadsOffsetsBeyond32Bits) {
  using namespace atlas_codec;
  TemporaryAtlasFiles files;
  const Config config = makeConfig();
  PointCloudMatcher matcher(config);
  LocalizationAtlas atlas(config, matcher);
  std::string error;
  ASSERT_TRUE(atlas.compileAndSave(files.map().string(), makeKeyframes(),
                                  files.atlas().string(), false, nullptr, &error)) << error;
  const auto point_count = atlas.globalMap()->size();
  const AtlasTestFile original(files.atlas());
  for (int corruption = 0; corruption < 8; ++corruption) {
    SCOPED_TRACE(corruption);
    auto file = original;
    auto& first_plane = file.section(SectionTag::PlaneLevel, 0);
    auto& second_plane = file.section(SectionTag::PlaneLevel, 1);
    if (corruption == 0) first_plane.size = UINT64_MAX;
    if (corruption == 1) second_plane.offset = first_plane.offset;
    if (corruption == 2) ++second_plane.offset;
    if (corruption == 3) second_plane.flags = 0;
    if (corruption == 4) second_plane.flags = 2;
    if (corruption == 5) --second_plane.size;
    if (corruption == 6) file.bytes.pop_back();
    if (corruption == 7) --file.section(SectionTag::GlobalPoints).size;
    ASSERT_TRUE(file.write(files.atlas()));
    EXPECT_FALSE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error));
    EXPECT_FALSE(atlas.loaded());
  }

  auto large = original;
  auto& metadata = large.section(SectionTag::Metadata);
  const auto contents = large.bytes.substr(metadata.offset, metadata.size);
  large.bytes.resize(metadata.offset);
  metadata.offset = (std::uint64_t{1} << 32) + 16;
  ASSERT_TRUE(large.write(files.atlas()));
  {
    std::fstream stream(files.atlas(), std::ios::binary | std::ios::in | std::ios::out);
    stream.seekp(metadata.offset);
    ASSERT_TRUE(stream.write(contents.data(), contents.size()));
  }
  ASSERT_TRUE(atlas.load(files.map().string(), files.atlas().string(), nullptr, &error)) << error;
  EXPECT_EQ(atlas.globalMap()->size(), point_count);
}

}
}
