#include "n3mapping/relocalizability_analyzer.h"

#include <chrono>
#include <fstream>
#include <iterator>

#include <gtest/gtest.h>
#include <json/json.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include "n3mapping/config.h"
#include "n3mapping/free_space_grid.h"
#include "n3mapping/global_map_cache.h"
#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

Keyframe::PointCloudT::Ptr geometry(bool corner) {
  auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
  pcl::PointXYZI point;
  point.intensity = 1.0f;
  for (int x = -50; x <= 50; ++x) {
    for (int z = -10; z <= 18; ++z) {
      point.x = x * 0.1f;
      point.z = z * 0.1f;
      for (const float y : {-2.0f, 1.7f}) {
        point.y = y;
        cloud->push_back(point);
      }
    }
    for (int y = -20; y <= 17; ++y) {
      point.y = y * 0.1f;
      point.z = -1.0f;
      cloud->push_back(point);
    }
  }
  if (corner) {
    for (int y = -45; y <= 45; ++y) {
      for (int z = -10; z <= 18; ++z) {
        point.x = 3.0f;
        point.y = y * 0.1f;
        point.z = z * 0.1f;
        cloud->push_back(point);
      }
    }
  }
  return cloud;
}

std::vector<Keyframe::Ptr> scans(bool corner, double floor_z = 0.0, MapSessionId session = 0) {
  const auto world = geometry(corner);
  std::vector<Keyframe::Ptr> frames;
  for (int index = 0; index < 5; ++index) {
    Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
    local.translation().x() = (index - 2) * 0.25;
    auto cloud = pcl::make_shared<Keyframe::PointCloudT>();
    const Eigen::Matrix4f inverse = local.inverse().matrix().cast<float>();
    pcl::transformPointCloud(*world, *cloud, inverse);
    auto pose = local;
    pose.translation().z() = floor_z;
    auto frame = Keyframe::create(10 * session + 2 * index, index, pose, cloud);
    frame->session_id = session;
    frames.push_back(std::move(frame));
  }
  return frames;
}

Config analysisConfig() {
  Config config;
  config.num_threads = 2;
  return config;
}

TEST(RelocalizabilityAnalyzerTest, TwinFloorsRemainRemoteAndFlagEachOther) {
  auto frames = scans(true);
  const auto upstairs = scans(true, 4.0, 1);
  frames.insert(frames.end(), upstairs.begin(), upstairs.end());
  const auto results = analyzeRelocalizability(analysisConfig(), frames, FreeSpaceGrid{}, {});
  ASSERT_EQ(results.size(), 10u);
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    ASSERT_TRUE(result.top_confuser);
    ASSERT_TRUE(result.directional_confusers[0] && result.directional_confusers[1]);
    EXPECT_FALSE(result.directional_confusers[0]->reversed_180);
    EXPECT_TRUE(result.directional_confusers[1]->reversed_180);
    EXPECT_EQ(result.top_confuser->keyframe_id, frames[(index + 5) % 10]->id);
    EXPECT_DOUBLE_EQ(std::abs(result.top_confuser->dz_m), 4.0);
    EXPECT_DOUBLE_EQ(result.top_confuser->xy_dist_m, 0.0);
    EXPECT_EQ(result.color, RelocalizabilityColor::Red);
    EXPECT_EQ(result.reasons.front(), "remote_floor_alias");
  }
}

TEST(RelocalizabilityAnalyzerTest, CorridorWeakAxisAndCornerSupport) {
  const auto config = analysisConfig();
  const auto corridor = analyzeRelocalizability(config, scans(false), FreeSpaceGrid{}, {});
  const auto corner = analyzeRelocalizability(config, scans(true), FreeSpaceGrid{}, {});
  ASSERT_TRUE(corridor[1].planar_observability && corner[1].planar_observability);
  EXPECT_LT(*corridor[1].planar_observability, *corner[1].planar_observability / 5.0);
  ASSERT_TRUE(corridor[1].weakest_axis_yaw_rad);
  EXPECT_NEAR(*corridor[1].weakest_axis_yaw_rad, 0.0, 0.05);
  EXPECT_GT(corridor[1].num_inliers, 100u);
  EXPECT_NE(corridor[1].color, RelocalizabilityColor::Green);
  EXPECT_EQ(corridor[1].reasons.front(), "corridor_degeneracy");
  EXPECT_EQ(corner[1].color, RelocalizabilityColor::Green);
  EXPECT_EQ(corner[1].reasons.front(), "good");
}

TEST(RelocalizabilityAnalyzerTest, ExportsJsonAndColoredPointClouds) {
  const auto frames = scans(true);
  RelocalizabilityReport report;
  report.map_sha256 = std::string(64, 'a');
  report.map_frame = "test_map";
  report.keyframes = analyzeRelocalizability(analysisConfig(), frames, FreeSpaceGrid{}, {});
  const auto global = GlobalMapCache::build(frames, 0.2);
  struct OutputDirectory {
    std::filesystem::path path;
    ~OutputDirectory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } output{std::filesystem::temp_directory_path() /
      ("n3mapping_heatmap_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))};

  std::string error;
  ASSERT_TRUE(saveRelocalizabilityReport(report, *global, output.path, error)) << error;
  Json::Value json;
  std::ifstream input(output.path / "relocalizability.json");
  ASSERT_TRUE(Json::parseFromStream(Json::CharReaderBuilder{}, input, &json, &error)) << error;
  EXPECT_EQ(json["map_sha256"].asString(), report.map_sha256);
  EXPECT_EQ(json["coordinate_frame"].asString(), "test_map");
  EXPECT_TRUE(json.isMember("algorithm_version") && json.isMember("thresholds"));
  ASSERT_EQ(json["keyframes"].size(), 5u);
  const auto& row = json["keyframes"][1];
  EXPECT_EQ(row["color"].asString(), "GREEN");
  EXPECT_TRUE(row.isMember("top_confuser") && row.isMember("remote_heading_distance"));
  EXPECT_TRUE(row["planar_observability"].isDouble());
  EXPECT_EQ(row["planar_information_xy_yaw"].size(), 3u);
  EXPECT_EQ(json["summary"]["colors"]["GREEN"]["count"].asInt(), 5);

  for (const auto* name : {"relocalizability_keyframes.pcd", "relocalizability_map.pcd"}) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    ASSERT_EQ(pcl::io::loadPCDFile((output.path / name).string(), cloud), 0);
    ASSERT_FALSE(cloud.empty());
    for (const auto& point : cloud) {
      EXPECT_EQ(point.r, 33);
      EXPECT_EQ(point.g, 163);
      EXPECT_EQ(point.b, 111);
    }
  }
}

TEST(RelocalizabilityAnalyzerTest, RotationSymmetryWithConstrainedTranslationFlagsYaw) {
  auto config = analysisConfig();
  config.gicp_num_neighbors = 5;
  config.gicp_downsampling_resolution = 0.05;
  auto target = pcl::make_shared<Keyframe::PointCloudT>();
  auto sparse_query = pcl::make_shared<Keyframe::PointCloudT>();
  for (int height = -18; height <= 18; ++height) {
    for (int side = 0; side < 4; ++side) {
      const bool on_x_axis = side % 2 == 0;
      const float radius = side < 2 ? 3.f : -3.f;
      pcl::PointXYZI center;
      center.x = on_x_axis ? radius : 0.0f;
      center.y = on_x_axis ? 0.0f : radius;
      center.z = 0.125f * height;
      center.intensity = 1.0f;
      sparse_query->push_back(center);
      for (int sample = -4; sample <= 4; ++sample) {
        auto point = center;
        if (on_x_axis) point.y += 0.0625f * sample;
        else point.x += 0.0625f * sample;
        target->push_back(point);
      }
    }
  }
  std::vector<Keyframe::Ptr> frames;
  for (int i = 0; i < 5; ++i)
    frames.push_back(Keyframe::create(i, i, Eigen::Isometry3d::Identity(),
                                     i == 2 ? sparse_query : target));
  const auto result = analyzeRelocalizability(config, frames, FreeSpaceGrid{}, {})[2];
  ASSERT_TRUE(result.planar_observability);
  EXPECT_GT(*result.planar_observability, 0.2);
  EXPECT_TRUE(result.yaw_unobservable);
  EXPECT_NE(result.color, RelocalizabilityColor::Green);
  EXPECT_NE(std::find(result.reasons.begin(), result.reasons.end(), "yaw_unobservable"),
            result.reasons.end());
}

TEST(RelocalizabilityAnalyzerTest, ParallelReportsAreIdenticalAndScanOrderIgnoresIds) {
  auto frames = scans(true);
  const auto upstairs = scans(true, 4.0, 1);
  frames.insert(frames.end(), upstairs.begin(), upstairs.end());
  struct Output {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("n3mapping_parallel_heatmap_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Output() { std::filesystem::remove_all(path); }
  } output;
  RelocalizabilityReport report;
  auto config = analysisConfig();
  config.num_threads = 1;
  report.keyframes = analyzeRelocalizability(config, frames, FreeSpaceGrid{}, {});
  const auto serial = report.keyframes;
  const auto global = GlobalMapCache::build(frames, 0.2);
  std::string error;
  ASSERT_TRUE(saveRelocalizabilityReport(report, *global, output.path / "serial", error)) << error;
  config.num_threads = 4;
  report.keyframes = analyzeRelocalizability(config, frames, FreeSpaceGrid{}, {});
  ASSERT_TRUE(saveRelocalizabilityReport(report, *global, output.path / "parallel", error)) << error;
  for (const auto* name : {"relocalizability.json", "relocalizability_keyframes.pcd",
                           "relocalizability_map.pcd"}) {
    std::ifstream a(output.path / "serial" / name, std::ios::binary);
    std::ifstream b(output.path / "parallel" / name, std::ios::binary);
    const std::string a_contents{std::istreambuf_iterator<char>(a), {}};
    const std::string b_contents{std::istreambuf_iterator<char>(b), {}};
    EXPECT_TRUE(a_contents == b_contents) << name;
  }

  for (std::size_t i = 0; i < frames.size(); ++i) frames[i]->id = i;
  const auto renumbered = analyzeRelocalizability(config, frames, FreeSpaceGrid{}, {});
  for (std::size_t i = 0; i < frames.size(); ++i) {
    EXPECT_EQ(renumbered[i].planar_observability, serial[i].planar_observability);
    EXPECT_EQ(renumbered[i].local_submap_keyframes, serial[i].local_submap_keyframes);
    EXPECT_EQ(renumbered[i].num_inliers, serial[i].num_inliers);
    EXPECT_EQ(renumbered[i].reasons, serial[i].reasons);
  }
}

}
}
