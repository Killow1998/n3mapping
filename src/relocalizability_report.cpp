#include "n3mapping/relocalizability_analyzer.h"

#include <algorithm>
#include <cmath>
#include <fstream>

#include <json/json.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {
constexpr std::array<const char*, 3> kColorNames{"GREEN", "YELLOW", "RED"};
constexpr std::array<std::array<std::uint8_t, 3>, 3> kPalette{{
    {33, 163, 111}, {234, 183, 53}, {220, 67, 70}}};

Json::Value confuserJson(const std::optional<RelocalizabilityConfuser>& confuser) {
  if (!confuser) return {};
  Json::Value value(Json::objectValue);
  value["keyframe_id"] = Json::Int64(confuser->keyframe_id);
  value["session_id"] = Json::UInt64(confuser->session_id);
  value["x"] = confuser->position.x();
  value["y"] = confuser->position.y();
  value["z"] = confuser->position.z();
  value["xy_dist_m"] = confuser->xy_dist_m;
  value["dz_m"] = confuser->dz_m;
  value["yaw_diff_rad"] = confuser->yaw_diff_rad;
  value["reversed_180"] = confuser->reversed_180;
  value["rhpd_distance"] = confuser->rhpd_distance;
  value["heading_distance"] = confuser->heading_distance;
  value["fused_score"] = confuser->fused_score;
  return value;
}

pcl::PointXYZRGB colorPoint(float x, float y, float z, RelocalizabilityColor color) {
  pcl::PointXYZRGB point;
  point.x = x;
  point.y = y;
  point.z = z;
  const auto& rgb = kPalette[static_cast<std::size_t>(color)];
  point.r = rgb[0];
  point.g = rgb[1];
  point.b = rgb[2];
  point.a = 255;
  return point;
}
}

bool saveRelocalizabilityReport(const RelocalizabilityReport& report,
                                 const pcl::PointCloud<pcl::PointXYZI>& global_map,
                                 const std::filesystem::path& output_dir, std::string& error) {
  if (report.keyframes.empty() || global_map.empty()) {
    error = "Cannot export an empty heatmap";
    return false;
  }
  std::error_code directory_error;
  std::filesystem::create_directories(output_dir, directory_error);
  if (directory_error) {
    error = directory_error.message();
    return false;
  }

  Json::Value root(Json::objectValue);
  root["algorithm_version"] = "n3mapping_relocalizability_v2";
  root["map_sha256"] = report.map_sha256;
  root["coordinate_frame"] = report.map_frame;
  root["used_atlas"] = report.used_atlas;
  root["free_space_valid"] = report.free_space_valid;
  root["map_voxel_size_m"] = report.map_voxel_size_m;
  auto& thresholds = root["thresholds"];
  thresholds["red_alias_score"] = report.thresholds.red_alias_score;
  thresholds["max_alias_score"] = report.thresholds.max_alias_score;
  thresholds["red_alias_margin"] = report.thresholds.red_alias_margin;
  thresholds["yellow_alias_margin"] = report.thresholds.yellow_alias_margin;
  thresholds["red_observability"] = report.thresholds.red_observability;
  thresholds["yellow_observability"] = report.thresholds.yellow_observability;
  thresholds["min_inliers"] = report.thresholds.min_inliers;
  thresholds["min_inlier_ratio"] = report.thresholds.min_inlier_ratio
      ? Json::Value(*report.thresholds.min_inlier_ratio) : Json::Value();
  thresholds["min_relative_inlier_ratio"] = report.thresholds.min_relative_inlier_ratio;

  std::array<Json::UInt64, 3> counts{};
  Json::Value keyframes(Json::arrayValue);
  pcl::PointCloud<pcl::PointXYZRGB> keyframe_cloud;
  keyframe_cloud.reserve(report.keyframes.size());
  auto positions = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  positions->reserve(report.keyframes.size());
  const auto number = [](const std::optional<double>& value) {
    return value && std::isfinite(*value) ? Json::Value(*value) : Json::Value();
  };
  for (const auto& row : report.keyframes) {
    const auto color = static_cast<std::size_t>(row.color);
    if (row.reasons.empty() || !row.position.cast<float>().allFinite() ||
        color >= counts.size() ||
        (row.top_confuser && !row.top_confuser->position.allFinite()) ||
        std::any_of(row.directional_confusers.begin(), row.directional_confusers.end(),
                    [](const auto& candidate) { return candidate && !candidate->position.allFinite(); })) {
      error = "Invalid heatmap keyframe reasons, coordinates or color";
      return false;
    }
    Json::Value value(Json::objectValue);
    value["keyframe_id"] = Json::Int64(row.keyframe_id);
    value["session_id"] = Json::UInt64(row.session_id);
    value["x"] = row.position.x();
    value["y"] = row.position.y();
    value["z"] = row.position.z();
    value["local_neighbor_fused_score"] = number(row.local_neighbor_fused_score);
    value["remote_rhpd_distance"] = number(row.top_confuser ?
        std::optional<double>(row.top_confuser->rhpd_distance) : std::nullopt);
    value["remote_heading_distance"] = number(row.top_confuser ?
        std::optional<double>(row.top_confuser->heading_distance) : std::nullopt);
    value["remote_fused_score"] = number(row.top_confuser ?
        std::optional<double>(row.top_confuser->fused_score) : std::nullopt);
    value["distinctiveness_margin"] = number(row.distinctiveness_margin);
    value["top_confuser"] = confuserJson(row.top_confuser);
    value["best_forward_confuser"] = confuserJson(row.directional_confusers[0]);
    value["best_reversed_confuser"] = confuserJson(row.directional_confusers[1]);
    value["planar_observability"] = number(row.planar_observability);
    value["minimum_planar_information_per_inlier"] = number(row.minimum_planar_information_per_inlier);
    value["yaw_unobservable"] = row.yaw_unobservable;
    value["xy_condition_number"] = number(row.xy_condition_number);
    value["weakest_axis_yaw_rad"] = number(row.weakest_axis_yaw_rad);
    value["num_inliers"] = Json::UInt64(row.num_inliers);
    value["inlier_ratio"] = row.inlier_ratio;
    value["local_median_inlier_ratio"] = number(row.local_median_inlier_ratio);
    value["relative_inlier_ratio"] = number(row.relative_inlier_ratio);
    value["local_submap_keyframes"] = Json::UInt64(row.local_submap_keyframes);
    value["planar_information_xy_yaw"] = Json::Value();
    value["yaw_information_per_inlier"] = Json::Value();
    if (row.planar_information) {
      Json::Value matrix(Json::arrayValue);
      for (int i = 0; i < 3; ++i) {
        Json::Value entries(Json::arrayValue);
        for (int j = 0; j < 3; ++j) entries.append((*row.planar_information)(i, j));
        matrix.append(std::move(entries));
      }
      value["planar_information_xy_yaw"] = std::move(matrix);
      value["yaw_information_per_inlier"] = (*row.planar_information)(2, 2);
    }
    ++counts[color];
    value["color"] = kColorNames[color];
    value["reason"] = row.reasons.front();
    auto& reasons = value["reason_codes"];
    for (const auto& reason : row.reasons) reasons.append(reason);
    keyframes.append(std::move(value));
    keyframe_cloud.push_back(colorPoint(row.position.x(), row.position.y(), row.position.z(), row.color));
    positions->push_back(pcl::PointXYZ(row.position.x(), row.position.y(), row.position.z()));
  }
  root["keyframes"] = std::move(keyframes);
  root["summary"]["keyframes"] = Json::UInt64(report.keyframes.size());
  root["summary"]["global_map_points"] = Json::UInt64(global_map.size());
  for (std::size_t color = 0; color < counts.size(); ++color) {
    auto& entry = root["summary"]["colors"][kColorNames[color]];
    entry["count"] = counts[color];
    entry["fraction"] = double(counts[color]) / report.keyframes.size();
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> tree;
  tree.setInputCloud(positions);
  pcl::PointCloud<pcl::PointXYZRGB> colored_map;
  colored_map.reserve(global_map.size());
  std::vector<int> nearest(1);
  std::vector<float> squared_distance(1);
  for (const auto& point : global_map) {
    if (!point.getVector3fMap().allFinite() ||
        tree.nearestKSearch(pcl::PointXYZ(point.x, point.y, point.z), 1, nearest, squared_distance) < 1) {
      error = "Invalid map point or no nearest heatmap keyframe";
      return false;
    }
    const auto color = report.keyframes[nearest.front()].color;
    colored_map.push_back(colorPoint(point.x, point.y, point.z, color));
  }

  if (pcl::io::savePCDFileBinary((output_dir / "relocalizability_keyframes.pcd").string(), keyframe_cloud) != 0 ||
      pcl::io::savePCDFileBinary((output_dir / "relocalizability_map.pcd").string(), colored_map) != 0) {
    error = "Failed to write heatmap PCDs";
    return false;
  }
  std::ofstream json(output_dir / "relocalizability.json");
  json << Json::writeString(Json::StreamWriterBuilder{}, root) << '\n';
  json.close();
  if (!json) {
    error = "Failed to write relocalizability.json";
    return false;
  }
  return true;
}

}
