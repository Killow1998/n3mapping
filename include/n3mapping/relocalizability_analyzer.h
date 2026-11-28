#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "n3mapping/keyframe.h"

namespace n3mapping {
struct Config;
class FreeSpaceGrid;
class LocalizationAtlas;

enum class RelocalizabilityColor { Green, Yellow, Red };

struct RelocalizabilityThresholds {
  double red_alias_score = 0.18;
  double max_alias_score = 0.24;
  double red_alias_margin = 0.02;
  double yellow_alias_margin = 0.05;
  double red_observability = 0.025;
  double yellow_observability = 0.075;
  int min_inliers = 100;
  std::optional<double> min_inlier_ratio;
  double min_relative_inlier_ratio = 0.65;
};

struct RelocalizabilityConfuser {
  int64_t keyframe_id;
  MapSessionId session_id;
  Eigen::Vector3d position;
  double xy_dist_m;
  double dz_m;
  double yaw_diff_rad;
  bool reversed_180;
  double rhpd_distance;
  double heading_distance;
  double fused_score;
};

struct RelocalizabilityKeyframe {
  int64_t keyframe_id;
  MapSessionId session_id;
  Eigen::Vector3d position;
  std::optional<double> local_neighbor_fused_score;
  std::optional<double> distinctiveness_margin;
  std::array<std::optional<RelocalizabilityConfuser>, 2> directional_confusers;
  std::optional<RelocalizabilityConfuser> top_confuser;
  std::optional<double> planar_observability;
  std::optional<double> xy_condition_number;
  std::optional<double> weakest_axis_yaw_rad;
  std::optional<Eigen::Matrix3d> planar_information;
  std::optional<double> minimum_planar_information_per_inlier;
  bool yaw_unobservable = false;
  std::size_t num_inliers = 0;
  double inlier_ratio = 0.0;
  std::optional<double> local_median_inlier_ratio;
  std::optional<double> relative_inlier_ratio;
  std::size_t local_submap_keyframes = 0;
  RelocalizabilityColor color = RelocalizabilityColor::Green;
  std::vector<std::string> reasons;
};

struct RelocalizabilityReport {
  std::string map_sha256;
  std::string map_frame;
  bool used_atlas = false;
  bool free_space_valid = false;
  double map_voxel_size_m = 0.0;
  RelocalizabilityThresholds thresholds;
  std::vector<RelocalizabilityKeyframe> keyframes;
};

std::vector<RelocalizabilityKeyframe> analyzeRelocalizability(
    const Config& config, const std::vector<Keyframe::Ptr>& frames,
    const FreeSpaceGrid& free_space, const RelocalizabilityThresholds& thresholds,
    const LocalizationAtlas* atlas = nullptr);

bool saveRelocalizabilityReport(
    const RelocalizabilityReport& report,
    const pcl::PointCloud<pcl::PointXYZI>& global_map,
    const std::filesystem::path& output_dir, std::string& error);
}
