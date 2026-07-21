#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/pcl_compat.h"
#include "n3mapping/synthetic_relocalization_query.h"

namespace n3mapping {
namespace {

using Cloud = core::LioFrame::PointCloud;

struct Options {
    std::string map_path;
    std::string output_dir;
    int max_queries = 100;
    int stride = 0;
    double dropout = 0.0;
    double noise_sigma = 0.0;
    double fake_odom_yaw_deg = 90.0;
    double fake_odom_roll_deg = 0.0;
    double fake_odom_pitch_deg = 0.0;
    double fake_odom_tx = 20.0;
    double fake_odom_ty = -10.0;
    double fake_odom_tz = 1.0;
    double query_pose_xy_jitter_m = 0.0;
    double query_pose_z_jitter_m = 0.0;
    double query_pose_yaw_jitter_deg = 0.0;
    double query_pose_roll_pitch_jitter_deg = 0.0;
    double range_min = 0.5;
    double range_max = 30.0;
    double fov_azimuth_deg = 0.0;
    double fov_vertical_deg = 0.0;
    double raycast_azimuth_resolution_deg = 1.0;
    double raycast_vertical_resolution_deg = 1.0;
    int occlusion_dilation_bins = 1;
    double occlusion_depth_tolerance_m = 0.3;
    int query_submap_radius = 2;
    double query_voxel_size = 0.12;
    double pose_translation_threshold_m = 1.0;
    double pose_yaw_threshold_deg = 10.0;
    double pose_roll_pitch_threshold_deg = 5.0;
    std::string query_source = "same_keyframe";
    std::string query_pose_manifest;
    std::string review_pcd_dir;
    std::string eval_profile = "relaxed_smoke";
    bool reloc_debug = false;
    bool strict = false;
};

struct QueryResult {
    int64_t query_id = -1;
    int64_t episode_id = -1;
    int64_t frame_index = 0;
    int64_t keyframe_id = -1;
    int64_t matched_keyframe_id = -1;
    std::uint32_t renderer_seed = 0;
    bool generation_valid = false;
    bool localization_attempted = false;
    bool tracking_processed = false;
    bool locked_before_frame = false;
    bool locked_after_frame = false;
    bool ever_locked_before_frame = false;
    bool tracking_loss_event = false;
    std::string lock_event_type = "none";
    std::string generation_reason = "not_generated";
    std::string runtime_stage = "generation";
    bool success = false;
    bool relocalization_locked = false;
    double query_x_m = std::numeric_limits<double>::quiet_NaN();
    double query_y_m = std::numeric_limits<double>::quiet_NaN();
    double query_z_m = std::numeric_limits<double>::quiet_NaN();
    double query_roll_deg = std::numeric_limits<double>::quiet_NaN();
    double query_pitch_deg = std::numeric_limits<double>::quiet_NaN();
    double query_yaw_deg = std::numeric_limits<double>::quiet_NaN();
    double translation_error_m = std::numeric_limits<double>::quiet_NaN();
    double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    double roll_pitch_error_deg = std::numeric_limits<double>::quiet_NaN();
    double input_odom_translation_error_m = std::numeric_limits<double>::quiet_NaN();
    double input_odom_yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    std::size_t num_query_points = 0;
    std::size_t num_query_points_before_voxel = 0;
    bool query_cloud_fingerprint_available = false;
    std::uint64_t query_cloud_fingerprint_fnv1a64 = 14695981039346656037ULL;
    synthetic::QueryVisibilityStats visibility;
    double renderer_elapsed_ms = 0.0;
    double elapsed_ms = 0.0;
    double scorer_elapsed_ms = 0.0;
};

struct QuerySpec {
    int64_t query_id = -1;
    int64_t episode_id = -1;
    int64_t frame_index = 0;
    std::size_t reference_keyframe_index = 0;
    std::uint32_t renderer_seed = 0;
    Eigen::Isometry3d T_map_lidar_gt = Eigen::Isometry3d::Identity();
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
};

void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " --map /path/to/n3map.pbstream [options]\n"
        << "Options:\n"
        << "  --output DIR              Output directory. Default: <map_dir>/synthetic_relocalization_eval\n"
        << "  --max_queries N           Maximum sampled keyframes. Default: 100\n"
        << "  --stride N                Query every N keyframes. Default: auto from max_queries\n"
        << "  --dropout R               Random point dropout ratio [0,1). Default: 0\n"
        << "  --noise_sigma M           XYZ Gaussian noise in meters. Default: 0\n"
        << "  --fake_odom_yaw_deg DEG   Fake map->odom yaw. Default: 90\n"
        << "  --fake_odom_roll_deg DEG  Fake map->odom roll. Default: 0\n"
        << "  --fake_odom_pitch_deg DEG Fake map->odom pitch. Default: 0\n"
        << "  --fake_odom_tx M          Fake map->odom x. Default: 20\n"
        << "  --fake_odom_ty M          Fake map->odom y. Default: -10\n"
        << "  --fake_odom_tz M          Fake map->odom z. Default: 1\n"
        << "  --query_pose_xy_jitter_m M      Random query pose XY jitter around selected keyframe. Default: 0\n"
        << "  --query_pose_z_jitter_m M       Random query pose Z jitter around selected keyframe. Default: 0\n"
        << "  --query_pose_yaw_jitter_deg DEG Random query pose yaw jitter. Default: 0\n"
        << "  --query_pose_roll_pitch_jitter_deg DEG Random query pose roll/pitch jitter. Default: 0\n"
        << "  --query_source MODE       same_keyframe, local_submap, or global_map. Default: same_keyframe\n"
        << "  --query_pose_manifest CSV Deterministic absolute 6DoF poses. Requires local_submap or global_map.\n"
        << "                            Standalone header: query_id,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
        << "                            Episode header: query_id,episode_id,frame_index,x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg\n"
        << "  --review_pcd_dir DIR      Export one colored map-frame PCD for the last frame of each episode.\n"
        << "                            Gray=global map, green=query at GT, red=query at a locked estimate.\n"
        << "  --reloc_debug             Write relocalization_debug.jsonl inside --output.\n"
        << "  --eval_profile PROFILE    relaxed_smoke or product_default. Default: relaxed_smoke\n"
        << "                            product_default requires an episode manifest and remains synthetic evidence.\n"
        << "  --range_min M             Min range for map-query synthesis. Default: 0.5\n"
        << "  --range_max M             Max range for map-query synthesis. Default: 30\n"
        << "  --fov_azimuth_deg DEG     Horizontal FOV override; <=0 infers from keyframe cloud. Default: 0\n"
        << "  --fov_vertical_deg DEG    Vertical FOV override; <=0 infers from keyframe cloud. Default: 0\n"
        << "  --raycast_azimuth_resolution_deg DEG   Raycast azimuth bin size, <=0 disables. Default: 1\n"
        << "  --raycast_vertical_resolution_deg DEG  Raycast vertical bin size, <=0 disables. Default: 1\n"
        << "  --occlusion_dilation_bins N            Neighbor bins used to suppress background through sparse foregrounds. Default: 1\n"
        << "  --occlusion_depth_tolerance M          Depth tolerance for neighbor occlusion. Default: 0.3\n"
        << "  --query_submap_radius N   Keyframe radius for local_submap queries. Default: 2\n"
        << "  --query_voxel_size M      Voxel size for map-query synthesis, <=0 disables. Default: 0.12\n"
        << "  --pose_translation_threshold M  Pose-success translation threshold. Default: 1\n"
        << "  --pose_yaw_threshold_deg DEG    Pose-success yaw threshold. Default: 10\n"
        << "  --pose_roll_pitch_threshold_deg DEG  Pose-success roll/pitch threshold. Default: 5\n"
        << "  --strict                  Return nonzero if default smoke criteria fail\n";
}

bool parseArgs(int argc, char** argv, Options* options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "--map") {
            if (const char* v = needValue(arg)) options->map_path = v; else return false;
        } else if (arg == "--output") {
            if (const char* v = needValue(arg)) options->output_dir = v; else return false;
        } else if (arg == "--max_queries") {
            if (const char* v = needValue(arg)) options->max_queries = std::max(1, std::stoi(v)); else return false;
        } else if (arg == "--stride") {
            if (const char* v = needValue(arg)) options->stride = std::max(1, std::stoi(v)); else return false;
        } else if (arg == "--dropout") {
            if (const char* v = needValue(arg)) options->dropout = std::clamp(std::stod(v), 0.0, 0.95); else return false;
        } else if (arg == "--noise_sigma") {
            if (const char* v = needValue(arg)) options->noise_sigma = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--fake_odom_yaw_deg") {
            if (const char* v = needValue(arg)) options->fake_odom_yaw_deg = std::stod(v); else return false;
        } else if (arg == "--fake_odom_roll_deg") {
            if (const char* v = needValue(arg)) options->fake_odom_roll_deg = std::stod(v); else return false;
        } else if (arg == "--fake_odom_pitch_deg") {
            if (const char* v = needValue(arg)) options->fake_odom_pitch_deg = std::stod(v); else return false;
        } else if (arg == "--fake_odom_tx") {
            if (const char* v = needValue(arg)) options->fake_odom_tx = std::stod(v); else return false;
        } else if (arg == "--fake_odom_ty") {
            if (const char* v = needValue(arg)) options->fake_odom_ty = std::stod(v); else return false;
        } else if (arg == "--fake_odom_tz") {
            if (const char* v = needValue(arg)) options->fake_odom_tz = std::stod(v); else return false;
        } else if (arg == "--query_pose_xy_jitter_m") {
            if (const char* v = needValue(arg)) options->query_pose_xy_jitter_m = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--query_pose_z_jitter_m") {
            if (const char* v = needValue(arg)) options->query_pose_z_jitter_m = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--query_pose_yaw_jitter_deg") {
            if (const char* v = needValue(arg)) options->query_pose_yaw_jitter_deg = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--query_pose_roll_pitch_jitter_deg") {
            if (const char* v = needValue(arg)) options->query_pose_roll_pitch_jitter_deg = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--query_source") {
            if (const char* v = needValue(arg)) options->query_source = v; else return false;
            if (options->query_source != "same_keyframe" &&
                options->query_source != "local_submap" &&
                options->query_source != "global_map") {
                std::cerr << "--query_source must be same_keyframe, local_submap, or global_map\n";
                return false;
            }
        } else if (arg == "--query_pose_manifest") {
            if (const char* v = needValue(arg)) options->query_pose_manifest = v; else return false;
        } else if (arg == "--review_pcd_dir") {
            if (const char* v = needValue(arg)) options->review_pcd_dir = v; else return false;
        } else if (arg == "--reloc_debug") {
            options->reloc_debug = true;
        } else if (arg == "--eval_profile") {
            if (const char* v = needValue(arg)) options->eval_profile = v; else return false;
            if (options->eval_profile != "relaxed_smoke" && options->eval_profile != "product_default") {
                std::cerr << "--eval_profile must be relaxed_smoke or product_default\n";
                return false;
            }
        } else if (arg == "--range_min") {
            if (const char* v = needValue(arg)) options->range_min = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--range_max") {
            if (const char* v = needValue(arg)) options->range_max = std::max(0.5, std::stod(v)); else return false;
        } else if (arg == "--fov_azimuth_deg") {
            if (const char* v = needValue(arg)) options->fov_azimuth_deg = std::clamp(std::stod(v), 0.0, 360.0); else return false;
        } else if (arg == "--fov_vertical_deg") {
            if (const char* v = needValue(arg)) options->fov_vertical_deg = std::clamp(std::stod(v), 0.0, 180.0); else return false;
        } else if (arg == "--raycast_azimuth_resolution_deg") {
            if (const char* v = needValue(arg)) options->raycast_azimuth_resolution_deg = std::stod(v); else return false;
        } else if (arg == "--raycast_vertical_resolution_deg") {
            if (const char* v = needValue(arg)) options->raycast_vertical_resolution_deg = std::stod(v); else return false;
        } else if (arg == "--occlusion_dilation_bins") {
            if (const char* v = needValue(arg)) options->occlusion_dilation_bins = std::max(0, std::stoi(v)); else return false;
        } else if (arg == "--occlusion_depth_tolerance") {
            if (const char* v = needValue(arg)) options->occlusion_depth_tolerance_m = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--query_submap_radius") {
            if (const char* v = needValue(arg)) options->query_submap_radius = std::max(0, std::stoi(v)); else return false;
        } else if (arg == "--query_voxel_size") {
            if (const char* v = needValue(arg)) options->query_voxel_size = std::stod(v); else return false;
        } else if (arg == "--pose_translation_threshold") {
            if (const char* v = needValue(arg)) options->pose_translation_threshold_m = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--pose_yaw_threshold_deg") {
            if (const char* v = needValue(arg)) options->pose_yaw_threshold_deg = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--pose_roll_pitch_threshold_deg") {
            if (const char* v = needValue(arg)) options->pose_roll_pitch_threshold_deg = std::max(0.0, std::stod(v)); else return false;
        } else if (arg == "--strict") {
            options->strict = true;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (options->map_path.empty()) {
        std::cerr << "--map is required\n";
        return false;
    }
    if (!options->query_pose_manifest.empty() && options->query_source == "same_keyframe") {
        std::cerr << "--query_pose_manifest requires --query_source local_submap or global_map\n";
        return false;
    }
    if (options->eval_profile == "product_default" && options->query_pose_manifest.empty()) {
        std::cerr << "--eval_profile product_default requires --query_pose_manifest with episode_id,frame_index\n";
        return false;
    }
    if (options->eval_profile == "product_default" &&
        (!std::isfinite(options->raycast_azimuth_resolution_deg) ||
         !std::isfinite(options->raycast_vertical_resolution_deg) ||
         options->raycast_azimuth_resolution_deg <= 0.0 ||
         options->raycast_vertical_resolution_deg <= 0.0)) {
        std::cerr << "--eval_profile product_default requires finite positive raycast resolutions\n";
        return false;
    }
    if (options->output_dir.empty()) {
        options->output_dir =
            (std::filesystem::path(options->map_path).parent_path() / "synthetic_relocalization_eval").string();
    }
    return true;
}

Config makeEvalConfig(const std::string& profile)
{
    Config config;
    if (profile == "product_default") {
        return config;
    }
    config.gicp_max_iterations = 60;
    config.gicp_fitness_threshold = 0.8;
    config.gicp_max_correspondence_distance = 2.0;
    config.gicp_submap_size = 1;
    config.rhpd_enabled = true;
    config.rhpd_dist_threshold = 100.0;
    config.rhpd_num_candidates = 10;
    config.rhpd_preselect_candidates = 100;
    config.rhpd_yaw_hypotheses = 4;
    config.reloc_num_candidates = 10;
    config.reloc_temporal_window_size = 1;
    config.reloc_lock_log_likelihood_threshold = -100.0;
    config.reloc_lock_min_winner_streak = 1;
    config.reloc_lock_min_converged_updates = 1;
    config.reloc_lock_min_margin = 0.0;
    config.reloc_min_confidence = 0.0;
    config.reloc_min_inlier_ratio = 0.0;
    config.reloc_static_agg_enable = false;
    config.reloc_ambiguity_min_basin_separation = 1000.0;
    return config;
}

Eigen::Isometry3d makeFakeMapOdom(const Options& options)
{
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(options.fake_odom_tx, options.fake_odom_ty, options.fake_odom_tz);
    pose.linear() =
        (Eigen::AngleAxisd(options.fake_odom_yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(options.fake_odom_pitch_deg * M_PI / 180.0, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(options.fake_odom_roll_deg * M_PI / 180.0, Eigen::Vector3d::UnitX())).toRotationMatrix();
    return pose;
}

synthetic::QuerySynthesisOptions makeQuerySynthesisOptions(const Options& options)
{
    synthetic::QuerySynthesisOptions out;
    out.dropout = options.dropout;
    out.noise_sigma = options.noise_sigma;
    out.range_min = options.range_min;
    out.range_max = options.range_max;
    out.fov_azimuth_deg = options.fov_azimuth_deg;
    out.fov_vertical_deg = options.fov_vertical_deg;
    out.raycast_azimuth_resolution_deg = options.raycast_azimuth_resolution_deg;
    out.raycast_vertical_resolution_deg = options.raycast_vertical_resolution_deg;
    out.occlusion_dilation_bins = options.occlusion_dilation_bins;
    out.occlusion_depth_tolerance_m = options.occlusion_depth_tolerance_m;
    return out;
}

synthetic::PoseJitterOptions makePoseJitterOptions(const Options& options)
{
    synthetic::PoseJitterOptions out;
    out.xy_m = options.query_pose_xy_jitter_m;
    out.z_m = options.query_pose_z_jitter_m;
    out.yaw_deg = options.query_pose_yaw_jitter_deg;
    out.roll_pitch_deg = options.query_pose_roll_pitch_jitter_deg;
    return out;
}

Cloud::Ptr perturbCloud(const Cloud::Ptr& input,
                        const Options& options,
                        std::uint32_t seed,
                        synthetic::QueryVisibilityStats* visibility_stats = nullptr)
{
    if (visibility_stats) {
        *visibility_stats = synthetic::QueryVisibilityStats{};
        visibility_stats->input_map_points = input ? input->size() : 0U;
        visibility_stats->finite_points = input ? input->size() : 0U;
        visibility_stats->range_eligible_points = input ? input->size() : 0U;
        visibility_stats->fov_eligible_points = input ? input->size() : 0U;
    }
    auto output = pcl::make_shared<Cloud>();
    if (!input) {
        return output;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, options.noise_sigma);
    output->reserve(input->size());

    for (const auto& point : input->points) {
        if (keep_dist(rng) < options.dropout) {
            if (visibility_stats) ++visibility_stats->dropout_suppressed_points;
            continue;
        }
        pcl::PointXYZI out = point;
        out.x = static_cast<float>(out.x + noise(rng));
        out.y = static_cast<float>(out.y + noise(rng));
        out.z = static_cast<float>(out.z + noise(rng));
        output->push_back(out);
    }

    output->width = static_cast<std::uint32_t>(output->size());
    output->height = 1;
    output->is_dense = input->is_dense;
    if (visibility_stats) visibility_stats->output_points = output->size();
    return output;
}

Cloud::Ptr downsampleCloud(const Cloud::Ptr& input, double voxel_size)
{
    if (!input || input->empty() || voxel_size <= 0.0) {
        return input ? input : pcl::make_shared<Cloud>();
    }
    Cloud::Ptr output;
    if (!safeVoxelGridFilter<pcl::PointXYZI>(input, voxel_size, &output) || !output) {
        return input;
    }
    return output;
}

Cloud::Ptr buildLocalSubmapInMapFrame(const std::vector<Keyframe::Ptr>& keyframes,
                                      std::size_t center_index,
                                      int radius)
{
    auto map_cloud = pcl::make_shared<Cloud>();
    if (keyframes.empty() || center_index >= keyframes.size()) {
        return map_cloud;
    }

    const std::size_t begin = center_index > static_cast<std::size_t>(radius)
        ? center_index - static_cast<std::size_t>(radius)
        : 0;
    const std::size_t end = std::min(keyframes.size(), center_index + static_cast<std::size_t>(radius) + 1);

    for (std::size_t i = begin; i < end; ++i) {
        const auto& kf = keyframes[i];
        if (!kf || !kf->cloud || kf->cloud->empty()) continue;
        Cloud transformed;
        pcl::transformPointCloud(*kf->cloud, transformed, kf->pose_optimized.matrix().cast<float>());
        *map_cloud += transformed;
    }
    map_cloud->width = static_cast<std::uint32_t>(map_cloud->size());
    map_cloud->height = 1;
    map_cloud->is_dense = true;
    return map_cloud;
}

Cloud::Ptr synthesizeBodyCloudFromMapCloud(const Cloud::Ptr& map_cloud,
                                           const Eigen::Isometry3d& T_map_lidar,
                                           const Options& options,
                                           std::uint32_t seed,
                                           const Cloud::Ptr& reference_cloud,
                                           synthetic::QueryVisibilityStats* visibility_stats)
{
    const auto rendered = synthetic::synthesizeBodyCloudFromMapCloud(
        map_cloud,
        T_map_lidar,
        makeQuerySynthesisOptions(options),
        seed,
        reference_cloud,
        visibility_stats);
    return downsampleCloud(rendered, options.query_voxel_size);
}

core::LioFrame makeFrame(std::int64_t stamp_nsec,
                         const Eigen::Isometry3d& T_odom_lidar,
                         const Cloud::Ptr& query_cloud)
{
    core::LioFrame frame;
    frame.stamp.nsec = stamp_nsec;
    frame.T_world_lidar = T_odom_lidar;
    frame.undistorted_cloud = query_cloud;
    frame.pose_valid = true;
    return frame;
}

bool isValidRigidPose(const Eigen::Isometry3d& pose)
{
    if (!pose.matrix().array().isFinite().all()) return false;
    const Eigen::Matrix3d rotation = pose.rotation();
    return std::abs(rotation.determinant() - 1.0) <= 1e-6 &&
           (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <= 1e-6;
}

void poseEulerDeg(const Eigen::Isometry3d& pose,
                  double* roll_deg,
                  double* pitch_deg,
                  double* yaw_deg)
{
    const Eigen::Matrix3d rotation = pose.rotation();
    const double roll = std::atan2(rotation(2, 1), rotation(2, 2));
    const double pitch = std::atan2(-rotation(2, 0), std::hypot(rotation(2, 1), rotation(2, 2)));
    const double yaw = std::atan2(rotation(1, 0), rotation(0, 0));
    if (roll_deg) *roll_deg = roll * 180.0 / M_PI;
    if (pitch_deg) *pitch_deg = pitch * 180.0 / M_PI;
    if (yaw_deg) *yaw_deg = yaw * 180.0 / M_PI;
}

std::size_t nearestKeyframeIndex(const std::vector<Keyframe::Ptr>& keyframes,
                                 const Eigen::Isometry3d& query_pose)
{
    std::size_t best_index = 0;
    double best_squared_distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < keyframes.size(); ++i) {
        const double squared_distance =
            (keyframes[i]->pose_optimized.translation() - query_pose.translation()).squaredNorm();
        if (squared_distance < best_squared_distance) {
            best_squared_distance = squared_distance;
            best_index = i;
        }
    }
    return best_index;
}

std::uint32_t rendererSeed(std::int64_t query_id)
{
    const std::uint64_t value = static_cast<std::uint64_t>(query_id);
    std::uint64_t hash = 14695981039346656037ULL;
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= static_cast<std::uint8_t>((value >> shift) & 0xffU);
        hash *= 1099511628211ULL;
    }
    return static_cast<std::uint32_t>((hash ^ (hash >> 32U)) & 0xffffffffULL);
}

std::string emptyGenerationReason(const QueryResult& result)
{
    if (result.visibility.fov_eligible_points == 0 ||
        (result.visibility.raycast_enabled && result.visibility.occupied_ray_bins == 0)) {
        return "empty_visibility";
    }
    if (result.visibility.output_points == 0 && result.visibility.dropout_suppressed_points > 0) {
        return "empty_after_dropout";
    }
    return "empty_after_voxel";
}

double planarYawErrorDeg(const Eigen::Isometry3d& estimated, const Eigen::Isometry3d& expected)
{
    const Eigen::Vector3d estimated_x = estimated.rotation().col(0);
    const Eigen::Vector3d expected_x = expected.rotation().col(0);
    const double yaw_est = std::atan2(estimated_x.y(), estimated_x.x());
    const double yaw_exp = std::atan2(expected_x.y(), expected_x.x());
    double diff = yaw_est - yaw_exp;
    while (diff > M_PI) diff -= 2.0 * M_PI;
    while (diff < -M_PI) diff += 2.0 * M_PI;
    return std::abs(diff) * 180.0 / M_PI;
}

double rollPitchErrorDeg(const Eigen::Isometry3d& estimated, const Eigen::Isometry3d& expected)
{
    const Eigen::Vector3d estimated_z = estimated.rotation().col(2).normalized();
    const Eigen::Vector3d expected_z = expected.rotation().col(2).normalized();
    const double dot = std::clamp(estimated_z.dot(expected_z), -1.0, 1.0);
    return std::acos(dot) * 180.0 / M_PI;
}

bool poseAccurate(const QueryResult& r, const Options& options)
{
    return r.localization_attempted &&
           r.relocalization_locked &&
           std::isfinite(r.translation_error_m) &&
           std::isfinite(r.yaw_error_deg) &&
           std::isfinite(r.roll_pitch_error_deg) &&
           r.translation_error_m <= options.pose_translation_threshold_m &&
           r.yaw_error_deg <= options.pose_yaw_threshold_deg &&
           r.roll_pitch_error_deg <= options.pose_roll_pitch_threshold_deg;
}

using ReviewPoint = pcl::PointXYZRGB;
using ReviewCloud = pcl::PointCloud<ReviewPoint>;

struct ReviewCloudCounts {
    std::size_t map_points = 0;
    std::size_t gt_query_points = 0;
    std::size_t estimated_query_points = 0;
};

std::size_t appendReviewCloud(const Cloud& source,
                              const Eigen::Isometry3d& T_map_source,
                              std::uint8_t red,
                              std::uint8_t green,
                              std::uint8_t blue,
                              ReviewCloud* destination)
{
    if (!destination) return 0;
    const std::size_t size_before = destination->size();
    destination->reserve(destination->size() + source.size());
    for (const auto& point : source) {
        if (!pcl::isFinite(point)) continue;
        const Eigen::Vector3d transformed =
            T_map_source * Eigen::Vector3d(point.x, point.y, point.z);
        if (!transformed.array().isFinite().all()) continue;
        ReviewPoint review_point{};
        review_point.x = static_cast<float>(transformed.x());
        review_point.y = static_cast<float>(transformed.y());
        review_point.z = static_cast<float>(transformed.z());
        review_point.r = red;
        review_point.g = green;
        review_point.b = blue;
        destination->push_back(review_point);
    }
    return destination->size() - size_before;
}

bool writeReviewPcd(const std::filesystem::path& path,
                    const Cloud& map_cloud,
                    const Cloud& query_cloud,
                    const Eigen::Isometry3d& T_map_lidar_gt,
                    bool estimated_pose_available,
                    const Eigen::Isometry3d& T_map_lidar_estimated,
                    ReviewCloudCounts* counts)
{
    if (!counts) return false;
    auto composite = pcl::make_shared<ReviewCloud>();
    counts->map_points = appendReviewCloud(
        map_cloud, Eigen::Isometry3d::Identity(), 96, 96, 96, composite.get());
    counts->gt_query_points = appendReviewCloud(
        query_cloud, T_map_lidar_gt, 0, 255, 0, composite.get());
    if (estimated_pose_available) {
        counts->estimated_query_points = appendReviewCloud(
            query_cloud, T_map_lidar_estimated, 255, 0, 0, composite.get());
    }
    composite->width = static_cast<std::uint32_t>(composite->size());
    composite->height = 1;
    composite->is_dense = true;
    return pcl::io::savePCDFileBinaryCompressed(path.string(), *composite) == 0;
}

bool prepareFreshDirectory(const std::filesystem::path& path, const std::string& label)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) {
        std::cerr << "Failed to inspect " << label << ": " << error.message() << "\n";
        return false;
    }
    if (exists) {
        if (!std::filesystem::is_directory(path, error) || error) {
            std::cerr << label << " exists and is not a readable directory: " << path << "\n";
            return false;
        }
        const auto begin = std::filesystem::directory_iterator(path, error);
        if (error) {
            std::cerr << "Failed to inspect " << label << " contents: " << error.message() << "\n";
            return false;
        }
        if (begin != std::filesystem::directory_iterator{}) {
            std::cerr << label << " must be fresh (missing or empty); refusing to overwrite: "
                      << path << "\n";
            return false;
        }
        return true;
    }
    std::filesystem::create_directories(path, error);
    if (error) {
        std::cerr << "Failed to create " << label << ": " << error.message() << "\n";
        return false;
    }
    return true;
}

bool writeReviewReadme(const std::filesystem::path& path, const Options& options)
{
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << "Synthetic relocalization visual review\n\n"
         << "Every PCD is in the global map frame and contains:\n"
         << "  gray  RGB(96,96,96): global map\n"
         << "  green RGB(0,255,0): exact localizer input query transformed by manifest GT pose\n"
         << "  red   RGB(255,0,0): same input query transformed by the locked output pose\n"
         << "No red points means the episode ended without a lock; no estimate is invented.\n\n"
         << "Interpretation:\n"
         << "  gray vs green checks whether the rendered query and GT placement are plausible.\n"
         << "  green vs red checks the locked pose. Overlap means agreement; separation means pose error.\n\n"
         << "Inputs:\n"
         << "  map: " << options.map_path << "\n"
         << "  query_pose_manifest: " << options.query_pose_manifest << "\n"
         << "  query_source: " << options.query_source << "\n"
         << "  evidence_class: synthetic_map_render\n"
         << "This evaluator does not read a rosbag; bag provenance of the map is not established here.\n";
    return file.good();
}

bool writeReviewIndexHeader(std::ofstream* file)
{
    if (!file || !file->is_open()) return false;
    *file << "pcd_file,episode_id,query_id,frame_index,status,locked_after_frame,lock_event_type,"
             "matched_keyframe_id,map_points,gt_query_points,estimated_query_points,"
             "gt_x_m,gt_y_m,gt_z_m,gt_roll_deg,gt_pitch_deg,gt_yaw_deg,"
             "estimated_x_m,estimated_y_m,estimated_z_m,estimated_roll_deg,estimated_pitch_deg,estimated_yaw_deg,"
             "translation_error_m,yaw_error_deg,roll_pitch_error_deg,"
             "map_rgb,gt_query_rgb,estimated_query_rgb\n";
    return file->good();
}

bool writeReviewIndexRow(std::ofstream* file,
                         const std::string& pcd_file,
                         const QuerySpec& spec,
                         const QueryResult& result,
                         const Options& options,
                         bool estimated_pose_available,
                         const Eigen::Isometry3d& T_map_lidar_estimated,
                         const ReviewCloudCounts& counts)
{
    if (!file || !file->is_open()) return false;
    double estimated_roll_deg = std::numeric_limits<double>::quiet_NaN();
    double estimated_pitch_deg = std::numeric_limits<double>::quiet_NaN();
    double estimated_yaw_deg = std::numeric_limits<double>::quiet_NaN();
    double translation_error_m = std::numeric_limits<double>::quiet_NaN();
    double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    double roll_pitch_error_deg = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d estimated_translation = Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    std::string status = "no_lock";
    if (estimated_pose_available) {
        estimated_translation = T_map_lidar_estimated.translation();
        poseEulerDeg(
            T_map_lidar_estimated,
            &estimated_roll_deg,
            &estimated_pitch_deg,
            &estimated_yaw_deg);
        translation_error_m =
            (estimated_translation - spec.T_map_lidar_gt.translation()).norm();
        yaw_error_deg = planarYawErrorDeg(T_map_lidar_estimated, spec.T_map_lidar_gt);
        roll_pitch_error_deg = rollPitchErrorDeg(T_map_lidar_estimated, spec.T_map_lidar_gt);
        const bool accurate =
            translation_error_m <= options.pose_translation_threshold_m &&
            yaw_error_deg <= options.pose_yaw_threshold_deg &&
            roll_pitch_error_deg <= options.pose_roll_pitch_threshold_deg;
        status = accurate ? "correct_lock" : "false_lock";
    }

    *file << std::setprecision(17)
          << pcd_file << ',' << spec.episode_id << ',' << spec.query_id << ',' << spec.frame_index << ','
          << status << ',' << result.locked_after_frame << ',' << result.lock_event_type << ','
          << result.matched_keyframe_id << ',' << counts.map_points << ',' << counts.gt_query_points << ','
          << counts.estimated_query_points << ','
          << spec.T_map_lidar_gt.translation().x() << ','
          << spec.T_map_lidar_gt.translation().y() << ','
          << spec.T_map_lidar_gt.translation().z() << ','
          << spec.roll_deg << ',' << spec.pitch_deg << ',' << spec.yaw_deg << ','
          << estimated_translation.x() << ',' << estimated_translation.y() << ','
          << estimated_translation.z() << ',' << estimated_roll_deg << ','
          << estimated_pitch_deg << ',' << estimated_yaw_deg << ','
          << translation_error_m << ',' << yaw_error_deg << ',' << roll_pitch_error_deg << ','
          << "96|96|96,0|255|0," << (estimated_pose_available ? "255|0|0" : "none") << '\n';
    return file->good();
}

double percentile(std::vector<double> values, double q)
{
    values.erase(std::remove_if(values.begin(), values.end(), [](double v) { return !std::isfinite(v); }), values.end());
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double idx = std::clamp(q, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) {
        return values[lo];
    }
    const double alpha = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

std::string queryCloudFingerprintText(const QueryResult& result);

bool writePerQueryCsv(const std::filesystem::path& path,
                      const std::vector<QueryResult>& results,
                      const Options& options)
{
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << "query_keyframe_id,success,relocalization_locked,translation_error_m,yaw_error_deg,"
            "roll_pitch_error_deg,pose_accurate,input_odom_translation_error_m,input_odom_yaw_error_deg,"
            "matched_keyframe_id,self_match,"
            "num_query_points,dropout_ratio,noise_sigma,elapsed_ms,"
            "renderer_elapsed_ms,localizer_elapsed_ms,scorer_elapsed_ms,"
            "query_id,episode_id,frame_index,reference_keyframe_id,renderer_seed,"
            "generation_valid,generation_reason,localization_attempted,tracking_processed,"
            "locked_before_frame,locked_after_frame,ever_locked_before_frame,"
            "lock_event_type,tracking_loss_event,runtime_stage,"
            "query_x_m,query_y_m,query_z_m,query_roll_deg,query_pitch_deg,query_yaw_deg,"
            "query_cloud_fingerprint_available,query_cloud_fingerprint_algorithm,query_cloud_fingerprint_fnv1a64,"
            "input_map_points,finite_points,range_eligible_points,fov_eligible_points,"
            "azimuth_bins,vertical_bins,occupied_ray_bins,"
            "same_ray_occluded_points,dropout_suppressed_points,occlusion_suppressed_bins,query_points_before_voxel,"
            "raycast_enabled,envelope_valid,envelope_azimuth_full,envelope_azimuth_start_deg,"
            "envelope_azimuth_span_deg,envelope_vertical_min_deg,envelope_vertical_max_deg\n";
    file << std::setprecision(synthetic::kEvaluationCsvDoublePrecision);
    for (const auto& r : results) {
        file << r.keyframe_id << ','
             << r.success << ','
             << r.relocalization_locked << ','
             << r.translation_error_m << ','
             << r.yaw_error_deg << ','
             << r.roll_pitch_error_deg << ','
             << poseAccurate(r, options) << ','
             << r.input_odom_translation_error_m << ','
             << r.input_odom_yaw_error_deg << ','
             << r.matched_keyframe_id << ','
             << (r.matched_keyframe_id == r.keyframe_id) << ','
             << r.num_query_points << ','
             << options.dropout << ','
             << options.noise_sigma << ','
             << r.elapsed_ms << ','
             << r.renderer_elapsed_ms << ','
             << r.elapsed_ms << ','
             << r.scorer_elapsed_ms << ','
             << r.query_id << ','
             << r.episode_id << ','
             << r.frame_index << ','
             << r.keyframe_id << ','
             << r.renderer_seed << ','
             << r.generation_valid << ','
             << r.generation_reason << ','
             << r.localization_attempted << ','
             << r.tracking_processed << ','
             << r.locked_before_frame << ','
             << r.locked_after_frame << ','
             << r.ever_locked_before_frame << ','
             << r.lock_event_type << ','
             << r.tracking_loss_event << ','
             << r.runtime_stage << ','
             << r.query_x_m << ','
             << r.query_y_m << ','
             << r.query_z_m << ','
             << r.query_roll_deg << ','
             << r.query_pitch_deg << ','
             << r.query_yaw_deg << ','
             << r.query_cloud_fingerprint_available << ','
             << synthetic::kQueryCloudFingerprintAlgorithm << ','
             << queryCloudFingerprintText(r) << ','
             << r.visibility.input_map_points << ','
             << r.visibility.finite_points << ','
             << r.visibility.range_eligible_points << ','
             << r.visibility.fov_eligible_points << ','
             << r.visibility.azimuth_bins << ','
             << r.visibility.vertical_bins << ','
             << r.visibility.occupied_ray_bins << ','
             << r.visibility.same_ray_occluded_points << ','
             << r.visibility.dropout_suppressed_points << ','
             << r.visibility.occlusion_suppressed_bins << ','
             << r.num_query_points_before_voxel << ','
             << r.visibility.raycast_enabled << ','
             << r.visibility.envelope_valid << ','
             << r.visibility.envelope_azimuth_full << ','
             << r.visibility.envelope_azimuth_start_deg << ','
             << r.visibility.envelope_azimuth_span_deg << ','
             << r.visibility.envelope_vertical_min_deg << ','
             << r.visibility.envelope_vertical_max_deg << '\n';
    }
    return file.good();
}

std::string jsonEscape(const std::string& input)
{
    std::ostringstream out;
    for (const char c : input) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }
    return out.str();
}

std::string normalizedQuerySource(const std::string& query_source)
{
    if (query_source == "global_map") return "global_map_render";
    if (query_source == "local_submap") return "local_submap_render";
    return "same_keyframe";
}

std::string queryCloudFingerprintText(const QueryResult& result)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16)
        << result.query_cloud_fingerprint_fnv1a64;
    return out.str();
}

struct AggregateStats {
    int tested = 0;
    int generation_valid = 0;
    int invalid_pose = 0;
    int empty_visibility = 0;
    int empty_after_dropout = 0;
    int empty_after_voxel = 0;
    int localization_attempted = 0;
    int tracking_processed = 0;
    int locks = 0;
    int initial_lock_events = 0;
    int reentry_lock_events = 0;
    int tracking_loss_events = 0;
    int pose_success = 0;
    int self_matches = 0;
    int episodes = 0;
    int episode_locks = 0;
    int episode_pose_success = 0;
    std::vector<double> translation_errors;
    std::vector<double> yaw_errors;
    std::vector<double> roll_pitch_errors;
    std::vector<int64_t> failed_ids;
    std::vector<int64_t> inaccurate_ids;
    std::vector<int64_t> generation_failure_ids;
};

AggregateStats aggregateResults(const std::vector<QueryResult>& results, const Options& options)
{
    AggregateStats stats;
    stats.tested = static_cast<int>(results.size());
    struct EpisodeState {
        bool locked = false;
        bool pose_success = false;
    };
    std::map<int64_t, EpisodeState> episodes;
    for (const auto& r : results) {
        auto& episode = episodes[r.episode_id];
        if (r.generation_valid) ++stats.generation_valid;
        if (r.generation_reason == "invalid_pose") ++stats.invalid_pose;
        if (r.generation_reason == "empty_visibility") ++stats.empty_visibility;
        if (r.generation_reason == "empty_after_dropout") ++stats.empty_after_dropout;
        if (r.generation_reason == "empty_after_voxel") ++stats.empty_after_voxel;
        if (!r.generation_valid) stats.generation_failure_ids.push_back(r.query_id);
        if (r.localization_attempted) ++stats.localization_attempted;
        if (r.tracking_processed) ++stats.tracking_processed;
        if (r.lock_event_type == "initial_lock") ++stats.initial_lock_events;
        if (r.lock_event_type == "reentry_lock") ++stats.reentry_lock_events;
        if (r.tracking_loss_event) ++stats.tracking_loss_events;
        if (r.localization_attempted && r.relocalization_locked) {
            ++stats.locks;
            episode.locked = true;
            stats.translation_errors.push_back(r.translation_error_m);
            stats.yaw_errors.push_back(r.yaw_error_deg);
            stats.roll_pitch_errors.push_back(r.roll_pitch_error_deg);
            if (r.matched_keyframe_id == r.keyframe_id) {
                ++stats.self_matches;
            }
        } else if (r.localization_attempted) {
            stats.failed_ids.push_back(r.query_id);
        }
        if (poseAccurate(r, options)) {
            ++stats.pose_success;
            episode.pose_success = true;
        } else if (r.localization_attempted) {
            stats.inaccurate_ids.push_back(r.query_id);
        }
    }
    stats.episodes = static_cast<int>(episodes.size());
    for (const auto& item : episodes) {
        if (item.second.locked) ++stats.episode_locks;
        if (item.second.pose_success) ++stats.episode_pose_success;
    }
    return stats;
}

void writeJsonNumber(std::ostream& file, double value)
{
    if (std::isfinite(value)) file << value;
    else file << "null";
}

void writeJsonIntArray(std::ostream& file, const std::vector<int64_t>& values)
{
    file << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) file << ", ";
        file << values[i];
    }
    file << ']';
}

bool writeSummaryJson(const std::filesystem::path& path,
                      const std::vector<QueryResult>& results,
                      const Options& options)
{
    const AggregateStats stats = aggregateResults(results, options);

    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << std::setprecision(10);
    file << "{\n";
    file << "  \"schema_version\": 2,\n";
    file << "  \"evidence_class\": \"synthetic_map_render\",\n";
    file << "  \"odom_source\": \"oracle_gt_odom\",\n";
    file << "  \"gt_runtime_access\": true,\n";
    file << "  \"verdict\": \"SHADOW_ONLY\",\n";
    file << "  \"query_cloud_fingerprint_algorithm\": \""
         << synthetic::kQueryCloudFingerprintAlgorithm << "\",\n";
    file << "  \"eval_profile\": \"" << jsonEscape(options.eval_profile) << "\",\n";
    file << "  \"map_path\": \"" << jsonEscape(options.map_path) << "\",\n";
    file << "  \"tested\": " << stats.tested << ",\n";
    file << "  \"generation_valid\": " << stats.generation_valid << ",\n";
    file << "  \"invalid_pose\": " << stats.invalid_pose << ",\n";
    file << "  \"empty_visibility\": " << stats.empty_visibility << ",\n";
    file << "  \"empty_after_dropout\": " << stats.empty_after_dropout << ",\n";
    file << "  \"empty_after_voxel\": " << stats.empty_after_voxel << ",\n";
    file << "  \"localization_attempted\": " << stats.localization_attempted << ",\n";
    file << "  \"tracking_processed\": " << stats.tracking_processed << ",\n";
    file << "  \"lock_success\": " << stats.locks << ",\n";
    file << "  \"initial_lock_events\": " << stats.initial_lock_events << ",\n";
    file << "  \"reentry_lock_events\": " << stats.reentry_lock_events << ",\n";
    file << "  \"tracking_loss_events\": " << stats.tracking_loss_events << ",\n";
    file << "  \"pose_success\": " << stats.pose_success << ",\n";
    file << "  \"self_matches\": " << stats.self_matches << ",\n";
    file << "  \"episode_count\": " << stats.episodes << ",\n";
    file << "  \"episode_lock_success\": " << stats.episode_locks << ",\n";
    file << "  \"episode_pose_success\": " << stats.episode_pose_success << ",\n";
    file << "  \"lock_success_rate\": "
         << (stats.tested > 0 ? static_cast<double>(stats.locks) / stats.tested : 0.0) << ",\n";
    file << "  \"localization_lock_rate\": "
         << (stats.localization_attempted > 0
                 ? static_cast<double>(stats.locks) / stats.localization_attempted
                 : 0.0) << ",\n";
    file << "  \"pose_success_rate\": "
         << (stats.tested > 0 ? static_cast<double>(stats.pose_success) / stats.tested : 0.0) << ",\n";
    file << "  \"episode_pose_success_rate\": "
         << (stats.episodes > 0 ? static_cast<double>(stats.episode_pose_success) / stats.episodes : 0.0) << ",\n";
    file << "  \"median_translation_error_m\": ";
    writeJsonNumber(file, percentile(stats.translation_errors, 0.5));
    file << ",\n  \"p95_translation_error_m\": ";
    writeJsonNumber(file, percentile(stats.translation_errors, 0.95));
    file << ",\n  \"median_yaw_error_deg\": ";
    writeJsonNumber(file, percentile(stats.yaw_errors, 0.5));
    file << ",\n  \"p95_yaw_error_deg\": ";
    writeJsonNumber(file, percentile(stats.yaw_errors, 0.95));
    file << ",\n  \"median_roll_pitch_error_deg\": ";
    writeJsonNumber(file, percentile(stats.roll_pitch_errors, 0.5));
    file << ",\n  \"p95_roll_pitch_error_deg\": ";
    writeJsonNumber(file, percentile(stats.roll_pitch_errors, 0.95));
    file << ",\n";
    file << "  \"dropout_ratio\": " << options.dropout << ",\n";
    file << "  \"noise_sigma\": " << options.noise_sigma << ",\n";
    file << "  \"query_source\": \"" << normalizedQuerySource(options.query_source) << "\",\n";
    file << "  \"pose_translation_threshold_m\": " << options.pose_translation_threshold_m << ",\n";
    file << "  \"pose_yaw_threshold_deg\": " << options.pose_yaw_threshold_deg << ",\n";
    file << "  \"pose_roll_pitch_threshold_deg\": " << options.pose_roll_pitch_threshold_deg << ",\n";
    file << "  \"failed_query_ids\": ";
    writeJsonIntArray(file, stats.failed_ids);
    file << ",\n  \"inaccurate_query_ids\": ";
    writeJsonIntArray(file, stats.inaccurate_ids);
    file << ",\n  \"generation_failure_query_ids\": ";
    writeJsonIntArray(file, stats.generation_failure_ids);
    file << "\n";
    file << "}\n";
    return file.good();
}

bool writeConfigUsedJson(const std::filesystem::path& path,
                         const Options& options,
                         const Config& config)
{
    std::ofstream file(path);
    if (!file.is_open()) return false;
    const bool raycast_enabled =
        options.query_source != "same_keyframe" &&
        std::isfinite(options.raycast_azimuth_resolution_deg) &&
        std::isfinite(options.raycast_vertical_resolution_deg) &&
        options.raycast_azimuth_resolution_deg > 0.0 &&
        options.raycast_vertical_resolution_deg > 0.0;
    const char* visibility_model = options.query_source == "same_keyframe"
        ? "same_keyframe_perturbation_v1"
        : (raycast_enabled
               ? "map_conditioned_point_zbuffer_v1"
               : "range_fov_passthrough_v1");
    const auto generation_thresholds = synthetic::queryGenerationThresholds(
        options.eval_profile == "product_default");
    file << std::setprecision(10);
    file << "{\n";
    file << "  \"schema_version\": 2,\n";
    file << "  \"evidence_class\": \"synthetic_map_render\",\n";
    file << "  \"odom_source\": \"oracle_gt_odom\",\n";
    file << "  \"gt_runtime_access\": true,\n";
    file << "  \"verdict\": \"SHADOW_ONLY\",\n";
    file << "  \"query_cloud_fingerprint_algorithm\": \""
         << synthetic::kQueryCloudFingerprintAlgorithm << "\",\n";
    file << "  \"eval_profile\": \"" << jsonEscape(options.eval_profile) << "\",\n";
    file << "  \"map_path\": \"" << jsonEscape(options.map_path) << "\",\n";
    file << "  \"query_pose_manifest\": \"" << jsonEscape(options.query_pose_manifest) << "\",\n";
    file << "  \"query_source\": \"" << normalizedQuerySource(options.query_source) << "\",\n";
    file << "  \"raycast_enabled\": " << (raycast_enabled ? "true" : "false") << ",\n";
    file << "  \"visibility_model\": \"" << visibility_model << "\",\n";
    file << "  \"normal_visibility\": \"not_modeled\",\n";
    file << "  \"sensor_profile\": \"reference_envelope_only\",\n";
    file << "  \"scan_pattern_model\": \"not_modeled\",\n";
    file << "  \"motion_distortion_model\": \"not_modeled\",\n";
    file << "  \"seed_derivation\": \"query_id_fnv_mix_v1\",\n";
    file << "  \"max_queries\": " << options.max_queries << ",\n";
    file << "  \"stride\": " << options.stride << ",\n";
    file << "  \"dropout_ratio\": " << options.dropout << ",\n";
    file << "  \"noise_sigma\": " << options.noise_sigma << ",\n";
    file << "  \"range_max\": " << options.range_max << ",\n";
    file << "  \"range_min\": " << options.range_min << ",\n";
    file << "  \"fov_azimuth_deg\": " << options.fov_azimuth_deg << ",\n";
    file << "  \"fov_vertical_deg\": " << options.fov_vertical_deg << ",\n";
    file << "  \"raycast_azimuth_resolution_deg\": " << options.raycast_azimuth_resolution_deg << ",\n";
    file << "  \"raycast_vertical_resolution_deg\": " << options.raycast_vertical_resolution_deg << ",\n";
    file << "  \"occlusion_dilation_bins\": " << options.occlusion_dilation_bins << ",\n";
    file << "  \"occlusion_depth_tolerance_m\": " << options.occlusion_depth_tolerance_m << ",\n";
    file << "  \"query_submap_radius\": " << options.query_submap_radius << ",\n";
    file << "  \"query_voxel_size\": " << options.query_voxel_size << ",\n";
    file << "  \"minimum_query_points\": "
         << generation_thresholds.minimum_query_points << ",\n";
    file << "  \"minimum_occupied_ray_bins\": "
         << generation_thresholds.minimum_occupied_ray_bins << ",\n";
    file << "  \"fake_odom_yaw_deg\": " << options.fake_odom_yaw_deg << ",\n";
    file << "  \"fake_odom_roll_deg\": " << options.fake_odom_roll_deg << ",\n";
    file << "  \"fake_odom_pitch_deg\": " << options.fake_odom_pitch_deg << ",\n";
    file << "  \"fake_odom_translation\": [" << options.fake_odom_tx << ", "
         << options.fake_odom_ty << ", " << options.fake_odom_tz << "],\n";
    file << "  \"query_pose_xy_jitter_m\": " << options.query_pose_xy_jitter_m << ",\n";
    file << "  \"query_pose_z_jitter_m\": " << options.query_pose_z_jitter_m << ",\n";
    file << "  \"query_pose_yaw_jitter_deg\": " << options.query_pose_yaw_jitter_deg << ",\n";
    file << "  \"query_pose_roll_pitch_jitter_deg\": " << options.query_pose_roll_pitch_jitter_deg << ",\n";
    file << "  \"pose_translation_threshold_m\": " << options.pose_translation_threshold_m << ",\n";
    file << "  \"pose_yaw_threshold_deg\": " << options.pose_yaw_threshold_deg << ",\n";
    file << "  \"pose_roll_pitch_threshold_deg\": " << options.pose_roll_pitch_threshold_deg << ",\n";
    file << "  \"gicp_max_iterations\": " << config.gicp_max_iterations << ",\n";
    file << "  \"gicp_max_correspondence_distance\": "
         << config.gicp_max_correspondence_distance << ",\n";
    file << "  \"gicp_fitness_threshold\": " << config.gicp_fitness_threshold << ",\n";
    file << "  \"gicp_submap_size\": " << config.gicp_submap_size << ",\n";
    file << "  \"rhpd_enabled\": " << (config.rhpd_enabled ? "true" : "false") << ",\n";
    file << "  \"rhpd_num_candidates\": " << config.rhpd_num_candidates << ",\n";
    file << "  \"rhpd_preselect_candidates\": " << config.rhpd_preselect_candidates << ",\n";
    file << "  \"reloc_num_candidates\": " << config.reloc_num_candidates << ",\n";
    file << "  \"reloc_temporal_window_size\": " << config.reloc_temporal_window_size << "\n";
    file << "}\n";
    return file.good();
}

bool writeResolvedQueriesCsv(const std::filesystem::path& path,
                             const std::vector<QueryResult>& results)
{
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << "query_id,episode_id,frame_index,reference_keyframe_id,renderer_seed,"
            "x_m,y_m,z_m,roll_deg,pitch_deg,yaw_deg,generation_valid,generation_reason,"
            "query_cloud_fingerprint_available,query_cloud_fingerprint_algorithm,query_cloud_fingerprint_fnv1a64\n";
    file << std::setprecision(17);
    for (const auto& r : results) {
        file << r.query_id << ',' << r.episode_id << ',' << r.frame_index << ','
             << r.keyframe_id << ',' << r.renderer_seed << ','
             << r.query_x_m << ',' << r.query_y_m << ',' << r.query_z_m << ','
             << r.query_roll_deg << ',' << r.query_pitch_deg << ',' << r.query_yaw_deg << ','
             << r.generation_valid << ',' << r.generation_reason << ','
             << r.query_cloud_fingerprint_available << ','
             << synthetic::kQueryCloudFingerprintAlgorithm << ','
             << queryCloudFingerprintText(r) << '\n';
    }
    return file.good();
}

bool writeEvalCompleteMarker(const std::filesystem::path& path)
{
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << "evaluator_schema_version=2\nstatus=complete\nevidence_class=synthetic_map_render\nverdict=SHADOW_ONLY\n";
    return file.good();
}

}  // namespace
}  // namespace n3mapping

int main(int argc, char** argv)
{
    using namespace n3mapping;

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = false;
    FLAGS_alsologtostderr = false;
    FLAGS_minloglevel = 2;

    Options options;
    if (!parseArgs(argc, argv, &options)) {
        return 1;
    }

    Config config = makeEvalConfig(options.eval_profile);
    if (options.reloc_debug) {
        config.reloc_debug_enable = true;
        config.reloc_debug_path =
            (std::filesystem::path(options.output_dir) / "relocalization_debug.jsonl").string();
    }
    N3MappingCore catalog(config);
    if (!catalog.loadMap(options.map_path)) {
        std::cerr << "Failed to load map: " << options.map_path << "\n";
        return 1;
    }

    auto keyframes = catalog.getAllKeyframes();
    keyframes.erase(std::remove_if(keyframes.begin(), keyframes.end(), [](const Keyframe::Ptr& kf) {
        return !kf || !kf->cloud || kf->cloud->empty();
    }), keyframes.end());
    std::sort(keyframes.begin(), keyframes.end(), [](const Keyframe::Ptr& a, const Keyframe::Ptr& b) {
        return a->id < b->id;
    });
    if (keyframes.empty()) {
        std::cerr << "Loaded map has no keyframes with point clouds\n";
        return 1;
    }

    std::vector<QuerySpec> query_specs;
    synthetic::QueryPoseManifest pose_manifest;
    if (!options.query_pose_manifest.empty()) {
        std::string manifest_error;
        if (!synthetic::readQueryPoseManifestCsv(options.query_pose_manifest, &pose_manifest, &manifest_error)) {
            std::cerr << "Invalid query pose manifest: " << manifest_error << "\n";
            return 1;
        }
        if (options.eval_profile == "product_default" && !pose_manifest.has_episode_columns) {
            std::cerr << "product_default requires the episode manifest header with episode_id,frame_index\n";
            return 1;
        }
        query_specs.reserve(pose_manifest.entries.size());
        for (const auto& entry : pose_manifest.entries) {
            QuerySpec spec;
            spec.query_id = entry.query_id;
            spec.episode_id = entry.episode_id;
            spec.frame_index = entry.frame_index;
            spec.reference_keyframe_index = nearestKeyframeIndex(keyframes, entry.T_map_lidar);
            spec.renderer_seed = rendererSeed(entry.query_id);
            spec.T_map_lidar_gt = entry.T_map_lidar;
            spec.roll_deg = entry.roll_deg;
            spec.pitch_deg = entry.pitch_deg;
            spec.yaw_deg = entry.yaw_deg;
            query_specs.push_back(spec);
        }
    } else {
        const int auto_stride = std::max(1, static_cast<int>(std::ceil(
            static_cast<double>(keyframes.size()) / static_cast<double>(std::max(1, options.max_queries)))));
        const int stride = options.stride > 0 ? options.stride : auto_stride;
        query_specs.reserve(std::min<std::size_t>(
            keyframes.size(), static_cast<std::size_t>(options.max_queries)));
        for (std::size_t idx = 0;
             idx < keyframes.size() && static_cast<int>(query_specs.size()) < options.max_queries;
             idx += stride) {
            const auto& kf = keyframes[idx];
            QuerySpec spec;
            spec.query_id = kf->id;
            spec.episode_id = kf->id;
            spec.frame_index = 0;
            spec.reference_keyframe_index = idx;
            spec.renderer_seed = static_cast<std::uint32_t>(1000 + kf->id);
            spec.T_map_lidar_gt = kf->pose_optimized;
            if (options.query_source != "same_keyframe") {
                std::mt19937 pose_rng(static_cast<std::uint32_t>(5000 + kf->id));
                spec.T_map_lidar_gt = synthetic::applyUniformPoseJitter(
                    kf->pose_optimized, makePoseJitterOptions(options), &pose_rng);
            }
            poseEulerDeg(spec.T_map_lidar_gt, &spec.roll_deg, &spec.pitch_deg, &spec.yaw_deg);
            query_specs.push_back(spec);
        }
    }

    if (options.eval_profile == "product_default") {
        std::map<int64_t, int64_t> episode_frame_counts;
        for (const auto& spec : query_specs) {
            episode_frame_counts[spec.episode_id] = std::max(
                episode_frame_counts[spec.episode_id], spec.frame_index + 1);
        }
        const int minimum_frames = std::max(1, config.reloc_temporal_window_size);
        for (const auto& item : episode_frame_counts) {
            if (item.second < minimum_frames) {
                std::cerr << "product_default episode " << item.first << " has " << item.second
                          << " frames; requires at least reloc_temporal_window_size=" << minimum_frames << "\n";
                return 1;
            }
        }
    }

    const std::filesystem::path output_dir(options.output_dir);
    if (!prepareFreshDirectory(output_dir, "output directory")) {
        return 1;
    }

    std::filesystem::path review_pcd_dir;
    std::ofstream review_index;
    std::map<int64_t, std::size_t> episode_review_positions;
    if (!options.review_pcd_dir.empty()) {
        review_pcd_dir = std::filesystem::path(options.review_pcd_dir);
        if (review_pcd_dir.lexically_normal() == output_dir.lexically_normal()) {
            std::cerr << "--review_pcd_dir must differ from --output\n";
            return 1;
        }
        if (!prepareFreshDirectory(review_pcd_dir, "review PCD directory")) {
            return 1;
        }
        if (!writeReviewReadme(review_pcd_dir / "README.txt", options)) {
            std::cerr << "Failed to write review PCD README\n";
            return 1;
        }
        review_index.open(review_pcd_dir / "review_index.csv");
        if (!writeReviewIndexHeader(&review_index)) {
            std::cerr << "Failed to write review PCD index header\n";
            return 1;
        }
        for (std::size_t i = 0; i < query_specs.size(); ++i) {
            episode_review_positions[query_specs[i].episode_id] = i;
        }
    }

    const Eigen::Isometry3d T_map_odom_fake = makeFakeMapOdom(options);
    Cloud::Ptr global_map;
    if (options.query_source == "global_map" || !options.review_pcd_dir.empty()) {
        global_map = downsampleCloud(catalog.buildGlobalMap(), options.query_voxel_size);
        if (!global_map || global_map->empty()) {
            std::cerr << "Failed to build global map for query synthesis or review export\n";
            return 1;
        }
    }

    std::vector<QueryResult> results;
    results.reserve(query_specs.size());
    std::unique_ptr<N3MappingCore> localizer;
    int64_t active_episode_id = std::numeric_limits<int64_t>::min();
    bool current_locked = false;
    bool ever_locked = false;
    const auto generation_thresholds = synthetic::queryGenerationThresholds(
        options.eval_profile == "product_default");

    for (std::size_t spec_index = 0; spec_index < query_specs.size(); ++spec_index) {
        const auto& spec = query_specs[spec_index];
        const bool export_review_frame = !options.review_pcd_dir.empty() &&
            episode_review_positions.at(spec.episode_id) == spec_index;
        const std::size_t idx = spec.reference_keyframe_index;
        const auto& kf = keyframes[idx];
        if (active_episode_id != spec.episode_id) {
            localizer.reset();
            active_episode_id = spec.episode_id;
            current_locked = false;
            ever_locked = false;
        }
        const bool locked_before_frame = current_locked;
        const bool ever_locked_before_frame = ever_locked;
        QueryResult qr;
        qr.query_id = spec.query_id;
        qr.episode_id = spec.episode_id;
        qr.frame_index = spec.frame_index;
        qr.keyframe_id = kf->id;
        qr.renderer_seed = spec.renderer_seed;
        qr.query_x_m = spec.T_map_lidar_gt.translation().x();
        qr.query_y_m = spec.T_map_lidar_gt.translation().y();
        qr.query_z_m = spec.T_map_lidar_gt.translation().z();
        qr.query_roll_deg = spec.roll_deg;
        qr.query_pitch_deg = spec.pitch_deg;
        qr.query_yaw_deg = spec.yaw_deg;
        qr.locked_before_frame = locked_before_frame;
        qr.locked_after_frame = locked_before_frame;
        qr.ever_locked_before_frame = ever_locked_before_frame;
        const auto renderer_start = std::chrono::steady_clock::now();
        if (!isValidRigidPose(spec.T_map_lidar_gt)) {
            qr.renderer_elapsed_ms = std::max(
                0.0,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - renderer_start).count());
            qr.generation_reason = "invalid_pose";
            results.push_back(qr);
            if (export_review_frame) {
                std::cerr << "Cannot export review PCD for episode " << spec.episode_id
                          << ": final query pose is invalid\n";
                return 1;
            }
            continue;
        }

        Cloud::Ptr query_cloud;
        if (options.query_source == "global_map") {
            query_cloud = synthesizeBodyCloudFromMapCloud(
                global_map,
                spec.T_map_lidar_gt,
                options,
                spec.renderer_seed,
                kf->cloud,
                &qr.visibility);
        } else if (options.query_source == "local_submap") {
            auto local_map = buildLocalSubmapInMapFrame(keyframes, idx, options.query_submap_radius);
            query_cloud = synthesizeBodyCloudFromMapCloud(
                local_map,
                spec.T_map_lidar_gt,
                options,
                spec.renderer_seed,
                kf->cloud,
                &qr.visibility);
        } else {
            query_cloud = perturbCloud(kf->cloud, options, spec.renderer_seed, &qr.visibility);
        }
        qr.renderer_elapsed_ms = std::max(
            0.0,
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - renderer_start).count());
        qr.num_query_points_before_voxel = qr.visibility.output_points;
        qr.num_query_points = query_cloud->size();
        qr.query_cloud_fingerprint_available = true;
        qr.query_cloud_fingerprint_fnv1a64 =
            synthetic::fingerprintPointSequenceFNV1a64(*query_cloud);
        if (query_cloud->empty()) {
            qr.generation_reason = emptyGenerationReason(qr);
            results.push_back(qr);
            if (export_review_frame) {
                std::cerr << "Cannot export review PCD for episode " << spec.episode_id
                          << ": final query cloud is empty\n";
                return 1;
            }
            continue;
        }
        const auto generation_support = synthetic::evaluateQueryGenerationSupport(
            qr.num_query_points,
            qr.visibility.occupied_ray_bins,
            options.eval_profile == "product_default" || qr.visibility.raycast_enabled,
            generation_thresholds);
        if (generation_support != synthetic::QueryGenerationSupport::SUFFICIENT) {
            qr.generation_reason = synthetic::queryGenerationSupportReason(generation_support);
            results.push_back(qr);
            if (export_review_frame) {
                std::cerr << "Cannot export review PCD for episode " << spec.episode_id
                          << ": final query generation support is insufficient\n";
                return 1;
            }
            continue;
        }
        qr.generation_valid = true;
        qr.generation_reason = "ok";

        if (!localizer) {
            localizer = std::make_unique<N3MappingCore>(config);
            if (!localizer->loadMap(options.map_path)) {
                std::cerr << "Failed to reload map for episode " << spec.episode_id << "\n";
                return 1;
            }
        }

        const Eigen::Isometry3d T_odom_lidar_input = T_map_odom_fake.inverse() * spec.T_map_lidar_gt;
        qr.input_odom_translation_error_m =
            (T_odom_lidar_input.translation() - spec.T_map_lidar_gt.translation()).norm();
        qr.input_odom_yaw_error_deg = planarYawErrorDeg(T_odom_lidar_input, spec.T_map_lidar_gt);

        const auto start = std::chrono::steady_clock::now();
        const auto output = localizer->processLocalizationFrame(
            makeFrame((spec.frame_index + 1) * 1000000000LL, T_odom_lidar_input, query_cloud));
        const auto end = std::chrono::steady_clock::now();
        const auto transition = synthetic::advanceEpisodeFrameState(
            locked_before_frame,
            ever_locked_before_frame,
            output.success,
            output.relocalization_locked);
        qr.localization_attempted = transition.localization_attempted;
        qr.tracking_processed = transition.tracking_processed;
        qr.locked_after_frame = transition.locked_after_frame;
        qr.lock_event_type = synthetic::episodeLockEventTypeName(transition.stage);
        qr.tracking_loss_event = transition.tracking_loss_event;
        qr.runtime_stage = synthetic::episodeFrameStageName(transition.stage);
        current_locked = transition.locked_after_frame;
        ever_locked = transition.ever_locked_after_frame;
        qr.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        qr.success = output.success;
        qr.relocalization_locked = output.relocalization_locked;
        qr.matched_keyframe_id = output.matched_keyframe_id;
        if (transition.lock_event) {
            const auto scorer_start = std::chrono::steady_clock::now();
            qr.translation_error_m =
                (output.T_world_lidar.translation() - spec.T_map_lidar_gt.translation()).norm();
            qr.yaw_error_deg = planarYawErrorDeg(output.T_world_lidar, spec.T_map_lidar_gt);
            qr.roll_pitch_error_deg = rollPitchErrorDeg(output.T_world_lidar, spec.T_map_lidar_gt);
            qr.scorer_elapsed_ms = std::max(
                0.0,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - scorer_start).count());
        }
        if (export_review_frame) {
            const bool estimated_pose_available = output.success && transition.locked_after_frame;
            if (estimated_pose_available && !isValidRigidPose(output.T_world_lidar)) {
                std::cerr << "Cannot export review PCD for episode " << spec.episode_id
                          << ": locked output pose is invalid\n";
                return 1;
            }
            const double review_translation_error = estimated_pose_available
                ? (output.T_world_lidar.translation() - spec.T_map_lidar_gt.translation()).norm()
                : std::numeric_limits<double>::quiet_NaN();
            const double review_yaw_error = estimated_pose_available
                ? planarYawErrorDeg(output.T_world_lidar, spec.T_map_lidar_gt)
                : std::numeric_limits<double>::quiet_NaN();
            const double review_roll_pitch_error = estimated_pose_available
                ? rollPitchErrorDeg(output.T_world_lidar, spec.T_map_lidar_gt)
                : std::numeric_limits<double>::quiet_NaN();
            const bool accurate = estimated_pose_available &&
                review_translation_error <= options.pose_translation_threshold_m &&
                review_yaw_error <= options.pose_yaw_threshold_deg &&
                review_roll_pitch_error <= options.pose_roll_pitch_threshold_deg;
            const std::string status = !estimated_pose_available
                ? "no_lock"
                : (accurate ? "correct_lock" : "false_lock");
            const std::string pcd_file =
                "episode_" + std::to_string(spec.episode_id) +
                "_frame_" + std::to_string(spec.frame_index) +
                "_" + status + ".pcd";
            ReviewCloudCounts counts;
            if (!writeReviewPcd(
                    review_pcd_dir / pcd_file,
                    *global_map,
                    *query_cloud,
                    spec.T_map_lidar_gt,
                    estimated_pose_available,
                    output.T_world_lidar,
                    &counts)) {
                std::cerr << "Failed to write review PCD: " << (review_pcd_dir / pcd_file) << "\n";
                return 1;
            }
            if (!writeReviewIndexRow(
                    &review_index,
                    pcd_file,
                    spec,
                    qr,
                    options,
                    estimated_pose_available,
                    output.T_world_lidar,
                    counts)) {
                std::cerr << "Failed to write review PCD index row\n";
                return 1;
            }
        }
        results.push_back(qr);
    }

    if (review_index.is_open()) {
        review_index.close();
        if (!review_index) {
            std::cerr << "Failed to finalize review PCD index\n";
            return 1;
        }
    }

    bool artifacts_ok = true;
    artifacts_ok &= writePerQueryCsv(output_dir / "per_query.csv", results, options);
    artifacts_ok &= writeResolvedQueriesCsv(output_dir / "resolved_queries.csv", results);
    artifacts_ok &= writeSummaryJson(output_dir / "summary.json", results, options);
    artifacts_ok &= writeSummaryJson(output_dir / "metrics.json", results, options);
    artifacts_ok &= writeConfigUsedJson(output_dir / "config_used.json", options, config);
    artifacts_ok &= writeConfigUsedJson(output_dir / "renderer_config.json", options, config);

    std::ofstream failed(output_dir / "failed_queries.txt");
    artifacts_ok &= failed.is_open();
    for (const auto& r : results) {
        if (r.localization_attempted && !r.relocalization_locked) {
            failed << r.query_id << '\n';
        }
    }
    artifacts_ok &= failed.good();
    failed.close();
    if (!artifacts_ok) {
        std::cerr << "Failed to write one or more synthetic evaluator artifacts; EVAL_COMPLETE not written\n";
        return 1;
    }
    if (!writeEvalCompleteMarker(output_dir / "EVAL_COMPLETE")) {
        std::cerr << "Failed to write EVAL_COMPLETE marker\n";
        return 1;
    }

    const AggregateStats stats = aggregateResults(results, options);
    const double lock_success_rate = results.empty()
        ? 0.0
        : static_cast<double>(stats.locks) / static_cast<double>(results.size());
    const double pose_success_rate = results.empty()
        ? 0.0
        : static_cast<double>(stats.pose_success) / static_cast<double>(results.size());
    const double episode_pose_success_rate = stats.episodes > 0
        ? static_cast<double>(stats.episode_pose_success) / static_cast<double>(stats.episodes)
        : 0.0;
    std::cout << "tested=" << results.size()
              << " generation_valid=" << stats.generation_valid
              << " localization_attempted=" << stats.localization_attempted
              << " tracking_processed=" << stats.tracking_processed
              << " lock_success=" << stats.locks
              << " pose_success=" << stats.pose_success
              << " lock_success_rate=" << lock_success_rate
              << " pose_success_rate=" << pose_success_rate
              << " episode_pose_success_rate=" << episode_pose_success_rate
              << " eval_profile=" << options.eval_profile
              << " evidence_class=synthetic_map_render"
              << " formal_product_gate_eligible=0"
              << " output=" << options.output_dir << "\n";

    if (options.strict) {
        const double selected_success_rate = options.eval_profile == "product_default"
            ? episode_pose_success_rate
            : pose_success_rate;
        const bool strict_failed =
            stats.generation_valid != stats.tested ||
            selected_success_rate < 0.95 ||
            percentile(stats.translation_errors, 0.5) > options.pose_translation_threshold_m ||
            percentile(stats.yaw_errors, 0.5) > options.pose_yaw_threshold_deg;
        std::cout << "strict_result=" << (strict_failed ? "FAIL" : "PASS")
                  << " strict_scope="
                  << (options.eval_profile == "relaxed_smoke"
                          ? "synthetic_relaxed_smoke_only"
                          : "synthetic_product_config_nonformal")
                  << "\n";
        if (strict_failed) return 2;
    }
    return 0;
}
