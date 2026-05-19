#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <glog/logging.h>

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/pcl_compat.h"

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
    double fake_odom_tx = 20.0;
    double fake_odom_ty = -10.0;
    double fake_odom_tz = 1.0;
    double range_max = 30.0;
    int query_submap_radius = 2;
    double query_voxel_size = 0.12;
    double pose_translation_threshold_m = 1.0;
    double pose_yaw_threshold_deg = 10.0;
    double pose_roll_pitch_threshold_deg = 5.0;
    std::string query_source = "same_keyframe";
    bool strict = false;
};

struct QueryResult {
    int64_t keyframe_id = -1;
    int64_t matched_keyframe_id = -1;
    bool success = false;
    bool relocalization_locked = false;
    double translation_error_m = std::numeric_limits<double>::quiet_NaN();
    double yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    double roll_pitch_error_deg = std::numeric_limits<double>::quiet_NaN();
    double input_odom_translation_error_m = std::numeric_limits<double>::quiet_NaN();
    double input_odom_yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    std::size_t num_query_points = 0;
    double elapsed_ms = 0.0;
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
        << "  --fake_odom_tx M          Fake map->odom x. Default: 20\n"
        << "  --fake_odom_ty M          Fake map->odom y. Default: -10\n"
        << "  --fake_odom_tz M          Fake map->odom z. Default: 1\n"
        << "  --query_source MODE       same_keyframe, local_submap, or global_map. Default: same_keyframe\n"
        << "  --range_max M             Max range for map-query synthesis. Default: 30\n"
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
        } else if (arg == "--fake_odom_tx") {
            if (const char* v = needValue(arg)) options->fake_odom_tx = std::stod(v); else return false;
        } else if (arg == "--fake_odom_ty") {
            if (const char* v = needValue(arg)) options->fake_odom_ty = std::stod(v); else return false;
        } else if (arg == "--fake_odom_tz") {
            if (const char* v = needValue(arg)) options->fake_odom_tz = std::stod(v); else return false;
        } else if (arg == "--query_source") {
            if (const char* v = needValue(arg)) options->query_source = v; else return false;
            if (options->query_source != "same_keyframe" &&
                options->query_source != "local_submap" &&
                options->query_source != "global_map") {
                std::cerr << "--query_source must be same_keyframe, local_submap, or global_map\n";
                return false;
            }
        } else if (arg == "--range_max") {
            if (const char* v = needValue(arg)) options->range_max = std::max(0.5, std::stod(v)); else return false;
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
    if (options->output_dir.empty()) {
        options->output_dir =
            (std::filesystem::path(options->map_path).parent_path() / "synthetic_relocalization_eval").string();
    }
    return true;
}

Config makeEvalConfig()
{
    Config config;
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
    pose.linear() = Eigen::AngleAxisd(options.fake_odom_yaw_deg * M_PI / 180.0,
                                      Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return pose;
}

Cloud::Ptr perturbCloud(const Cloud::Ptr& input, const Options& options, std::uint32_t seed)
{
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
    return output;
}

Cloud::Ptr downsampleCloud(const Cloud::Ptr& input, double voxel_size)
{
    if (!input || input->empty() || voxel_size <= 0.0) {
        return input ? input : pcl::make_shared<Cloud>();
    }
    auto output = pcl::make_shared<Cloud>();
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(static_cast<float>(voxel_size),
                      static_cast<float>(voxel_size),
                      static_cast<float>(voxel_size));
    voxel.setInputCloud(input);
    voxel.filter(*output);
    output->is_dense = true;
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
                                           std::uint32_t seed)
{
    auto body = pcl::make_shared<Cloud>();
    if (!map_cloud) {
        return body;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> keep_dist(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, options.noise_sigma);
    const Eigen::Isometry3d T_lidar_map = T_map_lidar.inverse();
    body->reserve(map_cloud->size());

    for (const auto& point_map : map_cloud->points) {
        const Eigen::Vector3d p_body = T_lidar_map * Eigen::Vector3d(point_map.x, point_map.y, point_map.z);
        const double range = p_body.norm();
        if (range < 0.5 || range > options.range_max) {
            continue;
        }
        if (p_body.z() < -2.0 || p_body.z() > 6.0) {
            continue;
        }
        if (keep_dist(rng) < options.dropout) {
            continue;
        }
        pcl::PointXYZI out;
        out.x = static_cast<float>(p_body.x() + noise(rng));
        out.y = static_cast<float>(p_body.y() + noise(rng));
        out.z = static_cast<float>(p_body.z() + noise(rng));
        out.intensity = point_map.intensity;
        body->push_back(out);
    }

    body->width = static_cast<std::uint32_t>(body->size());
    body->height = 1;
    body->is_dense = true;
    return downsampleCloud(body, options.query_voxel_size);
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
    return r.relocalization_locked &&
           std::isfinite(r.translation_error_m) &&
           std::isfinite(r.yaw_error_deg) &&
           std::isfinite(r.roll_pitch_error_deg) &&
           r.translation_error_m <= options.pose_translation_threshold_m &&
           r.yaw_error_deg <= options.pose_yaw_threshold_deg &&
           r.roll_pitch_error_deg <= options.pose_roll_pitch_threshold_deg;
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

void writePerQueryCsv(const std::filesystem::path& path,
                      const std::vector<QueryResult>& results,
                      const Options& options)
{
    std::ofstream file(path);
    file << "query_keyframe_id,success,relocalization_locked,translation_error_m,yaw_error_deg,"
            "roll_pitch_error_deg,pose_accurate,input_odom_translation_error_m,input_odom_yaw_error_deg,"
            "matched_keyframe_id,self_match,"
            "num_query_points,dropout_ratio,noise_sigma,elapsed_ms\n";
    file << std::setprecision(10);
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
             << r.elapsed_ms << '\n';
    }
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

void writeSummaryJson(const std::filesystem::path& path,
                      const std::vector<QueryResult>& results,
                      const Options& options)
{
    const int tested = static_cast<int>(results.size());
    const int locks = static_cast<int>(std::count_if(results.begin(), results.end(), [](const QueryResult& r) {
        return r.relocalization_locked;
    }));
    std::vector<double> translation_errors;
    std::vector<double> yaw_errors;
    std::vector<double> roll_pitch_errors;
    std::vector<int64_t> failed_ids;
    std::vector<int64_t> inaccurate_ids;
    int self_matches = 0;
    int pose_success = 0;
    for (const auto& r : results) {
        if (r.relocalization_locked) {
            translation_errors.push_back(r.translation_error_m);
            yaw_errors.push_back(r.yaw_error_deg);
            roll_pitch_errors.push_back(r.roll_pitch_error_deg);
            if (r.matched_keyframe_id == r.keyframe_id) {
                ++self_matches;
            }
        } else {
            failed_ids.push_back(r.keyframe_id);
        }
        if (poseAccurate(r, options)) {
            ++pose_success;
        } else {
            inaccurate_ids.push_back(r.keyframe_id);
        }
    }

    std::ofstream file(path);
    file << std::setprecision(10);
    file << "{\n";
    file << "  \"map_path\": \"" << jsonEscape(options.map_path) << "\",\n";
    file << "  \"tested\": " << tested << ",\n";
    file << "  \"lock_success\": " << locks << ",\n";
    file << "  \"pose_success\": " << pose_success << ",\n";
    file << "  \"self_matches\": " << self_matches << ",\n";
    file << "  \"lock_success_rate\": " << (tested > 0 ? static_cast<double>(locks) / tested : 0.0) << ",\n";
    file << "  \"pose_success_rate\": " << (tested > 0 ? static_cast<double>(pose_success) / tested : 0.0) << ",\n";
    file << "  \"median_translation_error_m\": " << percentile(translation_errors, 0.5) << ",\n";
    file << "  \"p95_translation_error_m\": " << percentile(translation_errors, 0.95) << ",\n";
    file << "  \"median_yaw_error_deg\": " << percentile(yaw_errors, 0.5) << ",\n";
    file << "  \"p95_yaw_error_deg\": " << percentile(yaw_errors, 0.95) << ",\n";
    file << "  \"median_roll_pitch_error_deg\": " << percentile(roll_pitch_errors, 0.5) << ",\n";
    file << "  \"p95_roll_pitch_error_deg\": " << percentile(roll_pitch_errors, 0.95) << ",\n";
    file << "  \"dropout_ratio\": " << options.dropout << ",\n";
    file << "  \"noise_sigma\": " << options.noise_sigma << ",\n";
    file << "  \"query_source\": \"" << jsonEscape(options.query_source) << "\",\n";
    file << "  \"pose_translation_threshold_m\": " << options.pose_translation_threshold_m << ",\n";
    file << "  \"pose_yaw_threshold_deg\": " << options.pose_yaw_threshold_deg << ",\n";
    file << "  \"pose_roll_pitch_threshold_deg\": " << options.pose_roll_pitch_threshold_deg << ",\n";
    file << "  \"failed_query_ids\": [";
    for (std::size_t i = 0; i < failed_ids.size(); ++i) {
        if (i > 0) file << ", ";
        file << failed_ids[i];
    }
    file << "],\n";
    file << "  \"inaccurate_query_ids\": [";
    for (std::size_t i = 0; i < inaccurate_ids.size(); ++i) {
        if (i > 0) file << ", ";
        file << inaccurate_ids[i];
    }
    file << "]\n";
    file << "}\n";
}

void writeConfigUsedJson(const std::filesystem::path& path,
                         const Options& options,
                         const Config& config)
{
    std::ofstream file(path);
    file << std::setprecision(10);
    file << "{\n";
    file << "  \"map_path\": \"" << jsonEscape(options.map_path) << "\",\n";
    file << "  \"query_source\": \"" << jsonEscape(options.query_source) << "\",\n";
    file << "  \"max_queries\": " << options.max_queries << ",\n";
    file << "  \"stride\": " << options.stride << ",\n";
    file << "  \"dropout_ratio\": " << options.dropout << ",\n";
    file << "  \"noise_sigma\": " << options.noise_sigma << ",\n";
    file << "  \"range_max\": " << options.range_max << ",\n";
    file << "  \"query_submap_radius\": " << options.query_submap_radius << ",\n";
    file << "  \"query_voxel_size\": " << options.query_voxel_size << ",\n";
    file << "  \"fake_odom_yaw_deg\": " << options.fake_odom_yaw_deg << ",\n";
    file << "  \"fake_odom_translation\": [" << options.fake_odom_tx << ", "
         << options.fake_odom_ty << ", " << options.fake_odom_tz << "],\n";
    file << "  \"pose_translation_threshold_m\": " << options.pose_translation_threshold_m << ",\n";
    file << "  \"pose_yaw_threshold_deg\": " << options.pose_yaw_threshold_deg << ",\n";
    file << "  \"pose_roll_pitch_threshold_deg\": " << options.pose_roll_pitch_threshold_deg << ",\n";
    file << "  \"rhpd_enabled\": " << (config.rhpd_enabled ? "true" : "false") << ",\n";
    file << "  \"rhpd_num_candidates\": " << config.rhpd_num_candidates << ",\n";
    file << "  \"rhpd_preselect_candidates\": " << config.rhpd_preselect_candidates << ",\n";
    file << "  \"reloc_num_candidates\": " << config.reloc_num_candidates << ",\n";
    file << "  \"reloc_temporal_window_size\": " << config.reloc_temporal_window_size << "\n";
    file << "}\n";
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

    Config config = makeEvalConfig();
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

    const int auto_stride = std::max(1, static_cast<int>(std::ceil(
        static_cast<double>(keyframes.size()) / static_cast<double>(std::max(1, options.max_queries)))));
    const int stride = options.stride > 0 ? options.stride : auto_stride;
    const Eigen::Isometry3d T_map_odom_fake = makeFakeMapOdom(options);
    Cloud::Ptr global_map;
    if (options.query_source == "global_map") {
        global_map = downsampleCloud(catalog.buildGlobalMap(), options.query_voxel_size);
        if (!global_map || global_map->empty()) {
            std::cerr << "Failed to build global map for query synthesis\n";
            return 1;
        }
    }

    std::vector<QueryResult> results;
    results.reserve(std::min<std::size_t>(keyframes.size(), static_cast<std::size_t>(options.max_queries)));

    for (std::size_t idx = 0; idx < keyframes.size() && static_cast<int>(results.size()) < options.max_queries; idx += stride) {
        const auto& kf = keyframes[idx];
        QueryResult qr;
        qr.keyframe_id = kf->id;

        Cloud::Ptr query_cloud;
        if (options.query_source == "global_map") {
            query_cloud = synthesizeBodyCloudFromMapCloud(
                global_map, kf->pose_optimized, options, static_cast<std::uint32_t>(1000 + kf->id));
        } else if (options.query_source == "local_submap") {
            auto local_map = buildLocalSubmapInMapFrame(keyframes, idx, options.query_submap_radius);
            query_cloud = synthesizeBodyCloudFromMapCloud(
                local_map, kf->pose_optimized, options, static_cast<std::uint32_t>(1000 + kf->id));
        } else {
            query_cloud = perturbCloud(kf->cloud, options, static_cast<std::uint32_t>(1000 + kf->id));
        }
        qr.num_query_points = query_cloud->size();
        if (query_cloud->empty()) {
            results.push_back(qr);
            continue;
        }

        N3MappingCore localizer(config);
        if (!localizer.loadMap(options.map_path)) {
            std::cerr << "Failed to reload map for query " << kf->id << "\n";
            return 1;
        }

        const Eigen::Isometry3d T_map_lidar_gt = kf->pose_optimized;
        const Eigen::Isometry3d T_odom_lidar_input = T_map_odom_fake.inverse() * T_map_lidar_gt;
        qr.input_odom_translation_error_m =
            (T_odom_lidar_input.translation() - T_map_lidar_gt.translation()).norm();
        qr.input_odom_yaw_error_deg = planarYawErrorDeg(T_odom_lidar_input, T_map_lidar_gt);

        const auto start = std::chrono::steady_clock::now();
        const auto output = localizer.processLocalizationFrame(
            makeFrame(static_cast<std::int64_t>(idx + 1) * 1000000000LL, T_odom_lidar_input, query_cloud));
        const auto end = std::chrono::steady_clock::now();
        qr.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        qr.success = output.success;
        qr.relocalization_locked = output.relocalization_locked;
        qr.matched_keyframe_id = output.matched_keyframe_id;
        if (output.relocalization_locked) {
            qr.translation_error_m = (output.T_world_lidar.translation() - T_map_lidar_gt.translation()).norm();
            qr.yaw_error_deg = planarYawErrorDeg(output.T_world_lidar, T_map_lidar_gt);
            qr.roll_pitch_error_deg = rollPitchErrorDeg(output.T_world_lidar, T_map_lidar_gt);
        }
        results.push_back(qr);
    }

    std::filesystem::create_directories(options.output_dir);
    writePerQueryCsv(std::filesystem::path(options.output_dir) / "per_query.csv", results, options);
    writeSummaryJson(std::filesystem::path(options.output_dir) / "summary.json", results, options);
    writeConfigUsedJson(std::filesystem::path(options.output_dir) / "config_used.json", options, config);

    std::ofstream failed(std::filesystem::path(options.output_dir) / "failed_queries.txt");
    for (const auto& r : results) {
        if (!r.relocalization_locked) {
            failed << r.keyframe_id << '\n';
        }
    }

    const int locks = static_cast<int>(std::count_if(results.begin(), results.end(), [](const QueryResult& r) {
        return r.relocalization_locked;
    }));
    const int pose_success = static_cast<int>(std::count_if(results.begin(), results.end(), [&](const QueryResult& r) {
        return poseAccurate(r, options);
    }));
    const double lock_success_rate = results.empty() ? 0.0 : static_cast<double>(locks) / static_cast<double>(results.size());
    const double pose_success_rate = results.empty() ? 0.0 : static_cast<double>(pose_success) / static_cast<double>(results.size());
    std::cout << "tested=" << results.size()
              << " lock_success=" << locks
              << " pose_success=" << pose_success
              << " lock_success_rate=" << lock_success_rate
              << " pose_success_rate=" << pose_success_rate
              << " output=" << options.output_dir << "\n";

    if (options.strict && (pose_success_rate < 0.95 || percentile([&] {
            std::vector<double> values;
            for (const auto& r : results) {
                if (r.relocalization_locked) values.push_back(r.translation_error_m);
            }
            return values;
        }(), 0.5) > options.pose_translation_threshold_m || percentile([&] {
            std::vector<double> values;
            for (const auto& r : results) {
                if (r.relocalization_locked) values.push_back(r.yaw_error_deg);
            }
            return values;
        }(), 0.5) > options.pose_yaw_threshold_deg)) {
        return 2;
    }
    return 0;
}
