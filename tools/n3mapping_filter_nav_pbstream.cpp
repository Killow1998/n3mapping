#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "n3map.pb.h"
#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/pcl_compat.h"

namespace {

struct Options {
    std::string input_pbstream;
    std::string output_pbstream;
    std::string debug_dir;
    bool rear_sector_enable = true;
    double rear_sector_center_deg = 180.0;
    double rear_sector_width_deg = 45.0;
    std::string forward_axis = "x";
    double min_range = 0.0;
    double max_range = 1.0e9;
    bool self_filter_enable = false;
    double self_box_x_min = -0.5;
    double self_box_x_max = 0.5;
    double self_box_y_min = -0.5;
    double self_box_y_max = 0.5;
    double self_box_z_min = -0.5;
    double self_box_z_max = 0.5;
    double removed_voxel_size = 0.1;
};

struct KeyframeFilterStats {
    int64_t id = -1;
    std::size_t original_points = 0;
    std::size_t kept_points = 0;
    std::size_t removed_points = 0;
};

struct FilterStats {
    std::size_t keyframes = 0;
    std::size_t original_points = 0;
    std::size_t kept_points = 0;
    std::size_t removed_points = 0;
    std::size_t removed_non_finite = 0;
    std::size_t removed_range = 0;
    std::size_t removed_self = 0;
    std::size_t removed_rear = 0;
    std::vector<KeyframeFilterStats> per_keyframe;
    std::vector<int64_t> empty_result_keyframes;
};

void printUsage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <input.pbstream> [output.pbstream] [options]\n"
        << "       " << argv0 << " --input IN --output OUT [options]\n"
        << "Options:\n"
        << "  Default behavior is rear-sector-only filtering; range and self filters require explicit options.\n"
        << "  --debug-dir DIR\n"
        << "  --rear-sector-enable true|false\n"
        << "  --rear_sector_center_deg DEG | --rear-sector-center-deg DEG\n"
        << "  --rear_sector_width_deg DEG  | --rear-sector-width-deg DEG\n"
        << "  --forward-axis x|y\n"
        << "  --min-range M --max-range M\n"
        << "  --self-filter-enable true|false\n"
        << "  --self-box-x-min M --self-box-x-max M --self-box-y-min M --self-box-y-max M --self-box-z-min M --self-box-z-max M\n";
}

bool parseBool(const std::string& value, bool* out) {
    if (!out) return false;
    if (value == "true" || value == "1" || value == "yes") {
        *out = true;
        return true;
    }
    if (value == "false" || value == "0" || value == "no") {
        *out = false;
        return true;
    }
    return false;
}

bool parseDouble(const std::string& value, double* out) {
    if (!out) return false;
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(parsed)) return false;
        *out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseArgs(int argc, char** argv, Options* options) {
    if (!options || argc < 2) return false;
    int positional = 0;
    for (int argi = 1; argi < argc; ++argi) {
        const std::string arg = argv[argi];
        auto needValue = [&](double* out) {
            if (argi + 1 >= argc || !out) return false;
            return parseDouble(argv[++argi], out);
        };
        auto needString = [&](std::string* out) {
            if (argi + 1 >= argc || !out) return false;
            *out = argv[++argi];
            return true;
        };
        auto needBool = [&](bool* out) {
            if (argi + 1 >= argc || !out) return false;
            return parseBool(argv[++argi], out);
        };

        if (arg == "--input") {
            if (!needString(&options->input_pbstream)) return false;
        } else if (arg == "--output") {
            if (!needString(&options->output_pbstream)) return false;
        } else if (arg == "--debug-dir") {
            if (!needString(&options->debug_dir)) return false;
        } else if (arg == "--rear-sector-enable") {
            if (!needBool(&options->rear_sector_enable)) return false;
        } else if (arg == "--rear_sector_center_deg" || arg == "--rear-sector-center-deg") {
            if (!needValue(&options->rear_sector_center_deg)) return false;
        } else if (arg == "--rear_sector_width_deg" || arg == "--rear-sector-width-deg") {
            if (!needValue(&options->rear_sector_width_deg)) return false;
        } else if (arg == "--forward-axis") {
            if (!needString(&options->forward_axis)) return false;
        } else if (arg == "--min-range") {
            if (!needValue(&options->min_range)) return false;
        } else if (arg == "--max-range") {
            if (!needValue(&options->max_range)) return false;
        } else if (arg == "--self-filter-enable") {
            if (!needBool(&options->self_filter_enable)) return false;
        } else if (arg == "--self-box-x-min") {
            if (!needValue(&options->self_box_x_min)) return false;
        } else if (arg == "--self-box-x-max") {
            if (!needValue(&options->self_box_x_max)) return false;
        } else if (arg == "--self-box-y-min") {
            if (!needValue(&options->self_box_y_min)) return false;
        } else if (arg == "--self-box-y-max") {
            if (!needValue(&options->self_box_y_max)) return false;
        } else if (arg == "--self-box-z-min") {
            if (!needValue(&options->self_box_z_min)) return false;
        } else if (arg == "--self-box-z-max") {
            if (!needValue(&options->self_box_z_max)) return false;
        } else if (arg == "--removed-voxel-size") {
            if (!needValue(&options->removed_voxel_size)) return false;
        } else if (arg.rfind("--", 0) == 0) {
            return false;
        } else {
            if (positional == 0) {
                options->input_pbstream = arg;
            } else if (positional == 1) {
                options->output_pbstream = arg;
            } else {
                return false;
            }
            ++positional;
        }
    }

    if (options->input_pbstream.empty()) return false;
    if (options->output_pbstream.empty()) {
        const std::filesystem::path input(options->input_pbstream);
        options->output_pbstream = (input.parent_path() / "n3map_nav_filtered.pbstream").string();
    }
    if (options->debug_dir.empty()) {
        const std::filesystem::path output(options->output_pbstream);
        options->debug_dir = (output.parent_path().empty()
            ? std::filesystem::current_path()
            : output.parent_path()).string();
    }
    try {
        if (std::filesystem::absolute(options->input_pbstream) ==
            std::filesystem::absolute(options->output_pbstream)) {
            return false;
        }
    } catch (...) {
        return false;
    }

    return std::isfinite(options->rear_sector_center_deg) &&
           std::isfinite(options->rear_sector_width_deg) &&
           options->rear_sector_width_deg > 0.0 &&
           options->rear_sector_width_deg <= 360.0 &&
           (options->forward_axis == "x" || options->forward_axis == "y") &&
           std::isfinite(options->min_range) &&
           std::isfinite(options->max_range) &&
           options->min_range >= 0.0 &&
           options->max_range >= options->min_range &&
           std::isfinite(options->self_box_x_min) &&
           std::isfinite(options->self_box_x_max) &&
           std::isfinite(options->self_box_y_min) &&
           std::isfinite(options->self_box_y_max) &&
           std::isfinite(options->self_box_z_min) &&
           std::isfinite(options->self_box_z_max) &&
           options->self_box_x_min <= options->self_box_x_max &&
           options->self_box_y_min <= options->self_box_y_max &&
           options->self_box_z_min <= options->self_box_z_max &&
           std::isfinite(options->removed_voxel_size) &&
           options->removed_voxel_size >= 0.0;
}

double normalizeDeg(double angle) {
    while (angle > 180.0) angle -= 360.0;
    while (angle <= -180.0) angle += 360.0;
    return angle;
}

bool isFinitePoint(const pcl::PointXYZI& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && std::isfinite(point.intensity);
}

bool isRearSectorPoint(const pcl::PointXYZI& point, const Options& options) {
    const double forward = options.forward_axis == "y" ? static_cast<double>(point.y) : static_cast<double>(point.x);
    const double left = options.forward_axis == "y" ? -static_cast<double>(point.x) : static_cast<double>(point.y);
    const double angle_deg = std::atan2(left, forward) * 180.0 / M_PI;
    const double diff = normalizeDeg(angle_deg - options.rear_sector_center_deg);
    constexpr double kBoundaryEpsilonDeg = 1e-6;
    return std::abs(diff) <= options.rear_sector_width_deg * 0.5 + kBoundaryEpsilonDeg;
}

bool isOutOfRange(const pcl::PointXYZI& point, const Options& options) {
    const double range = std::hypot(static_cast<double>(point.x), static_cast<double>(point.y));
    return range < options.min_range || range > options.max_range;
}

bool isInSelfBox(const pcl::PointXYZI& point, const Options& options) {
    return options.self_filter_enable &&
           point.x >= options.self_box_x_min && point.x <= options.self_box_x_max &&
           point.y >= options.self_box_y_min && point.y <= options.self_box_y_max &&
           point.z >= options.self_box_z_min && point.z <= options.self_box_z_max;
}

bool shouldRemovePoint(const pcl::PointXYZI& point, const Options& options, FilterStats* stats) {
    if (!isFinitePoint(point)) {
        if (stats) ++stats->removed_non_finite;
        return true;
    }
    if (isOutOfRange(point, options)) {
        if (stats) ++stats->removed_range;
        return true;
    }
    if (isInSelfBox(point, options)) {
        if (stats) ++stats->removed_self;
        return true;
    }
    if (options.rear_sector_enable && isRearSectorPoint(point, options)) {
        if (stats) ++stats->removed_rear;
        return true;
    }
    return false;
}

std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char c : value) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

std::string buildPolicyString(const Options& options) {
    std::ostringstream ss;
    ss << "rear_sector_enable=" << (options.rear_sector_enable ? "true" : "false")
       << ";rear_sector_center_deg=" << options.rear_sector_center_deg
       << ";rear_sector_width_deg=" << options.rear_sector_width_deg
       << ";forward_axis=" << options.forward_axis
       << ";min_range=" << options.min_range
       << ";max_range=" << options.max_range
       << ";self_filter_enable=" << (options.self_filter_enable ? "true" : "false");
    return ss.str();
}

void voxelizeCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, double voxel_size) {
    if (!cloud || cloud->empty() || voxel_size <= 0.0) return;
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel.setInputCloud(cloud);
    auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    voxel.filter(*filtered);
    *cloud = *filtered;
}

bool atomicWriteProto(const std::filesystem::path& path, const n3mapping::N3Map& map_proto) {
    const std::filesystem::path tmp = path.string() + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs.is_open()) return false;
        if (!map_proto.SerializeToOstream(&ofs)) {
            ofs.close();
            std::filesystem::remove(tmp);
            return false;
        }
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::filesystem::remove(tmp);
            return false;
        }
    }
    std::filesystem::rename(tmp, path);
    return true;
}

bool readMapProto(const std::filesystem::path& path, n3mapping::N3Map* map_proto) {
    if (!map_proto) return false;
    std::ifstream ifs(path, std::ios::binary);
    return ifs.is_open() && map_proto->ParseFromIstream(&ifs);
}

bool updateOutputMetadata(const std::filesystem::path& output,
                          const n3mapping::MapMetadata& input_metadata,
                          const Options& options,
                          const FilterStats& stats) {
    std::ifstream ifs(output, std::ios::binary);
    if (!ifs.is_open()) return false;
    n3mapping::N3Map map_proto;
    if (!map_proto.ParseFromIstream(&ifs)) return false;

    auto* meta = map_proto.mutable_metadata();
    if (!input_metadata.map_frame().empty()) {
        meta->set_map_frame(input_metadata.map_frame());
    }
    if (!input_metadata.body_frame().empty()) {
        meta->set_body_frame(input_metadata.body_frame());
    }
    meta->set_nav_cloud_filter_applied(true);
    meta->set_nav_cloud_filter_policy(buildPolicyString(options));
    meta->set_descriptors_recomputed_from_filtered_cloud(true);
    meta->set_nav_filter_raw_points(stats.original_points);
    meta->set_nav_filter_kept_points(stats.kept_points);
    meta->set_nav_filter_removed_points(stats.removed_points);
    if (meta->dense_trajectory_source().empty()) {
        meta->set_dense_trajectory_source(map_proto.dense_optimized_trajectory_size() > 0 ? "native" : "none");
        meta->set_dense_trajectory_degraded(map_proto.dense_optimized_trajectory_size() == 0);
    }
    return atomicWriteProto(output, map_proto);
}

void writeReport(const std::filesystem::path& path,
                 const Options& options,
                 const FilterStats& stats) {
    const double removed_ratio = stats.original_points == 0
        ? 0.0
        : static_cast<double>(stats.removed_points) / static_cast<double>(stats.original_points);
    std::ofstream ofs(path);
    ofs << "{\n"
        << "  \"input_pbstream\": \"" << jsonEscape(options.input_pbstream) << "\",\n"
        << "  \"output_pbstream\": \"" << jsonEscape(options.output_pbstream) << "\",\n"
        << "  \"debug_dir\": \"" << jsonEscape(options.debug_dir) << "\",\n"
        << "  \"rear_sector_enable\": " << (options.rear_sector_enable ? "true" : "false") << ",\n"
        << "  \"rear_sector_center_deg\": " << options.rear_sector_center_deg << ",\n"
        << "  \"rear_sector_width_deg\": " << options.rear_sector_width_deg << ",\n"
        << "  \"forward_axis\": \"" << options.forward_axis << "\",\n"
        << "  \"min_range\": " << options.min_range << ",\n"
        << "  \"max_range\": " << options.max_range << ",\n"
        << "  \"self_filter_enable\": " << (options.self_filter_enable ? "true" : "false") << ",\n"
        << "  \"keyframes\": " << stats.keyframes << ",\n"
        << "  \"original_points\": " << stats.original_points << ",\n"
        << "  \"kept_points\": " << stats.kept_points << ",\n"
        << "  \"removed_points\": " << stats.removed_points << ",\n"
        << "  \"removed_ratio\": " << removed_ratio << ",\n"
        << "  \"removed_non_finite\": " << stats.removed_non_finite << ",\n"
        << "  \"removed_range\": " << stats.removed_range << ",\n"
        << "  \"removed_self\": " << stats.removed_self << ",\n"
        << "  \"removed_rear\": " << stats.removed_rear << ",\n"
        << "  \"empty_result_keyframes\": [";
    for (std::size_t i = 0; i < stats.empty_result_keyframes.size(); ++i) {
        if (i > 0) ofs << ", ";
        ofs << stats.empty_result_keyframes[i];
    }
    ofs << "],\n"
        << "  \"per_keyframe\": [\n";
    for (std::size_t i = 0; i < stats.per_keyframe.size(); ++i) {
        const auto& kf = stats.per_keyframe[i];
        const double kf_removed_ratio = kf.original_points == 0
            ? 0.0
            : static_cast<double>(kf.removed_points) / static_cast<double>(kf.original_points);
        ofs << "    {\"id\": " << kf.id
            << ", \"original_points\": " << kf.original_points
            << ", \"kept_points\": " << kf.kept_points
            << ", \"removed_points\": " << kf.removed_points
            << ", \"removed_ratio\": " << kf_removed_ratio << "}";
        if (i + 1 < stats.per_keyframe.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n"
        << "}\n";
}

bool savePcdOrEmptyHeader(const std::filesystem::path& path,
                          const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    if (!cloud.empty()) {
        return pcl::io::savePCDFileBinary(path.string(), cloud) != -1;
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\n"
        << "FIELDS x y z intensity\n"
        << "SIZE 4 4 4 4\n"
        << "TYPE F F F F\n"
        << "COUNT 1 1 1 1\n"
        << "WIDTH 0\n"
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS 0\n"
        << "DATA ascii\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parseArgs(argc, argv, &options)) {
        printUsage(argv[0]);
        return 2;
    }

    n3mapping::Config config;
    std::string config_error;
    if (!config.validate(&config_error)) {
        std::cerr << "Invalid default config: " << config_error << "\n";
        return 2;
    }

    n3mapping::KeyframeManager keyframe_manager(config);
    n3mapping::LoopDetector loop_detector(config);
    n3mapping::GraphOptimizer optimizer(config);
    n3mapping::MapSerializer serializer(config);
    std::vector<n3mapping::core::DenseTrajectoryPose> dense_trajectory;
    n3mapping::core::DenseTrajectoryMetadata dense_trajectory_metadata;
    n3mapping::N3Map input_map_proto;

    if (!readMapProto(options.input_pbstream, &input_map_proto)) {
        std::cerr << "Failed to parse input pbstream: " << options.input_pbstream << "\n";
        return 1;
    }

    if (!serializer.loadMap(options.input_pbstream,
                            keyframe_manager,
                            loop_detector,
                            optimizer,
                            &dense_trajectory,
                            &dense_trajectory_metadata)) {
        std::cerr << "Failed to load input pbstream: " << options.input_pbstream << "\n";
        return 1;
    }

    const auto keyframes = keyframe_manager.getAllKeyframes();
    auto removed_global = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    FilterStats stats;
    stats.keyframes = keyframes.size();
    stats.per_keyframe.reserve(keyframes.size());

    for (const auto& keyframe : keyframes) {
        if (!keyframe || !keyframe->cloud) continue;
        auto kept = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto removed_local = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        kept->reserve(keyframe->cloud->size());
        removed_local->reserve(keyframe->cloud->size());

        KeyframeFilterStats kf_stats;
        kf_stats.id = keyframe->id;
        for (const auto& point : keyframe->cloud->points) {
            ++stats.original_points;
            ++kf_stats.original_points;
            if (shouldRemovePoint(point, options, &stats)) {
                ++stats.removed_points;
                ++kf_stats.removed_points;
                if (isFinitePoint(point)) {
                    removed_local->push_back(point);
                }
            } else {
                kept->push_back(point);
                ++stats.kept_points;
                ++kf_stats.kept_points;
            }
        }

        kept->width = static_cast<uint32_t>(kept->size());
        kept->height = 1;
        kept->is_dense = false;
        if (kept->empty()) {
            stats.empty_result_keyframes.push_back(keyframe->id);
            stats.per_keyframe.push_back(kf_stats);
            std::filesystem::create_directories(options.debug_dir);
            writeReport(std::filesystem::path(options.debug_dir) / "nav_filter_report.json", options, stats);
            std::cerr << "Filtering would leave keyframe " << keyframe->id
                      << " with an empty cloud; refusing to write inconsistent pbstream.\n";
            return 1;
        }
        keyframe->cloud = kept;
        stats.per_keyframe.push_back(kf_stats);

        if (!removed_local->empty()) {
            pcl::PointCloud<pcl::PointXYZI> removed_world;
            pcl::transformPointCloud(*removed_local,
                                     removed_world,
                                     keyframe->pose_optimized.matrix().cast<float>());
            removed_global->points.insert(removed_global->points.end(),
                                          removed_world.points.begin(),
                                          removed_world.points.end());
        }
    }

    loop_detector.clear();
    for (const auto& keyframe : keyframes) {
        if (!keyframe || !keyframe->cloud || keyframe->cloud->empty()) continue;
        keyframe->sc_descriptor = loop_detector.addDescriptor(keyframe->id, keyframe->cloud);
        keyframe->rhpd_descriptor = loop_detector.addRHPD(keyframe->id, keyframe->cloud);
    }

    if (!serializer.saveMap(options.output_pbstream,
                            keyframe_manager,
                            loop_detector,
                            optimizer,
                            dense_trajectory,
                            dense_trajectory_metadata)) {
        std::cerr << "Failed to save output pbstream: " << options.output_pbstream << "\n";
        return 1;
    }
    if (!updateOutputMetadata(options.output_pbstream, input_map_proto.metadata(), options, stats)) {
        std::cerr << "Failed to update output metadata: " << options.output_pbstream << "\n";
        return 1;
    }

    std::filesystem::create_directories(options.debug_dir);
    const std::filesystem::path debug_dir(options.debug_dir);
    const auto filtered_global = serializer.buildGlobalMap(keyframe_manager, config.save_global_map_voxel_size);
    if (filtered_global &&
        !savePcdOrEmptyHeader(debug_dir / "global_map_nav_filtered_debug.pcd", *filtered_global)) {
        std::cerr << "Failed to write global_map_nav_filtered_debug.pcd\n";
        return 1;
    }

    removed_global->width = static_cast<uint32_t>(removed_global->size());
    removed_global->height = 1;
    removed_global->is_dense = false;
    if (!savePcdOrEmptyHeader(debug_dir / "removed_by_nav_filter.pcd", *removed_global)) {
        std::cerr << "Failed to write removed_by_nav_filter.pcd\n";
        return 1;
    }
    auto removed_voxel = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*removed_global);
    voxelizeCloud(removed_voxel, options.removed_voxel_size);
    if (!savePcdOrEmptyHeader(debug_dir / "removed_by_nav_filter_voxel.pcd", *removed_voxel)) {
        std::cerr << "Failed to write removed_by_nav_filter_voxel.pcd\n";
        return 1;
    }
    writeReport(debug_dir / "nav_filter_report.json", options, stats);

    std::cout << "Wrote " << options.output_pbstream
              << " kept_points=" << stats.kept_points
              << " removed_points=" << stats.removed_points << "\n";
    return 0;
}
