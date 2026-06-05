#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/memory.h>

#include "n3mapping/config.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/graph_optimizer.h"

namespace {

struct Options {
    std::string input_pbstream;
    std::string output_pbstream;
    double rear_sector_center_deg = 180.0;
    double rear_sector_width_deg = 45.0;
};

struct FilterStats {
    std::size_t keyframes = 0;
    std::size_t original_points = 0;
    std::size_t kept_points = 0;
    std::size_t removed_points = 0;
};

void printUsage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " <input.pbstream> [output.pbstream] "
        << "[--rear_sector_center_deg DEG] [--rear_sector_width_deg DEG]\n";
}

bool parseArgs(int argc, char** argv, Options* options) {
    if (!options || argc < 2) return false;
    options->input_pbstream = argv[1];
    int argi = 2;
    if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
        options->output_pbstream = argv[argi++];
    }
    for (; argi < argc; ++argi) {
        const std::string arg = argv[argi];
        auto needValue = [&](double* out) {
            if (argi + 1 >= argc || !out) return false;
            *out = std::stod(argv[++argi]);
            return true;
        };
        if (arg == "--rear_sector_center_deg") {
            if (!needValue(&options->rear_sector_center_deg)) return false;
        } else if (arg == "--rear_sector_width_deg") {
            if (!needValue(&options->rear_sector_width_deg)) return false;
        } else {
            return false;
        }
    }
    if (options->output_pbstream.empty()) {
        const std::filesystem::path input(options->input_pbstream);
        options->output_pbstream =
            (input.parent_path() / "n3map_nav_filtered.pbstream").string();
    }
    return std::isfinite(options->rear_sector_center_deg) &&
           std::isfinite(options->rear_sector_width_deg) &&
           options->rear_sector_width_deg > 0.0 &&
           options->rear_sector_width_deg <= 360.0;
}

double normalizeDeg(double angle) {
    while (angle > 180.0) angle -= 360.0;
    while (angle <= -180.0) angle += 360.0;
    return angle;
}

bool isRearSectorPoint(const pcl::PointXYZI& point, double center_deg, double width_deg) {
    const double angle_deg = std::atan2(static_cast<double>(point.y), static_cast<double>(point.x)) * 180.0 / M_PI;
    const double diff = normalizeDeg(angle_deg - center_deg);
    return std::abs(diff) <= width_deg * 0.5;
}

void writeReport(const std::filesystem::path& path,
                 const Options& options,
                 const FilterStats& stats) {
    std::ofstream ofs(path);
    ofs << "{\n"
        << "  \"input_pbstream\": \"" << options.input_pbstream << "\",\n"
        << "  \"output_pbstream\": \"" << options.output_pbstream << "\",\n"
        << "  \"rear_sector_center_deg\": " << options.rear_sector_center_deg << ",\n"
        << "  \"rear_sector_width_deg\": " << options.rear_sector_width_deg << ",\n"
        << "  \"keyframes\": " << stats.keyframes << ",\n"
        << "  \"original_points\": " << stats.original_points << ",\n"
        << "  \"kept_points\": " << stats.kept_points << ",\n"
        << "  \"removed_points\": " << stats.removed_points << "\n"
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

    if (!serializer.loadMap(options.input_pbstream,
                            keyframe_manager,
                            loop_detector,
                            optimizer,
                            &dense_trajectory)) {
        std::cerr << "Failed to load input pbstream: " << options.input_pbstream << "\n";
        return 1;
    }

    const auto keyframes = keyframe_manager.getAllKeyframes();
    auto removed_global = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    FilterStats stats;
    stats.keyframes = keyframes.size();

    for (const auto& keyframe : keyframes) {
        if (!keyframe || !keyframe->cloud) continue;
        auto kept = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        auto removed_local = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        kept->reserve(keyframe->cloud->size());
        removed_local->reserve(keyframe->cloud->size());

        for (const auto& point : keyframe->cloud->points) {
            ++stats.original_points;
            if (isRearSectorPoint(point, options.rear_sector_center_deg, options.rear_sector_width_deg)) {
                removed_local->push_back(point);
                ++stats.removed_points;
            } else {
                kept->push_back(point);
                ++stats.kept_points;
            }
        }

        kept->width = static_cast<uint32_t>(kept->size());
        kept->height = 1;
        kept->is_dense = false;
        keyframe->cloud = kept;

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
                            dense_trajectory)) {
        std::cerr << "Failed to save output pbstream: " << options.output_pbstream << "\n";
        return 1;
    }

    const std::filesystem::path output_path(options.output_pbstream);
    const std::filesystem::path output_dir = output_path.parent_path().empty()
        ? std::filesystem::current_path()
        : output_path.parent_path();
    const auto filtered_global = serializer.buildGlobalMap(keyframe_manager, config.save_global_map_voxel_size);
    if (filtered_global &&
        !savePcdOrEmptyHeader(output_dir / "global_map_nav_filtered_debug.pcd", *filtered_global)) {
        std::cerr << "Failed to write global_map_nav_filtered_debug.pcd\n";
        return 1;
    }
    removed_global->width = static_cast<uint32_t>(removed_global->size());
    removed_global->height = 1;
    removed_global->is_dense = false;
    if (!savePcdOrEmptyHeader(output_dir / "removed_by_nav_filter.pcd", *removed_global)) {
        std::cerr << "Failed to write removed_by_nav_filter.pcd\n";
        return 1;
    }
    writeReport(output_dir / "nav_filter_report.json", options, stats);

    std::cout << "Wrote " << options.output_pbstream
              << " kept_points=" << stats.kept_points
              << " removed_points=" << stats.removed_points << "\n";
    return 0;
}
