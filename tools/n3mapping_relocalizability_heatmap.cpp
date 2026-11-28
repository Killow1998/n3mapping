#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>

#include <glog/logging.h>

#include "n3mapping/global_map_cache.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3mapping/relocalizability_analyzer.h"

namespace {
struct Options {
  std::string map_path;
  std::string atlas_path;
  std::filesystem::path output_dir;
  n3mapping::RelocalizabilityThresholds thresholds;
  std::optional<double> map_voxel;
  int threads = n3mapping::Config{}.num_threads;
};

void usage(const char* program) {
  std::cout << "Usage: " << program
            << " --map MAP.pbstream [--atlas ATLAS.pb] --output-dir DIR\n"
            << "  [--red-alias-score N] [--max-alias-score N]\n"
            << "  [--red-alias-margin N] [--yellow-alias-margin N]\n"
            << "  [--red-observability N] [--yellow-observability N]\n"
            << "  [--min-inliers N] [--min-inlier-ratio N] [--min-relative-inlier-ratio N]\n"
            << "  [--map-voxel METRES] [--threads N]\n";
}

bool parseArgs(int argc, char** argv, Options& options) {
  auto& thresholds = options.thresholds;
  const std::map<std::string, double*> controls{
      {"--red-alias-score", &thresholds.red_alias_score},
      {"--max-alias-score", &thresholds.max_alias_score},
      {"--red-alias-margin", &thresholds.red_alias_margin},
      {"--yellow-alias-margin", &thresholds.yellow_alias_margin},
      {"--red-observability", &thresholds.red_observability},
      {"--yellow-observability", &thresholds.yellow_observability},
      {"--min-relative-inlier-ratio", &thresholds.min_relative_inlier_ratio}};
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) return false;
    const char* value = argv[++index];
    if (option == "--map") options.map_path = value;
    else if (option == "--atlas") options.atlas_path = value;
    else if (option == "--output-dir") options.output_dir = value;
    else {
      char* end = nullptr;
      const double number = std::strtod(value, &end);
      if (end == value || *end != '\0' || !std::isfinite(number)) return false;
      const auto field = controls.find(option);
      if (field != controls.end()) {
        *field->second = number;
      } else if (option == "--min-inlier-ratio") {
        thresholds.min_inlier_ratio = number;
      } else if (option == "--map-voxel") {
        options.map_voxel = number;
      } else if (option == "--min-inliers" || option == "--threads") {
        if (number < 1.0 || number > std::numeric_limits<int>::max() ||
            std::floor(number) != number) return false;
        auto& field = option == "--min-inliers" ? thresholds.min_inliers : options.threads;
        field = static_cast<int>(number);
      } else {
        return false;
      }
    }
  }
  return !options.map_path.empty() && !options.output_dir.empty() &&
      thresholds.red_alias_score >= 0.0 &&
      thresholds.red_alias_score <= thresholds.max_alias_score &&
      thresholds.red_alias_margin <= thresholds.yellow_alias_margin &&
      thresholds.red_observability >= 0.0 &&
      thresholds.red_observability <= thresholds.yellow_observability &&
      (!thresholds.min_inlier_ratio ||
       (*thresholds.min_inlier_ratio >= 0.0 && *thresholds.min_inlier_ratio <= 1.0)) &&
      thresholds.min_relative_inlier_ratio >= 0.0 &&
      thresholds.min_relative_inlier_ratio <= 1.0 &&
      (!options.map_voxel || *options.map_voxel >= 0.0);
}
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
    usage(argv[0]);
    return 0;
  }
  Options options;
  if (!parseArgs(argc, argv, options)) {
    usage(argv[0]);
    return 2;
  }

  using namespace n3mapping;
  const auto fail = [](const std::string& error) {
    std::cerr << error << '\n';
    return 1;
  };
  try {
    Config config;
    config.num_threads = options.threads;
    RelocalizabilityReport report;
    report.thresholds = options.thresholds;
    if (!report.thresholds.min_inlier_ratio)
      report.thresholds.min_inlier_ratio = config.reloc_min_inlier_ratio;
    std::vector<Keyframe::Ptr> frames;
    std::string error;
    {
      N3Map map;
      {
        std::ifstream input(options.map_path, std::ios::binary | std::ios::ate);
        const auto size = input.tellg();
        if (size <= 0 || static_cast<uint64_t>(size) > defaultPbstreamResourceLimits().max_file_bytes)
          return fail("Cannot read pbstream snapshot within the file size limit");
        std::string bytes(static_cast<std::size_t>(size), '\0');
        if (!input.seekg(0).read(bytes.data(), size))
          return fail("Failed to read pbstream snapshot");
        report.map_sha256 = LocalizationAtlas::sha256(bytes);
        if (report.map_sha256.empty()) return fail("Failed to hash pbstream snapshot");
        if (!decodeN3MapProto(bytes, &map, &error)) return fail(error);
      }
      report.map_frame = extractPbstreamMetadata(map.metadata()).map_frame;
      std::vector<ParsedKeyframeProto> parsed;
      std::unordered_set<int64_t> ids;
      if (!parseKeyframesFromProto(map, {}, &parsed, &ids, &error)) return fail(error);
      frames.reserve(parsed.size());
      for (const auto& entry : parsed) {
        auto frame = Keyframe::create(entry.id, entry.timestamp, entry.pose_odom, entry.cloud);
        frame->session_id = entry.session_id;
        frame->pose_optimized = entry.pose_optimized;
        frames.push_back(std::move(frame));
      }
    }

    if (frames.empty()) return fail("Map has no keyframes");
    PointCloudMatcher matcher(config);
    LocalizationAtlas atlas(config, matcher);
    if (!options.atlas_path.empty() &&
        !atlas.loadForMapSha256(report.map_sha256, options.atlas_path, nullptr, &error))
      return fail(error);
    report.used_atlas = atlas.loaded();
    report.map_voxel_size_m = options.map_voxel.value_or(config.global_map_voxel_size);
    const auto grid_map = report.used_atlas ? atlas.globalMap()
                                           : GlobalMapCache::build(frames, config.global_map_voxel_size);
    auto free_space = atlas.takeFreeSpaceGrid();
    if (!free_space.valid()) {
      std::vector<Eigen::Vector3d> origins;
      origins.reserve(frames.size());
      for (const auto& frame : frames) origins.push_back(frame->getPosition());
      free_space.build(*grid_map, origins, config.reloc_free_space_resolution,
                       config.reloc_free_space_max_ray_length,
                       config.reloc_free_space_occupied_min_points);
    }
    report.free_space_valid = free_space.valid();
    std::clog << "Analyzing " << frames.size() << " keyframes in " << report.map_frame << '\n';
    report.keyframes = analyzeRelocalizability(config, frames, free_space, options.thresholds,
                                               report.used_atlas ? &atlas : nullptr);
    const auto output_map = report.map_voxel_size_m == config.global_map_voxel_size
        ? grid_map : GlobalMapCache::build(frames, report.map_voxel_size_m);
    if (!saveRelocalizabilityReport(report, *output_map, options.output_dir, error))
      return fail(error);

    std::cout << "Heatmap saved to " << options.output_dir << '\n';
    return 0;
  } catch (const std::exception& error) {
    return fail(error.what());
  }
}
