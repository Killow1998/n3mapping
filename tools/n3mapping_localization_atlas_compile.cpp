#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include <glog/logging.h>

#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/localization_atlas.h"
#include "n3mapping/point_cloud_matcher.h"

namespace {

struct Options {
  std::string map_path;
  std::string output_path;
  bool force = false;
};

void printUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --map MAP.pbstream [--output FILE] [--force]\n";
}

bool parseArgs(int argc, char **argv, Options *options) {
  if (!options)
    return false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto needValue = [&]() -> const char * {
      return i + 1 < argc ? argv[++i] : nullptr;
    };
    if (arg == "--map") {
      const char *value = needValue();
      if (!value)
        return false;
      options->map_path = value;
    } else if (arg == "--output") {
      const char *value = needValue();
      if (!value)
        return false;
      options->output_path = value;
    } else if (arg == "--force") {
      options->force = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  if (options->map_path.empty())
    return false;
  if (options->output_path.empty()) {
    options->output_path =
        n3mapping::LocalizationAtlas::defaultAtlasPath(options->map_path);
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  Options options;
  if (!parseArgs(argc, argv, &options)) {
    printUsage(argv[0]);
    return 2;
  }
  if (!std::filesystem::is_regular_file(options.map_path)) {
    std::cerr << "Map does not exist: " << options.map_path << "\n";
    return 2;
  }

  n3mapping::Config config;
  config.mode = "localization";
  config.map_path = options.map_path;
  config.reloc_atlas_enable = false;
  n3mapping::N3MappingCore core(config);
  if (!core.loadMap(options.map_path)) {
    std::cerr << "Failed to load map: " << options.map_path << "\n";
    return 1;
  }

  n3mapping::PointCloudMatcher matcher(config);
  n3mapping::LocalizationAtlas atlas(config, matcher);
  n3mapping::LocalizationAtlasStats stats;
  std::string error;
  if (!atlas.compileAndSave(options.map_path, core.getAllKeyframes(),
                            options.output_path, options.force, &stats,
                            &error)) {
    std::cerr << "Failed to compile localization atlas: " << error << "\n";
    return 1;
  }
  const std::string map_sha256 = atlas.mapSha256();
  const std::string config_sha256 = atlas.configSha256();
  atlas.clear();
  n3mapping::PointCloudMatcher verifier_matcher(config);
  n3mapping::LocalizationAtlas verified(config, verifier_matcher);
  n3mapping::LocalizationAtlasStats load_stats;
  if (!verified.load(options.map_path, options.output_path, &load_stats,
                     &error)) {
    std::cerr << "Compiled atlas failed verification load: " << error << "\n";
    return 1;
  }

  std::cout << std::fixed << std::setprecision(3)
            << "{\n"
            << "  \"output\": \"" << options.output_path << "\",\n"
            << "  \"map_sha256\": \"" << map_sha256 << "\",\n"
            << "  \"config_sha256\": \"" << config_sha256 << "\",\n"
            << "  \"keyframes\": " << stats.keyframe_count << ",\n"
            << "  \"global_points\": " << stats.global_point_count << ",\n"
            << "  \"prepared_points\": " << stats.prepared_point_count << ",\n"
            << "  \"sidecar_bytes\": " << stats.sidecar_bytes << ",\n"
            << "  \"global_map_ms\": " << stats.global_map_ms << ",\n"
            << "  \"prepare_ms\": " << stats.prepare_ms << ",\n"
            << "  \"serialize_ms\": " << stats.serialize_ms << ",\n"
            << "  \"verify_load_ms\": " << load_stats.load_ms << ",\n"
            << "  \"verify_kdtree_ms\": " << load_stats.kdtree_ms << "\n"
            << "}\n";
  return 0;
}
