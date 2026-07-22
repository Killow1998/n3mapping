#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <pcl/common/point_tests.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/core/n3mapping_core.h"

namespace fs = std::filesystem;

namespace n3mapping {
namespace {

using Cloud = core::LioFrame::PointCloud;
using ReviewPoint = pcl::PointXYZRGB;
using ReviewCloud = pcl::PointCloud<ReviewPoint>;

struct Options {
  fs::path map_path;
  fs::path atlas_path;
  fs::path manifest_path;
  fs::path output_dir;
  bool reloc_debug = false;
  double input_voxel_size_m = 0.0;
  double reference_map_x_m = std::numeric_limits<double>::quiet_NaN();
  double reference_map_y_m = std::numeric_limits<double>::quiet_NaN();
  double reference_map_yaw_deg = std::numeric_limits<double>::quiet_NaN();
};

struct FrameRecord {
  std::string episode_id;
  int64_t frame_index = -1;
  int64_t stamp_ns = 0;
  fs::path pcd_path;
  Eigen::Isometry3d T_odom_body = Eigen::Isometry3d::Identity();
};

struct BufferedFrame {
  FrameRecord record;
  Cloud::Ptr cloud;
};

struct FrameResult {
  int64_t frame_index = -1;
  int64_t stamp_ns = 0;
  bool success = false;
  bool lock = false;
  int64_t seed_keyframe_id = -1;
  int64_t support_keyframe_id = -1;
  int64_t matched_keyframe_id = -1;
  Eigen::Isometry3d T_map_body = Eigen::Isometry3d::Identity();
};

void printUsage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --map MAP.pbstream --manifest frames.csv --output DIR [options]\n"
      << "Options:\n"
      << "  --atlas FILE     Enable a map-bound localization atlas sidecar.\n"
      << "  --input-voxel-size METERS  Match evaluator-side query preprocessing.\n"
      << "  --reference-map-x METERS --reference-map-y METERS "
         "--reference-map-yaw DEG\n"
      << "                    Add a blue oracle reference layer for eval review only.\n"
      << "  --reloc-debug    Write relocalization_debug.jsonl in the output "
         "directory.\n";
}

double parseFiniteDouble(const std::string &text, const char *field_name);

bool parseArgs(int argc, char **argv, Options *options) {
  if (!options)
    return false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto needValue = [&]() -> const char * {
      if (i + 1 >= argc)
        return nullptr;
      return argv[++i];
    };
    if (arg == "--map") {
      const char *value = needValue();
      if (!value)
        return false;
      options->map_path = value;
    } else if (arg == "--atlas") {
      const char *value = needValue();
      if (!value)
        return false;
      options->atlas_path = value;
    } else if (arg == "--manifest") {
      const char *value = needValue();
      if (!value)
        return false;
      options->manifest_path = value;
    } else if (arg == "--output") {
      const char *value = needValue();
      if (!value)
        return false;
      options->output_dir = value;
    } else if (arg == "--reloc-debug") {
      options->reloc_debug = true;
    } else if (arg == "--input-voxel-size") {
      const char *value = needValue();
      if (!value)
        return false;
      options->input_voxel_size_m = parseFiniteDouble(value, "input_voxel_size");
      if (options->input_voxel_size_m < 0.0)
        return false;
    } else if (arg == "--reference-map-x") {
      const char *value = needValue();
      if (!value)
        return false;
      options->reference_map_x_m = parseFiniteDouble(value, "reference_map_x");
    } else if (arg == "--reference-map-y") {
      const char *value = needValue();
      if (!value)
        return false;
      options->reference_map_y_m = parseFiniteDouble(value, "reference_map_y");
    } else if (arg == "--reference-map-yaw") {
      const char *value = needValue();
      if (!value)
        return false;
      options->reference_map_yaw_deg =
          parseFiniteDouble(value, "reference_map_yaw");
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }
  const int reference_fields =
      static_cast<int>(std::isfinite(options->reference_map_x_m)) +
      static_cast<int>(std::isfinite(options->reference_map_y_m)) +
      static_cast<int>(std::isfinite(options->reference_map_yaw_deg));
  return !options->map_path.empty() && !options->manifest_path.empty() &&
         !options->output_dir.empty() &&
         (reference_fields == 0 || reference_fields == 3);
}

std::vector<std::string> splitCsv(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ','))
    fields.push_back(field);
  if (!line.empty() && line.back() == ',')
    fields.emplace_back();
  return fields;
}

double parseFiniteDouble(const std::string &text, const char *field_name) {
  std::size_t parsed = 0;
  const double value = std::stod(text, &parsed);
  if (parsed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error(std::string("invalid ") + field_name + ": " +
                             text);
  }
  return value;
}

int64_t parseInteger(const std::string &text, const char *field_name) {
  std::size_t parsed = 0;
  const long long value = std::stoll(text, &parsed);
  if (parsed != text.size()) {
    throw std::runtime_error(std::string("invalid ") + field_name + ": " +
                             text);
  }
  return static_cast<int64_t>(value);
}

std::vector<FrameRecord> readManifest(const fs::path &path) {
  std::ifstream stream(path);
  if (!stream.is_open())
    throw std::runtime_error("failed to open manifest: " + path.string());
  std::string line;
  if (!std::getline(stream, line))
    throw std::runtime_error("manifest is empty");
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  const std::vector<std::string> expected_header = {
      "episode_id", "frame_index", "stamp_ns", "pcd_path", "tx", "ty",
      "tz",         "qx",          "qy",       "qz",       "qw"};
  if (splitCsv(line) != expected_header) {
    throw std::runtime_error("unexpected manifest header");
  }

  std::vector<FrameRecord> records;
  const fs::path base = path.parent_path();
  int64_t expected_index = 0;
  std::string episode_id;
  int64_t previous_stamp = std::numeric_limits<int64_t>::min();
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    const auto fields = splitCsv(line);
    if (fields.size() != expected_header.size()) {
      throw std::runtime_error("manifest row must have 11 fields");
    }
    FrameRecord record;
    record.episode_id = fields[0];
    record.frame_index = parseInteger(fields[1], "frame_index");
    record.stamp_ns = parseInteger(fields[2], "stamp_ns");
    record.pcd_path = fs::path(fields[3]);
    if (record.pcd_path.is_relative())
      record.pcd_path = base / record.pcd_path;
    const Eigen::Vector3d translation(parseFiniteDouble(fields[4], "tx"),
                                      parseFiniteDouble(fields[5], "ty"),
                                      parseFiniteDouble(fields[6], "tz"));
    Eigen::Quaterniond quaternion(
        parseFiniteDouble(fields[10], "qw"), parseFiniteDouble(fields[7], "qx"),
        parseFiniteDouble(fields[8], "qy"), parseFiniteDouble(fields[9], "qz"));
    if (quaternion.norm() <= 1e-9)
      throw std::runtime_error("zero odometry quaternion");
    quaternion.normalize();
    record.T_odom_body.linear() = quaternion.toRotationMatrix();
    record.T_odom_body.translation() = translation;

    if (record.frame_index != expected_index++) {
      throw std::runtime_error(
          "frame_index must be contiguous and start at zero");
    }
    if (record.stamp_ns <= previous_stamp) {
      throw std::runtime_error("stamp_ns must be strictly increasing");
    }
    previous_stamp = record.stamp_ns;
    if (episode_id.empty())
      episode_id = record.episode_id;
    if (record.episode_id != episode_id) {
      throw std::runtime_error(
          "one manifest must contain exactly one episode_id");
    }
    if (!fs::is_regular_file(record.pcd_path)) {
      throw std::runtime_error("PCD does not exist: " +
                               record.pcd_path.string());
    }
    records.push_back(std::move(record));
  }
  if (records.empty())
    throw std::runtime_error("manifest contains no frames");
  return records;
}

void ensureFreshOutput(const fs::path &path) {
  std::error_code error;
  if (fs::exists(path, error)) {
    if (error || !fs::is_directory(path, error)) {
      throw std::runtime_error("output exists and is not a directory: " +
                               path.string());
    }
    if (fs::directory_iterator(path, error) != fs::directory_iterator{} ||
        error) {
      throw std::runtime_error("output directory must be empty: " +
                               path.string());
    }
    return;
  }
  if (!fs::create_directories(path, error) || error) {
    throw std::runtime_error("failed to create output directory: " +
                             path.string());
  }
}

Cloud::Ptr loadCloud(const fs::path &path, double voxel_size_m) {
  auto cloud = pcl::make_shared<Cloud>();
  if (pcl::io::loadPCDFile(path.string(), *cloud) != 0 || cloud->empty()) {
    throw std::runtime_error("failed to load non-empty PCD: " + path.string());
  }
  if (voxel_size_m > 1e-6) {
    Cloud::Ptr filtered;
    if (!safeVoxelGridFilter<pcl::PointXYZI>(cloud, voxel_size_m, &filtered) ||
        !filtered || filtered->empty()) {
      throw std::runtime_error("failed to voxelize review PCD: " +
                               path.string());
    }
    cloud = filtered;
  }
  return cloud;
}

core::LioFrame makeFrame(const FrameRecord &record, const Cloud::Ptr &cloud) {
  core::LioFrame frame;
  frame.stamp.nsec = record.stamp_ns;
  frame.T_world_lidar = record.T_odom_body;
  frame.undistorted_cloud = cloud;
  frame.pose_valid = true;
  return frame;
}

std::string jsonEscape(const std::string &value) {
  std::ostringstream stream;
  for (const unsigned char c : value) {
    switch (c) {
    case '\\':
      stream << "\\\\";
      break;
    case '"':
      stream << "\\\"";
      break;
    case '\n':
      stream << "\\n";
      break;
    case '\r':
      stream << "\\r";
      break;
    case '\t':
      stream << "\\t";
      break;
    default:
      if (c < 0x20) {
        stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<int>(c) << std::dec;
      } else {
        stream << c;
      }
    }
  }
  return stream.str();
}

void writePoseJson(std::ostream &stream, const Eigen::Isometry3d &pose) {
  const Eigen::Quaterniond q(pose.rotation());
  stream << std::setprecision(17) << "{\"tx\":" << pose.translation().x()
         << ",\"ty\":" << pose.translation().y()
         << ",\"tz\":" << pose.translation().z() << ",\"qx\":" << q.x()
         << ",\"qy\":" << q.y() << ",\"qz\":" << q.z() << ",\"qw\":" << q.w()
         << "}";
}

std::size_t appendCloud(const Cloud &source,
                        const Eigen::Isometry3d &T_map_source,
                        const std::array<std::uint8_t, 3> &color,
                        ReviewCloud *destination) {
  const std::size_t before = destination->size();
  destination->reserve(destination->size() + source.size());
  for (const auto &point : source) {
    if (!pcl::isFinite(point))
      continue;
    const Eigen::Vector3d transformed =
        T_map_source * Eigen::Vector3d(point.x, point.y, point.z);
    if (!transformed.array().isFinite().all())
      continue;
    ReviewPoint output;
    output.x = static_cast<float>(transformed.x());
    output.y = static_cast<float>(transformed.y());
    output.z = static_cast<float>(transformed.z());
    output.r = color[0];
    output.g = color[1];
    output.b = color[2];
    destination->push_back(output);
  }
  return destination->size() - before;
}

struct ReviewCounts {
  std::size_t map_points = 0;
  std::size_t seed_keyframe_points = 0;
  std::size_t support_keyframe_points = 0;
  std::size_t query_points = 0;
  std::size_t reference_query_points = 0;
};

Eigen::Isometry3d planarMapToOdom(double x_m, double y_m, double yaw_deg) {
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(x_m, y_m, 0.0);
  pose.linear() = Eigen::AngleAxisd(yaw_deg * M_PI / 180.0,
                                    Eigen::Vector3d::UnitZ())
                      .toRotationMatrix();
  return pose;
}

ReviewCounts writeReviewPcd(const fs::path &path, const Cloud &map_cloud,
                            const Keyframe::Ptr &seed_keyframe,
                            const Keyframe::Ptr &support_keyframe,
                            const std::deque<BufferedFrame> &buffered_frames,
                            const Eigen::Isometry3d &T_map_odom) {
  static constexpr std::array<std::array<std::uint8_t, 3>, 5> palette = {
      {{{255, 0, 0}},
       {{255, 128, 0}},
       {{255, 255, 0}},
       {{255, 0, 255}},
       {{0, 255, 255}}}};
  ReviewCounts counts;
  auto review = pcl::make_shared<ReviewCloud>();
  counts.map_points = appendCloud(map_cloud, Eigen::Isometry3d::Identity(),
                                  {{90, 90, 90}}, review.get());
  if (seed_keyframe && seed_keyframe->cloud) {
    counts.seed_keyframe_points =
        appendCloud(*seed_keyframe->cloud, seed_keyframe->pose_optimized,
                    {{0, 96, 255}}, review.get());
  }
  if (support_keyframe && support_keyframe->cloud &&
      (!seed_keyframe || support_keyframe->id != seed_keyframe->id)) {
    counts.support_keyframe_points =
        appendCloud(*support_keyframe->cloud, support_keyframe->pose_optimized,
                    {{0, 255, 96}}, review.get());
  }
  std::size_t palette_index = 0;
  for (const auto &frame : buffered_frames) {
    counts.query_points +=
        appendCloud(*frame.cloud, T_map_odom * frame.record.T_odom_body,
                    palette[palette_index % palette.size()], review.get());
    ++palette_index;
  }
  review->width = static_cast<std::uint32_t>(review->size());
  review->height = 1;
  review->is_dense = true;
  if (pcl::io::savePCDFileBinaryCompressed(path.string(), *review) != 0) {
    throw std::runtime_error("failed to write review PCD: " + path.string());
  }
  return counts;
}

ReviewCounts writeComparisonPcd(
    const fs::path &path, const Cloud &map_cloud,
    const std::deque<BufferedFrame> &buffered_frames,
    const Eigen::Isometry3d &T_estimated_map_odom, bool locked,
    const Eigen::Isometry3d &T_reference_map_odom) {
  static constexpr std::array<std::array<std::uint8_t, 3>, 5> palette = {
      {{{255, 0, 0}},
       {{255, 128, 0}},
       {{255, 255, 0}},
       {{255, 0, 255}},
       {{0, 255, 255}}}};
  ReviewCounts counts;
  auto review = pcl::make_shared<ReviewCloud>();
  counts.map_points = appendCloud(map_cloud, Eigen::Isometry3d::Identity(),
                                  {{90, 90, 90}}, review.get());
  std::size_t palette_index = 0;
  for (const auto &frame : buffered_frames) {
    if (locked) {
      counts.query_points += appendCloud(
          *frame.cloud, T_estimated_map_odom * frame.record.T_odom_body,
          palette[palette_index % palette.size()], review.get());
    }
    counts.reference_query_points += appendCloud(
        *frame.cloud, T_reference_map_odom * frame.record.T_odom_body,
        {{0, 96, 255}}, review.get());
    ++palette_index;
  }
  review->width = static_cast<std::uint32_t>(review->size());
  review->height = 1;
  review->is_dense = true;
  if (pcl::io::savePCDFileBinaryCompressed(path.string(), *review) != 0) {
    throw std::runtime_error("failed to write comparison PCD: " +
                             path.string());
  }
  return counts;
}

void writeFrameResults(const fs::path &path,
                       const std::vector<FrameResult> &results) {
  std::ofstream stream(path);
  if (!stream.is_open())
    throw std::runtime_error("failed to write frame results");
  stream << "frame_index,stamp_ns,success,relocalization_locked,seed_"
            "keyframe_id,support_keyframe_id,matched_keyframe_id,"
            "map_tx,map_ty,map_tz,map_qx,map_qy,map_qz,map_qw\n";
  stream << std::setprecision(17);
  for (const auto &result : results) {
    const Eigen::Quaterniond q(result.T_map_body.rotation());
    stream << result.frame_index << ',' << result.stamp_ns << ','
           << (result.success ? 1 : 0) << ',' << (result.lock ? 1 : 0) << ','
           << result.seed_keyframe_id << ',' << result.support_keyframe_id
           << ',' << result.matched_keyframe_id << ','
           << result.T_map_body.translation().x() << ','
           << result.T_map_body.translation().y() << ','
           << result.T_map_body.translation().z() << ',' << q.x() << ','
           << q.y() << ',' << q.z() << ',' << q.w() << '\n';
  }
}

void writeReadme(const fs::path &path, bool locked, bool has_reference) {
  std::ofstream stream(path);
  stream << "Manual review contract\n"
         << "======================\n"
         << "*_alignment.pcd is authoritative for pose review: gray global "
            "map plus colored real query scans only.\n"
         << "*_provenance.pcd is authoritative for hypothesis provenance: "
            "gray global map plus blue seed and green support keyframes.\n"
         << "*_locked.pcd keeps all layers together for backward-compatible "
            "inspection, but blue-to-query overlap is not a pose gate.\n"
         << "Gray: global map loaded from the specified pbstream.\n"
         << "Blue: causal seed map keyframe that initialized the winning "
            "place hypothesis.\n"
         << "Green: final local support keyframe used before lock (omitted "
            "when equal to seed).\n"
         << "Red/orange/yellow/magenta/cyan: consecutive real query scans "
            "transformed by odometry "
            "and the reported map-to-odom transform.\n"
         << "*_comparison.pcd is present only when an evaluator reference "
            "map-to-odom was explicitly supplied. Blue is the same real "
            "query at that oracle reference pose; colored points are emitted "
            "only for an algorithm lock. The reference layer is never passed "
            "to the localizer.\n"
         << "algorithm_lock=" << (locked ? "true" : "false") << "\n"
         << "oracle_reference_layer=" << (has_reference ? "true" : "false")
         << "\n"
         << (has_reference
                 ? "The blue reference is evaluator-only ground truth; it is "
                   "not localizer input and does not change the lock.\n"
                 : "No ground truth is available. This tool never labels a "
                   "lock correct or false; inspect colored query geometry "
                   "against the gray global map in *_alignment.pcd.\n");
}

int run(const Options &options) {
  if (!fs::is_regular_file(options.map_path)) {
    throw std::runtime_error("map does not exist: " +
                             options.map_path.string());
  }
  if (!fs::is_regular_file(options.manifest_path)) {
    throw std::runtime_error("manifest does not exist: " +
                             options.manifest_path.string());
  }
  ensureFreshOutput(options.output_dir);
  const auto records = readManifest(options.manifest_path);

  Config config;
  config.mode = "localization";
  config.map_path = options.map_path.string();
  if (!options.atlas_path.empty()) {
    config.reloc_atlas_enable = true;
    config.reloc_atlas_path = options.atlas_path.string();
  }
  if (options.reloc_debug) {
    config.reloc_debug_enable = true;
    config.reloc_debug_path =
        (options.output_dir / "relocalization_debug.jsonl").string();
  }
  N3MappingCore core(config);
  if (!core.loadMap(options.map_path.string())) {
    throw std::runtime_error("failed to load map: " +
                             options.map_path.string());
  }
  const auto global_map = core.buildGlobalMap();
  if (!global_map || global_map->empty()) {
    throw std::runtime_error("loaded map has no global cloud");
  }

  std::vector<FrameResult> results;
  std::deque<BufferedFrame> buffered_frames;
  core::BackendOutput last_output;
  const FrameRecord *last_record = nullptr;
  bool locked = false;
  for (const auto &record : records) {
    auto cloud = loadCloud(record.pcd_path, options.input_voxel_size_m);
    buffered_frames.push_back({record, cloud});
    while (static_cast<int>(buffered_frames.size()) >
           config.reloc_static_agg_max_frames) {
      buffered_frames.pop_front();
    }
    last_output = core.processLocalizationFrame(makeFrame(record, cloud));
    last_record = &record;
    results.push_back({record.frame_index, record.stamp_ns, last_output.success,
                       last_output.relocalization_locked,
                       last_output.relocalization_seed_keyframe_id,
                       last_output.relocalization_support_keyframe_id,
                       last_output.matched_keyframe_id,
                       last_output.T_world_lidar});
    if (last_output.relocalization_locked) {
      locked = true;
      break;
    }
  }
  if (!last_record)
    throw std::runtime_error("no frames were processed");

  const Eigen::Isometry3d T_map_odom =
      last_output.T_world_lidar * last_record->T_odom_body.inverse();
  const bool has_reference = std::isfinite(options.reference_map_x_m);
  const auto seed_keyframe =
      locked ? core.getKeyframe(last_output.relocalization_seed_keyframe_id)
             : nullptr;
  const auto support_keyframe =
      locked ? core.getKeyframe(last_output.relocalization_support_keyframe_id)
             : nullptr;
  const std::string review_name =
      records.front().episode_id +
      (locked ? "_locked.pcd" : "_no_lock_unverified_pose.pcd");
  const std::string alignment_review_name =
      records.front().episode_id + "_alignment.pcd";
  const std::string provenance_review_name =
      records.front().episode_id + "_provenance.pcd";
  const std::string comparison_review_name =
      has_reference ? records.front().episode_id + "_comparison.pcd" : "";
  const ReviewCounts counts = writeReviewPcd(
      options.output_dir / review_name, *global_map, seed_keyframe,
      support_keyframe, buffered_frames, T_map_odom);
  writeReviewPcd(options.output_dir / alignment_review_name, *global_map,
                 nullptr, nullptr, buffered_frames, T_map_odom);
  writeReviewPcd(options.output_dir / provenance_review_name, *global_map,
                 seed_keyframe, support_keyframe, {}, T_map_odom);
  ReviewCounts comparison_counts;
  if (has_reference) {
    comparison_counts = writeComparisonPcd(
        options.output_dir / comparison_review_name, *global_map,
        buffered_frames, T_map_odom, locked,
        planarMapToOdom(options.reference_map_x_m, options.reference_map_y_m,
                        options.reference_map_yaw_deg));
  }
  writeFrameResults(options.output_dir / "frame_results.csv", results);
  writeReadme(options.output_dir / "README.txt", locked, has_reference);

  std::ofstream summary(options.output_dir / "result.json");
  if (!summary.is_open())
    throw std::runtime_error("failed to write result.json");
  summary << "{\n"
          << "  \"schema\": \"n3mapping_real_relocalization_result_v2\",\n"
          << "  \"episode_id\": \"" << jsonEscape(records.front().episode_id)
          << "\",\n"
          << "  \"map_path\": \""
          << jsonEscape(fs::absolute(options.map_path).string()) << "\",\n"
          << "  \"manifest_path\": \""
          << jsonEscape(fs::absolute(options.manifest_path).string()) << "\",\n"
          << "  \"input_frame_count\": " << records.size() << ",\n"
          << "  \"processed_frame_count\": " << results.size() << ",\n"
          << "  \"algorithm_lock\": " << (locked ? "true" : "false") << ",\n"
          << "  \"manual_alignment_correct\": null,\n"
          << "  \"lock_frame_index\": "
          << (locked ? last_record->frame_index : -1) << ",\n"
          << "  \"lock_stamp_ns\": " << (locked ? last_record->stamp_ns : 0)
          << ",\n"
          << "  \"matched_keyframe_id\": "
          << (locked ? last_output.matched_keyframe_id : -1) << ",\n"
          << "  \"relocalization_seed_keyframe_id\": "
          << (locked ? last_output.relocalization_seed_keyframe_id : -1)
          << ",\n"
          << "  \"relocalization_support_keyframe_id\": "
          << (locked ? last_output.relocalization_support_keyframe_id : -1)
          << ",\n"
          << "  \"reported_map_body_pose\": ";
  writePoseJson(summary, last_output.T_world_lidar);
  summary << ",\n  \"reported_map_odom_transform\": ";
  writePoseJson(summary, T_map_odom);
  summary
      << ",\n"
      << "  \"review_pcd\": \"" << jsonEscape(review_name) << "\",\n"
      << "  \"alignment_review_pcd\": \""
      << jsonEscape(alignment_review_name) << "\",\n"
      << "  \"provenance_review_pcd\": \""
      << jsonEscape(provenance_review_name) << "\",\n"
      << "  \"comparison_review_pcd\": \""
      << jsonEscape(comparison_review_name) << "\",\n"
      << "  \"review_counts\": {\"map\":" << counts.map_points
      << ",\"seed_keyframe\":" << counts.seed_keyframe_points
      << ",\"support_keyframe\":" << counts.support_keyframe_points
      << ",\"real_query\":" << counts.query_points << "},\n"
      << "  \"comparison_counts\": {\"map\":"
      << comparison_counts.map_points << ",\"estimated_query\":"
      << comparison_counts.query_points << ",\"reference_query\":"
      << comparison_counts.reference_query_points << "},\n"
      << "  \"config_source\": \"compiled_product_defaults_no_overrides\",\n"
      << "  \"input_voxel_size_m\": " << options.input_voxel_size_m << ",\n"
      << "  \"relocalization_contract\": {"
      << "\"temporal_window_size\":" << config.reloc_temporal_window_size << ','
      << "\"static_agg_max_frames\":" << config.reloc_static_agg_max_frames
      << ','
      << "\"lock_min_winner_streak\":" << config.reloc_lock_min_winner_streak
      << ',' << "\"lock_min_converged_updates\":"
      << config.reloc_lock_min_converged_updates << "}\n}\n";

  std::cout << "episode=" << records.front().episode_id
            << " algorithm_lock=" << (locked ? 1 : 0)
            << " processed_frames=" << results.size() << " seed_keyframe_id="
            << (locked ? last_output.relocalization_seed_keyframe_id : -1)
            << " support_keyframe_id="
            << (locked ? last_output.relocalization_support_keyframe_id : -1)
            << " review_pcd=" << (options.output_dir / review_name) << '\n';
  return locked ? 0 : 2;
}

} // namespace
} // namespace n3mapping

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  n3mapping::Options options;
  if (!n3mapping::parseArgs(argc, argv, &options)) {
    n3mapping::printUsage(argv[0]);
    return 1;
  }
  try {
    return n3mapping::run(options);
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
