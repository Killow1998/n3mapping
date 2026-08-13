#include "n3mapping/runtime_performance_debug_logger.h"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "n3mapping/relocalization_debug_logger.h"

namespace n3mapping {
namespace {

void appendComma(std::ostream &stream, bool *first) {
  if (*first) {
    *first = false;
  } else {
    stream << ',';
  }
}

std::string jsonEscape(const std::string &value) {
  std::ostringstream stream;
  for (const unsigned char ch : value) {
    switch (ch) {
    case '"':
      stream << "\\\"";
      break;
    case '\\':
      stream << "\\\\";
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
      if (ch < 0x20) {
        stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
               << static_cast<int>(ch) << std::dec << std::setfill(' ');
      } else {
        stream << static_cast<char>(ch);
      }
      break;
    }
  }
  return stream.str();
}

void appendString(std::ostream &stream, bool *first, const char *key,
                  const std::string &value) {
  appendComma(stream, first);
  stream << '"' << key << "\":\"" << jsonEscape(value) << '"';
}

void appendBool(std::ostream &stream, bool *first, const char *key,
                bool value) {
  appendComma(stream, first);
  stream << '"' << key << "\":" << (value ? "true" : "false");
}

void appendNumber(std::ostream &stream, bool *first, const char *key,
                  double value) {
  appendComma(stream, first);
  stream << '"' << key << "\":";
  if (std::isfinite(value)) {
    stream << std::setprecision(17) << value;
  } else {
    stream << "null";
  }
}

void appendInteger(std::ostream &stream, bool *first, const char *key,
                   int64_t value) {
  appendComma(stream, first);
  stream << '"' << key << "\":" << value;
}

void appendSize(std::ostream &stream, bool *first, const char *key,
                std::size_t value) {
  appendComma(stream, first);
  stream << '"' << key << "\":" << value;
}

} // namespace

RuntimePerformanceDebugLogger::RuntimePerformanceDebugLogger(
    const Config &config)
    : enabled_(config.reloc_debug_enable), path_(resolvePath(config)) {
  if (!enabled_) {
    return;
  }
  try {
    const std::filesystem::path path(path_);
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    stream_.open(path_, std::ios::out | std::ios::app);
    ready_ = stream_.is_open();
  } catch (const std::exception &) {
    ready_ = false;
  }
}

std::string RuntimePerformanceDebugLogger::resolvePath(const Config &config) {
  const std::filesystem::path reloc_path(
      RelocalizationDebugLogger::resolvePath(config));
  const auto parent = reloc_path.parent_path();
  return (parent / "runtime_performance_debug.jsonl").string();
}

bool RuntimePerformanceDebugLogger::append(
    const RuntimePerformanceDebugEvent &event) {
  if (!enabled_ || !ready_) {
    return false;
  }

  std::ostringstream record;
  bool first = true;
  record << '{';
  appendString(record, &first, "schema", "n3mapping_runtime_performance_v1");
  appendString(record, &first, "record_type", "map_extension_frame");
  appendNumber(record, &first, "processing_time", event.processing_time);
  appendSize(record, &first, "frame_index", event.frame_index);
  appendNumber(record, &first, "sensor_timestamp", event.sensor_timestamp);
  appendNumber(record, &first, "sensor_delta_ms", event.sensor_delta_ms);
  appendNumber(record, &first, "callback_interarrival_ms",
               event.callback_interarrival_ms);
  appendSize(record, &first, "input_points", event.input_points);
  appendBool(record, &first, "core_success", event.core_success);
  appendBool(record, &first, "accepted_keyframe", event.accepted_keyframe);
  appendInteger(record, &first, "keyframe_id", event.keyframe_id);
  appendInteger(record, &first, "matched_keyframe_id",
                event.matched_keyframe_id);
  appendString(record, &first, "relocalization_state",
               event.relocalization_state);
  appendString(record, &first, "pose_source", event.pose_source);
  appendString(record, &first, "relocalization_decision",
               event.relocalization_decision);
  appendBool(record, &first, "callback_skipped", event.callback_skipped);
  appendBool(record, &first, "published_global_pose",
             event.published_global_pose);
  appendBool(record, &first, "published_body_cloud",
             event.published_body_cloud);
  appendBool(record, &first, "published_world_cloud",
             event.published_world_cloud);
  appendNumber(record, &first, "callback_lock_wait_ms",
               event.callback_lock_wait_ms);
  appendNumber(record, &first, "ros_conversion_ms", event.ros_conversion_ms);
  appendNumber(record, &first, "core_frame_ms", event.core_frame_ms);
  appendNumber(record, &first, "initial_relocalization_ms",
               event.initial_relocalization_ms);
  appendNumber(record, &first, "loaded_map_tracking_ms",
               event.loaded_map_tracking_ms);
  appendNumber(record, &first, "keyframe_gate_ms", event.keyframe_gate_ms);
  appendNumber(record, &first, "keyframe_commit_ms",
               event.keyframe_commit_ms);
  appendNumber(record, &first, "graph_update_ms", event.graph_update_ms);
  appendNumber(record, &first, "descriptor_update_ms",
               event.descriptor_update_ms);
  appendNumber(record, &first, "post_commit_refresh_ms",
               event.post_commit_refresh_ms);
  appendNumber(record, &first, "authority_publish_ms",
               event.authority_publish_ms);
  appendNumber(record, &first, "odometry_path_publish_ms",
               event.odometry_path_publish_ms);
  appendNumber(record, &first, "callback_locked_ms",
               event.callback_locked_ms);
  appendNumber(record, &first, "cloud_publish_ms", event.cloud_publish_ms);
  appendNumber(record, &first, "callback_total_ms", event.callback_total_ms);
  record << '}';

  std::lock_guard<std::mutex> lock(mutex_);
  stream_ << record.str() << '\n';
  return stream_.good();
}

} // namespace n3mapping
