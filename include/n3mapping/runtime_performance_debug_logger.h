#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>

#include "n3mapping/config.h"

namespace n3mapping {

// PERF-ME-01B: one record per synchronized ROS frame. This is intentionally a
// separate JSONL stream from relocalization_debug.jsonl so existing evidence
// consumers never see a new record type. The logger is enabled by the existing
// reloc_debug_enable switch and does not participate in runtime decisions.
struct RuntimePerformanceDebugEvent {
  std::string mode;
  double processing_time = 0.0;
  uint64_t frame_index = 0;
  double sensor_timestamp = std::numeric_limits<double>::quiet_NaN();
  double sensor_delta_ms = std::numeric_limits<double>::quiet_NaN();
  double callback_interarrival_ms =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t input_points = 0;
  bool core_success = false;
  bool accepted_keyframe = false;
  int64_t keyframe_id = -1;
  int64_t matched_keyframe_id = -1;
  std::string relocalization_state;
  std::string pose_source;
  std::string relocalization_decision;
  bool relocalization_locked = false;
  bool tracking_attempted = false;
  bool callback_skipped = false;
  bool published_global_pose = false;
  bool published_body_cloud = false;
  bool published_world_cloud = false;

  double callback_lock_wait_ms =
      std::numeric_limits<double>::quiet_NaN();
  double ros_conversion_ms = std::numeric_limits<double>::quiet_NaN();
  double core_frame_ms = std::numeric_limits<double>::quiet_NaN();
  double initial_relocalization_ms =
      std::numeric_limits<double>::quiet_NaN();
  double loaded_map_tracking_ms =
      std::numeric_limits<double>::quiet_NaN();
  double keyframe_gate_ms = std::numeric_limits<double>::quiet_NaN();
  double keyframe_commit_ms = std::numeric_limits<double>::quiet_NaN();
  double graph_update_ms = std::numeric_limits<double>::quiet_NaN();
  double descriptor_update_ms = std::numeric_limits<double>::quiet_NaN();
  double post_commit_refresh_ms =
      std::numeric_limits<double>::quiet_NaN();
  double authority_publish_ms = std::numeric_limits<double>::quiet_NaN();
  double odometry_path_publish_ms =
      std::numeric_limits<double>::quiet_NaN();
  double callback_locked_ms = std::numeric_limits<double>::quiet_NaN();
  double cloud_publish_ms = std::numeric_limits<double>::quiet_NaN();
  double callback_total_ms = std::numeric_limits<double>::quiet_NaN();
};

// One record per mapping loop-timer callback. Empty cycles are retained so
// timer scheduling and lock contention remain observable; analyzers grade
// actual loop work separately from idle polling.
struct RuntimePerformanceLoopEvent {
  std::string mode;
  double processing_time = 0.0;
  uint64_t cycle_index = 0;
  std::size_t queued_keyframe_count = 0;
  std::size_t detected_candidate_count = 0;
  std::size_t place_candidate_count = 0;
  std::size_t accepted_loop_count = 0;
  std::size_t edge_count = 0;
  bool optimized = false;
  double lock_wait_ms = std::numeric_limits<double>::quiet_NaN();
  double core_ms = std::numeric_limits<double>::quiet_NaN();
  double publish_ms = std::numeric_limits<double>::quiet_NaN();
  double total_ms = std::numeric_limits<double>::quiet_NaN();
};

class RuntimePerformanceDebugLogger {
public:
  explicit RuntimePerformanceDebugLogger(const Config &config);

  static std::string resolvePath(const Config &config);
  bool enabled() const noexcept { return enabled_; }
  bool ready() const noexcept { return ready_; }
  const std::string &path() const noexcept { return path_; }
  bool append(const RuntimePerformanceDebugEvent &event);
  bool append(const RuntimePerformanceLoopEvent &event);

private:
  bool enabled_ = false;
  bool ready_ = false;
  std::string path_;
  std::ofstream stream_;
  std::mutex mutex_;
};

} // namespace n3mapping
