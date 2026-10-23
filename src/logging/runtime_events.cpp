#include "n3mapping/runtime_events.h"
#include "n3mapping/async_log.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/core/realtime_output.h"
#include <sstream>

namespace n3mapping {
void RuntimeEvents::backend(const core::BackendOutput& output, std::int64_t stamp, double ms) {
  auto& log = AsyncLog::instance();
  log.count("backend_frame", ms);
  log.count("decision/" + output.relocalization_decision);
  if (output.accepted_keyframe) log.count("keyframe_accepted");
  const auto state = std::string(relocalizationStateName(output.relocalization_state)) + "/" +
      poseSourceName(output.pose_source) + "/" + std::to_string(static_cast<int>(output.mapping_block_reason));
  if (state != backend_state_ || output.relocalization_locked) {
    std::ostringstream detail;
    detail << "stamp_ns=" << stamp << " from=" << backend_state_ << " to=" << state
           << " decision=" << output.relocalization_decision << " seed=" << output.relocalization_seed_keyframe_id
           << " support=" << output.relocalization_support_keyframe_id << " success=" << output.success;
    logEvent(output.relocalization_locked ? "relocalization_lock" : "backend_state", detail.str(), true);
    backend_state_ = state;
  }
}
void RuntimeEvents::live(const core::OdometryEmission& emission) {
  auto& log = AsyncLog::instance();
  log.count(emission.pose ? "live_published" : "live_suppressed");
  if (emission.reason != live_state_) {
    const auto& d = emission.diagnostic;
    logEvent("live_state", "from=" + live_state_ + " to=" + emission.reason +
        " stamp_ns=" + std::to_string(d.input_stamp_nsec) +
        " correction_ns=" + std::to_string(d.correction_stamp_nsec) +
        " generation=" + std::to_string(d.generation), true);
    live_state_ = emission.reason;
  }
}
void RuntimeEvents::loop(const CoreLoopClosureResult& result) {
  auto& log = AsyncLog::instance(); log.count("loop_cycle");
  if (result.optimized || !result.accepted_loops.empty() || result.graph_rebase_count || result.stale_job_count) {
    std::ostringstream detail;
    detail << "optimized=" << result.optimized << " edges=" << result.edge_count
           << " rebase=" << result.graph_rebase_count << " stale=" << result.stale_job_count
           << " pending=" << result.pending_query_count << " worker_ms=" << result.worker_wall_ms;
    for (const auto& edge : result.accepted_loops)
      detail << " pair=" << edge.query_id << ':' << edge.match_id;
    logEvent("loop_commit", detail.str(), true);
  }
}
} // namespace n3mapping
