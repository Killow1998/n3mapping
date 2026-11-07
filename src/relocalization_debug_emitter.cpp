#include "n3mapping/relocalization_debug_emitter.h"

#include <chrono>
#include "n3mapping/config.h"
#include "n3mapping/relocalization_debug_logger.h"
#include "n3mapping/relocalization_hypothesis_manager.h"

namespace n3mapping {

RelocalizationDebugEmitter::RelocalizationDebugEmitter(const Config &config)
    : enabled_(config.reloc_debug_enable),
      path_(RelocalizationDebugLogger::resolvePath(config)) {}

bool RelocalizationDebugEmitter::enabled() const noexcept { return enabled_; }

RelocalizationDebugEvent RelocalizationDebugEmitter::beginRelocalization() {
  RelocalizationDebugEvent event;
  if (!enabled_) {
    return event;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  event.query_index = ++relocalization_query_index_;
  event.processing_time = processingTimeSeconds();
  return event;
}

RelocTrackingDebugEvent RelocalizationDebugEmitter::beginTracking() {
  RelocTrackingDebugEvent event;
  if (!enabled_) {
    return event;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  event.query_index = ++tracking_query_index_;
  event.processing_time = processingTimeSeconds();
  return event;
}

void RelocalizationDebugEmitter::captureHypotheses(
    RelocalizationDebugEvent& event,
    const RelocalizationHypothesisManager& hypotheses,
    const Eigen::Isometry3d& odom_pose) {
  event.hypotheses.clear();
  event.hypotheses.reserve(hypotheses.size());
  for (const auto &hyp : hypotheses.hypotheses()) {
    RelocDebugHypothesisSummary summary;
    summary.tracking_proposal = hyp.tracking_proposal;
    summary.seed_match_id = hyp.seed_match_id;
    summary.last_match_id = hyp.last_match_id;
    summary.pose_in_map = hyp.T_map_odom * odom_pose;
    summary.cumulative_log_likelihood = hyp.cumulative_log_likelihood;
    summary.num_updates = hyp.num_updates;
    summary.observations = hypotheses.windowCount() - hyp.first_observation + 1;
    summary.converged_updates = hyp.converged_updates;
    summary.visibility_updates = hyp.visibility_updates;
    summary.mean_visibility_consistency =
        hyp.visibility_updates > 0
            ? hyp.visibility_consistency_sum /
                  static_cast<double>(hyp.visibility_updates)
            : std::numeric_limits<double>::quiet_NaN();
    summary.mean_visibility_evidence =
        hyp.visibility_updates > 0
            ? hyp.visibility_evidence_sum /
                  static_cast<double>(hyp.visibility_updates)
            : std::numeric_limits<double>::quiet_NaN();
    // Initial hypotheses duplicate the basin-best registrations emitted in
    // the same event. Keep one complete copy to bound shadow logging cost.
    if (event.basin_best_results.empty()) {
      summary.registration_observability =
          hyp.last_registration_observability;
    }
    summary.alive = hyp.alive;
    event.hypotheses.push_back(std::move(summary));
  }
}

void RelocalizationDebugEmitter::finishRelocalization(
    RelocalizationDebugEvent &event, const std::string &lock_result,
    const std::string &reject_reason) {
  if (!enabled_) {
    return;
  }

  event.processing_time = processingTimeSeconds();
  event.lock_result = lock_result;
  event.lock_accepted = (lock_result == "accepted");
  event.reject_reason = reject_reason;

  std::lock_guard<std::mutex> lock(mutex_);
  RelocalizationDebugLogger::appendRelocalization(path_, event);
}

void RelocalizationDebugEmitter::finishTracking(
    RelocTrackingDebugEvent &event, bool result_success,
    int consecutive_track_failures, const std::string &reject_reason) {
  if (!enabled_) {
    return;
  }

  event.processing_time = processingTimeSeconds();
  event.result_success = result_success;
  event.consecutive_track_failures = consecutive_track_failures;
  event.reject_reason = reject_reason;

  std::lock_guard<std::mutex> lock(mutex_);
  RelocalizationDebugLogger::appendTracking(path_, event);
}

double RelocalizationDebugEmitter::processingTimeSeconds() {
  using Clock = std::chrono::system_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

} // namespace n3mapping
