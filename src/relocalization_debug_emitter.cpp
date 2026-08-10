#include "n3mapping/relocalization_debug_emitter.h"

#include <chrono>

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
