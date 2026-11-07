#pragma once

#include <cstdint>
#include <mutex>
#include <string>

#include <Eigen/Geometry>

namespace n3mapping {

struct Config;
struct RelocalizationDebugEvent;
struct RelocTrackingDebugEvent;
class RelocalizationHypothesisManager;

// Owns debug-event sequencing and persistence. Callers populate the event
// payload; this class adds the per-stream index and terminal fields without
// participating in relocalization decisions.
class RelocalizationDebugEmitter {
public:
  explicit RelocalizationDebugEmitter(const Config &config);

  bool enabled() const noexcept;

  RelocalizationDebugEvent beginRelocalization();
  RelocTrackingDebugEvent beginTracking();
  static void captureHypotheses(RelocalizationDebugEvent& event,
      const RelocalizationHypothesisManager& hypotheses,
      const Eigen::Isometry3d& odom_pose);

  void finishRelocalization(RelocalizationDebugEvent &event,
                            const std::string &lock_result,
                            const std::string &reject_reason);
  void finishTracking(RelocTrackingDebugEvent &event, bool result_success,
                      int consecutive_track_failures,
                      const std::string &reject_reason);

private:
  static double processingTimeSeconds();

  bool enabled_ = false;
  std::string path_;
  uint64_t relocalization_query_index_ = 0;
  uint64_t tracking_query_index_ = 0;
  std::mutex mutex_;
};

} // namespace n3mapping
