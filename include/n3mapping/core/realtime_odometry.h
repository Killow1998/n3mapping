#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "n3mapping/relocalization_state.h"

namespace n3mapping::core {

struct RealtimeOdometryPose {
  int64_t stamp_nsec;
  int64_t correction_stamp_nsec;
  Eigen::Isometry3d pose;
};

struct MapOdometryCorrection {
  int64_t stamp_nsec;
  std::string odom_frame;
  Eigen::Isometry3d map_to_odom;
};

enum class RealtimeOdometryReason {
  Ready,
  AwaitingCorrection,
  InvalidCorrection,
  InvalidInputTime,
  InputTimeRegressed,
  InvalidInputPose,
  DuplicateInput,
  SourceFrameMismatch,
  ChildFrameMismatch,
  InputBeforeCorrection,
  InvalidOutputPose,
  OdometryDiverged,
};

inline const char* realtimeOdometryReasonName(RealtimeOdometryReason reason) {
  switch (reason) {
    case RealtimeOdometryReason::Ready: return "ready";
    case RealtimeOdometryReason::AwaitingCorrection: return "awaiting_correction";
    case RealtimeOdometryReason::OdometryDiverged: return "odometry_diverged";
    case RealtimeOdometryReason::InvalidCorrection: return "invalid_correction";
    case RealtimeOdometryReason::InvalidInputTime: return "invalid_input_time";
    case RealtimeOdometryReason::InputTimeRegressed: return "input_time_regressed";
    case RealtimeOdometryReason::InvalidInputPose: return "invalid_input_pose";
    case RealtimeOdometryReason::DuplicateInput: return "duplicate_input";
    case RealtimeOdometryReason::SourceFrameMismatch: return "source_frame_mismatch";
    case RealtimeOdometryReason::ChildFrameMismatch: return "child_frame_mismatch";
    case RealtimeOdometryReason::InputBeforeCorrection: return "input_before_correction";
    case RealtimeOdometryReason::InvalidOutputPose: return "invalid_output_pose";
  }
  return "invalid_correction";
}

struct RealtimeOdometryDiagnostic {
  RealtimeOdometryReason reason = RealtimeOdometryReason::AwaitingCorrection;
  int64_t input_stamp_nsec = 0;
  int64_t correction_stamp_nsec = 0;
  int64_t now_nsec = 0;
};

// Only the small correction snapshot is shared. Registration, graph updates,
// cloud conversion, and publication must never run under this mutex.
class RealtimeOdometry {
 public:
  std::optional<MapOdometryCorrection> updateCorrection(
                        int64_t stamp_nsec, const std::string& source_frame,
                        const std::string& child_frame,
                        const Eigen::Isometry3d& raw_pose,
                        const Eigen::Isometry3d& map_pose, bool authorized) {
    std::lock_guard<std::mutex> lock(mutex_);
    // OdometrySanity is latched for the lifetime of the mapping session. A
    // trusted correction cannot make the raw input valid again; rebuilding the
    // SLAM unit creates a fresh RealtimeOdometry instance instead.
    if (correction_status_ == RealtimeOdometryReason::OdometryDiverged) {
      return std::nullopt;
    }
    if (!authorized) {
      // A rejected observation cannot replace or revoke the last trusted anchor.
      // Input/frame failures have their own explicit invalidation path.
      return std::nullopt;
    }
    if (stamp_nsec <= 0 || source_frame.empty() ||
        child_frame.empty() || !isFiniteRigidPose(raw_pose) ||
        !isFiniteRigidPose(map_pose)) {
      invalidate(stamp_nsec, RealtimeOdometryReason::InvalidCorrection);
      return std::nullopt;
    }
    if (stamp_nsec <= invalidated_at_nsec_ ||
        stamp_nsec <= correction_.stamp_nsec) {
      return std::nullopt;
    }
    const Eigen::Isometry3d map_to_odom = map_pose * raw_pose.inverse();
    if (!isFiniteRigidPose(map_to_odom)) {
      invalidate(stamp_nsec, RealtimeOdometryReason::InvalidCorrection);
      return std::nullopt;
    }
    correction_ = Correction{stamp_nsec, source_frame, child_frame, map_to_odom};
    correction_status_ = RealtimeOdometryReason::Ready;
    return MapOdometryCorrection{stamp_nsec, source_frame, map_to_odom};
  }

  // Invalidate the live projection when the mapping backend has latched the
  // existing raw-odometry divergence detector. This is deliberately separate
  // from an ordinary rejected geometric observation, which retains the last
  // trusted correction above.
  void invalidateForOdometryDivergence(int64_t stamp_nsec) {
    std::lock_guard<std::mutex> lock(mutex_);
    correction_status_ = RealtimeOdometryReason::OdometryDiverged;
    invalidated_at_nsec_ = std::max(invalidated_at_nsec_, stamp_nsec);
  }

  std::optional<RealtimeOdometryPose> project(
      int64_t stamp_nsec, int64_t now_nsec, const std::string& source_frame,
      const std::string& child_frame, const Eigen::Isometry3d& raw_pose,
      bool pose_valid = true, RealtimeOdometryDiagnostic* diagnostic = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto report = [&](RealtimeOdometryReason reason) {
      if (diagnostic) {
        *diagnostic = {reason, stamp_nsec, correction_.stamp_nsec, now_nsec};
      }
    };
    const auto reject = [&](RealtimeOdometryReason reason) {
      invalidate(std::max(last_input_stamp_nsec_, std::min(stamp_nsec, now_nsec)), reason);
      report(reason);
      return std::nullopt;
    };
    if (correction_status_ == RealtimeOdometryReason::OdometryDiverged) {
      report(RealtimeOdometryReason::OdometryDiverged);
      return std::nullopt;
    }
    if (stamp_nsec <= 0 || stamp_nsec > now_nsec) {
      return reject(RealtimeOdometryReason::InvalidInputTime);
    }
    if (stamp_nsec < last_input_stamp_nsec_) {
      return reject(RealtimeOdometryReason::InputTimeRegressed);
    }
    if (!pose_valid || !isFiniteRigidPose(raw_pose)) {
      return reject(RealtimeOdometryReason::InvalidInputPose);
    }
    if (stamp_nsec == last_input_stamp_nsec_) {
      report(RealtimeOdometryReason::DuplicateInput);
      return std::nullopt;
    }
    last_input_stamp_nsec_ = stamp_nsec;
    if (correction_status_ != RealtimeOdometryReason::Ready) {
      report(correction_status_);
      return std::nullopt;
    }
    if (source_frame != correction_.source_frame) {
      return reject(RealtimeOdometryReason::SourceFrameMismatch);
    }
    if (child_frame != correction_.child_frame) {
      return reject(RealtimeOdometryReason::ChildFrameMismatch);
    }
    if (now_nsec < correction_.stamp_nsec) {
      return reject(RealtimeOdometryReason::InvalidCorrection);
    }
    // The independent backend queue may install a newer correction before an
    // already-received raw callback runs. Drop that sample, not the correction.
    if (stamp_nsec < correction_.stamp_nsec) {
      report(RealtimeOdometryReason::InputBeforeCorrection);
      return std::nullopt;
    }
    const Eigen::Isometry3d map_pose = correction_.map_to_odom * raw_pose;
    if (!isFiniteRigidPose(map_pose)) {
      return reject(RealtimeOdometryReason::InvalidOutputPose);
    }
    report(RealtimeOdometryReason::Ready);
    return RealtimeOdometryPose{stamp_nsec, correction_.stamp_nsec, map_pose};
  }

  // Clouds use the paired scan pose, which may precede the latest live odometry.
  // This read-only projection must not mutate the input stream's ordering state.
  std::optional<RealtimeOdometryPose> projectScan(
      int64_t stamp_nsec, int64_t now_nsec, const std::string& source_frame,
      const std::string& child_frame, const Eigen::Isometry3d& raw_pose) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (correction_status_ != RealtimeOdometryReason::Ready ||
        stamp_nsec <= 0 || stamp_nsec > now_nsec ||
        stamp_nsec < correction_.stamp_nsec ||
        source_frame != correction_.source_frame ||
        child_frame != correction_.child_frame || !isFiniteRigidPose(raw_pose)) {
      return std::nullopt;
    }
    const Eigen::Isometry3d pose = correction_.map_to_odom * raw_pose;
    if (!isFiniteRigidPose(pose)) return std::nullopt;
    return RealtimeOdometryPose{stamp_nsec, correction_.stamp_nsec, pose};
  }

 private:
  struct Correction {
    int64_t stamp_nsec = 0;
    std::string source_frame;
    std::string child_frame;
    Eigen::Isometry3d map_to_odom = Eigen::Isometry3d::Identity();
  };

  void invalidate(int64_t stamp_nsec, RealtimeOdometryReason reason) {
    correction_status_ = reason;
    invalidated_at_nsec_ = std::max(invalidated_at_nsec_, stamp_nsec);
  }

  mutable std::mutex mutex_;
  Correction correction_;
  RealtimeOdometryReason correction_status_ = RealtimeOdometryReason::AwaitingCorrection;
  int64_t last_input_stamp_nsec_ = 0;
  int64_t invalidated_at_nsec_ = 0;
};

}  // namespace n3mapping::core
