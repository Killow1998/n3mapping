#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>

#include "n3mapping/relocalization_state.h"

namespace n3mapping::core {

struct RealtimeOdometryPose {
  int64_t stamp_nsec;
  int64_t correction_stamp_nsec;
  Eigen::Isometry3d pose;
  uint64_t generation = 0;
  uint64_t basis_revision = 0;
  Eigen::Isometry3d map_from_source = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d source_from_body = Eigen::Isometry3d::Identity();
  std::string source_frame;
  std::string body_frame;
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
  ProjectionExpired,
  SourceObservationMismatch,
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
    case RealtimeOdometryReason::ProjectionExpired: return "projection_expired";
    case RealtimeOdometryReason::SourceObservationMismatch: return "source_observation_mismatch";
  }
  return "invalid_correction";
}

struct RealtimeOdometryDiagnostic {
  RealtimeOdometryReason reason = RealtimeOdometryReason::AwaitingCorrection;
  int64_t input_stamp_nsec = 0;
  int64_t correction_stamp_nsec = 0;
  int64_t now_nsec = 0;
  uint64_t generation = 0;
};

// Only bounded pose/correction records are shared. Registration, graph updates,
// cloud conversion, and publication must never run under this mutex.
class RealtimeOdometry {
 public:
  // Match the adapter's existing synchronization retention, not a new timeout.
  explicit RealtimeOdometry(std::size_t retention = 1)
      : retention_(std::max<std::size_t>(1, retention)) {}

  void setSynchronizationRetention(std::size_t retention) {
    std::lock_guard<std::mutex> lock(mutex_);
    retention_ = std::max<std::size_t>(1, retention);
    trimProjections();
  }

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
    if (correction_.stamp_nsec > 0 &&
        (source_frame != correction_.source_frame || child_frame != correction_.child_frame)) {
      // This authorized correction begins the new basis at stamp_nsec. Retire
      // older observations, including any already frozen old-frame projection,
      // without expiring the new correction's own paired observation.
      invalidate(stamp_nsec - 1, source_frame != correction_.source_frame
          ? RealtimeOdometryReason::SourceFrameMismatch
          : RealtimeOdometryReason::ChildFrameMismatch);
    }
    correction_ = Correction{stamp_nsec, source_frame, child_frame, map_to_odom};
    ++basis_revision_;
    correction_status_ = RealtimeOdometryReason::Ready;
    return MapOdometryCorrection{stamp_nsec, source_frame, map_to_odom};
  }

  // Invalidate the live projection when the mapping backend has latched the
  // existing raw-odometry divergence detector. This is deliberately separate
  // from an ordinary rejected geometric observation, which retains the last
  // trusted correction above.
  void invalidateForOdometryDivergence(int64_t stamp_nsec) {
    std::lock_guard<std::mutex> lock(mutex_);
    invalidate(stamp_nsec, RealtimeOdometryReason::OdometryDiverged);
  }

  std::optional<RealtimeOdometryPose> project(
      int64_t stamp_nsec, int64_t now_nsec, const std::string& source_frame,
      const std::string& child_frame, const Eigen::Isometry3d& raw_pose,
      bool pose_valid = true, RealtimeOdometryDiagnostic* diagnostic = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    int64_t projection_correction_stamp = correction_.stamp_nsec;
    const auto report = [&](RealtimeOdometryReason reason) {
      if (diagnostic) {
        *diagnostic = {reason, stamp_nsec, projection_correction_stamp, now_nsec, generation_};
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
      const auto existing = projections_.find(stamp_nsec);
      if (existing != projections_.end() &&
          !existing->second.matches(source_frame, child_frame, raw_pose)) {
        return reject(RealtimeOdometryReason::SourceObservationMismatch);
      }
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
    const auto existing = projections_.find(stamp_nsec);
    if (existing != projections_.end()) {
      if (!existing->second.matches(source_frame, child_frame, raw_pose)) {
        return reject(RealtimeOdometryReason::SourceObservationMismatch);
      }
      projection_correction_stamp = existing->second.output.correction_stamp_nsec;
      report(RealtimeOdometryReason::Ready);
      return existing->second.output;
    }
    if (stamp_nsec <= retired_through_nsec_) {
      report(RealtimeOdometryReason::ProjectionExpired);
      return std::nullopt;
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
    return retainProjection(stamp_nsec, source_frame, child_frame, raw_pose, map_pose);
  }

  // The first valid odom OR paired scan freezes this observation's map pose.
  // Delayed scans reuse that basis even after a newer correction, but never
  // recreate an evicted basis. Scan arrival does not advance odom ordering.
  std::optional<RealtimeOdometryPose> projectScan(
      int64_t stamp_nsec, int64_t now_nsec, const std::string& source_frame,
      const std::string& child_frame, const Eigen::Isometry3d& raw_pose) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (correction_status_ != RealtimeOdometryReason::Ready ||
        stamp_nsec <= 0 || stamp_nsec > now_nsec ||
        source_frame != correction_.source_frame ||
        child_frame != correction_.child_frame || !isFiniteRigidPose(raw_pose)) {
      return std::nullopt;
    }
    const auto existing = projections_.find(stamp_nsec);
    if (existing != projections_.end()) {
      if (!existing->second.matches(source_frame, child_frame, raw_pose)) return std::nullopt;
      return existing->second.output;
    }
    if (stamp_nsec <= retired_through_nsec_ || stamp_nsec < correction_.stamp_nsec) {
      return std::nullopt;
    }
    const Eigen::Isometry3d pose = correction_.map_to_odom * raw_pose;
    if (!isFiniteRigidPose(pose)) return std::nullopt;
    return retainProjection(stamp_nsec, source_frame, child_frame, raw_pose, pose);
  }

  // Heavy cloud work happens outside the lock; invalidation cancels its result.
  // Ordinary accepted corrections deliberately leave this generation intact.
  bool isCurrent(const RealtimeOdometryPose& projection) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return correction_status_ == RealtimeOdometryReason::Ready &&
           projection.generation == generation_;
  }

  uint64_t invalidateUnpublishableInput(int64_t stamp_nsec) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (correction_status_ == RealtimeOdometryReason::OdometryDiverged) return generation_;
    invalidate(stamp_nsec, RealtimeOdometryReason::InvalidInputPose);
    return generation_;
  }

 private:
  struct Projection {
    std::string source_frame;
    std::string child_frame;
    Eigen::Isometry3d raw_pose;
    RealtimeOdometryPose output;

    bool matches(const std::string& source, const std::string& child,
                 const Eigen::Isometry3d& raw) const {
      return source == source_frame && child == child_frame &&
             (raw.matrix().array() == raw_pose.matrix().array()).all();
    }
  };

  RealtimeOdometryPose retainProjection(
      int64_t stamp, const std::string& source, const std::string& child,
      const Eigen::Isometry3d& raw, const Eigen::Isometry3d& map_pose) {
    RealtimeOdometryPose output;
    output.stamp_nsec = stamp;
    output.correction_stamp_nsec = correction_.stamp_nsec;
    output.pose = map_pose;
    output.generation = generation_;
    output.basis_revision = basis_revision_;
    output.map_from_source = correction_.map_to_odom;
    output.source_from_body = raw;
    output.source_frame = source;
    output.body_frame = child;
    projections_.emplace(stamp, Projection{source, child, raw, output});
    trimProjections();
    return output;
  }

  void trimProjections() {
    while (projections_.size() > retention_) {
      retired_through_nsec_ = std::max(retired_through_nsec_, projections_.begin()->first);
      projections_.erase(projections_.begin());
    }
  }

  struct Correction {
    int64_t stamp_nsec = 0;
    std::string source_frame;
    std::string child_frame;
    Eigen::Isometry3d map_to_odom = Eigen::Isometry3d::Identity();
  };

  void invalidate(int64_t stamp_nsec, RealtimeOdometryReason reason) {
    correction_status_ = reason;
    invalidated_at_nsec_ = std::max(invalidated_at_nsec_, stamp_nsec);
    retired_through_nsec_ = std::max(retired_through_nsec_, invalidated_at_nsec_);
    if (!projections_.empty()) {
      retired_through_nsec_ = std::max(retired_through_nsec_, projections_.rbegin()->first);
    }
    projections_.clear();
    ++generation_;
  }

  mutable std::mutex mutex_;
  Correction correction_;
  RealtimeOdometryReason correction_status_ = RealtimeOdometryReason::AwaitingCorrection;
  int64_t last_input_stamp_nsec_ = 0;
  int64_t invalidated_at_nsec_ = 0;
  std::size_t retention_;
  std::map<int64_t, Projection> projections_;
  int64_t retired_through_nsec_ = 0;
  uint64_t generation_ = 0;
  uint64_t basis_revision_ = 0;
};

}  // namespace n3mapping::core
