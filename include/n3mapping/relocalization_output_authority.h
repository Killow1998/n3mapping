// ROS-free, stateful authority for relocalization output publication.
#pragma once

#include <cstdint>

#include "n3mapping/core/types.h"

namespace n3mapping {

enum class RelocalizationOutputMode : std::uint8_t {
  OTHER = 0,
  LOCALIZATION = 1,
  MAP_EXTENSION = 2,
};

struct RelocalizationPublicationPlan {
  bool publish_status = false;
  bool publish_global_pose = false;
  bool publish_world_cloud = false;
  bool publish_authoritative_pose = false;
  bool publish_legacy_lock = false;
  bool inconsistent_lock_event = false;
  bool invalid_usable_pose_suppressed = false;
  std::uint64_t lock_epoch = 0;
};

class RelocalizationOutputAuthority {
 public:
  RelocalizationPublicationPlan process(RelocalizationOutputMode mode,
                                        core::BackendOutput& output) {
    const bool localization_mode =
        mode == RelocalizationOutputMode::LOCALIZATION;
    const bool map_extension_mode =
        mode == RelocalizationOutputMode::MAP_EXTENSION;
    const bool backend_lock_event = output.relocalization_locked;
    const bool pose_is_finite_rigid =
        isFiniteRigidPose(output.T_world_lidar);
    const bool invalid_usable_pose =
        localization_mode &&
        hasUsableGlobalRelocalizationPose(output.relocalization_state,
                                          output.pose_source) &&
        !pose_is_finite_rigid;

    if (invalid_usable_pose) {
      output.success = false;
      output.relocalization_locked = false;
      output.relocalization_state = RelocalizationState::SEARCHING;
      output.pose_source = PoseSource::NONE;
      output.relocalization_decision = "invalid_nonrigid_pose";
    }

    const bool current_authoritative_full =
        localization_mode && pose_is_finite_rigid &&
        hasAuthoritativeRelocalizationInitializationPose(
            output.relocalization_state, output.pose_source);
    const bool usable_pose =
        !localization_mode ||
        (pose_is_finite_rigid &&
         hasUsableGlobalRelocalizationPose(output.relocalization_state,
                                           output.pose_source));
    const bool authoritative_edge =
        current_authoritative_full && !previous_authoritative_full_;
    const bool publish_legacy_lock =
        authoritative_edge ||
        (map_extension_mode && backend_lock_event && pose_is_finite_rigid);

    if (publish_legacy_lock) {
      ++lock_epoch_;
    }
    previous_authoritative_full_ = current_authoritative_full;

    return {
        localization_mode,
        usable_pose,
        usable_pose,
        authoritative_edge,
        publish_legacy_lock,
        localization_mode && backend_lock_event &&
            !current_authoritative_full,
        invalid_usable_pose,
        lock_epoch_,
    };
  }

  std::uint64_t lockEpoch() const { return lock_epoch_; }

 private:
  bool previous_authoritative_full_ = false;
  std::uint64_t lock_epoch_ = 0;
};

}  // namespace n3mapping
