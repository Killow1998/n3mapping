// Acquisition identity and keyframe sampling for extension. Replaying data
// still produces tracking outputs, but cannot add another independent factor.
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/keyframe_manager.h"
#include <glog/logging.h>
#include <cmath>
namespace n3mapping {
bool MappingResuming::shouldAddKeyframe(
    const Eigen::Isometry3d& pose_in_session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shouldAddKeyframeNoLock(pose_in_session);
}

bool MappingResuming::shouldAddKeyframeNoLock(
    const Eigen::Isometry3d& pose_in_session) const {
    if ((state_ != MappingResumingState::RELOCALIZED &&
         state_ != MappingResumingState::EXTENDING) ||
        current_session_id_ == kInvalidMapSessionId ||
        !pose_in_session.matrix().allFinite()) {
        return false;
    }
    if (reused_gate_keyframe_id_ >= 0) {
        const auto previous = keyframe_manager_.getKeyframe(reused_gate_keyframe_id_);
        if (!previous) return false;
        const double distance = (pose_in_session.translation() - previous->pose_odom.translation()).norm();
        const double angle = std::abs(Eigen::AngleAxisd(
            previous->pose_odom.rotation().transpose() * pose_in_session.rotation()).angle());
        return distance >= config_.keyframe_distance_threshold ||
               angle >= config_.keyframe_angle_threshold;
    }
    if (first_new_keyframe_pending_) return true;
    return keyframe_manager_.shouldAddKeyframeInSession(
        current_session_id_, pose_in_session);
}

int64_t MappingResuming::reuseAcquisitionNoLock(
    double timestamp, const Eigen::Isometry3d& odom_pose,
    const PointCloudT::Ptr& cloud) {
    const Keyframe acquisition(-1, timestamp, odom_pose, cloud);
    const auto previous_acquisitions = acquisitions_by_stamp_.find(timestamp);
    if (previous_acquisitions != acquisitions_by_stamp_.end()) {
        for (const auto& previous : previous_acquisitions->second) {
            if (!acquisition.sameAcquisition(*previous)) continue;
            // Advance sampling without adding another copy of its geometry,
            // odometry or tracking factor. A later fresh frame can still extend.
            reused_gate_keyframe_id_ = previous->id;
            LOG(INFO) << "[MappingResuming] Reused acquisition keyframe=" << previous->id;
            return previous->id;
        }
    }
    return -1;
}
} // namespace n3mapping
