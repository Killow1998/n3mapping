#include "n3mapping/mapping_resuming.h"
#include "n3mapping/world_localizing.h"
#include "n3mapping/submap_builder.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/rotation_utils.h"
#include <glog/logging.h>

namespace n3mapping {

MappingResuming::MappingResuming(const Config& config,
                                 KeyframeManager& keyframe_manager,
                                 GraphOptimizer& optimizer,
                                 WorldLocalizing& world_localizing,
                                 SubmapBuilder* submap_builder)
    : config_(config)
    , keyframe_manager_(keyframe_manager)
    , optimizer_(optimizer)
    , world_localizing_(world_localizing)
    , submap_builder_(submap_builder)
    , state_(MappingResumingState::NOT_INITIALIZED)
    , original_keyframe_count_(0)
    , original_max_keyframe_id_(-1)
    , relocalization_anchor_keyframe_id_(-1)
    , previous_new_keyframe_id_(-1)
    , previous_new_odom_pose_(Eigen::Isometry3d::Identity())
    , current_session_id_(kInvalidMapSessionId)
    , T_map_session_initial_(Eigen::Isometry3d::Identity())
    , first_new_keyframe_pending_(false) {}

bool MappingResuming::initializeFromLoadedMap() {
    std::lock_guard<std::mutex> lock(mutex_);
    return initializeFromLoadedMapNoLock();
}

bool MappingResuming::initializeFromLoadedMapNoLock() {
    acquisitions_by_stamp_.clear();
    reused_gate_keyframe_id_ = -1;
    original_keyframe_count_ = keyframe_manager_.size();
    if (original_keyframe_count_ == 0) {
        state_ = MappingResumingState::NOT_INITIALIZED;
        original_max_keyframe_id_ = -1;
        relocalization_anchor_keyframe_id_ = -1;
        previous_new_keyframe_id_ = -1;
        previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
        current_session_id_ = kInvalidMapSessionId;
        current_session_source_frame_id_.clear();
        T_map_session_initial_ = Eigen::Isometry3d::Identity();
        first_new_keyframe_pending_ = false;
        return false;
    }

    const auto all_keyframes = keyframe_manager_.getAllKeyframes();
    std::set<int64_t> loaded_ids;
    for (const auto& kf : all_keyframes) {
        kf->is_from_loaded_map = true;
        loaded_ids.insert(kf->id);
    }
    original_max_keyframe_id_ = *loaded_ids.rbegin();
    optimizer_.setFixedNodes(loaded_ids);

    state_ = MappingResumingState::MAP_LOADED;
    relocalization_anchor_keyframe_id_ = -1;
    previous_new_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    current_session_id_ = kInvalidMapSessionId;
    current_session_source_frame_id_.clear();
    T_map_session_initial_ = Eigen::Isometry3d::Identity();
    first_new_keyframe_pending_ = false;
    return true;
}

bool MappingResuming::performInitialRelocalization(
    const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose,
    const std::string& source_frame_id, RelocalizationPerformance* performance) {
    if (performance) *performance = {};
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MappingResumingState::MAP_LOADED || !cloud || cloud->empty() ||
        !odom_pose.matrix().allFinite()) {
        return false;
    }

    RelocResult result = world_localizing_.relocalize(cloud, odom_pose);
    if (performance) *performance = result.performance;
    if (!result.success) return false;

    const Eigen::Isometry3d T_map_odom = result.pose_in_map * odom_pose.inverse();
    if (!beginSessionNoLock(T_map_odom, result.matched_keyframe_id, source_frame_id))
        return false;
    world_localizing_.setMapToOdomTransform(T_map_odom);
    return true;
}

bool MappingResuming::adoptCurrentLocalization(const std::string& source_frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MappingResumingState::MAP_LOADED || !world_localizing_.isRelocalized())
        return false;
    return beginSessionNoLock(world_localizing_.getMapToOdomTransform(),
                              world_localizing_.getLastMatchedKeyframeId(), source_frame_id);
}

bool MappingResuming::beginSessionNoLock(const Eigen::Isometry3d& T_map_odom,
                                        int64_t matched_keyframe_id,
                                        const std::string& source_frame_id) {
    auto anchor = keyframe_manager_.getKeyframe(matched_keyframe_id);
    if (!anchor || !optimizer_.hasNode(matched_keyframe_id) ||
        !anchor->pose_optimized.matrix().allFinite()) {
        return false;
    }

    if (!T_map_odom.matrix().allFinite()) return false;
    const MapSessionId next_session_id =
        keyframe_manager_.getNextSessionId();
    if (next_session_id == kInvalidMapSessionId) return false;

    world_localizing_.warmLoadedMapTrackingTarget(matched_keyframe_id);

    for (const auto& keyframe : keyframe_manager_.getAllKeyframes())
        if (keyframe) acquisitions_by_stamp_[keyframe->timestamp].push_back(keyframe);

    const Eigen::Vector3d rpy_deg = (canonicalRollPitchYaw(T_map_odom) * (180.0 / M_PI));
    LOG(INFO) << "[MappingResuming] Relocalization anchor id="
              << matched_keyframe_id
              << " T_map_odom_t=" << T_map_odom.translation().transpose()
              << " T_map_odom_rpy_deg=" << rpy_deg.transpose();

    relocalization_anchor_keyframe_id_ = matched_keyframe_id;
    previous_new_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    current_session_id_ = next_session_id;
    current_session_source_frame_id_ = source_frame_id;
    T_map_session_initial_ = T_map_odom;
    first_new_keyframe_pending_ = true;
    reused_gate_keyframe_id_ = -1;

    state_ = MappingResumingState::RELOCALIZED;
    return true;
}

MappingResumingState MappingResuming::getState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

size_t MappingResuming::getOriginalKeyframeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return original_keyframe_count_;
}

size_t MappingResuming::getNewKeyframeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ == MappingResumingState::NOT_INITIALIZED) return 0u;
    const size_t total = keyframe_manager_.size();
    return total >= original_keyframe_count_
        ? total - original_keyframe_count_
        : 0u;
}

bool MappingResuming::isFromOriginalMap(int64_t keyframe_id) const {
    return keyframe_id >= 0 && keyframe_id <= original_max_keyframe_id_;
}

MapSessionId MappingResuming::getCurrentSessionId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_session_id_;
}

void MappingResuming::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (submap_builder_) {
        submap_builder_->closeActiveSubmap();
    }
    state_ = MappingResumingState::NOT_INITIALIZED;
    original_keyframe_count_ = 0;
    original_max_keyframe_id_ = -1;
    relocalization_anchor_keyframe_id_ = -1;
    previous_new_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    current_session_id_ = kInvalidMapSessionId;
    current_session_source_frame_id_.clear();
    T_map_session_initial_ = Eigen::Isometry3d::Identity();
    first_new_keyframe_pending_ = false;
    acquisitions_by_stamp_.clear();
    reused_gate_keyframe_id_ = -1;
    world_localizing_.reset();
}

}
