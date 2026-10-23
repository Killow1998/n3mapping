#include "n3mapping/mapping_resuming.h"
#include "n3mapping/rotation_utils.h"
#include <glog/logging.h>

namespace n3mapping {

MappingResuming::MappingResuming(const Config& config,
                                 KeyframeManager& keyframe_manager,
                                 LoopDetector& loop_detector,
                                 PointCloudMatcher& matcher,
                                 GraphOptimizer& optimizer,
                                 MapSerializer& serializer,
                                 WorldLocalizing& world_localizing,
                                 SubmapBuilder* submap_builder)
    : config_(config)
    , keyframe_manager_(keyframe_manager)
    , loop_detector_(loop_detector)
    , matcher_(matcher)
    , optimizer_(optimizer)
    , loop_closure_manager_(config)
    , serializer_(serializer)
    , world_localizing_(world_localizing)
    , submap_builder_(submap_builder)
    , state_(MappingResumingState::NOT_INITIALIZED)
    , original_keyframe_count_(0)
    , original_max_keyframe_id_(-1)
    , cross_loop_count_(0)
    , relocalization_anchor_keyframe_id_(-1)
    , previous_new_keyframe_id_(-1)
    , last_trusted_constraint_keyframe_id_(-1)
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
        last_trusted_constraint_keyframe_id_ = -1;
        previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
        current_session_id_ = kInvalidMapSessionId;
        current_session_source_frame_id_.clear();
        T_map_session_initial_ = Eigen::Isometry3d::Identity();
        first_new_keyframe_pending_ = false;
        return false;
    }

    auto all_keyframes = keyframe_manager_.getAllKeyframes();
    original_max_keyframe_id_ = -1;
    for (const auto& kf : all_keyframes) {
        if (kf && kf->id > original_max_keyframe_id_)
            original_max_keyframe_id_ = kf->id;
    }
    for (auto& kf : all_keyframes) {
        if (kf) kf->is_from_loaded_map = true;
    }

    state_ = MappingResumingState::MAP_LOADED;
    relocalization_anchor_keyframe_id_ = -1;
    previous_new_keyframe_id_ = -1;
    last_trusted_constraint_keyframe_id_ = -1;
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

    auto anchor = keyframe_manager_.getKeyframe(result.matched_keyframe_id);
    if (!anchor || !optimizer_.hasNode(result.matched_keyframe_id) ||
        !anchor->pose_optimized.matrix().allFinite() ||
        !result.pose_in_map.matrix().allFinite()) {
        return false;
    }

    Eigen::Isometry3d T_map_odom = result.pose_in_map * odom_pose.inverse();
    if (!T_map_odom.matrix().allFinite()) return false;
    const MapSessionId next_session_id =
        keyframe_manager_.getNextSessionId();
    if (next_session_id == kInvalidMapSessionId) return false;
    world_localizing_.setMapToOdomTransform(T_map_odom);

    // Prepare the authoritative anchor before the first tracking callback.
    world_localizing_.warmLoadedMapTrackingTarget(result.matched_keyframe_id);

    // Ordinary localization also initializes a loaded session, but never
    // inserts observations. Prepare this index only when extension starts.
    for (const auto& keyframe : keyframe_manager_.getAllKeyframes())
        if (keyframe) acquisitions_by_stamp_[keyframe->timestamp].push_back(keyframe);

    const Eigen::Vector3d rpy_deg = (canonicalRollPitchYaw(T_map_odom) * (180.0 / M_PI));
    LOG(INFO) << "[MappingResuming] Relocalization anchor id="
              << result.matched_keyframe_id
              << " T_map_odom_t=" << T_map_odom.translation().transpose()
              << " T_map_odom_rpy_deg=" << rpy_deg.transpose();

    relocalization_anchor_keyframe_id_ = result.matched_keyframe_id;
    previous_new_keyframe_id_ = -1;
    last_trusted_constraint_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    current_session_id_ = next_session_id;
    current_session_source_frame_id_ = source_frame_id;
    T_map_session_initial_ = T_map_odom;
    first_new_keyframe_pending_ = true;
    reused_gate_keyframe_id_ = -1;

    state_ = MappingResumingState::RELOCALIZED;
    return true;
}

bool MappingResuming::saveExtendedMap(const std::string& map_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!refreshSubmapPosesNoLock("save_extended_map")) {
        return false;
    }
    return serializer_.saveMap(map_path, keyframe_manager_, loop_detector_,
                               optimizer_, submap_builder_);
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

size_t MappingResuming::getCrossLoopCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cross_loop_count_;
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
    cross_loop_count_ = 0;
    relocalization_anchor_keyframe_id_ = -1;
    previous_new_keyframe_id_ = -1;
    last_trusted_constraint_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    current_session_id_ = kInvalidMapSessionId;
    current_session_source_frame_id_.clear();
    T_map_session_initial_ = Eigen::Isometry3d::Identity();
    first_new_keyframe_pending_ = false;
    acquisitions_by_stamp_.clear();
    reused_gate_keyframe_id_ = -1;
    world_localizing_.reset();
}

}  // namespace n3mapping
