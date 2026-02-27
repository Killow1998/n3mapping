#include "n3mapping/mapping_resuming.h"

#include <algorithm>

namespace n3mapping {

MappingResuming::MappingResuming(const Config& config,
                                 KeyframeManager& keyframe_manager,
                                 LoopDetector& loop_detector,
                                 PointCloudMatcher& matcher,
                                 GraphOptimizer& optimizer,
                                 MapSerializer& serializer,
                                 WorldLocalizing& world_localizing)
  : config_(config)
  , keyframe_manager_(keyframe_manager)
  , loop_detector_(loop_detector)
  , matcher_(matcher)
  , optimizer_(optimizer)
  , loop_closure_manager_(config)
  , serializer_(serializer)
  , world_localizing_(world_localizing)
  , state_(MappingResumingState::NOT_INITIALIZED)
  , original_keyframe_count_(0)
  , original_max_keyframe_id_(-1)
  , cross_loop_count_(0)
  , last_keyframe_pose_(Eigen::Isometry3d::Identity())
  , last_keyframe_id_(-1)
{
}

bool
MappingResuming::loadExistingMap(const std::string& map_path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!serializer_.loadMap(map_path, keyframe_manager_, loop_detector_, optimizer_)) {
        return false;
    }

    original_keyframe_count_ = keyframe_manager_.size();

    auto all_keyframes = keyframe_manager_.getAllKeyframes();
    original_max_keyframe_id_ = -1;
    for (const auto& kf : all_keyframes) {
        if (kf && kf->id > original_max_keyframe_id_) {
            original_max_keyframe_id_ = kf->id;
        }
    }

    for (auto& kf : all_keyframes) {
        if (kf) {
            kf->is_from_loaded_map = true;
        }
    }

    state_ = MappingResumingState::MAP_LOADED;
    return true;
}

bool
MappingResuming::performInitialRelocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ != MappingResumingState::MAP_LOADED) {
        return false;
    }

    RelocResult result = world_localizing_.relocalize(cloud, odom_pose);

    if (!result.success) {
        return false;
    }

    Eigen::Isometry3d T_map_odom = result.pose_in_map * odom_pose.inverse();
    world_localizing_.setMapToOdomTransform(T_map_odom);

    last_keyframe_pose_ = result.pose_in_map;
    last_keyframe_id_ = result.matched_keyframe_id;

    state_ = MappingResumingState::RELOCALIZED;
    return true;
}

int64_t
MappingResuming::processNewKeyframe(double timestamp, const Eigen::Isometry3d& odom_pose, const PointCloudT::Ptr& cloud)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ != MappingResumingState::RELOCALIZED && state_ != MappingResumingState::EXTENDING) {
        return -1;
    }

    Eigen::Isometry3d T_map_odom = world_localizing_.getMapToOdomTransform();
    Eigen::Isometry3d pose_in_map = T_map_odom * odom_pose;

    if (!keyframe_manager_.shouldAddKeyframe(pose_in_map)) {
        return -1;
    }

    int64_t new_kf_id = keyframe_manager_.addKeyframe(timestamp, pose_in_map, cloud);

    loop_detector_.addDescriptor(new_kf_id, cloud);

    if (last_keyframe_id_ >= 0) {
        auto last_kf = keyframe_manager_.getKeyframe(last_keyframe_id_);
        if (last_kf) {
            EdgeInfo edge;
            edge.from_id = last_keyframe_id_;
            edge.to_id = new_kf_id;
            edge.measurement = last_kf->pose_optimized.inverse() * pose_in_map;
            edge.information = Eigen::Matrix<double, 6, 6>::Identity();
            edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
            edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
            edge.type = EdgeType::ODOMETRY;

            optimizer_.addOdometryEdge(edge);
        }
    } else {
        int64_t matched_id = world_localizing_.getLastMatchedKeyframeId();
        if (matched_id >= 0) {
            auto matched_kf = keyframe_manager_.getKeyframe(matched_id);
            if (matched_kf) {
                addRelocalizationConstraint(new_kf_id, matched_id, matched_kf->pose_optimized.inverse() * pose_in_map);
            }
        }
    }

    last_keyframe_pose_ = pose_in_map;
    last_keyframe_id_ = new_kf_id;
    state_ = MappingResumingState::EXTENDING;

    return new_kf_id;
}

int
MappingResuming::detectCrossLoops(int64_t new_keyframe_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ != MappingResumingState::EXTENDING) {
        return 0;
    }

    auto new_kf = keyframe_manager_.getKeyframe(new_keyframe_id);
    if (!new_kf || !new_kf->cloud) {
        return 0;
    }

    std::vector<LoopCandidate> candidates = loop_detector_.detectLoopCandidates(new_keyframe_id);

    int cross_loops_found = 0;
    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        if (!isFromOriginalMap(candidate.match_id)) {
            continue;
        }

        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf || !match_kf->cloud) {
            continue;
        }

        VerifiedLoop verified = loop_detector_.verifyLoopCandidate(candidate, new_kf, match_kf, matcher_);
        verified_loops.push_back(verified);
    }

    auto valid_loops = loop_closure_manager_.filterValidLoops(verified_loops);
    auto edges = loop_closure_manager_.buildLoopEdges(valid_loops, LoopEdgeDirection::MatchToQuery);
    if (!edges.empty()) {
        loop_closure_manager_.applyEdges(edges, optimizer_);
        cross_loops_found = static_cast<int>(edges.size());
        cross_loop_count_ += edges.size();

        auto optimized_poses = optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);
    }

    return cross_loops_found;
}

bool
MappingResuming::saveExtendedMap(const std::string& map_path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    return serializer_.saveMap(map_path, keyframe_manager_, loop_detector_, optimizer_);
}

MappingResumingState
MappingResuming::getState() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

size_t
MappingResuming::getOriginalKeyframeCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return original_keyframe_count_;
}

size_t
MappingResuming::getNewKeyframeCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframe_manager_.size() - original_keyframe_count_;
}

size_t
MappingResuming::getCrossLoopCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return cross_loop_count_;
}

bool
MappingResuming::isFromOriginalMap(int64_t keyframe_id) const
{
    return keyframe_id <= original_max_keyframe_id_;
}

void
MappingResuming::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);

    state_ = MappingResumingState::NOT_INITIALIZED;
    original_keyframe_count_ = 0;
    original_max_keyframe_id_ = -1;
    cross_loop_count_ = 0;
    last_keyframe_pose_ = Eigen::Isometry3d::Identity();
    last_keyframe_id_ = -1;

    world_localizing_.reset();
}

void
MappingResuming::addRelocalizationConstraint(int64_t new_keyframe_id, int64_t matched_keyframe_id, const Eigen::Isometry3d& T_match_new)
{
    EdgeInfo edge;
    edge.from_id = matched_keyframe_id;
    edge.to_id = new_keyframe_id;
    edge.measurement = T_match_new;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.loop_noise_position * config_.loop_noise_position);
    edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.loop_noise_rotation * config_.loop_noise_rotation);
    edge.type = EdgeType::LOOP;

    optimizer_.addLoopEdge(edge);
}

} // namespace n3mapping
