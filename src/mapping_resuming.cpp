// MappingResuming: map extension — load existing map, relocalize, add new keyframes, detect cross-loops.
#include "n3mapping/mapping_resuming.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>

#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_verifier.h"

namespace n3mapping {
namespace {

bool isFiniteTransform(const Eigen::Isometry3d& pose) {
    return pose.matrix().allFinite();
}

Eigen::Matrix<double, 6, 6> diagonalInformation(double position_sigma,
                                                 double rotation_sigma) {
    Eigen::Matrix<double, 6, 6> information =
        Eigen::Matrix<double, 6, 6>::Identity();
    information.block<3, 3>(0, 0) *=
        1.0 / (position_sigma * position_sigma);
    information.block<3, 3>(3, 3) *=
        1.0 / (rotation_sigma * rotation_sigma);
    return information;
}

bool validNoise(double position_sigma, double rotation_sigma) {
    return std::isfinite(position_sigma) && position_sigma > 0.0 &&
           std::isfinite(rotation_sigma) && rotation_sigma > 0.0;
}

Eigen::Vector3d rotationRpyDegrees(const Eigen::Isometry3d& pose) {
    return pose.rotation().eulerAngles(0, 1, 2) * (180.0 / M_PI);
}

}  // namespace

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
    , relocalization_anchor_keyframe_id_(-1)
    , previous_new_keyframe_id_(-1)
    , previous_new_odom_pose_(Eigen::Isometry3d::Identity())
    , first_new_keyframe_pending_(false) {}

bool MappingResuming::initializeFromLoadedMap() {
    std::lock_guard<std::mutex> lock(mutex_);
    return initializeFromLoadedMapNoLock();
}

bool MappingResuming::initializeFromLoadedMapNoLock() {
    original_keyframe_count_ = keyframe_manager_.size();
    if (original_keyframe_count_ == 0) {
        state_ = MappingResumingState::NOT_INITIALIZED;
        original_max_keyframe_id_ = -1;
        relocalization_anchor_keyframe_id_ = -1;
        previous_new_keyframe_id_ = -1;
        previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
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
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    first_new_keyframe_pending_ = false;
    return true;
}

bool MappingResuming::performInitialRelocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MappingResumingState::MAP_LOADED || !cloud || cloud->empty() ||
        !isFiniteTransform(odom_pose)) {
        return false;
    }

    RelocResult result = world_localizing_.relocalize(cloud, odom_pose);
    if (!result.success) return false;

    auto anchor = keyframe_manager_.getKeyframe(result.matched_keyframe_id);
    if (!anchor || !optimizer_.hasNode(result.matched_keyframe_id) ||
        !isFiniteTransform(anchor->pose_optimized) ||
        !isFiniteTransform(result.pose_in_map)) {
        return false;
    }

    Eigen::Isometry3d T_map_odom = result.pose_in_map * odom_pose.inverse();
    if (!isFiniteTransform(T_map_odom)) return false;
    world_localizing_.setMapToOdomTransform(T_map_odom);

    const Eigen::Vector3d rpy_deg = rotationRpyDegrees(T_map_odom);
    LOG(INFO) << "[MappingResuming] Relocalization anchor id="
              << result.matched_keyframe_id
              << " T_map_odom_t=" << T_map_odom.translation().transpose()
              << " T_map_odom_rpy_deg=" << rpy_deg.transpose();

    relocalization_anchor_keyframe_id_ = result.matched_keyframe_id;
    previous_new_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    first_new_keyframe_pending_ = true;

    state_ = MappingResumingState::RELOCALIZED;
    return true;
}

int64_t MappingResuming::processNewKeyframe(double timestamp, const Eigen::Isometry3d& odom_pose,
                                            const PointCloudT::Ptr& cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((state_ != MappingResumingState::RELOCALIZED &&
         state_ != MappingResumingState::EXTENDING) ||
        !std::isfinite(timestamp) || !isFiniteTransform(odom_pose) ||
        !cloud || cloud->empty()) {
        return -1;
    }

    Eigen::Isometry3d T_map_odom = world_localizing_.getMapToOdomTransform();
    Eigen::Isometry3d pose_in_map = T_map_odom * odom_pose;
    if (!isFiniteTransform(T_map_odom) || !isFiniteTransform(pose_in_map)) {
        return -1;
    }

    if (!keyframe_manager_.shouldAddKeyframe(pose_in_map)) return -1;

    EdgeInfo edge;
    edge.measurement = Eigen::Isometry3d::Identity();
    bool is_first_new_keyframe = first_new_keyframe_pending_;
    if (is_first_new_keyframe) {
        auto anchor = keyframe_manager_.getKeyframe(
            relocalization_anchor_keyframe_id_);
        if (!anchor || !optimizer_.hasNode(relocalization_anchor_keyframe_id_) ||
            !isFiniteTransform(anchor->pose_optimized) ||
            !validNoise(config_.loop_noise_position,
                        config_.loop_noise_rotation)) {
            return -1;
        }
        edge.from_id = relocalization_anchor_keyframe_id_;
        edge.measurement = anchor->pose_optimized.inverse() * pose_in_map;
        edge.information = diagonalInformation(
            config_.loop_noise_position, config_.loop_noise_rotation);
        edge.type = EdgeType::SESSION_ANCHOR;
    } else {
        auto previous = keyframe_manager_.getKeyframe(previous_new_keyframe_id_);
        if (!previous || !optimizer_.hasNode(previous_new_keyframe_id_) ||
            !isFiniteTransform(previous_new_odom_pose_) ||
            !validNoise(config_.odom_noise_position,
                        config_.odom_noise_rotation)) {
            return -1;
        }
        edge.from_id = previous_new_keyframe_id_;
        edge.measurement = previous_new_odom_pose_.inverse() * odom_pose;
        edge.information = diagonalInformation(
            config_.odom_noise_position, config_.odom_noise_rotation);
        edge.type = EdgeType::ODOMETRY;
    }
    if (!isFiniteTransform(edge.measurement)) return -1;

    const int64_t new_kf_id =
        keyframe_manager_.addKeyframe(timestamp, pose_in_map, cloud);
    edge.to_id = new_kf_id;

    const bool edge_staged = is_first_new_keyframe
        ? optimizer_.addSessionAnchorEdge(edge)
        : (optimizer_.addOdometryEdge(edge), true);
    if (!edge_staged || !optimizer_.incrementalOptimize()) {
        if (!keyframe_manager_.removeLatestKeyframe(new_kf_id)) {
            LOG(ERROR) << "[MappingResuming] Failed to roll back keyframe id="
                       << new_kf_id;
        }
        return -1;
    }

    keyframe_manager_.updateOptimizedPoses(optimizer_.getOptimizedPoses());

    if (is_first_new_keyframe) {
        const Eigen::Vector3d measurement_rpy_deg =
            rotationRpyDegrees(edge.measurement);
        LOG(INFO) << "[MappingResuming] Committed SESSION_ANCHOR from="
                  << edge.from_id << " to=" << edge.to_id
                  << " measurement_t=" << edge.measurement.translation().transpose()
                  << " measurement_rpy_deg=" << measurement_rpy_deg.transpose();
    }

    // From this point the graph and keyframe are committed. Descriptor
    // generation is best-effort and can be rebuilt from the stored cloud.
    previous_new_keyframe_id_ = new_kf_id;
    previous_new_odom_pose_ = odom_pose;
    first_new_keyframe_pending_ = false;
    state_ = MappingResumingState::EXTENDING;

    try {
        const Eigen::MatrixXd descriptor =
            loop_detector_.addDescriptor(new_kf_id, cloud);
        if (descriptor.size() == 0) {
            LOG(WARNING) << "[MappingResuming] ScanContext descriptor missing for accepted keyframe id="
                         << new_kf_id;
        }

        auto new_kf = keyframe_manager_.getKeyframe(new_kf_id);
        if (new_kf) {
            const int submap_radius =
                std::max(0, config_.rhpd_submap_kf_radius);
            PointCloudT::Ptr rhpd_cloud = cloud;
            if (submap_radius > 0) {
                rhpd_cloud = keyframe_manager_.buildCausalSubmapInRootFrame(
                    new_kf_id, submap_radius, new_kf_id);
            }
            if (rhpd_cloud && !rhpd_cloud->empty() &&
                config_.rhpd_submap_voxel_size > 1e-4) {
                PointCloudT::Ptr filtered;
                if (safeVoxelGridFilter<pcl::PointXYZI>(
                        rhpd_cloud, config_.rhpd_submap_voxel_size, &filtered) &&
                    filtered && !filtered->empty()) {
                    rhpd_cloud = filtered;
                }
            }
            new_kf->rhpd_descriptor =
                loop_detector_.addRHPD(new_kf_id, rhpd_cloud);
        }
    } catch (const std::exception& error) {
        LOG(WARNING) << "[MappingResuming] Descriptor generation failed for accepted keyframe id="
                     << new_kf_id << ": " << error.what();
    } catch (...) {
        LOG(WARNING) << "[MappingResuming] Descriptor generation failed for accepted keyframe id="
                     << new_kf_id;
    }

    return new_kf_id;
}

int MappingResuming::detectCrossLoops(int64_t new_keyframe_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MappingResumingState::EXTENDING ||
        isFromOriginalMap(new_keyframe_id) ||
        !optimizer_.hasNode(new_keyframe_id)) {
        return 0;
    }

    auto new_kf = keyframe_manager_.getKeyframe(new_keyframe_id);
    if (!new_kf || !new_kf->cloud) return 0;

    // The first new keyframe is already tied to the loaded map by the explicit
    // SESSION_ANCHOR. Re-registering that same frame immediately is redundant
    // and, before this guard, allowed a raw ICP Hessian to overpower the anchor.
    if (new_keyframe_id == original_max_keyframe_id_ + 1) {
        return 0;
    }

    std::map<int64_t, Keyframe::Ptr> keyframe_map;
    for (const auto& keyframe : keyframe_manager_.getAllKeyframes()) {
        if (keyframe) {
            keyframe_map[keyframe->id] = keyframe;
        }
    }
    auto candidates =
        loop_detector_.detectLoopCandidates(new_keyframe_id, keyframe_map);

    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());
    LoopVerifier verifier(config_);

    for (const auto& candidate : candidates) {
        if (!isFromOriginalMap(candidate.match_id)) continue;

        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf || !match_kf->cloud) continue;

        LoopVerification verification =
            verifier.verifyKeyframesLegacy(candidate, new_kf, match_kf, matcher_);
        if (verification.loop.verified && !verification.geometry_ok) {
            LOG(WARNING) << "[MappingResuming] Reject cross-session loop query="
                         << candidate.query_id << " match=" << candidate.match_id
                         << " reason=session_anchor_disagreement"
                         << " icp_translation="
                         << verification.icp_translation_norm
                         << " icp_rotation=" << verification.icp_rotation_norm;
            continue;
        }
        verified_loops.push_back(std::move(verification.loop));
    }

    auto valid_loops = loop_closure_manager_.filterValidLoops(verified_loops);
    auto best_loops = loop_closure_manager_.selectBestPerQuery(valid_loops);
    auto edges = loop_closure_manager_.buildLoopEdges(
        best_loops, LoopEdgeDirection::MatchToQuery);
    if (!edges.empty()) {
        if (!loop_closure_manager_.applyEdges(edges, optimizer_)) {
            return 0;
        }
        cross_loop_count_ += edges.size();
        LOG(INFO) << "[MappingResuming] Committed cross-session loop query="
                  << new_keyframe_id << " edge_count=" << edges.size();

        auto optimized_poses = optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);
    }

    return static_cast<int>(edges.size());
}

bool MappingResuming::saveExtendedMap(const std::string& map_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    return serializer_.saveMap(map_path, keyframe_manager_, loop_detector_, optimizer_);
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

void MappingResuming::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = MappingResumingState::NOT_INITIALIZED;
    original_keyframe_count_ = 0;
    original_max_keyframe_id_ = -1;
    cross_loop_count_ = 0;
    relocalization_anchor_keyframe_id_ = -1;
    previous_new_keyframe_id_ = -1;
    previous_new_odom_pose_ = Eigen::Isometry3d::Identity();
    first_new_keyframe_pending_ = false;
    world_localizing_.reset();
}

} // namespace n3mapping
