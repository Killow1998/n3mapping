// MappingResuming: map extension — load existing map, relocalize, add new keyframes, detect cross-loops.
#include "n3mapping/mapping_resuming.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>

#include <glog/logging.h>

#include "n3mapping/cloud_utils.h"
#include "n3mapping/loop_verification_pipeline.h"

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

Eigen::Matrix<double, 6, 6> trackingInformation(const Config& config) {
    auto information = diagonalInformation(
        config.loaded_map_tracking_noise_position,
        config.loaded_map_tracking_noise_rotation);
    const double z_sigma = config.loop_noise_position_z > 0.0
        ? config.loop_noise_position_z
        : config.loop_noise_position;
    information(2, 2) = 1.0 / (z_sigma * z_sigma);
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
    const std::string& source_frame_id) {
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
    const MapSessionId next_session_id =
        keyframe_manager_.getNextSessionId();
    if (next_session_id == kInvalidMapSessionId) return false;
    world_localizing_.setMapToOdomTransform(T_map_odom);

    const Eigen::Vector3d rpy_deg = rotationRpyDegrees(T_map_odom);
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

    state_ = MappingResumingState::RELOCALIZED;
    return true;
}

bool MappingResuming::shouldAddKeyframe(
    const Eigen::Isometry3d& pose_in_session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((state_ != MappingResumingState::RELOCALIZED &&
         state_ != MappingResumingState::EXTENDING) ||
        current_session_id_ == kInvalidMapSessionId ||
        !isFiniteTransform(pose_in_session)) {
        return false;
    }
    if (first_new_keyframe_pending_) return true;
    return keyframe_manager_.shouldAddKeyframeInSession(
        current_session_id_, pose_in_session);
}

int64_t MappingResuming::processNewKeyframe(
    double timestamp, const Eigen::Isometry3d& odom_pose,
    const PointCloudT::Ptr& cloud, int64_t loaded_tracking_match_id,
    const Eigen::Isometry3d& tracked_pose_in_map) {
    std::lock_guard<std::mutex> lock(mutex_);
    if ((state_ != MappingResumingState::RELOCALIZED &&
         state_ != MappingResumingState::EXTENDING) ||
        !std::isfinite(timestamp) || !isFiniteTransform(odom_pose) ||
        !cloud || cloud->empty()) {
        return -1;
    }

    const bool has_tracking_constraint = loaded_tracking_match_id >= 0;
    Eigen::Isometry3d T_map_odom = world_localizing_.getMapToOdomTransform();
    Eigen::Isometry3d pose_in_map = has_tracking_constraint
        ? tracked_pose_in_map
        : T_map_odom * odom_pose;
    if (!isFiniteTransform(T_map_odom) || !isFiniteTransform(pose_in_map)) {
        return -1;
    }

    Keyframe::Ptr tracking_match;
    if (has_tracking_constraint) {
        tracking_match = keyframe_manager_.getKeyframe(
            loaded_tracking_match_id);
        if (!tracking_match || !tracking_match->is_from_loaded_map ||
            !optimizer_.hasNode(loaded_tracking_match_id) ||
            !isFiniteTransform(tracking_match->pose_optimized) ||
            !validNoise(config_.loaded_map_tracking_noise_position,
                        config_.loaded_map_tracking_noise_rotation)) {
            return -1;
        }
    }

    if (current_session_id_ == kInvalidMapSessionId) return -1;
    const bool is_first_new_keyframe = first_new_keyframe_pending_;
    if (!is_first_new_keyframe &&
        !keyframe_manager_.hasSession(current_session_id_)) {
        return -1;
    }
    if (!is_first_new_keyframe &&
        !keyframe_manager_.shouldAddKeyframeInSession(
            current_session_id_, odom_pose)) {
        return -1;
    }

    EdgeInfo edge;
    edge.measurement = Eigen::Isometry3d::Identity();
    if (is_first_new_keyframe) {
        auto anchor = keyframe_manager_.getKeyframe(
            relocalization_anchor_keyframe_id_);
        if (!anchor || !optimizer_.hasNode(relocalization_anchor_keyframe_id_) ||
            !isFiniteTransform(anchor->pose_optimized) ||
            !validNoise(config_.loaded_map_tracking_noise_position,
                        config_.loaded_map_tracking_noise_rotation)) {
            return -1;
        }
        edge.from_id = relocalization_anchor_keyframe_id_;
        edge.measurement = anchor->pose_optimized.inverse() * pose_in_map;
        edge.information = trackingInformation(config_);
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

    bool registered_session_now = false;
    if (is_first_new_keyframe) {
        MapSessionInfo session;
        session.id = current_session_id_;
        session.source_frame_id = current_session_source_frame_id_;
        session.start_timestamp = timestamp;
        session.T_map_session_initial = T_map_session_initial_;
        if (!keyframe_manager_.addSession(session)) return -1;
        registered_session_now = true;
    }

    const int64_t new_kf_id = keyframe_manager_.addKeyframe(
        timestamp, odom_pose, pose_in_map, cloud, current_session_id_);
    if (new_kf_id < 0) {
        if (registered_session_now) {
            keyframe_manager_.removeSessionIfEmpty(current_session_id_);
        }
        return -1;
    }
    edge.to_id = new_kf_id;

    bool edge_staged = false;
    if (is_first_new_keyframe) {
        edge_staged = optimizer_.addSessionAnchorEdge(edge);
    } else {
        edge_staged = optimizer_.addSessionOdometryEdge(edge, pose_in_map);
        if (edge_staged && has_tracking_constraint) {
            EdgeInfo tracking_edge;
            tracking_edge.from_id = loaded_tracking_match_id;
            tracking_edge.to_id = new_kf_id;
            tracking_edge.measurement =
                tracking_match->pose_optimized.inverse() * pose_in_map;
            tracking_edge.information = trackingInformation(config_);
            tracking_edge.type = EdgeType::LOOP;
            tracking_edge.constraint_mode = EdgeConstraintMode::FULL_6DOF;
            optimizer_.addLoopEdge(tracking_edge);
        }
    }
    if (!edge_staged || !optimizer_.incrementalOptimize()) {
        if (!keyframe_manager_.removeLatestKeyframe(new_kf_id)) {
            LOG(ERROR) << "[MappingResuming] Failed to roll back keyframe id="
                       << new_kf_id;
        }
        if (registered_session_now &&
            !keyframe_manager_.removeSessionIfEmpty(current_session_id_)) {
            LOG(ERROR) << "[MappingResuming] Failed to roll back session id="
                       << current_session_id_;
        }
        return -1;
    }

    keyframe_manager_.updateOptimizedPoses(optimizer_.getOptimizedPoses());

    if (is_first_new_keyframe) {
        const Eigen::Vector3d measurement_rpy_deg =
            rotationRpyDegrees(edge.measurement);
        LOG(INFO) << "[MappingResuming] Committed SESSION_ANCHOR from="
                  << edge.from_id << " to=" << edge.to_id
                  << " session=" << current_session_id_
                  << " measurement_t=" << edge.measurement.translation().transpose()
                  << " measurement_rpy_deg=" << measurement_rpy_deg.transpose();
        last_trusted_constraint_keyframe_id_ = new_kf_id;
    } else if (has_tracking_constraint) {
        ++cross_loop_count_;
        last_trusted_constraint_keyframe_id_ = new_kf_id;
        LOG(INFO) << "[MappingResuming] Committed loaded-map tracking loop "
                  << "match=" << loaded_tracking_match_id
                  << " query=" << new_kf_id;
    }

    const auto optimized_poses = optimizer_.getOptimizedPoses();
    const auto optimized_current = optimized_poses.find(new_kf_id);
    if (optimized_current != optimized_poses.end()) {
        const Eigen::Isometry3d corrected_map_odom =
            optimized_current->second * odom_pose.inverse();
        if (isFiniteTransform(corrected_map_odom)) {
            world_localizing_.setMapToOdomTransform(corrected_map_odom);
        }
    }

    // From this point the graph and keyframe are committed. Descriptor
    // generation is best-effort and can be rebuilt from the stored cloud.
    previous_new_keyframe_id_ = new_kf_id;
    previous_new_odom_pose_ = odom_pose;
    first_new_keyframe_pending_ = false;
    state_ = MappingResumingState::EXTENDING;

    if (submap_builder_ && submap_builder_->enabled()) {
        const auto committed_keyframe =
            keyframe_manager_.getKeyframe(new_kf_id);
        if (!submap_builder_->appendKeyframe(committed_keyframe)) {
            LOG(WARNING) << "[MappingResuming] Shadow submap append failed for committed keyframe id="
                         << new_kf_id;
        }
    }

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
    const auto accept_loaded_map = [&](int64_t match_id) {
        const auto it = keyframe_map.find(match_id);
        return it != keyframe_map.end() && it->second &&
               it->second->is_from_loaded_map;
    };
    auto candidates = loop_detector_.detectLoopCandidates(
        new_keyframe_id, keyframe_map, accept_loaded_map);
    if (config_.loop_spatial_candidates_enable) {
        auto spatial_candidates = loop_detector_.detectSpatialCandidates(
            new_keyframe_id, keyframe_map, accept_loaded_map);
        for (const auto& spatial : spatial_candidates) {
            auto duplicate = std::find_if(
                candidates.begin(), candidates.end(),
                [&](const LoopCandidate& existing) {
                    return existing.query_id == spatial.query_id &&
                           existing.match_id == spatial.match_id;
                });
            if (duplicate == candidates.end()) {
                candidates.push_back(spatial);
            } else {
                duplicate->source_flags |= spatial.source_flags;
                duplicate->spatial_score =
                    std::max(duplicate->spatial_score, spatial.spatial_score);
            }
        }
    }

    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());
    LoopVerificationPipeline verification_pipeline(
        config_, keyframe_manager_, matcher_, optimizer_,
        loop_closure_manager_);
    LoopVerificationContext verification_context;
    verification_context.cross_session = true;
    verification_context.pose_visibility_evaluator =
        [this, query_cloud = new_kf->cloud](
            const Eigen::Isometry3d& T_map_query) {
            return world_localizing_.evaluateLoadedMapPoseVisibility(
                query_cloud, T_map_query);
        };

    for (const auto& candidate : candidates) {
        if (!isFromOriginalMap(candidate.match_id) ||
            !accept_loaded_map(candidate.match_id)) continue;

        auto verification = verification_pipeline.evaluate(
            candidate, verification_context);
        if (!verification.loop.verified) {
            VLOG(1) << "[MappingResuming] Reject cross-session candidate query="
                    << candidate.query_id << " match=" << candidate.match_id
                    << " stage=" << verification.reject_stage
                    << " reason=" << verification.reject_reason
                    << " fitness="
                    << verification.verification.match_result.fitness_score
                    << " inlier="
                    << verification.verification.match_result.inlier_ratio
                    << " descriptor_seeded="
                    << verification.descriptor_seeded
                    << " yaw_seed=" << verification.selected_seed_yaw_rad
                    << " hypotheses="
                    << verification.registration_hypothesis_count
                    << " visibility="
                    << verification.pose_visibility.consistency_ratio
                    << " visibility_logodds="
                    << verification.pose_visibility.evidence_log_odds
                    << " visibility_conflict="
                    << verification.pose_visibility.foreground_conflict_ratio;
            continue;
        }
        VLOG(1) << "[MappingResuming] Verified cross-session candidate query="
                << candidate.query_id << " match=" << candidate.match_id
                << " fitness="
                << verification.verification.match_result.fitness_score
                << " inlier="
                << verification.verification.match_result.inlier_ratio
                << " descriptor_seeded="
                << verification.descriptor_seeded
                << " yaw_seed=" << verification.selected_seed_yaw_rad
                << " hypotheses="
                << verification.registration_hypothesis_count
                << " visibility="
                << verification.pose_visibility.consistency_ratio
                << " visibility_logodds="
                << verification.pose_visibility.evidence_log_odds
                << " visibility_conflict="
                << verification.pose_visibility.foreground_conflict_ratio;
        verified_loops.push_back(std::move(verification.loop));
    }

    auto valid_loops = loop_closure_manager_.filterValidLoops(verified_loops);
    std::vector<LoopConstraintPipelineResult> accepted_constraints;
    accepted_constraints.reserve(valid_loops.size());
    const LoopConstraintContext constraint_context{true};
    // A good single-frame score is not enough to choose the cross-session
    // correspondence. Gate every bounded candidate with independent neighbor
    // evidence first; otherwise one attractive alias can be selected, fail
    // consensus, and suppress a valid candidate from the same query.
    for (const auto& loop : valid_loops) {
        auto constraint = verification_pipeline.evaluateConstraint(
            loop, LoopEdgeDirection::MatchToQuery, constraint_context);
        VLOG(1) << "[MappingResuming] Cross-session consensus query="
                << loop.query_id << " match=" << loop.match_id
                << " decision="
                << loopConsensusDecisionName(constraint.consensus.decision)
                << " valid=" << constraint.consensus.valid_pair_count
                << " left=" << constraint.consensus.left_support_count
                << " right=" << constraint.consensus.right_support_count
                << " contradictions="
                << constraint.consensus.contradiction_count
                << " median_t="
                << constraint.consensus.median_translation_delta
                << " median_r="
                << constraint.consensus.median_rotation_delta
                << " estimator="
                << constraint.consensus.estimator_recommendation
                << " estimator_pairs="
                << constraint.consensus.estimator_pair_count;
        if (!constraint.accepted || !constraint.has_edge) {
            VLOG(1) << "[MappingResuming] Defer cross-session constraint query="
                    << loop.query_id << " match=" << loop.match_id
                    << " stage=" << constraint.reject_stage
                    << " reason=" << constraint.reject_reason
                    << " consensus="
                    << loopConsensusDecisionName(constraint.consensus.decision);
            continue;
        }
        accepted_constraints.push_back(std::move(constraint));
    }

    std::vector<VerifiedLoop> accepted_loops;
    accepted_loops.reserve(accepted_constraints.size());
    for (const auto& constraint : accepted_constraints) {
        accepted_loops.push_back(constraint.loop);
    }
    const auto best_loops =
        loop_closure_manager_.selectBestPerQuery(accepted_loops);
    std::vector<EdgeInfo> edges;
    edges.reserve(best_loops.size());
    for (const auto& best : best_loops) {
        const auto selected = std::find_if(
            accepted_constraints.begin(), accepted_constraints.end(),
            [&](const LoopConstraintPipelineResult& candidate) {
                return candidate.loop.query_id == best.query_id &&
                       candidate.loop.match_id == best.match_id;
            });
        if (selected != accepted_constraints.end()) {
            edges.push_back(selected->edge);
        }
    }
    if (!edges.empty()) {
        // A correct session-merge loop can be tens of metres from the raw
        // odometry chain. With a redescending kernel it would enter as an
        // outlier and barely move the graph. Seed the new-session component
        // at the accepted measurement, preserving every raw odometry edge,
        // then rebuild transactionally so the robust loop starts in-basin.
        if (edges.size() != 1) {
            LOG(ERROR) << "[MappingResuming] Expected one selected cross-session "
                          "edge, got " << edges.size();
            return 0;
        }
        const auto poses_before_rebase = optimizer_.getOptimizedPoses();
        const auto from_pose = poses_before_rebase.find(edges.front().from_id);
        const auto to_pose = poses_before_rebase.find(edges.front().to_id);
        if (from_pose == poses_before_rebase.end() ||
            to_pose == poses_before_rebase.end()) {
            return 0;
        }
        const Eigen::Isometry3d rebase_correction =
            from_pose->second * edges.front().measurement *
            to_pose->second.inverse();
        const int64_t first_unconstrained_node =
            std::max(original_max_keyframe_id_ + 1,
                     last_trusted_constraint_keyframe_id_ + 1);
        if (!optimizer_.rebaseSessionAndAddLoopEdge(
                edges.front(), first_unconstrained_node)) {
            LOG(WARNING) << "[MappingResuming] Cross-session graph rebase failed "
                         << "query=" << new_keyframe_id;
            return 0;
        }
        LOG(INFO) << "[MappingResuming] Rebased new session before loop commit "
                  << "query=" << new_keyframe_id
                  << " correction_t="
                  << rebase_correction.translation().norm()
                  << " correction_r="
                  << Eigen::AngleAxisd(rebase_correction.rotation()).angle();
        cross_loop_count_ += edges.size();
        last_trusted_constraint_keyframe_id_ = new_keyframe_id;
        LOG(INFO) << "[MappingResuming] Committed cross-session loop query="
                  << new_keyframe_id << " edge_count=" << edges.size();

        auto optimized_poses = optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);

        // Map-to-odom is a live correction, not a once-per-session constant.
        // Once a trusted old-map constraint moves the current query in the
        // graph, carry that correction into every non-keyframe output and the
        // initial value of the next keyframe while retaining raw odometry on
        // the ODOMETRY edge itself.
        const auto optimized_it = optimized_poses.find(new_keyframe_id);
        if (optimized_it != optimized_poses.end() &&
            previous_new_keyframe_id_ == new_keyframe_id &&
            isFiniteTransform(previous_new_odom_pose_)) {
            const Eigen::Isometry3d corrected_map_odom =
                optimized_it->second * previous_new_odom_pose_.inverse();
            if (isFiniteTransform(corrected_map_odom)) {
                world_localizing_.setMapToOdomTransform(corrected_map_odom);
                const Eigen::Vector3d corrected_rpy_deg =
                    rotationRpyDegrees(corrected_map_odom);
                LOG(INFO) << "[MappingResuming] Updated T_map_odom from "
                             "cross-session graph query="
                          << new_keyframe_id << " t="
                          << corrected_map_odom.translation().transpose()
                          << " rpy_deg=" << corrected_rpy_deg.transpose();
            }
        }
    }

    return static_cast<int>(edges.size());
}

bool MappingResuming::saveExtendedMap(const std::string& map_path) {
    std::lock_guard<std::mutex> lock(mutex_);
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
    world_localizing_.reset();
}

} // namespace n3mapping
