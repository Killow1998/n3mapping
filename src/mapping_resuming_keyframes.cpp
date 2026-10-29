#include "n3mapping/mapping_resuming.h"
#include "n3mapping/world_localizing.h"
#include "n3mapping/submap_builder.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/rotation_utils.h"
#include <glog/logging.h>
#include <chrono>
#include <exception>
#include "n3mapping/cloud_utils.h"

namespace n3mapping {

namespace {
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

class ScopedElapsedMilliseconds {
public:
    explicit ScopedElapsedMilliseconds(double* output)
        : output_(output), started_(output ? Clock::now() : Clock::time_point{}) {}

    ~ScopedElapsedMilliseconds() {
        if (output_) {
            *output_ = std::chrono::duration<double, std::milli>(
                Clock::now() - started_).count();
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    double* output_;
    Clock::time_point started_;
};

}  // namespace

int64_t MappingResuming::processNewKeyframe(
    double timestamp, const Eigen::Isometry3d& odom_pose,
    const PointCloudT::Ptr& cloud, int64_t loaded_tracking_match_id,
    const Eigen::Isometry3d& tracked_pose_in_map,
    MappingResumingTiming* timing, int64_t* reused_acquisition_id) {
    if (reused_acquisition_id) *reused_acquisition_id = -1;
    ScopedElapsedMilliseconds total_timer(timing ? &timing->total_ms : nullptr);
    std::lock_guard<std::mutex> lock(mutex_);
    if ((state_ != MappingResumingState::RELOCALIZED &&
         state_ != MappingResumingState::EXTENDING) ||
        !std::isfinite(timestamp) || !odom_pose.matrix().allFinite() ||
        !cloud || cloud->empty()) {
        return -1;
    }

    const bool has_tracking_constraint = loaded_tracking_match_id >= 0;
    Eigen::Isometry3d T_map_odom = world_localizing_.getMapToOdomTransform();
    Eigen::Isometry3d pose_in_map = has_tracking_constraint
        ? tracked_pose_in_map
        : T_map_odom * odom_pose;
    if (!T_map_odom.matrix().allFinite() || !pose_in_map.matrix().allFinite()) {
        return -1;
    }

    Keyframe::Ptr tracking_match;
    if (has_tracking_constraint) {
        tracking_match = keyframe_manager_.getKeyframe(
            loaded_tracking_match_id);
        if (!tracking_match || !tracking_match->is_from_loaded_map ||
            !optimizer_.hasNode(loaded_tracking_match_id) ||
            !tracking_match->pose_optimized.matrix().allFinite() ||
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
    if (!shouldAddKeyframeNoLock(odom_pose)) return -1;
    const int64_t reused = reuseAcquisitionNoLock(timestamp, odom_pose, cloud);
    if (reused >= 0) {
        if (reused_acquisition_id) *reused_acquisition_id = reused;
        return -1;
    }

    EdgeInfo edge;
    edge.measurement = Eigen::Isometry3d::Identity();
    if (is_first_new_keyframe) {
        const int64_t anchor_id = has_tracking_constraint && reused_gate_keyframe_id_ >= 0
            ? loaded_tracking_match_id : relocalization_anchor_keyframe_id_;
        auto anchor = keyframe_manager_.getKeyframe(anchor_id);
        if (!anchor || !optimizer_.hasNode(anchor_id) ||
            !anchor->pose_optimized.matrix().allFinite() ||
            !validNoise(config_.loaded_map_tracking_noise_position,
                        config_.loaded_map_tracking_noise_rotation)) {
            return -1;
        }
        edge.from_id = anchor_id;
        edge.measurement = anchor->pose_optimized.inverse() * pose_in_map;
        edge.information = trackingInformation(config_);
        edge.type = EdgeType::SESSION_ANCHOR;
    } else {
        auto previous = keyframe_manager_.getKeyframe(previous_new_keyframe_id_);
        if (!previous || !optimizer_.hasNode(previous_new_keyframe_id_) ||
            !previous_new_odom_pose_.matrix().allFinite() ||
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
    if (!edge.measurement.matrix().allFinite())
        return -1;

    int64_t new_kf_id = -1;
    {
        ScopedElapsedMilliseconds graph_timer(timing ? &timing->graph_update_ms : nullptr);
        bool registered_session_now = false;
        if (is_first_new_keyframe) {
            MapSessionInfo session;
            session.id = current_session_id_;
            session.source_frame_id = current_session_source_frame_id_;
            session.start_timestamp = timestamp;
            session.T_map_session_initial = reused_gate_keyframe_id_ >= 0
                ? pose_in_map * odom_pose.inverse() : T_map_session_initial_;
            if (!keyframe_manager_.addSession(session))
                return -1;
            registered_session_now = true;
        }

        new_kf_id = keyframe_manager_.addKeyframe(timestamp, odom_pose, pose_in_map, cloud,
                                                  current_session_id_);
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
                tracking_edge.measurement = tracking_match->pose_optimized.inverse() * pose_in_map;
                tracking_edge.information = trackingInformation(config_);
                tracking_edge.type = EdgeType::LOOP;
                tracking_edge.constraint_mode = EdgeConstraintMode::FULL_6DOF;
                optimizer_.addLoopEdge(tracking_edge);
            }
        }
        if (!edge_staged || !optimizer_.incrementalOptimize()) {
            if (!keyframe_manager_.removeLatestKeyframe(new_kf_id)) {
                LOG(ERROR) << "[MappingResuming] Failed to roll back keyframe id=" << new_kf_id;
            }
            if (registered_session_now &&
                !keyframe_manager_.removeSessionIfEmpty(current_session_id_)) {
                LOG(ERROR) << "[MappingResuming] Failed to roll back session id="
                           << current_session_id_;
            }
            return -1;
        }

        keyframe_manager_.updateOptimizedPoses(optimizer_.getOptimizedPoses());
        refreshSubmapPosesNoLock("keyframe_commit");

        if (is_first_new_keyframe) {
            const Eigen::Vector3d measurement_rpy_deg = (canonicalRollPitchYaw(edge.measurement) * (180.0 / M_PI));
            LOG(INFO) << "[MappingResuming] Committed SESSION_ANCHOR from=" << edge.from_id
                      << " to=" << edge.to_id << " session=" << current_session_id_
                      << " measurement_t=" << edge.measurement.translation().transpose()
                      << " measurement_rpy_deg=" << measurement_rpy_deg.transpose();
            last_trusted_constraint_keyframe_id_ = new_kf_id;
        } else if (has_tracking_constraint) {
            ++cross_loop_count_;
            last_trusted_constraint_keyframe_id_ = new_kf_id;
            LOG(INFO) << "[MappingResuming] Committed loaded-map tracking loop "
                      << "match=" << loaded_tracking_match_id << " query=" << new_kf_id;
        }

        const auto optimized_poses = optimizer_.getOptimizedPoses();
        const auto optimized_current = optimized_poses.find(new_kf_id);
        if (optimized_current != optimized_poses.end()) {
            // The append solve conditions old graph poses. Its compromise
            // with odometry is not a new registration against the live Atlas.
            // Preserve the verified tracking correction until the worker
            // commits the optimized reference and its coordinate correction.
            const Eigen::Isometry3d corrected_map_odom =
                (has_tracking_constraint ? tracked_pose_in_map : optimized_current->second) *
                odom_pose.inverse();
            if (corrected_map_odom.matrix().allFinite()) {
                world_localizing_.setMapToOdomTransform(corrected_map_odom);
            }
        }

        // From this point the graph and keyframe are committed. Descriptor
        // generation is best-effort and can be rebuilt from the stored cloud.
        previous_new_keyframe_id_ = new_kf_id;
        previous_new_odom_pose_ = odom_pose;
        first_new_keyframe_pending_ = false;
        reused_gate_keyframe_id_ = -1;
        acquisitions_by_stamp_[timestamp].push_back(keyframe_manager_.getKeyframe(new_kf_id));
        state_ = MappingResumingState::EXTENDING;

        if (submap_builder_ && submap_builder_->enabled()) {
            const auto committed_keyframe = keyframe_manager_.getKeyframe(new_kf_id);
            if (!submap_builder_->appendKeyframe(committed_keyframe)) {
                LOG(WARNING)
                    << "[MappingResuming] Shadow submap append failed for committed keyframe id="
                    << new_kf_id;
            } else {
                // The keyframe graph is committed before shadow membership. A
                // post-append refresh keeps the structural diagnostics current.
                refreshSubmapPosesNoLock("submap_append");
            }
        }
    }

    ScopedElapsedMilliseconds descriptor_timer(
        timing ? &timing->descriptor_update_ms : nullptr);
    try {
        const auto descriptor =
            loop_detector_.addDescriptor(new_kf_id, cloud);
        if (!descriptor) {
            LOG(WARNING) << "[MappingResuming] RHPD heading missing for accepted keyframe id="
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

}  // namespace n3mapping
