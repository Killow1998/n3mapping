#include "n3mapping/map_serializer.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/submap_builder.h"
#include "planar_geometry/serialization.h"
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace n3mapping {
namespace {
bool atomicWriteProto(const std::string& filepath, const n3mapping::N3Map& map_proto) {
    const std::filesystem::path final_path(filepath);
    const std::filesystem::path tmp_path = final_path.string() + ".tmp";
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        if (!ofs.is_open()) return false;
        if (!map_proto.SerializeToOstream(&ofs)) {
            ofs.close();
            std::filesystem::remove(tmp_path);
            return false;
        }
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::filesystem::remove(tmp_path);
            return false;
        }
    }
    std::filesystem::rename(tmp_path, final_path);
    return true;
}

void poseToProto(const Eigen::Isometry3d& pose, n3mapping::Pose3D* proto) {
    proto->set_tx(pose.translation().x()); proto->set_ty(pose.translation().y()); proto->set_tz(pose.translation().z());
    Eigen::Quaterniond q(pose.rotation());
    proto->set_qx(q.x()); proto->set_qy(q.y()); proto->set_qz(q.z()); proto->set_qw(q.w());
}

void edgeToProto(const EdgeInfo& edge, n3mapping::EdgeProto* proto) {
    proto->set_from_id(edge.from_id); proto->set_to_id(edge.to_id);
    poseToProto(edge.measurement, proto->mutable_measurement());
    auto* information = proto->mutable_information();
    information->mutable_values()->Reserve(21);
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) information->add_values(edge.information(i, j));
    n3mapping::EdgeProto::EdgeType proto_type = n3mapping::EdgeProto::ODOMETRY;
    if (edge.type == EdgeType::LOOP) {
        proto_type = n3mapping::EdgeProto::LOOP;
    } else if (edge.type == EdgeType::SESSION_ANCHOR) {
        proto_type = n3mapping::EdgeProto::SESSION_ANCHOR;
    }
    proto->set_type(proto_type);
    proto->set_constraint_mode(edge.constraint_mode == EdgeConstraintMode::XY_YAW
        ? n3mapping::EdgeProto::XY_YAW
        : n3mapping::EdgeProto::FULL_6DOF);
}

void denseTrajectoryToProto(const core::DenseTrajectoryPose& pose,
                                           n3mapping::DenseTrajectoryPose* proto) {
    proto->set_seq(pose.seq);
    proto->set_timestamp(pose.timestamp);
    poseToProto(pose.pose_world_lidar, proto->mutable_pose_world_lidar());
    if (pose.correction_source) {
        const auto& source = *pose.correction_source;
        auto* saved = proto->mutable_correction_source();
        poseToProto(source.pose_world_lidar_raw, saved->mutable_pose_world_lidar_raw());
        poseToProto(source.anchor_pose_world_lidar_raw,
                    saved->mutable_anchor_pose_world_lidar_raw());
        saved->set_anchor_keyframe_id(source.anchor_keyframe_id);
        saved->set_has_anchor(source.has_anchor);
        saved->set_use_bracketing_correction(source.use_bracketing_correction);
    }
}

} // namespace

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer) {
    static const std::vector<core::DenseTrajectoryPose> kEmptyDenseTrajectory;
    const core::DenseTrajectoryMetadata metadata;
    return saveMap(filepath, keyframe_manager, loop_detector, optimizer,
                   kEmptyDenseTrajectory, metadata, nullptr);
}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer,
                            const SubmapBuilder* submap_builder) {
    static const std::vector<core::DenseTrajectoryPose> kEmptyDenseTrajectory;
    const core::DenseTrajectoryMetadata metadata;
    return saveMap(filepath, keyframe_manager, loop_detector, optimizer,
                   kEmptyDenseTrajectory, metadata, submap_builder);
}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer,
                            const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory) {
    core::DenseTrajectoryMetadata metadata;
    metadata.source = dense_optimized_trajectory.empty() ? "none" : "native";
    metadata.degraded = dense_optimized_trajectory.empty();
    return saveMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, metadata, nullptr);
}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector& loop_detector,
                            const GraphOptimizer& optimizer,
                            const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory,
                            const core::DenseTrajectoryMetadata& dense_trajectory_metadata) {
    return saveMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, dense_trajectory_metadata,
                   nullptr);
}

bool MapSerializer::saveMap(const std::string& filepath,
                            const KeyframeManager& keyframe_manager,
                            const LoopDetector&,
                            const GraphOptimizer& optimizer,
                            const std::vector<core::DenseTrajectoryPose>& dense_optimized_trajectory,
                            const core::DenseTrajectoryMetadata& dense_trajectory_metadata,
                            const SubmapBuilder* submap_builder) {
    try {
        std::filesystem::path fp(filepath);
        if (fp.has_parent_path()) std::filesystem::create_directories(fp.parent_path());
        n3mapping::N3Map map_proto;
        auto* meta = map_proto.mutable_metadata();
        meta->set_version(kMapVersion);
        meta->set_creation_timestamp(std::time(nullptr));
        meta->set_map_frame(config_.world_frame);
        meta->set_body_frame(config_.body_frame);
        auto keyframes = keyframe_manager.getAllKeyframes();
        if (keyframes.empty()) {
            LOG(ERROR) << "[MapSerializer] Refuse to save map with no keyframes.";
            return false;
        }
        std::unordered_set<int64_t> saved_keyframe_ids;
        saved_keyframe_ids.reserve(keyframes.size());
        std::unordered_map<int64_t, MapSessionId> keyframe_sessions;
        keyframe_sessions.reserve(keyframes.size());
        const auto sessions = keyframe_manager.getSessions();
        if (sessions.empty()) {
            LOG(ERROR) << "[MapSerializer] Refuse to save map with no sessions.";
            return false;
        }
        std::unordered_set<MapSessionId> saved_session_ids;
        saved_session_ids.reserve(sessions.size());
        for (const auto& session : sessions) {
            if (session.id == kInvalidMapSessionId ||
                !std::isfinite(session.start_timestamp) ||
                !isFinitePose(session.T_map_session_initial) ||
                !saved_session_ids.insert(session.id).second) {
                LOG(ERROR) << "[MapSerializer] Refuse to save invalid or duplicate session id="
                           << session.id;
                return false;
            }
            auto* session_proto = map_proto.add_sessions();
            session_proto->set_id(session.id);
            session_proto->set_source_frame_id(session.source_frame_id);
            session_proto->set_start_timestamp(session.start_timestamp);
            poseToProto(session.T_map_session_initial,
                        session_proto->mutable_t_map_session_initial());
            session_proto->set_loaded(session.loaded);
        }
        for (const auto& kf : keyframes) {
            if (!kf || !kf->cloud || kf->cloud->empty()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe with missing or empty cloud.";
                return false;
            }
            if (kf->id < 0) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe with negative id=" << kf->id;
                return false;
            }
            if (!std::isfinite(kf->timestamp) ||
                !isFinitePose(kf->pose_odom) ||
                !isFinitePose(kf->pose_optimized)) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id=" << kf->id
                           << " with non-finite timestamp or pose.";
                return false;
            }
            if (saved_session_ids.find(kf->session_id) ==
                saved_session_ids.end()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id="
                           << kf->id << " with missing session id="
                           << kf->session_id;
                return false;
            }
            if (kf->cloud->size() > kMaxPbstreamPointsPerKeyframe) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id=" << kf->id
                           << " with too many points=" << kf->cloud->size();
                return false;
            }
            const uint64_t finite_points = countFinitePoints(kf->cloud);
            if (finite_points == 0 || finite_points != kf->cloud->size()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save keyframe id=" << kf->id
                           << " with invalid finite point count=" << finite_points;
                return false;
            }
            saved_keyframe_ids.insert(kf->id);
            keyframe_sessions.emplace(kf->id, kf->session_id);
        }

        const std::vector<Submap> submaps = submap_builder
            ? submap_builder->getSubmaps()
            : std::vector<Submap>{};
        if (submaps.size() > keyframes.size()) {
            LOG(ERROR) << "[MapSerializer] Refuse to save more submaps than keyframes.";
            return false;
        }
        std::unordered_set<SubmapId> saved_submap_ids;
        std::unordered_set<int64_t> assigned_submap_keyframes;
        SubmapId previous_submap_id = 0;
        bool have_previous_submap = false;
        bool saw_open_submap = false;
        for (std::size_t index = 0; index < submaps.size(); ++index) {
            const auto& submap = submaps[index];
            if (submap.id == kInvalidSubmapId ||
                !saved_submap_ids.insert(submap.id).second ||
                (have_previous_submap && submap.id <= previous_submap_id) ||
                saved_session_ids.find(submap.session_id) ==
                    saved_session_ids.end() ||
                submap.keyframe_ids.empty() ||
                !isFinitePose(submap.T_session_submap) ||
                !isFinitePose(submap.T_map_submap) ||
                submap.cloud_byte_budget < sizeof(pcl::PointXYZI) ||
                submap.cloud_point_count >
                    submap.cloud_byte_budget / sizeof(pcl::PointXYZI) ||
                submap.content_revision == 0) {
                LOG(ERROR) << "[MapSerializer] Refuse to save invalid submap id="
                           << submap.id;
                return false;
            }
            if (!submap.closed) {
                if (saw_open_submap || index + 1 != submaps.size()) {
                    LOG(ERROR) << "[MapSerializer] Refuse to save non-final open submap id="
                               << submap.id;
                    return false;
                }
                saw_open_submap = true;
            }
            for (const int64_t keyframe_id : submap.keyframe_ids) {
                const auto keyframe_session = keyframe_sessions.find(keyframe_id);
                if (keyframe_session == keyframe_sessions.end() ||
                    keyframe_session->second != submap.session_id ||
                    !assigned_submap_keyframes.insert(keyframe_id).second) {
                    LOG(ERROR) << "[MapSerializer] Refuse to save submap id="
                               << submap.id << " with invalid keyframe reference="
                               << keyframe_id;
                    return false;
                }
            }
            if (submap.descriptor_keyframe_id >= 0 &&
                std::find(submap.keyframe_ids.begin(),
                          submap.keyframe_ids.end(),
                          submap.descriptor_keyframe_id) ==
                    submap.keyframe_ids.end()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save submap id="
                           << submap.id << " with invalid descriptor keyframe.";
                return false;
            }
            if (submap.registration_cloud) {
                if (submap.visualization_cloud != submap.registration_cloud ||
                    submap.cloud_point_count !=
                        submap.registration_cloud->size() ||
                    submap.materializedCloudBytes() >
                        submap.cloud_byte_budget) {
                    LOG(ERROR) << "[MapSerializer] Refuse to save inconsistent materialized submap id="
                               << submap.id;
                    return false;
                }
            } else if (submap.visualization_cloud) {
                LOG(ERROR) << "[MapSerializer] Refuse to save half-materialized submap id="
                           << submap.id;
                return false;
            }

            auto* proto = map_proto.add_submaps();
            proto->set_id(submap.id);
            proto->set_session_id(submap.session_id);
            for (const int64_t keyframe_id : submap.keyframe_ids) {
                proto->add_keyframe_ids(keyframe_id);
            }
            poseToProto(submap.T_session_submap,
                        proto->mutable_t_session_submap());
            poseToProto(submap.T_map_submap,
                        proto->mutable_t_map_submap());
            proto->set_descriptor_keyframe_id(
                submap.descriptor_keyframe_id);
            proto->set_cloud_byte_budget(submap.cloud_byte_budget);
            proto->set_cloud_point_count(submap.cloud_point_count);
            proto->set_cloud_truncated(submap.cloud_truncated);
            proto->set_closed(submap.closed);
            proto->set_content_revision(submap.content_revision);
            previous_submap_id = submap.id;
            have_previous_submap = true;
        }
        for (const auto& pose : dense_optimized_trajectory) {
            if (!std::isfinite(pose.timestamp) || !isFinitePose(pose.pose_world_lidar)) {
                LOG(ERROR) << "[MapSerializer] Refuse to save non-finite dense trajectory pose seq=" << pose.seq;
                return false;
            }
            if (pose.correction_source) {
                const auto& source = *pose.correction_source;
                if (!isFinitePose(source.pose_world_lidar_raw) ||
                    !isFinitePose(source.anchor_pose_world_lidar_raw) ||
                    (source.has_anchor && !saved_keyframe_ids.count(source.anchor_keyframe_id))) {
                    LOG(ERROR) << "[MapSerializer] Refuse invalid dense trajectory correction source.";
                    return false;
                }
            }
        }
        meta->set_num_keyframes(keyframes.size());
        std::string dense_source = dense_trajectory_metadata.source.empty()
            ? (dense_optimized_trajectory.empty() ? "none" : "native")
            : dense_trajectory_metadata.source;
        bool dense_degraded = dense_trajectory_metadata.degraded;
        if (dense_optimized_trajectory.empty()) {
            dense_source = "none";
            dense_degraded = true;
        } else if (dense_source == "none") {
            LOG(ERROR) << "[MapSerializer] Refuse to save non-empty dense trajectory with source=none.";
            return false;
        }
        meta->set_dense_trajectory_source(dense_source);
        meta->set_dense_trajectory_degraded(dense_degraded);
        // Persist source geometry only. Both descriptor parts are rebuilt on load.
        for (const auto& kf : keyframes) {
            auto* proto = map_proto.add_keyframes();
            proto->set_id(kf->id);
            proto->set_timestamp(kf->timestamp);
            proto->set_session_id(kf->session_id);
            poseToProto(kf->pose_odom, proto->mutable_pose_odom());
            poseToProto(kf->pose_optimized, proto->mutable_pose_optimized());
            auto* cloud = proto->mutable_cloud();
            cloud->set_num_points(kf->cloud->size());
            cloud->mutable_points()->Reserve(static_cast<int>(kf->cloud->size() * 4));
            for (const auto& point : kf->cloud->points) {
                cloud->add_points(point.x); cloud->add_points(point.y);
                cloud->add_points(point.z); cloud->add_points(point.intensity);
            }
        }
        auto edges = optimizer.getEdges();
        int n_odom = 0, n_loop = 0, n_anchor = 0;
        for (const auto& e : edges) {
            std::string information_error;
            if (!isFinitePose(e.measurement) ||
                !isValidInformationMatrix(e.information, &information_error)) {
                LOG(ERROR) << "[MapSerializer] Refuse to save malformed edge from="
                           << e.from_id << " to=" << e.to_id
                           << " error=" << information_error;
                return false;
            }
            if (saved_keyframe_ids.find(e.from_id) == saved_keyframe_ids.end() ||
                saved_keyframe_ids.find(e.to_id) == saved_keyframe_ids.end()) {
                LOG(ERROR) << "[MapSerializer] Refuse to save edge with missing endpoint from="
                           << e.from_id << " to=" << e.to_id;
                return false;
            }
            edgeToProto(e, map_proto.add_edges());
            if (e.type == EdgeType::ODOMETRY) {
                ++n_odom;
            } else if (e.type == EdgeType::LOOP) {
                ++n_loop;
            } else {
                ++n_anchor;
            }
        }
        for (const auto& pose : dense_optimized_trajectory) {
            denseTrajectoryToProto(pose, map_proto.add_dense_optimized_trajectory());
        }
        meta->set_num_odometry_edges(n_odom);
        meta->set_num_loop_edges(n_loop);
        meta->set_num_session_anchor_edges(n_anchor);
        const auto& floor_constraints = optimizer.floorAttitudeConstraints();
        meta->set_num_floor_attitude_factors(
            static_cast<uint32_t>(floor_constraints.size()));
        for (const auto& fc : floor_constraints) {
            auto* floor_proto = map_proto.add_floor_attitude_factors();
            floor_proto->set_node_id(fc.node_id);
            floor_proto->set_nx(fc.normal_body.x());
            floor_proto->set_ny(fc.normal_body.y());
            floor_proto->set_nz(fc.normal_body.z());
            floor_proto->set_sigma_rad(fc.sigma_rad);
        }
        writeSharedPlanes(optimizer.sharedPlanes(), map_proto);
        if (optimizer.getNumNodes() > 0)
            poseToProto(optimizer.gaugePrior(), map_proto.mutable_graph_gauge_prior());
        return atomicWriteProto(filepath, map_proto);
    } catch (...) { return false; }
}

} // namespace n3mapping
