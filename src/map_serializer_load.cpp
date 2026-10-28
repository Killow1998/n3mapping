#include "n3mapping/map_serializer.h"
#include "n3mapping/n3map_proto_utils.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/submap_builder.h"
#include "planar_geometry/serialization.h"
#include <glog/logging.h>
#include "n3mapping/loop_detector.h"
#include <algorithm>
#include <chrono>
#include <sstream>
#include <unordered_set>

namespace n3mapping {
namespace {
struct SemVer {
    int major = 0;
    int minor = 0;
    int patch = 0;
};

bool parseSemVer(const std::string& version, SemVer* out) {
    if (!out) return false;
    std::stringstream ss(version);
    std::string major_str;
    std::string minor_str;
    std::string patch_str;
    if (!std::getline(ss, major_str, '.')) return false;
    if (!std::getline(ss, minor_str, '.')) return false;
    if (!std::getline(ss, patch_str, '.')) return false;
    try {
        out->major = std::stoi(major_str);
        out->minor = std::stoi(minor_str);
        out->patch = std::stoi(patch_str);
    } catch (...) {
        return false;
    }
    return true;
}

int compareSemVer(const SemVer& a, const SemVer& b) {
    if (a.major != b.major) return (a.major < b.major) ? -1 : 1;
    if (a.minor != b.minor) return (a.minor < b.minor) ? -1 : 1;
    if (a.patch != b.patch) return (a.patch < b.patch) ? -1 : 1;
    return 0;
}



Keyframe::Ptr keyframeFromParsedProto(const ParsedKeyframeProto& parsed) {
    auto keyframe = std::make_shared<Keyframe>();
    keyframe->id = parsed.id;
    keyframe->timestamp = parsed.timestamp;
    keyframe->session_id = parsed.session_id;
    keyframe->pose_odom = parsed.pose_odom;
    keyframe->pose_optimized = parsed.pose_optimized;
    keyframe->cloud = parsed.cloud;
    keyframe->is_from_loaded_map = true;
    return keyframe;
}

EdgeInfo edgeFromParsedProto(const ParsedEdgeProto& parsed) {
    EdgeInfo edge;
    edge.from_id = parsed.from_id;
    edge.to_id = parsed.to_id;
    edge.measurement = parsed.measurement;
    edge.information = parsed.information;
    edge.type = (parsed.type == PbstreamEdgeType::LOOP) ? EdgeType::LOOP : (parsed.type == PbstreamEdgeType::SESSION_ANCHOR) ? EdgeType::SESSION_ANCHOR : EdgeType::ODOMETRY;
    edge.constraint_mode = parsed.constraint_mode == PbstreamEdgeConstraintMode::XY_YAW
        ? EdgeConstraintMode::XY_YAW
        : EdgeConstraintMode::FULL_6DOF;
    return edge;
}
} // namespace

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer) {
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer, nullptr, nullptr, PbstreamLoadOptions{});
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            const PbstreamLoadOptions& options) {
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer, nullptr, nullptr, options);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory) {
    PbstreamLoadOptions options;
    options.allow_keyframe_fallback_dense = dense_optimized_trajectory != nullptr;
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, nullptr, options);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                            const PbstreamLoadOptions& options) {
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, nullptr, options);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                            core::DenseTrajectoryMetadata* dense_trajectory_metadata) {
    PbstreamLoadOptions options;
    options.allow_keyframe_fallback_dense = dense_optimized_trajectory != nullptr;
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, dense_trajectory_metadata, options);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                            core::DenseTrajectoryMetadata* dense_trajectory_metadata,
                            const PbstreamLoadOptions& options) {
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, dense_trajectory_metadata,
                   options, nullptr);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                            core::DenseTrajectoryMetadata* dense_trajectory_metadata,
                            SubmapBuilder* submap_builder) {
    PbstreamLoadOptions options;
    options.allow_keyframe_fallback_dense =
        dense_optimized_trajectory != nullptr;
    return loadMap(filepath, keyframe_manager, loop_detector, optimizer,
                   dense_optimized_trajectory, dense_trajectory_metadata,
                   options, submap_builder);
}

bool MapSerializer::loadMap(const std::string& filepath,
                            KeyframeManager& keyframe_manager,
                            LoopDetector& loop_detector,
                            GraphOptimizer& optimizer,
                            std::vector<core::DenseTrajectoryPose>* dense_optimized_trajectory,
                            core::DenseTrajectoryMetadata* dense_trajectory_metadata,
                            const PbstreamLoadOptions& options,
                            SubmapBuilder* submap_builder) {
    try {
        using Clock = std::chrono::steady_clock;
        const auto started = Clock::now();
        auto previous = started;
        const auto phase = [&](const char* name) {
            const auto now = Clock::now();
            LOG(INFO) << "[MapSerializer/Load] phase=" << name
                      << " wall_ms=" << std::chrono::duration<double, std::milli>(now - previous).count();
            previous = now;
        };
        n3mapping::N3Map map_proto;
        std::string read_error;
        if (!readN3MapProtoFromFile(filepath, &map_proto, &read_error)) {
            LOG(ERROR) << "[MapSerializer] Reject map: " << read_error;
            return false;
        }

        phase("read_protobuf");
        const std::string file_version = map_proto.metadata().version();
        SemVer file_semver, code_semver;
        if (parseSemVer(file_version, &file_semver) &&
            parseSemVer(kMapVersion, &code_semver) &&
            compareSemVer(file_semver, code_semver) > 0) {
            LOG(ERROR) << "[MapSerializer] Reject map: file version " << file_version
                       << " is newer than supported " << kMapVersion;
            return false;
        }

        std::string parse_error;
        std::vector<ParsedKeyframeProto> parsed_keyframes;
        std::unordered_set<int64_t> loaded_keyframe_ids;
        PbstreamKeyframeParseOptions keyframe_parse_options;
        keyframe_parse_options.policy = options.policy;
        if (!parseKeyframesFromProto(map_proto, keyframe_parse_options,
                                     &parsed_keyframes, &loaded_keyframe_ids, &parse_error)) {
            LOG(ERROR) << "[MapSerializer] Reject map: " << parse_error;
            return false;
        }

        std::vector<core::DenseTrajectoryPose> loaded_dense_trajectory;
        core::DenseTrajectoryMetadata loaded_dense_metadata;
        PbstreamLoadOptions dense_options = options;
        if (!parseDenseTrajectoryFromProto(map_proto, parsed_keyframes, dense_options,
                                           &loaded_dense_trajectory, &loaded_dense_metadata,
                                           &parse_error)) {
            LOG(ERROR) << "[MapSerializer] Reject map: " << parse_error;
            return false;
        }

        std::vector<Keyframe::Ptr> keyframes;
        keyframes.reserve(parsed_keyframes.size());
        for (const auto& parsed : parsed_keyframes) {
            keyframes.push_back(keyframeFromParsedProto(parsed));
        }

        std::vector<MapSessionInfo> sessions;
        if (map_proto.sessions_size() == 0) {
            MapSessionInfo legacy;
            legacy.source_frame_id = "legacy";
            legacy.loaded = true;
            legacy.start_timestamp = parsed_keyframes.front().timestamp;
            for (const auto& keyframe : parsed_keyframes) {
                legacy.start_timestamp =
                    std::min(legacy.start_timestamp, keyframe.timestamp);
            }
            sessions.push_back(std::move(legacy));
        } else {
            if (map_proto.sessions_size() >
                static_cast<int>(defaultPbstreamResourceLimits().max_keyframes)) {
                LOG(ERROR) << "[MapSerializer] Reject map: too many sessions.";
                return false;
            }
            std::unordered_set<MapSessionId> session_ids;
            session_ids.reserve(map_proto.sessions_size());
            sessions.reserve(map_proto.sessions_size());
            for (const auto& session_proto : map_proto.sessions()) {
                if (session_proto.id() == kInvalidMapSessionId ||
                    !session_ids.insert(session_proto.id()).second ||
                    !std::isfinite(session_proto.start_timestamp()) ||
                    !session_proto.has_t_map_session_initial() ||
                    !isFinitePoseProto(
                        session_proto.t_map_session_initial())) {
                    LOG(ERROR) << "[MapSerializer] Reject map: invalid or duplicate session id="
                               << session_proto.id();
                    return false;
                }
                MapSessionInfo session;
                session.id = session_proto.id();
                session.source_frame_id = session_proto.source_frame_id();
                session.start_timestamp = session_proto.start_timestamp();
                session.T_map_session_initial = poseFromProto(
                    session_proto.t_map_session_initial());
                session.loaded = session_proto.loaded();
                sessions.push_back(std::move(session));
            }
            for (const auto& keyframe : parsed_keyframes) {
                if (session_ids.find(keyframe.session_id) == session_ids.end()) {
                    LOG(ERROR) << "[MapSerializer] Reject map: keyframe id="
                               << keyframe.id << " references missing session id="
                               << keyframe.session_id;
                    return false;
                }
            }
        }

        if (map_proto.submaps_size() >
            static_cast<int>(defaultPbstreamResourceLimits().max_keyframes)) {
            LOG(ERROR) << "[MapSerializer] Reject map: too many submaps.";
            return false;
        }
        std::vector<Submap> submaps;
        submaps.reserve(map_proto.submaps_size());
        for (const auto& proto : map_proto.submaps()) {
            if (!proto.has_t_session_submap() ||
                !proto.has_t_map_submap() ||
                !isFinitePoseProto(proto.t_session_submap()) ||
                !isFinitePoseProto(proto.t_map_submap())) {
                LOG(ERROR) << "[MapSerializer] Reject map: invalid submap pose id="
                           << proto.id();
                return false;
            }
            Submap submap;
            submap.id = proto.id();
            submap.session_id = proto.session_id();
            submap.keyframe_ids.assign(proto.keyframe_ids().begin(),
                                       proto.keyframe_ids().end());
            submap.T_session_submap =
                poseFromProto(proto.t_session_submap());
            submap.T_map_submap = poseFromProto(proto.t_map_submap());
            submap.descriptor_keyframe_id =
                proto.descriptor_keyframe_id();
            submap.cloud_byte_budget = proto.cloud_byte_budget();
            submap.cloud_point_count = proto.cloud_point_count();
            submap.cloud_truncated = proto.cloud_truncated();
            submap.closed = proto.closed();
            submap.content_revision = proto.content_revision();
            submaps.push_back(std::move(submap));
        }

        std::vector<ParsedEdgeProto> parsed_edges;
        if (!parseEdgesFromProto(map_proto, loaded_keyframe_ids, options.policy, &parsed_edges, &parse_error)) {
            LOG(ERROR) << "[MapSerializer] Reject map: " << parse_error;
            return false;
        }

        std::vector<ParsedFloorAttitudeConstraint> parsed_floors;
        if (!parseFloorAttitudeFactorsFromProto(map_proto, loaded_keyframe_ids,
                                                options.policy, &parsed_floors,
                                                &parse_error)) {
            LOG(ERROR) << "[MapSerializer] Reject map: " << parse_error;
            return false;
        }

        phase("parse_validate");
        KeyframeManager temp_keyframe_manager(config_);
        LoopDetector temp_loop_detector(config_);
        GraphOptimizer temp_optimizer(config_);
        Config submap_config = config_;
        submap_config.submap_shadow_enable =
            submap_builder != nullptr && config_.submap_shadow_enable;
        SubmapBuilder temp_submap_builder(submap_config);
        if (!temp_keyframe_manager.loadKeyframes(keyframes, sessions)) {
            LOG(ERROR) << "[MapSerializer] Reject map: invalid session registry.";
            return false;
        }
        if (submaps.empty() && submap_config.submap_shadow_enable) {
            for (const auto& keyframe :
                 temp_keyframe_manager.getAllKeyframes()) {
                if (!temp_submap_builder.appendKeyframe(keyframe)) {
                    LOG(ERROR) << "[MapSerializer] Reject map: failed to rebuild shadow submaps.";
                    return false;
                }
            }
            // A loaded snapshot is immutable history. Any future extension
            // starts a new session/submap instead of appending to this tail.
            temp_submap_builder.closeActiveSubmap();
            LOG(INFO) << "[MapSerializer] Rebuilt shadow submaps for legacy map metadata.";
        } else if (!temp_submap_builder.loadSubmaps(
                       submaps, keyframes, sessions)) {
            LOG(ERROR) << "[MapSerializer] Reject map: invalid submap registry.";
            return false;
        }
        phase("restore_keyframes_submaps");
        if (options.rebuild_descriptors)
            temp_loop_detector.restoreMapDescriptors(temp_keyframe_manager);
        phase("rebuild_descriptors");
        std::vector<std::pair<int64_t, Eigen::Isometry3d>> nodes;
        for (const auto& kf : keyframes) {
            nodes.emplace_back(kf->id, kf->pose_optimized);
        }
        std::vector<EdgeInfo> edges;
        edges.reserve(parsed_edges.size());
        for (const auto& parsed : parsed_edges) {
            edges.push_back(edgeFromParsedProto(parsed));
        }
        std::vector<FloorAttitudeConstraint> floor_constraints;
        floor_constraints.reserve(parsed_floors.size());
        for (const auto& parsed : parsed_floors) {
            floor_constraints.push_back(FloorAttitudeConstraint{
                parsed.node_id, parsed.normal_body, parsed.sigma_rad});
        }
        std::vector<SharedPlane> planes;
        if (!readSharedPlanes(map_proto, planes)) return false;
        const bool restore = map_proto.has_graph_gauge_prior();
        if (!planes.empty() && !restore) return false;
        if (restore && !isFinitePoseProto(map_proto.graph_gauge_prior())) return false;
        const auto prior = restore ? poseFromProto(map_proto.graph_gauge_prior())
                                   : Eigen::Isometry3d::Identity();
        if (restore && !prior.matrix().allFinite()) return false;
        // Loading restores a checkpoint, including legacy maps. Re-solving here
        // would disagree with the saved clouds/Atlas before the first live frame.
        if (!temp_optimizer.loadGraph(nodes, edges, floor_constraints, planes,
                                     true, restore ? &prior : nullptr)) {
            LOG(ERROR) << "[MapSerializer] Reject map: graph optimizer failed to load graph.";
            return false;
        }

        phase("restore_graph");
        keyframe_manager.swapWith(temp_keyframe_manager);
        loop_detector.swapWith(temp_loop_detector);
        optimizer.swapWith(temp_optimizer);
        if (submap_builder) {
            submap_builder->swapWith(temp_submap_builder);
        }
        if (dense_trajectory_metadata) {
            *dense_trajectory_metadata = loaded_dense_metadata;
        }
        if (dense_optimized_trajectory) {
            *dense_optimized_trajectory = loaded_dense_trajectory;
        }
        return true;
    } catch (...) { return false; }
}

} // namespace n3mapping
