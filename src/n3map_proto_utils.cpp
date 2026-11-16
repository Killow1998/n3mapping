#include <glog/logging.h>
#include "n3mapping/n3map_proto_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_set>

#include <Eigen/Eigenvalues>

#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

bool setError(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
}

bool hasCompleteEdgeCountContract(const std::string& version) {
    std::stringstream stream(version);
    int major = 0;
    int minor = 0;
    char first_dot = '\0';
    if (!(stream >> major >> first_dot >> minor) || first_dot != '.') {
        return false;
    }
    return major > 2 || (major == 2 && minor >= 5);
}

bool validateKeyframeProto(const KeyframeProto& proto, std::string* error) {
    if (proto.id() < 0) return setError(error, "invalid keyframe id");
    if (!std::isfinite(proto.timestamp())) return setError(error, "non-finite keyframe timestamp");
    if (!isFinitePoseProto(proto.pose_odom()) || !isFinitePoseProto(proto.pose_optimized())) {
        return setError(error, "non-finite keyframe pose");
    }
    if (!proto.has_cloud()) return setError(error, "missing keyframe cloud");
    return true;
}

}

bool isFinitePoseProto(const Pose3D& proto) {
    const double values[] = {
        proto.tx(), proto.ty(), proto.tz(),
        proto.qx(), proto.qy(), proto.qz(), proto.qw()
    };
    for (double value : values) {
        if (!std::isfinite(value)) return false;
    }
    const double q_norm2 = proto.qx() * proto.qx() + proto.qy() * proto.qy() +
                           proto.qz() * proto.qz() + proto.qw() * proto.qw();
    return std::isfinite(q_norm2) && q_norm2 > 1e-12;
}

bool isFinitePose(const Eigen::Isometry3d& pose) {
    return pose.matrix().array().isFinite().all();
}

bool isFinitePoint(const pcl::PointXYZI& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z) && std::isfinite(point.intensity);
}

uint64_t countFinitePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    if (!cloud) return 0;
    uint64_t count = 0;
    for (const auto& point : cloud->points) {
        if (isFinitePoint(point)) ++count;
    }
    return count;
}

Eigen::Isometry3d poseFromProto(const Pose3D& proto) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(proto.tx(), proto.ty(), proto.tz());
    Eigen::Quaterniond q(proto.qw(), proto.qx(), proto.qy(), proto.qz());
    q.normalize();
    pose.linear() = q.toRotationMatrix();
    return pose;
}

PbstreamMetadata extractPbstreamMetadata(const MapMetadata& proto) {
    PbstreamMetadata metadata;
    metadata.version = proto.version();
    if (!proto.map_frame().empty()) metadata.map_frame = proto.map_frame();
    if (!proto.body_frame().empty()) metadata.body_frame = proto.body_frame();
    metadata.dense_trajectory_source = proto.dense_trajectory_source();
    metadata.dense_trajectory_degraded = proto.dense_trajectory_degraded();
    metadata.nav_cloud_filter_applied = proto.nav_cloud_filter_applied();
    metadata.nav_cloud_filter_policy = proto.nav_cloud_filter_policy();
    metadata.descriptors_recomputed_from_filtered_cloud =
        proto.descriptors_recomputed_from_filtered_cloud();
    metadata.nav_filter_raw_points = proto.nav_filter_raw_points();
    metadata.nav_filter_kept_points = proto.nav_filter_kept_points();
    metadata.nav_filter_removed_points = proto.nav_filter_removed_points();
    return metadata;
}

bool pointCloudFromProto(const PointCloudData& proto,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr* cloud,
                         std::string* error) {
    if (!cloud) return setError(error, "null point cloud output");
    const uint64_t required_values = static_cast<uint64_t>(proto.num_points()) * 4u;
    if (proto.num_points() > kMaxPbstreamPointsPerKeyframe ||
        required_values > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        proto.points_size() != static_cast<int>(required_values)) {
        return setError(error, "malformed point cloud repeated-field length");
    }

    auto parsed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    parsed->resize(proto.num_points());
    for (uint32_t i = 0; i < proto.num_points(); ++i) {
        auto& point = parsed->points[i];
        point.x = proto.points(i * 4);
        point.y = proto.points(i * 4 + 1);
        point.z = proto.points(i * 4 + 2);
        point.intensity = proto.points(i * 4 + 3);
        if (!isFinitePoint(point)) {
            return setError(error, "non-finite point in keyframe cloud");
        }
    }
    parsed->width = static_cast<uint32_t>(parsed->size());
    parsed->height = 1;
    parsed->is_dense = false;
    *cloud = parsed;
    return true;
}

bool informationFromProto(const InformationMatrix& proto,
                          Eigen::Matrix<double, 6, 6>* information,
                          std::string* error) {
    if (!information) return setError(error, "null information matrix output");
    if (proto.values_size() != 21) return setError(error, "malformed information matrix");

    Eigen::Matrix<double, 6, 6> parsed = Eigen::Matrix<double, 6, 6>::Identity();
    int index = 0;
    for (int row = 0; row < 6; ++row) {
        for (int col = row; col < 6; ++col) {
            const double value = proto.values(index++);
            if (!std::isfinite(value)) return setError(error, "non-finite information matrix value");
            parsed(row, col) = value;
            if (row != col) parsed(col, row) = value;
        }
    }
    if (!isValidInformationMatrix(parsed, error)) return false;
    *information = parsed;
    return true;
}

bool isValidInformationMatrix(const Eigen::Matrix<double, 6, 6>& information,
                              std::string* error) {
    if (!information.array().isFinite().all()) {
        return setError(error, "non-finite information matrix value");
    }
    if (!information.isApprox(information.transpose(), 1e-9)) {
        return setError(error, "non-symmetric information matrix");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> solver(information);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().array().isFinite().all()) {
        return setError(error, "invalid information matrix eigenvalues");
    }
    const double min_eigenvalue = solver.eigenvalues().minCoeff();
    const double max_eigenvalue = solver.eigenvalues().maxCoeff();
    if (min_eigenvalue <= kMinInformationEigenvalue) {
        return setError(error, "information matrix is not positive definite");
    }
    if (max_eigenvalue > kMaxInformationEigenvalue) {
        return setError(error, "information matrix eigenvalue exceeds limit");
    }
    if (max_eigenvalue / min_eigenvalue > kMaxInformationConditionNumber) {
        return setError(error, "information matrix condition number exceeds limit");
    }
    return true;
}

bool parseKeyframesFromProto(const N3Map& map_proto,
                             const PbstreamKeyframeParseOptions& options,
                             std::vector<ParsedKeyframeProto>* keyframes,
                             std::unordered_set<int64_t>* keyframe_ids,
                             std::string* error) {
    if (!keyframes || !keyframe_ids) return setError(error, "null keyframe output");
    if (map_proto.keyframes_size() > static_cast<int>(defaultPbstreamResourceLimits().max_keyframes)) {
        return setError(error, "pbstream has too many keyframes");
    }
    keyframes->clear();
    keyframe_ids->clear();
    keyframes->reserve(map_proto.keyframes_size());
    keyframe_ids->reserve(map_proto.keyframes_size());

    for (int i = 0; i < map_proto.keyframes_size(); ++i) {
        const auto& proto = map_proto.keyframes(i);
        std::string local_error;
        if (!validateKeyframeProto(proto, &local_error)) {
            if (options.policy == PbstreamLoadPolicy::STRICT) return setError(error, local_error);
            continue;
        }

        if (!keyframe_ids->insert(proto.id()).second) {
            return setError(error, "duplicate keyframe id");
        }

        ParsedKeyframeProto parsed;
        parsed.id = proto.id();
        parsed.session_id = proto.session_id();
        parsed.timestamp = proto.timestamp();
        parsed.pose_odom = poseFromProto(proto.pose_odom());
        parsed.pose_optimized = poseFromProto(proto.pose_optimized());
        if (!pointCloudFromProto(proto.cloud(), &parsed.cloud, &local_error)) {
            if (options.policy == PbstreamLoadPolicy::STRICT) return setError(error, local_error);
            keyframe_ids->erase(parsed.id);
            continue;
        }
        if (!parsed.cloud || parsed.cloud->empty()) {
            if (options.policy == PbstreamLoadPolicy::STRICT) return setError(error, "empty keyframe cloud");
            keyframe_ids->erase(parsed.id);
            continue;
        }
        keyframes->push_back(std::move(parsed));
    }

    if (keyframes->empty()) return setError(error, "pbstream_missing_keyframes");
    return true;
}

bool parseDenseTrajectoryFromProto(const N3Map& map_proto,
                                   const std::vector<ParsedKeyframeProto>& keyframes,
                                   const PbstreamLoadOptions& options,
                                   std::vector<core::DenseTrajectoryPose>* dense_trajectory,
                                   core::DenseTrajectoryMetadata* metadata,
                                   std::string* error) {
    if (!dense_trajectory || !metadata) return setError(error, "null dense trajectory output");
    if (map_proto.dense_optimized_trajectory_size() >
        static_cast<int>(defaultPbstreamResourceLimits().max_dense_trajectory_poses)) {
        return setError(error, "pbstream has too many dense trajectory poses");
    }
    dense_trajectory->clear();
    *metadata = core::DenseTrajectoryMetadata{};

    if (map_proto.dense_optimized_trajectory_size() > 0) {
        metadata->source = map_proto.metadata().dense_trajectory_source().empty()
            ? "native"
            : map_proto.metadata().dense_trajectory_source();
        if (metadata->source == "none") return setError(error, "invalid dense trajectory source");
        metadata->degraded = map_proto.metadata().dense_trajectory_degraded();
        dense_trajectory->reserve(static_cast<std::size_t>(map_proto.dense_optimized_trajectory_size()));
        std::unordered_set<int64_t> anchor_ids;
        for (const auto& kf : keyframes) anchor_ids.insert(kf.id);
        for (int i = 0; i < map_proto.dense_optimized_trajectory_size(); ++i) {
            const auto& proto = map_proto.dense_optimized_trajectory(i);
            if (!std::isfinite(proto.timestamp()) || !isFinitePoseProto(proto.pose_world_lidar())) {
                return setError(error, "non-finite dense trajectory pose");
            }
            core::DenseTrajectoryPose pose;
            pose.seq = proto.seq();
            pose.timestamp = proto.timestamp();
            pose.pose_world_lidar = poseFromProto(proto.pose_world_lidar());
            if (proto.has_correction_source()) {
                const auto& source = proto.correction_source();
                if (!isFinitePoseProto(source.pose_world_lidar_raw()) ||
                    !isFinitePoseProto(source.anchor_pose_world_lidar_raw()) ||
                    (source.has_anchor() && !anchor_ids.count(source.anchor_keyframe_id())))
                    return setError(error, "invalid dense trajectory correction source");
                core::AnchoredDenseTrajectorySample sample;
                sample.seq = pose.seq;
                sample.timestamp = pose.timestamp;
                sample.pose_world_lidar_raw = poseFromProto(source.pose_world_lidar_raw());
                sample.anchor_keyframe_id = source.anchor_keyframe_id();
                sample.anchor_pose_world_lidar_raw = poseFromProto(source.anchor_pose_world_lidar_raw());
                sample.has_anchor = source.has_anchor();
                sample.use_bracketing_correction = source.use_bracketing_correction();
                pose.correction_source = sample;
            }
            dense_trajectory->push_back(pose);
        }
        return true;
    }

    if (!options.allow_keyframe_fallback_dense) return true;

    metadata->source = "keyframe_fallback";
    metadata->degraded = true;
    dense_trajectory->reserve(keyframes.size());
    uint64_t seq = 0;
    for (const auto& keyframe : keyframes) {
        core::DenseTrajectoryPose pose;
        pose.seq = seq++;
        pose.timestamp = keyframe.timestamp;
        pose.pose_world_lidar = keyframe.pose_optimized;
        dense_trajectory->push_back(pose);
    }
    return true;
}

bool parseEdgesFromProto(const N3Map& map_proto,
                         const std::unordered_set<int64_t>& valid_keyframe_ids,
                         PbstreamLoadPolicy policy,
                         std::vector<ParsedEdgeProto>* edges,
                         std::string* error) {
    if (!edges) return setError(error, "null edge output");
    if (map_proto.edges_size() > static_cast<int>(defaultPbstreamResourceLimits().max_edges)) {
        return setError(error, "pbstream has too many edges");
    }
    edges->clear();
    edges->reserve(map_proto.edges_size());
    uint32_t odometry_edges = 0;
    uint32_t loop_edges = 0;
    uint32_t session_anchor_edges = 0;

    for (int i = 0; i < map_proto.edges_size(); ++i) {
        const auto& proto = map_proto.edges(i);
        ParsedEdgeProto edge;
        edge.from_id = proto.from_id();
        edge.to_id = proto.to_id();
        std::string local_error;
        if (!isFinitePoseProto(proto.measurement()) ||
            !informationFromProto(proto.information(), &edge.information, &local_error)) {
            if (policy == PbstreamLoadPolicy::STRICT) {
                return setError(error, local_error.empty() ? "malformed edge" : local_error);
            }
            continue;
        }
        if (valid_keyframe_ids.find(edge.from_id) == valid_keyframe_ids.end() ||
            valid_keyframe_ids.find(edge.to_id) == valid_keyframe_ids.end()) {
            if (policy == PbstreamLoadPolicy::STRICT) return setError(error, "malformed edge endpoint");
            continue;
        }
        edge.measurement = poseFromProto(proto.measurement());
        switch (proto.type()) {
        case EdgeProto::ODOMETRY:
            edge.type = PbstreamEdgeType::ODOMETRY;
            ++odometry_edges;
            break;
        case EdgeProto::LOOP:
            edge.type = PbstreamEdgeType::LOOP;
            ++loop_edges;
            break;
        case EdgeProto::SESSION_ANCHOR:
            edge.type = PbstreamEdgeType::SESSION_ANCHOR;
            ++session_anchor_edges;
            break;
        default:
            if (policy == PbstreamLoadPolicy::STRICT) {
                return setError(error, "unknown edge type");
            }
            LOG(ERROR) << "[MapSerializer] Salvage: skipping unknown edge type "
                      << static_cast<int>(proto.type()) << '\n';
            continue;
        }
        switch (proto.constraint_mode()) {
        case EdgeProto::FULL_6DOF:
            edge.constraint_mode = PbstreamEdgeConstraintMode::FULL_6DOF;
            break;
        case EdgeProto::XY_YAW:
            edge.constraint_mode = PbstreamEdgeConstraintMode::XY_YAW;
            break;
        default:
            if (policy == PbstreamLoadPolicy::STRICT) {
                return setError(error, "unknown edge constraint mode");
            }
            LOG(ERROR)
                << "[MapSerializer] Salvage: skipping unknown edge constraint mode "
                << static_cast<int>(proto.constraint_mode()) << '\n';
            if (edge.type == PbstreamEdgeType::ODOMETRY) {
                --odometry_edges;
            } else if (edge.type == PbstreamEdgeType::LOOP) {
                --loop_edges;
            } else {
                --session_anchor_edges;
            }
            continue;
        }
        edges->push_back(std::move(edge));
    }

    if (policy == PbstreamLoadPolicy::STRICT &&
        hasCompleteEdgeCountContract(map_proto.metadata().version()) &&
        (odometry_edges != map_proto.metadata().num_odometry_edges() ||
         loop_edges != map_proto.metadata().num_loop_edges() ||
         session_anchor_edges !=
             map_proto.metadata().num_session_anchor_edges())) {
        return setError(error, "edge metadata count mismatch");
    }
    return true;
}

bool parseFloorAttitudeFactorsFromProto(
    const N3Map& map_proto,
    const std::unordered_set<int64_t>& valid_keyframe_ids,
    PbstreamLoadPolicy policy,
    std::vector<ParsedFloorAttitudeConstraint>* floors,
    std::string* error) {
    if (!floors) return setError(error, "null floor output");
    const uint32_t declared =
        static_cast<uint32_t>(map_proto.metadata().num_floor_attitude_factors());
    if (static_cast<uint32_t>(map_proto.floor_attitude_factors_size()) != declared) {
        if (policy == PbstreamLoadPolicy::STRICT) {
            return setError(error, "floor attitude factor count mismatch");
        }
    }
    floors->clear();
    floors->reserve(map_proto.floor_attitude_factors_size());
    for (int i = 0; i < map_proto.floor_attitude_factors_size(); ++i) {
        const auto& proto = map_proto.floor_attitude_factors(i);
        ParsedFloorAttitudeConstraint floor;
        floor.node_id = proto.node_id();
        floor.normal_body =
            Eigen::Vector3d(proto.nx(), proto.ny(), proto.nz());
        floor.sigma_rad = proto.sigma_rad();
        std::string local_error;
        if (valid_keyframe_ids.find(floor.node_id) ==
            valid_keyframe_ids.end()) {
            local_error = "floor constraint node not in map";
        } else if (!floor.normal_body.allFinite() ||
                   floor.normal_body.norm() < 1e-6) {
            local_error = "floor constraint normal invalid";
        } else if (!std::isfinite(floor.sigma_rad) ||
                   floor.sigma_rad <= 0.0) {
            local_error = "floor constraint sigma invalid";
        }
        if (!local_error.empty()) {
            if (policy == PbstreamLoadPolicy::STRICT) {
                return setError(error, local_error);
            }
            LOG(ERROR) << "[MapSerializer] Salvage: skipping bad floor factor at "
                      << floor.node_id << ": " << local_error << '\n';
            continue;
        }
        floors->push_back(floor);
    }
    return true;
}

}
