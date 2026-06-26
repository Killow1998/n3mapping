#include "n3mapping/n3map_proto_utils.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>

#include <Eigen/Eigenvalues>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

bool setError(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
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

bool addChecked(uint64_t value, uint64_t* total, uint64_t limit, const std::string& message, std::string* error) {
    if (!total) return setError(error, "null resource counter");
    if (value > limit || *total > limit - value) return setError(error, message);
    *total += value;
    return true;
}

bool validateResourceCounts(const N3Map& map_proto,
                            const PbstreamResourceLimits& limits,
                            std::string* error) {
    if (limits.max_file_bytes == 0 ||
        limits.max_keyframes == 0 ||
        limits.max_edges == 0 ||
        limits.max_dense_trajectory_poses == 0 ||
        limits.max_total_points == 0 ||
        limits.max_total_descriptor_values == 0 ||
        limits.protobuf_recursion_limit <= 0) {
        return setError(error, "invalid pbstream resource limits");
    }
    if (map_proto.keyframes_size() > static_cast<int>(limits.max_keyframes)) {
        return setError(error, "pbstream has too many keyframes");
    }
    if (map_proto.edges_size() > static_cast<int>(limits.max_edges)) {
        return setError(error, "pbstream has too many edges");
    }
    if (map_proto.dense_optimized_trajectory_size() >
        static_cast<int>(limits.max_dense_trajectory_poses)) {
        return setError(error, "pbstream has too many dense trajectory poses");
    }

    uint64_t total_points = 0;
    uint64_t total_descriptor_values = 0;
    for (int i = 0; i < map_proto.keyframes_size(); ++i) {
        const auto& keyframe = map_proto.keyframes(i);
        if (keyframe.has_cloud()) {
            const uint64_t encoded_points =
                (static_cast<uint64_t>(keyframe.cloud().points_size()) + 3u) / 4u;
            const uint64_t counted_points =
                std::max(static_cast<uint64_t>(keyframe.cloud().num_points()), encoded_points);
            if (!addChecked(counted_points,
                            &total_points,
                            limits.max_total_points,
                            "pbstream has too many total points",
                            error)) {
                return false;
            }
        }
        if (keyframe.has_sc_descriptor()) {
            if (!addChecked(static_cast<uint64_t>(keyframe.sc_descriptor().values_size()),
                            &total_descriptor_values,
                            limits.max_total_descriptor_values,
                            "pbstream has too many descriptor values",
                            error)) {
                return false;
            }
        }
        if (keyframe.has_rhpd_descriptor()) {
            if (!addChecked(static_cast<uint64_t>(keyframe.rhpd_descriptor().values_size()),
                            &total_descriptor_values,
                            limits.max_total_descriptor_values,
                            "pbstream has too many descriptor values",
                            error)) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

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

PbstreamResourceLimits defaultPbstreamResourceLimits() {
    return PbstreamResourceLimits{};
}

bool readN3MapProtoFromFile(const std::string& filepath,
                            N3Map* map_proto,
                            std::string* error) {
    return readN3MapProtoFromFile(filepath, defaultPbstreamResourceLimits(), map_proto, error);
}

bool readN3MapProtoFromFile(const std::string& filepath,
                            const PbstreamResourceLimits& limits,
                            N3Map* map_proto,
                            std::string* error) {
    if (!map_proto) return setError(error, "null map proto output");
    map_proto->Clear();
    if (limits.max_file_bytes == 0 ||
        limits.max_keyframes == 0 ||
        limits.max_edges == 0 ||
        limits.max_dense_trajectory_poses == 0 ||
        limits.max_total_points == 0 ||
        limits.max_total_descriptor_values == 0 ||
        limits.protobuf_recursion_limit <= 0) {
        return setError(error, "invalid pbstream resource limits");
    }

    std::error_code ec;
    const auto status = std::filesystem::status(filepath, ec);
    if (ec || !std::filesystem::exists(status)) return setError(error, "pbstream does not exist");
    if (!std::filesystem::is_regular_file(status)) return setError(error, "pbstream is not a regular file");

    const auto file_size = std::filesystem::file_size(filepath, ec);
    if (ec) return setError(error, "failed to stat pbstream");
    if (file_size > limits.max_file_bytes) return setError(error, "pbstream exceeds file size limit");
    if (limits.max_file_bytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return setError(error, "pbstream byte limit exceeds protobuf API limit");
    }

    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs.is_open()) return setError(error, "failed to open pbstream");

    google::protobuf::io::IstreamInputStream zero_copy_input(&ifs);
    google::protobuf::io::CodedInputStream coded_input(&zero_copy_input);
    coded_input.SetTotalBytesLimit(static_cast<int>(limits.max_file_bytes));
    coded_input.SetRecursionLimit(limits.protobuf_recursion_limit);
    if (!map_proto->ParseFromCodedStream(&coded_input) || !coded_input.ConsumedEntireMessage()) {
        map_proto->Clear();
        return setError(error, "failed to parse pbstream");
    }
    if (!validateResourceCounts(*map_proto, limits, error)) {
        map_proto->Clear();
        return false;
    }
    return true;
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

bool scanContextFromProto(const ScanContextDescriptor& proto,
                          Eigen::MatrixXd* descriptor,
                          std::string* error) {
    if (!descriptor) return setError(error, "null ScanContext descriptor output");
    descriptor->resize(0, 0);
    const uint64_t required_values =
        static_cast<uint64_t>(proto.rows()) * static_cast<uint64_t>(proto.cols());
    if (proto.rows() == 0 || proto.cols() == 0 ||
        required_values > kMaxPbstreamDescriptorValues ||
        required_values > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        proto.values_size() != static_cast<int>(required_values)) {
        return setError(error, "malformed ScanContext descriptor");
    }

    Eigen::MatrixXd parsed(proto.rows(), proto.cols());
    int index = 0;
    for (uint32_t row = 0; row < proto.rows(); ++row) {
        for (uint32_t col = 0; col < proto.cols(); ++col) {
            const double value = proto.values(index++);
            if (!std::isfinite(value)) {
                return setError(error, "non-finite ScanContext descriptor value");
            }
            parsed(row, col) = value;
        }
    }
    *descriptor = parsed;
    return true;
}

bool rhpdFromProto(const RHPDDescriptor& proto,
                   int expected_dim,
                   Eigen::VectorXd* descriptor,
                   std::string* error) {
    if (!descriptor) return setError(error, "null RHPD descriptor output");
    *descriptor = Eigen::VectorXd();
    if (expected_dim <= 0 ||
        proto.dim() != static_cast<uint32_t>(expected_dim) ||
        proto.values_size() != static_cast<int>(proto.dim())) {
        return setError(error, "malformed RHPD descriptor");
    }

    Eigen::VectorXd parsed = Eigen::VectorXd::Zero(proto.dim());
    for (uint32_t i = 0; i < proto.dim(); ++i) {
        const double value = proto.values(i);
        if (!std::isfinite(value)) return setError(error, "non-finite RHPD descriptor value");
        parsed(i) = value;
    }
    *descriptor = parsed;
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
        if (options.parse_descriptors && proto.has_sc_descriptor()) {
            if (!scanContextFromProto(proto.sc_descriptor(), &parsed.sc_descriptor, &local_error)) {
                parsed.sc_descriptor = Eigen::MatrixXd();
            }
        }
        if (options.parse_descriptors &&
            proto.has_rhpd_descriptor() &&
            proto.rhpd_descriptor().values_size() > 0) {
            if (!rhpdFromProto(proto.rhpd_descriptor(), options.expected_rhpd_dim,
                               &parsed.rhpd_descriptor, &local_error)) {
                parsed.rhpd_descriptor = Eigen::VectorXd();
            }
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
        for (int i = 0; i < map_proto.dense_optimized_trajectory_size(); ++i) {
            const auto& proto = map_proto.dense_optimized_trajectory(i);
            if (!std::isfinite(proto.timestamp()) || !isFinitePoseProto(proto.pose_world_lidar())) {
                return setError(error, "non-finite dense trajectory pose");
            }
            core::DenseTrajectoryPose pose;
            pose.seq = proto.seq();
            pose.timestamp = proto.timestamp();
            pose.pose_world_lidar = poseFromProto(proto.pose_world_lidar());
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
        edge.type = (proto.type() == EdgeProto::LOOP) ? PbstreamEdgeType::LOOP : PbstreamEdgeType::ODOMETRY;
        edge.constraint_mode = proto.constraint_mode() == EdgeProto::XY_YAW
            ? PbstreamEdgeConstraintMode::XY_YAW
            : PbstreamEdgeConstraintMode::FULL_6DOF;
        edges->push_back(std::move(edge));
    }
    return true;
}

}  // namespace n3mapping
