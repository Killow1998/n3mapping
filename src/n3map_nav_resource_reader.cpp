#include "n3mapping/n3map_nav_resource_reader.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <unordered_set>

#include "n3map.pb.h"

namespace n3mapping {
namespace {

constexpr uint32_t MAX_POINTS_PER_KEYFRAME = 5000000u;

bool setError(std::string* error, const std::string& message) {
    if (error) *error = message;
    return false;
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

Eigen::Isometry3d readPose(const Pose3D& proto) {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.translation() = Eigen::Vector3d(proto.tx(), proto.ty(), proto.tz());
    Eigen::Quaterniond q(proto.qw(), proto.qx(), proto.qy(), proto.qz());
    q.normalize();
    pose.linear() = q.toRotationMatrix();
    return pose;
}

bool isFinitePoint(const pcl::PointXYZI& pt) {
    return std::isfinite(pt.x) && std::isfinite(pt.y) &&
           std::isfinite(pt.z) && std::isfinite(pt.intensity);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readCloud(const PointCloudData& proto, std::string* error) {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    const uint64_t required_values = static_cast<uint64_t>(proto.num_points()) * 4u;
    if (proto.num_points() > MAX_POINTS_PER_KEYFRAME ||
        required_values > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        proto.points_size() != static_cast<int>(required_values)) {
        setError(error, "malformed point cloud repeated-field length");
        return nullptr;
    }
    cloud->resize(proto.num_points());
    for (uint32_t i = 0; i < proto.num_points(); ++i) {
        auto& pt = cloud->points[i];
        pt.x = proto.points(i * 4);
        pt.y = proto.points(i * 4 + 1);
        pt.z = proto.points(i * 4 + 2);
        pt.intensity = proto.points(i * 4 + 3);
        if (!isFinitePoint(pt)) {
            setError(error, "non-finite point in keyframe cloud");
            return nullptr;
        }
    }
    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;
    return cloud;
}

}  // namespace

bool readN3NavResource(const std::string& pbstream_path,
                       N3NavResource* out,
                       std::string* error) {
    return readN3NavResource(pbstream_path, N3NavReaderOptions{}, out, error);
}

bool readN3NavResource(const std::string& pbstream_path,
                       const N3NavReaderOptions& options,
                       N3NavResource* out,
                       std::string* error) {
    if (!out) return setError(error, "null output resource");

    std::ifstream ifs(pbstream_path, std::ios::binary);
    if (!ifs.is_open()) return setError(error, "failed to open pbstream");

    N3Map map_proto;
    if (!map_proto.ParseFromIstream(&ifs)) return setError(error, "failed to parse pbstream");

    N3NavResource resource;
    resource.version = map_proto.metadata().version();
    if (!map_proto.metadata().map_frame().empty()) {
        resource.map_frame = map_proto.metadata().map_frame();
    }
    if (!map_proto.metadata().body_frame().empty()) {
        resource.body_frame = map_proto.metadata().body_frame();
    }
    resource.dense_trajectory_source = map_proto.metadata().dense_trajectory_source();
    resource.dense_trajectory_degraded = map_proto.metadata().dense_trajectory_degraded();
    resource.nav_cloud_filter_applied = map_proto.metadata().nav_cloud_filter_applied();
    resource.nav_cloud_filter_policy = map_proto.metadata().nav_cloud_filter_policy();
    resource.descriptors_recomputed_from_filtered_cloud =
        map_proto.metadata().descriptors_recomputed_from_filtered_cloud();
    resource.nav_filter_raw_points = map_proto.metadata().nav_filter_raw_points();
    resource.nav_filter_kept_points = map_proto.metadata().nav_filter_kept_points();
    resource.nav_filter_removed_points = map_proto.metadata().nav_filter_removed_points();

    resource.keyframes.reserve(map_proto.keyframes_size());
    std::unordered_set<int64_t> keyframe_ids;
    keyframe_ids.reserve(map_proto.keyframes_size());
    for (int i = 0; i < map_proto.keyframes_size(); ++i) {
        const auto& proto_kf = map_proto.keyframes(i);
        if (proto_kf.id() < 0) {
            return setError(error, "invalid keyframe id");
        }
        if (!std::isfinite(proto_kf.timestamp())) {
            return setError(error, "non-finite keyframe timestamp");
        }
        if (!isFinitePoseProto(proto_kf.pose_odom()) || !isFinitePoseProto(proto_kf.pose_optimized())) {
            return setError(error, "non-finite keyframe pose");
        }
        if (!proto_kf.has_cloud()) {
            return setError(error, "missing keyframe cloud");
        }
        if (!keyframe_ids.insert(proto_kf.id()).second) {
            return setError(error, "duplicate keyframe id");
        }

        N3NavKeyframe keyframe;
        keyframe.id = proto_kf.id();
        keyframe.timestamp = proto_kf.timestamp();
        keyframe.pose_odom = readPose(proto_kf.pose_odom());
        keyframe.pose_optimized = readPose(proto_kf.pose_optimized());
        keyframe.cloud = readCloud(proto_kf.cloud(), error);
        if (!keyframe.cloud) return false;
        if (keyframe.cloud->empty()) {
            return setError(error, "empty keyframe cloud");
        }

        resource.optimized_poses[keyframe.id] = keyframe.pose_optimized;
        resource.keyframes.push_back(std::move(keyframe));
    }
    if (resource.keyframes.empty()) {
        return setError(error, "pbstream_missing_keyframes");
    }

    if (map_proto.dense_optimized_trajectory_size() > 0) {
        if (resource.dense_trajectory_source.empty()) {
            resource.dense_trajectory_source = "native";
            resource.dense_trajectory_degraded = false;
        }
        if (resource.dense_trajectory_source == "none") {
            return setError(error, "invalid dense trajectory source");
        }
        resource.dense_optimized_trajectory.reserve(map_proto.dense_optimized_trajectory_size());
        for (int i = 0; i < map_proto.dense_optimized_trajectory_size(); ++i) {
            const auto& proto_pose = map_proto.dense_optimized_trajectory(i);
            if (!std::isfinite(proto_pose.timestamp()) || !isFinitePoseProto(proto_pose.pose_world_lidar())) {
                return setError(error, "non-finite dense trajectory pose");
            }
            core::DenseTrajectoryPose pose;
            pose.seq = proto_pose.seq();
            pose.timestamp = proto_pose.timestamp();
            pose.pose_world_lidar = readPose(proto_pose.pose_world_lidar());
            resource.dense_optimized_trajectory.push_back(pose);
        }
        resource.has_native_dense_trajectory =
            !resource.dense_optimized_trajectory.empty() &&
            resource.dense_trajectory_source == "native" &&
            !resource.dense_trajectory_degraded;
        resource.dense_trajectory_from_keyframe_fallback =
            resource.dense_trajectory_source == "keyframe_fallback" ||
            resource.dense_trajectory_source == "mixed_keyframe_fallback_and_high_rate";
    } else {
        if (!options.allow_keyframe_fallback) {
            return setError(error, "pbstream_missing_dense_trajectory");
        }
        resource.dense_trajectory_source = "keyframe_fallback";
        resource.dense_trajectory_degraded = true;
        resource.dense_trajectory_from_keyframe_fallback = true;
        resource.dense_optimized_trajectory.reserve(resource.keyframes.size());
        uint64_t seq = 0;
        for (const auto& keyframe : resource.keyframes) {
            core::DenseTrajectoryPose pose;
            pose.seq = seq++;
            pose.timestamp = keyframe.timestamp;
            pose.pose_world_lidar = keyframe.pose_optimized;
            resource.dense_optimized_trajectory.push_back(pose);
        }
    }

    *out = std::move(resource);
    return true;
}

}  // namespace n3mapping
