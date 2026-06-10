// Shared ROS-free pbstream proto validation and parsing utilities.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/types.h"
#include "n3map.pb.h"

namespace n3mapping {

enum class PbstreamLoadPolicy {
    STRICT,
    SALVAGE
};

struct PbstreamLoadOptions {
    PbstreamLoadPolicy policy = PbstreamLoadPolicy::STRICT;
    bool allow_keyframe_fallback_dense = false;
};

struct PbstreamResourceLimits {
    uint64_t max_file_bytes = 1024ull * 1024ull * 1024ull;
    uint32_t max_keyframes = 200000u;
    uint32_t max_edges = 1000000u;
    uint32_t max_dense_trajectory_poses = 5000000u;
    uint64_t max_total_points = 50000000ull;
    uint64_t max_total_descriptor_values = 50000000ull;
    int protobuf_recursion_limit = 100;
};

struct PbstreamKeyframeParseOptions {
    PbstreamLoadPolicy policy = PbstreamLoadPolicy::STRICT;
    int expected_rhpd_dim = 0;
    bool parse_descriptors = true;
};

struct PbstreamMetadata {
    std::string version;
    std::string map_frame = "map";
    std::string body_frame = "body";
    std::string dense_trajectory_source = "none";
    bool dense_trajectory_degraded = true;
    bool nav_cloud_filter_applied = false;
    std::string nav_cloud_filter_policy;
    bool descriptors_recomputed_from_filtered_cloud = false;
    uint64_t nav_filter_raw_points = 0;
    uint64_t nav_filter_kept_points = 0;
    uint64_t nav_filter_removed_points = 0;
};

enum class PbstreamEdgeType {
    ODOMETRY,
    LOOP
};

struct ParsedKeyframeProto {
    int64_t id = -1;
    double timestamp = 0.0;
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose_optimized = Eigen::Isometry3d::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    Eigen::MatrixXd sc_descriptor;
    Eigen::VectorXd rhpd_descriptor;
};

struct ParsedEdgeProto {
    int64_t from_id = -1;
    int64_t to_id = -1;
    Eigen::Isometry3d measurement = Eigen::Isometry3d::Identity();
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    PbstreamEdgeType type = PbstreamEdgeType::ODOMETRY;
};

constexpr uint32_t kMaxPbstreamPointsPerKeyframe = 5000000u;
constexpr uint64_t kMaxPbstreamDescriptorValues = 1000000u;
constexpr double kMinInformationEigenvalue = 1e-12;
constexpr double kMaxInformationEigenvalue = 1e12;
constexpr double kMaxInformationConditionNumber = 1e12;

bool isFinitePoseProto(const Pose3D& proto);
bool isFinitePose(const Eigen::Isometry3d& pose);
bool isFinitePoint(const pcl::PointXYZI& point);
uint64_t countFinitePoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
Eigen::Isometry3d poseFromProto(const Pose3D& proto);

PbstreamMetadata extractPbstreamMetadata(const MapMetadata& proto);
PbstreamResourceLimits defaultPbstreamResourceLimits();

bool readN3MapProtoFromFile(const std::string& filepath,
                            N3Map* map_proto,
                            std::string* error);
bool readN3MapProtoFromFile(const std::string& filepath,
                            const PbstreamResourceLimits& limits,
                            N3Map* map_proto,
                            std::string* error);

bool pointCloudFromProto(const PointCloudData& proto,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr* cloud,
                         std::string* error);
bool scanContextFromProto(const ScanContextDescriptor& proto,
                          Eigen::MatrixXd* descriptor,
                          std::string* error);
bool rhpdFromProto(const RHPDDescriptor& proto,
                   int expected_dim,
                   Eigen::VectorXd* descriptor,
                   std::string* error);
bool informationFromProto(const InformationMatrix& proto,
                          Eigen::Matrix<double, 6, 6>* information,
                          std::string* error);
bool isValidInformationMatrix(const Eigen::Matrix<double, 6, 6>& information,
                              std::string* error = nullptr);

bool parseKeyframesFromProto(const N3Map& map_proto,
                             const PbstreamKeyframeParseOptions& options,
                             std::vector<ParsedKeyframeProto>* keyframes,
                             std::unordered_set<int64_t>* keyframe_ids,
                             std::string* error);

bool parseDenseTrajectoryFromProto(const N3Map& map_proto,
                                   const std::vector<ParsedKeyframeProto>& keyframes,
                                   const PbstreamLoadOptions& options,
                                   std::vector<core::DenseTrajectoryPose>* dense_trajectory,
                                   core::DenseTrajectoryMetadata* metadata,
                                   std::string* error);

bool parseEdgesFromProto(const N3Map& map_proto,
                         const std::unordered_set<int64_t>& valid_keyframe_ids,
                         PbstreamLoadPolicy policy,
                         std::vector<ParsedEdgeProto>* edges,
                         std::string* error);

}  // namespace n3mapping
