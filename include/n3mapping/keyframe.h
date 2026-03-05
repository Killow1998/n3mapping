// keyframe.h — Keyframe data structure: id, timestamps, poses, point cloud, and ScanContext descriptor.
#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace n3mapping {

struct Keyframe {
    using Ptr = std::shared_ptr<Keyframe>;
    using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

    int64_t id = -1;
    double timestamp = 0.0;
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose_optimized = Eigen::Isometry3d::Identity();
    PointCloudT::Ptr cloud = nullptr;
    Eigen::MatrixXd sc_descriptor;
    Eigen::VectorXd rhpd_descriptor;   // RHPD descriptor for relocalization
    bool is_from_loaded_map = false;

    Keyframe() = default;

    Keyframe(int64_t id, double timestamp, const Eigen::Isometry3d& pose, const PointCloudT::Ptr& cloud)
        : id(id), timestamp(timestamp), pose_odom(pose), pose_optimized(pose), cloud(cloud) {}

    static Ptr create(int64_t id, double ts, const Eigen::Isometry3d& pose, const PointCloudT::Ptr& cloud) {
        return std::make_shared<Keyframe>(id, ts, pose, cloud);
    }

    bool isValid() const { return id >= 0 && cloud && !cloud->empty(); }
    Eigen::Vector3d getPosition() const { return pose_optimized.translation(); }
    Eigen::Matrix3d getRotation() const { return pose_optimized.rotation(); }
};

} // namespace n3mapping
