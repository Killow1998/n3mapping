// Lightweight ROS-free pbstream reader for downstream navigation consumers.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/core/types.h"

namespace n3mapping {

struct N3NavKeyframe {
    int64_t id = -1;
    double timestamp = 0.0;
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose_optimized = Eigen::Isometry3d::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
};

struct N3NavResource {
    std::string version;
    std::string map_frame = "map";
    std::string body_frame = "body";
    std::vector<N3NavKeyframe> keyframes;
    std::map<int64_t, Eigen::Isometry3d> optimized_poses;
    std::vector<core::DenseTrajectoryPose> dense_optimized_trajectory;
};

bool readN3NavResource(const std::string& pbstream_path,
                       N3NavResource* out,
                       std::string* error = nullptr);

}  // namespace n3mapping
