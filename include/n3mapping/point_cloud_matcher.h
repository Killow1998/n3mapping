// PointCloudMatcher: small_gicp-based point cloud registration with multi-scale PLANE_ICP and optional GICP refinement.
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <small_gicp/ann/kdtree.hpp>
#include <small_gicp/points/point_cloud.hpp>
#include <small_gicp/registration/registration_helper.hpp>

#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"

namespace n3mapping {

struct MatchResult {
    bool success = false;
    bool converged = false;
    Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
    double fitness_score = std::numeric_limits<double>::max();
    size_t num_inliers = 0;
    double inlier_ratio = 0.0;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
};

class PointCloudMatcher {
public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;
    using SmallGicpCloud = small_gicp::PointCloud;
    using SmallGicpKdTree = small_gicp::KdTree<SmallGicpCloud>;

    explicit PointCloudMatcher(const Config& config);
    ~PointCloudMatcher() = default;

    MatchResult align(const Keyframe::Ptr& target, const Keyframe::Ptr& source,
                      const Eigen::Isometry3d& init_guess = Eigen::Isometry3d::Identity());

    std::vector<MatchResult> alignBatch(
        const std::vector<std::pair<Keyframe::Ptr, Keyframe::Ptr>>& pairs,
        const std::vector<Eigen::Isometry3d>& init_guesses);

    MatchResult alignCloud(const PointCloudT::Ptr& target_cloud, const PointCloudT::Ptr& source_cloud,
                           const Eigen::Isometry3d& init_guess = Eigen::Isometry3d::Identity());

    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>> preprocessPointCloud(const PointCloudT::Ptr& cloud);
    const small_gicp::RegistrationSetting& getSettings() const { return setting_; }
    void setSettings(const small_gicp::RegistrationSetting& setting) { setting_ = setting; }

private:
    SmallGicpCloud::Ptr convertToSmallGicp(const PointCloudT::Ptr& pcl_cloud);
    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>>
    preprocessTargetPointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution);
    SmallGicpCloud::Ptr preprocessSourcePointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution);

    Config config_;
    small_gicp::RegistrationSetting setting_;
};

} // namespace n3mapping
