// PointCloudMatcher: small_gicp-based point cloud registration with multi-scale PLANE_ICP and optional GICP refinement.
#pragma once

#include <memory>
#include <string>
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

enum class MatchTermination {
    Invalid,
    Converged,
    MaxIterations,
    Stalled
};

const char* matchTerminationName(MatchTermination termination);
MatchTermination classifyMatchTermination(bool converged,
                                          size_t iterations,
                                          int max_iterations,
                                          bool valid_result);

struct MatchStageResult {
    std::string stage;
    double resolution = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
    size_t iterations = 0;
    double optimizer_error = std::numeric_limits<double>::quiet_NaN();
    double fitness_score = std::numeric_limits<double>::max();
    size_t num_inliers = 0;
    double inlier_ratio = 0.0;
    MatchTermination termination = MatchTermination::Invalid;
};

struct MatchResult {
    bool success = false;
    bool converged = false;
    Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
    double fitness_score = std::numeric_limits<double>::max();
    size_t num_inliers = 0;
    size_t iterations = 0;
    double optimizer_error = std::numeric_limits<double>::quiet_NaN();
    MatchTermination termination = MatchTermination::Invalid;
    double inlier_ratio = 0.0;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    std::vector<MatchStageResult> stages;
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
    MatchResult alignCloud(const PointCloudT::Ptr& target_cloud, const PointCloudT::Ptr& source_cloud,
                           const Eigen::Isometry3d& init_guess,
                           const small_gicp::RegistrationSetting& setting);

    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>> preprocessPointCloud(const PointCloudT::Ptr& cloud);
    const small_gicp::RegistrationSetting& getSettings() const { return setting_; }
    void setSettings(const small_gicp::RegistrationSetting& setting) { setting_ = setting; }

private:
    SmallGicpCloud::Ptr convertToSmallGicp(const PointCloudT::Ptr& pcl_cloud, double downsampling_resolution);
    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>>
    preprocessTargetPointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution);
    SmallGicpCloud::Ptr preprocessSourcePointCloud(const PointCloudT::Ptr& cloud, double downsampling_resolution);
    MatchResult alignCloudWithSetting(const PointCloudT::Ptr& target_cloud,
                                      const PointCloudT::Ptr& source_cloud,
                                      const Eigen::Isometry3d& init_guess,
                                      const small_gicp::RegistrationSetting& setting);

    Config config_;
    small_gicp::RegistrationSetting setting_;
};

} // namespace n3mapping
