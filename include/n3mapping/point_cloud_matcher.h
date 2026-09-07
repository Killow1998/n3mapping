// PointCloudMatcher: small_gicp-based point cloud registration with multi-scale PLANE_ICP and optional GICP refinement.
#pragma once

#include <cstddef>
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
    Stalled,
    // Residuals/correspondences evaluated without running an optimizer.
    FixedPoseEvaluation
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

struct MatchMetric {
    small_gicp::RegistrationSetting::RegistrationType type = small_gicp::RegistrationSetting::PLANE_ICP;
    double resolution = std::numeric_limits<double>::quiet_NaN();
    double max_correspondence_distance = 0.0;
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
    // Objective, resolution and correspondence gate of the selected optimizer
    // stage. Fixed-pose evaluation must use these same metric semantics.
    MatchMetric metric;
};

class PointCloudMatcher {
public:
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;
    using SmallGicpCloud = small_gicp::PointCloud;
    using SmallGicpKdTree = small_gicp::KdTree<SmallGicpCloud>;

    struct PreparedTargetLevel {
        double resolution = 0.0;
        SmallGicpCloud::Ptr cloud;
        std::shared_ptr<SmallGicpKdTree> kdtree;
    };

    struct PreparedSourceLevel {
        double resolution = 0.0;
        SmallGicpCloud::Ptr cloud;
    };

    // Immutable, call-scoped registration inputs. Preparing them once avoids
    // repeating voxelization, target KD-tree construction, and normal/covariance
    // estimation for every yaw seed or retry while leaving the optimizer inputs
    // and settings unchanged.
    struct PreparedTarget {
        std::vector<PreparedTargetLevel> plane_levels;
        PreparedTargetLevel refine_level;
        bool has_refine_level = false;
    };

    struct PreparedSource {
        std::vector<PreparedSourceLevel> plane_levels;
        SmallGicpCloud::Ptr refine_cloud;
        bool has_refine_cloud = false;
    };

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

    PreparedTarget prepareTargetCloud(const PointCloudT::Ptr& cloud);
    PreparedSource prepareSourceCloud(const PointCloudT::Ptr& cloud);
    MatchResult alignPrepared(const PreparedTarget& target,
                              const PreparedSource& source,
                              const Eigen::Isometry3d& init_guess = Eigen::Isometry3d::Identity());
    MatchResult alignPrepared(const PreparedTarget& target,
                              const PreparedSource& source,
                              const Eigen::Isometry3d& init_guess,
                              const small_gicp::RegistrationSetting& setting);
    // Does not change pose or claim optimizer convergence. The result has
    // FixedPoseEvaluation termination only when usable residuals were measured.
    MatchResult evaluatePreparedPose(const PreparedTarget& target,
                                    const PreparedSource& source,
                                    const Eigen::Isometry3d& pose,
                                    const MatchMetric& metric) const;

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
    MatchResult alignPreparedWithSetting(const PreparedTarget& target,
                                         const PreparedSource& source,
                                         const Eigen::Isometry3d& init_guess,
                                         const small_gicp::RegistrationSetting& setting);

    Config config_;
    small_gicp::RegistrationSetting setting_;
};

// Conservative retained-memory estimate for a prepared target. small_gicp
// does not expose KD-tree heap usage, so the estimate includes the exact
// point/normal/covariance payload plus a per-point tree/index allowance. The
// arithmetic saturates instead of wrapping, preserving byte-budget safety for
// unexpectedly large targets.
std::size_t estimatePreparedTargetMemoryBytes(
    const PointCloudMatcher::PreparedTarget& target);

} // namespace n3mapping
