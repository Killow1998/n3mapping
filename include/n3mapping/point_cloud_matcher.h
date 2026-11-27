#pragma once

#include <cstddef>
#include <memory>
#include <optional>
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
namespace registration_detail { struct SearchBounds; class LazyTargetNormals; struct MappedTargetLevel; }

enum class MatchTermination {
  Invalid,
  Converged,
  MaxIterations,
  Stalled,

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
        std::shared_ptr<const registration_detail::SearchBounds> search_bounds = nullptr;
        std::shared_ptr<registration_detail::LazyTargetNormals> lazy_normals = nullptr;
        std::shared_ptr<const registration_detail::MappedTargetLevel> mapped = nullptr;
        bool use_occupied_bounds = true;
        std::size_t size() const;
        bool ready() const;
    };

    struct PreparedSourceLevel {
        double resolution = 0.0;
        mutable SmallGicpCloud::Ptr cloud;
    };

    struct PreparedTarget {
        std::vector<PreparedTargetLevel> plane_levels;
        PreparedTargetLevel refine_level;
        bool has_refine_level = false;

        bool sharesRefineLevel() const;
    };

    struct PreparedSource {
        struct PendingRefine {
            SmallGicpCloud::Ptr points;
            double resolution;
            int neighbors;
            int threads;
        };

        std::vector<PreparedSourceLevel> plane_levels;
        mutable SmallGicpCloud::Ptr refine_cloud;
        bool has_refine_cloud = false;
        int lazy_coarse_threads = 0;
        mutable std::optional<PendingRefine> pending_refine;

        const SmallGicpCloud* planeCloud(double resolution) const;
        const SmallGicpCloud::Ptr& refineCloud() const;
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

    enum class TargetFeatures { Eager, OnDemand, SharedRefine };

    PreparedTarget prepareTargetCloud(
        const PointCloudT::Ptr& cloud, TargetFeatures features = TargetFeatures::Eager);
    PreparedSource prepareSourceCloud(const PointCloudT::Ptr& cloud,
                                      bool tracking = false);
    MatchResult alignPrepared(const PreparedTarget& target,
                              const PreparedSource& source,
                              const Eigen::Isometry3d& init_guess = Eigen::Isometry3d::Identity());

    MatchResult alignPrepared(const PreparedTarget& target,
                              const PreparedSource& source,
                              const Eigen::Isometry3d& init_guess,
                              const small_gicp::RegistrationSetting& setting);

    MatchResult alignPreparedReciprocal(const PreparedTarget& target,
                                       const PreparedSource& source,
                                       const Eigen::Isometry3d& init_guess,
                                       const small_gicp::RegistrationSetting& setting);

    MatchResult resumePrepared(const PreparedTarget& target,
                               const PreparedSource& source,
                               const Eigen::Isometry3d& init_guess,
                               const MatchMetric& metric,
                               bool refine_with_gicp = true);

    MatchResult evaluatePreparedPose(const PreparedTarget& target,
                                    const PreparedSource& source,
                                    const Eigen::Isometry3d& pose,
                                    const MatchMetric& metric) const;

    MatchResult evaluatePreparedPlaneGeometry(const PreparedTarget &target,
                                              const PreparedSource &source,
                                              const Eigen::Isometry3d &pose,
                                              const MatchMetric &metric) const;

    std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>> preprocessPointCloud(const PointCloudT::Ptr& cloud);
    const small_gicp::RegistrationSetting& getSettings() const { return setting_; }
    void setSettings(const small_gicp::RegistrationSetting& setting) { setting_ = setting; }

private:
  MatchResult evaluatePreparedPoseImpl(const PreparedTarget &target,
                                       const PreparedSource &source,
                                       const Eigen::Isometry3d &pose,
                                       const MatchMetric &metric,
                                       bool plane_geometry) const;
  std::pair<SmallGicpCloud::Ptr, std::shared_ptr<SmallGicpKdTree>>
  prepareTargetGeometry(const PointCloudT &cloud, double resolution);
  SmallGicpCloud::Ptr convertToSmallGicp(const PointCloudT &pcl_cloud,
                                         double downsampling_resolution);
  SmallGicpCloud::Ptr preprocessSourcePointCloud(
      const PointCloudT& cloud, double resolution);
  MatchResult
  alignCloudWithSetting(const PointCloudT::Ptr &target_cloud,
                        const PointCloudT::Ptr &source_cloud,
                        const Eigen::Isometry3d &init_guess,
                        const small_gicp::RegistrationSetting &setting);
  MatchResult alignPreparedWithSetting(
      const PreparedTarget &target, const PreparedSource &source,
      const Eigen::Isometry3d &init_guess,
      const small_gicp::RegistrationSetting &setting,
      double start_resolution = std::numeric_limits<double>::infinity(),
      bool reciprocal = false, bool refine_with_gicp = true);

  Config config_;
  small_gicp::RegistrationSetting setting_;
};

std::size_t estimatePreparedTargetMemoryBytes(
    const PointCloudMatcher::PreparedTarget& target);

}
