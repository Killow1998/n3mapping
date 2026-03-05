// LoopDetector: ScanContext-based loop candidate detection and ICP verification.
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "Scancontext/HybridScancontext.h"
#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

struct LoopCandidate {
    int64_t query_id = -1;
    int64_t match_id = -1;
    double sc_distance = std::numeric_limits<double>::max();
    float yaw_diff_rad = 0.0f;
    bool isValid() const { return query_id >= 0 && match_id >= 0; }
};

struct VerifiedLoop {
    int64_t query_id = -1;
    int64_t match_id = -1;
    Eigen::Isometry3d T_match_query = Eigen::Isometry3d::Identity();
    double fitness_score = std::numeric_limits<double>::max();
    double inlier_ratio = 0.0;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    bool verified = false;
    bool isValid() const { return verified && query_id >= 0 && match_id >= 0; }
};

class LoopDetector {
public:
    using Ptr = std::shared_ptr<LoopDetector>;
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;

    explicit LoopDetector(const Config& config);
    ~LoopDetector() = default;

    Eigen::MatrixXd makeScanContext(const PointCloudT::Ptr& cloud);
    Eigen::MatrixXd addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud);
    void addDescriptor(int64_t keyframe_id, const Eigen::MatrixXd& descriptor);
    std::vector<LoopCandidate> detectLoopCandidates(int64_t query_id);

    VerifiedLoop verifyLoopCandidate(const LoopCandidate& candidate,
                                     const Keyframe::Ptr& query_keyframe,
                                     const Keyframe::Ptr& match_keyframe,
                                     PointCloudMatcher& matcher);

    std::vector<VerifiedLoop> verifyLoopCandidatesBatch(
        const std::vector<LoopCandidate>& candidates,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        PointCloudMatcher& matcher);

    void rebuildTree();
    std::vector<std::pair<int64_t, Eigen::MatrixXd>> getDescriptors() const;
    void loadDescriptors(const std::vector<std::pair<int64_t, Eigen::MatrixXd>>& descriptors);
    size_t size() const;
    void clear();
    std::pair<int, int> getDescriptorDimensions() const;
    Eigen::MatrixXd getDescriptor(int64_t keyframe_id) const;
    int getNumExcludeRecent() const { return config_.sc_num_exclude_recent; }
    std::pair<double, int> computeDistance(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2);

    // RHPD interface
    Eigen::VectorXd computeRHPD(const PointCloudT::Ptr& cloud) const;
    Eigen::VectorXd addRHPD(int64_t kf_id, const PointCloudT::Ptr& cloud);
    const RHPDManager& getRHPDManager() const { return rhpd_manager_; }
    RHPDManager& getRHPDManager() { return rhpd_manager_; }

private:
    Config config_;
    HybridSCManager sc_manager_;
    RHPDManager rhpd_manager_;
    std::map<int64_t, size_t> id_to_index_;
    std::vector<int64_t> index_to_id_;
    std::vector<Eigen::MatrixXd> descriptors_;
    mutable std::mutex mutex_;
};

} // namespace n3mapping
