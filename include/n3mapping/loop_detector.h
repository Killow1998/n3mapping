// LoopDetector: ScanContext-based loop candidate detection and ICP verification.
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "Scancontext/HybridScancontext.h"
#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/config.h"
#include "n3mapping/loop_types.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

// Descriptor yaw seeds shared by global relocalization and drift-independent
// cross-session loop registration. Keeping the hypotheses identical prevents
// the two global matching paths from silently developing different basins.
std::vector<double> buildDescriptorYawHypotheses(
    const LoopCandidate& candidate, const Config& config);

class LoopDetector {
public:
    using Ptr = std::shared_ptr<LoopDetector>;
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;
    using MatchFilter = std::function<bool(int64_t)>;

    explicit LoopDetector(const Config& config);
    ~LoopDetector() = default;

    Eigen::MatrixXd makeScanContext(const PointCloudT::Ptr& cloud);
    Eigen::MatrixXd addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud);
    void addDescriptor(int64_t keyframe_id, const Eigen::MatrixXd& descriptor);
    bool isScanContextDescriptorCompatible(const Eigen::MatrixXd& descriptor) const;
    // keyframes supplies the odometry poses the path-length exclusion needs;
    // pass the same map used for spatial candidates.
    std::vector<LoopCandidate> detectLoopCandidates(
        int64_t query_id,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match = {});
    std::vector<LoopCandidate> detectSpatialCandidates(
        int64_t query_id,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match = {}) const;

    void rebuildTree();
    std::vector<std::pair<int64_t, Eigen::MatrixXd>> getDescriptors() const;
    void loadDescriptors(const std::vector<std::pair<int64_t, Eigen::MatrixXd>>& descriptors);
    void swapWith(LoopDetector& other);
    size_t size() const;
    void clear();
    std::pair<int, int> getDescriptorDimensions() const;
    double getScanContextSectorAngleDeg() const { return sc_manager_.PC_UNIT_SECTORANGLE; }
    Eigen::MatrixXd getDescriptor(int64_t keyframe_id) const;
    int getNumExcludeRecent() const { return config_.sc_num_exclude_recent; }
    std::pair<double, int> computeDistance(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2);

    // RHPD interface
    Eigen::VectorXd computeRHPD(const PointCloudT::Ptr& cloud) const;
    Eigen::VectorXd addRHPD(int64_t kf_id, const PointCloudT::Ptr& cloud);
    void loadRHPDDescriptors(const std::vector<std::pair<int64_t, Eigen::VectorXd>>& descriptors);
    void clearRHPD();
    const RHPDManager& getRHPDManager() const { return rhpd_manager_; }
    const RHPDescriptor::Params& getRHPDParams() const { return rhpd_manager_.getDescriptorParams(); }

private:
    Config config_;
    HybridSCManager sc_manager_;
    RHPDManager rhpd_manager_;
    std::map<int64_t, size_t> id_to_index_;
    std::vector<int64_t> index_to_id_;
    std::vector<Eigen::MatrixXd> descriptors_;
    mutable std::mutex mutex_;

    void rebuildTreeUnlocked();
};

} // namespace n3mapping
