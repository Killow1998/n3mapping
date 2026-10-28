// LoopDetector: RHPD place/heading candidates and spatial candidate selection.
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

#include "n3mapping/RHPDescriptor.h"
#include "n3mapping/config.h"
#include "n3mapping/loop_types.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

// Shared descriptor phase, full-circle yaw grid and periodic deduplication.
// Loop verification uses all generated starts. Relocalization selects its
// proposal budget in relocalization_seed_plan.h without changing loop policy.
inline constexpr double kDescriptorYawMergeTolerance = 1e-6;
std::vector<double> buildDescriptorYawHypotheses(
    const LoopCandidate& candidate, const Config& config);

class KeyframeManager;

class LoopDetector {
public:
    using Ptr = std::shared_ptr<LoopDetector>;
    using PointT = pcl::PointXYZI;
    using PointCloudT = pcl::PointCloud<PointT>;
    using MatchFilter = std::function<bool(int64_t)>;

    explicit LoopDetector(const Config& config);
    ~LoopDetector() = default;

    // Used by standalone map loading and the Atlas-backed core transaction.
    void restoreMapDescriptors(const KeyframeManager& keyframes,
                               const RHPDPlaceCache& submaps = {},
                               const std::map<int64_t, RHPDHeadingPtr>& headings = {});
    RHPDHeadingPtr computeHeading(const PointCloudT::Ptr& cloud);
    RHPDHeadingPtr addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud);
    void addDescriptor(int64_t keyframe_id, const RHPDHeadingPtr& descriptor);
    bool isHeadingShapeValid(const RHPDHeadingPtr& descriptor) const;
    // keyframes supplies the odometry poses the path-length exclusion needs;
    // pass the same map used for spatial candidates.
    // Retain the original top-k even if its predicted pose later changes, but
    // supplement it with one usable candidate if every original is out of range.
    std::vector<LoopCandidate> detectLoopCandidates(
        int64_t query_id,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match = {},
        const MatchFilter& within_work_bound = {});
    // Optional boundary reserve adds at most one extra top-k budget within a
    // registration voxel of the cutoff. Every candidate still obeys the radius
    // and caller eligibility filter; geometric verification remains mandatory.
    std::vector<LoopCandidate> detectSpatialCandidates(
        int64_t query_id,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match = {},
        bool include_boundary_ties = false) const;

    std::vector<std::pair<int64_t, RHPDHeadingPtr>> getDescriptors() const;
    void swapWith(LoopDetector& other);
    size_t size() const;
    void clear();
    std::pair<int, int> getDescriptorDimensions() const;
    double getHeadingSectorAngleDeg() const { return getRHPDParams().heading.sectorAngleDeg(); }
    RHPDHeadingPtr getDescriptor(int64_t keyframe_id) const;
    int getNumExcludeRecent() const { return config_.sc_num_exclude_recent; }
    std::pair<double, int> computeDistance(const RHPDHeadingPtr& sc1, const RHPDHeadingPtr& sc2);

    RHPDescriptor::Features computeFeatures(const PointCloudT::Ptr& cloud) const;

    // RHPD interface
    RHPDPlace computeRHPD(const PointCloudT::Ptr& cloud) const;
    RHPDPlace addRHPD(int64_t kf_id, const PointCloudT::Ptr& cloud);
    const RHPDManager& getRHPDManager() const { return rhpd_manager_; }
    const RHPDescriptor::Params& getRHPDParams() const { return rhpd_manager_.getDescriptorParams(); }

private:
    // Caller holds mutex_; shared ranking for the original and bounded search.
    std::vector<LoopCandidate> rankLoopCandidates(
        int64_t query_id, const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match);
    Config config_;
    RHPDManager rhpd_manager_;
    std::map<int64_t, size_t> id_to_index_;
    std::vector<int64_t> index_to_id_;
    std::vector<RHPDHeadingPtr> descriptors_;
    mutable std::mutex mutex_;

    void addDescriptorUnlocked(int64_t id, const RHPDHeadingPtr& descriptor);
};

} // namespace n3mapping
