#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
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

    void restoreMapDescriptors(const KeyframeManager& keyframes,
                               const RHPDPlaceCache& submaps = {},
                               const std::map<int64_t, RHPDHeadingPtr>& headings = {});
    void refreshMapDescriptors(const KeyframeManager& keyframes,
                               const std::set<int64_t>& changed_clouds);
    RHPDHeadingPtr computeHeading(const PointCloudT::Ptr& cloud);
    RHPDHeadingPtr addDescriptor(int64_t keyframe_id, const PointCloudT::Ptr& cloud);
    void addDescriptor(int64_t keyframe_id, const RHPDHeadingPtr& descriptor);
    bool isHeadingShapeValid(const RHPDHeadingPtr& descriptor) const;

    std::vector<LoopCandidate> detectLoopCandidates(
        int64_t query_id,
        const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match = {},
        const MatchFilter& within_work_bound = {});

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

    RHPDPlace computeRHPD(const PointCloudT::Ptr& cloud) const;
    RHPDPlace addRHPD(int64_t kf_id, const PointCloudT::Ptr& cloud);
    const RHPDManager& getRHPDManager() const { return rhpd_manager_; }
    const RHPDescriptor::Params& getRHPDParams() const { return rhpd_manager_.getDescriptorParams(); }

private:
    std::vector<LoopCandidate> rankLoopCandidates(
        int64_t query_id, const std::map<int64_t, Keyframe::Ptr>& keyframes,
        const MatchFilter& accept_match);
    Config config_;
    RHPDManager rhpd_manager_;
    std::map<int64_t, size_t> id_to_index_;
    std::vector<int64_t> index_to_id_;
    std::vector<RHPDHeadingPtr> descriptors_;
    mutable std::mutex mutex_;

    void addDescriptorUnlocked(int64_t id, const RHPDHeadingPtr& descriptor, bool replace = false);
};

}
