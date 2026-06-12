// LoopDetector: ScanContext-based loop candidate detection and ICP verification.
#pragma once

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
#include "n3mapping/keyframe.h"
#include "n3mapping/point_cloud_matcher.h"

namespace n3mapping {

struct LoopCandidate {
    static constexpr uint8_t SOURCE_SC = 1u;
    static constexpr uint8_t SOURCE_RHPD = 2u;
    enum class Source : uint8_t {
        Unknown = 0,
        RhpdPrimary = 1,
        ScanContextFallback = 2,
        RhpdFrame = 3
    };

    int64_t query_id = -1;
    int64_t match_id = -1;
    double rhpd_distance = std::numeric_limits<double>::max();
    double sc_distance = std::numeric_limits<double>::max();
    float yaw_diff_rad = 0.0f;
    uint8_t source_flags = 0u;
    Source candidate_source = Source::Unknown;
    double fused_score = std::numeric_limits<double>::max();
    int fused_rank = -1;
    int sc_rank = -1;
    int rhpd_rank = -1;
    bool isValid() const { return query_id >= 0 && match_id >= 0; }
    bool fromSC() const { return (source_flags & SOURCE_SC) != 0u; }
    bool fromRHPD() const { return (source_flags & SOURCE_RHPD) != 0u; }
};

enum class LoopEdgeMode {
    Full6Dof,
    PlanarXYYaw,
    RejectedVerticalInconsistent,
    RejectedYawInconsistent
};

inline const char* loopEdgeModeName(LoopEdgeMode mode)
{
    switch (mode) {
        case LoopEdgeMode::Full6Dof:
            return "full6dof";
        case LoopEdgeMode::PlanarXYYaw:
            return "planar_xy_yaw";
        case LoopEdgeMode::RejectedVerticalInconsistent:
            return "rejected_vertical_inconsistent";
        case LoopEdgeMode::RejectedYawInconsistent:
            return "rejected_yaw_inconsistent";
        default:
            return "unknown";
    }
}

struct VerifiedLoop {
    int64_t query_id = -1;
    int64_t match_id = -1;
    Eigen::Isometry3d T_match_query = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d candidate_residual = Eigen::Isometry3d::Identity();
    double candidate_yaw_diff_rad = 0.0;
    double fitness_score = std::numeric_limits<double>::max();
    double inlier_ratio = 0.0;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
    LoopEdgeMode edge_mode = LoopEdgeMode::Full6Dof;
    double vertical_observability_score = 1.0;
    bool vertical_downweighted = false;
    double source_z_span = std::numeric_limits<double>::quiet_NaN();
    double target_z_span = std::numeric_limits<double>::quiet_NaN();
    double z_overlap_ratio_before = std::numeric_limits<double>::quiet_NaN();
    double z_overlap_ratio_after = std::numeric_limits<double>::quiet_NaN();
    double source_z_robust_span = std::numeric_limits<double>::quiet_NaN();
    double target_z_robust_span = std::numeric_limits<double>::quiet_NaN();
    double z_robust_overlap_ratio_before = std::numeric_limits<double>::quiet_NaN();
    double z_robust_overlap_ratio_after = std::numeric_limits<double>::quiet_NaN();
    double source_target_z_centroid_delta_before = std::numeric_limits<double>::quiet_NaN();
    double source_target_z_centroid_delta_after = std::numeric_limits<double>::quiet_NaN();
    double vertical_information_ratio = std::numeric_limits<double>::quiet_NaN();
    int vertical_hypothesis_count = 0;
    double best_z_offset_m = std::numeric_limits<double>::quiet_NaN();
    double best_z_offset_fitness = std::numeric_limits<double>::quiet_NaN();
    double zero_z_fitness = std::numeric_limits<double>::quiet_NaN();
    double fitness_gap_zero_vs_best = std::numeric_limits<double>::quiet_NaN();
    double z_hypothesis_spread_m = std::numeric_limits<double>::quiet_NaN();
    double vertical_ambiguity_score = std::numeric_limits<double>::quiet_NaN();
    std::string vertical_hypothesis_edge_recommendation = "not_available";
    int heightmap_overlap_cell_count = 0;
    double heightmap_overlap_ratio = 0.0;
    double heightmap_ground_dz_median = std::numeric_limits<double>::quiet_NaN();
    double heightmap_ground_dz_p90 = std::numeric_limits<double>::quiet_NaN();
    double heightmap_ground_dz_max = std::numeric_limits<double>::quiet_NaN();
    double heightmap_ground_support_ratio = 0.0;
    double heightmap_vertical_consistency_score = 0.0;
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
    bool isScanContextDescriptorCompatible(const Eigen::MatrixXd& descriptor) const;
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
