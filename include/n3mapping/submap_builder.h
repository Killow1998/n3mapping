// Shadow-only submap data scaffold. It does not participate in the pose graph.
#pragma once

#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/map_session.h"

namespace n3mapping {

using SubmapId = std::uint64_t;
constexpr SubmapId kInvalidSubmapId = std::numeric_limits<SubmapId>::max();

struct Submap {
    using PointCloudT = Keyframe::PointCloudT;

    SubmapId id = kInvalidSubmapId;
    MapSessionId session_id = kInvalidMapSessionId;
    std::vector<int64_t> keyframe_ids;
    Eigen::Isometry3d T_session_submap = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_map_submap = Eigen::Isometry3d::Identity();
    PointCloudT::Ptr registration_cloud;
    // The first scaffold intentionally aliases the same immutable snapshot;
    // later visualization filtering must use copy-on-write.
    PointCloudT::Ptr visualization_cloud;
    int64_t descriptor_keyframe_id = -1;
    std::uint64_t cloud_byte_budget = 0;
    std::uint64_t cloud_point_count = 0;
    bool cloud_truncated = false;
    bool closed = false;
    std::uint64_t content_revision = 0;

    std::uint64_t materializedCloudBytes() const;
};

struct SubmapBuilderOptions {
    bool enable = false;
    std::size_t max_keyframes = 20;
    std::uint64_t cloud_max_bytes = 16ull * 1024ull * 1024ull;
};

// Shadow comparison between the current keyframe graph and the rigid
// submap-origin decomposition. Residuals are observations only: they never
// update keyframe poses or graph constraints.
struct SubmapPoseProjectionDiagnostics {
    bool valid = false;
    std::string failure_reason;
    std::size_t submap_count = 0;
    std::size_t projected_keyframe_count = 0;
    std::size_t unassigned_keyframe_count = 0;
    std::size_t refreshed_submap_count = 0;
    double mean_translation_residual_m = 0.0;
    double max_translation_residual_m = 0.0;
    double mean_rotation_residual_rad = 0.0;
    double max_rotation_residual_rad = 0.0;
};

class SubmapBuilder {
public:
    explicit SubmapBuilder(const Config& config);
    explicit SubmapBuilder(SubmapBuilderOptions options);

    bool appendKeyframe(const Keyframe::Ptr& keyframe);
    bool closeActiveSubmap();
    bool loadSubmaps(const std::vector<Submap>& submaps,
                     const std::vector<Keyframe::Ptr>& keyframes,
                     const std::vector<MapSessionInfo>& sessions);
    bool materialize(SubmapId id,
                     const std::vector<Keyframe::Ptr>& keyframes);
    // Re-anchor every T_map_submap from its descriptor keyframe (or the first
    // member for legacy metadata), then report how well one rigid submap pose
    // projects all member keyframes. The update is all-or-nothing.
    SubmapPoseProjectionDiagnostics refreshMapPoses(
        const std::vector<Keyframe::Ptr>& keyframes);
    SubmapPoseProjectionDiagnostics evaluatePoseProjection(
        const std::vector<Keyframe::Ptr>& keyframes) const;
    std::vector<Submap> getSubmaps() const;
    void swapWith(SubmapBuilder& other);
    void clear();
    bool enabled() const { return options_.enable; }

private:
    bool closeActiveSubmapNoLock();
    bool materializeNoLock(Submap* submap,
                           const std::vector<Keyframe::Ptr>& keyframes,
                           bool verify_serialized_metadata);
    bool appendCloudNoLock(Submap* submap,
                           const Keyframe::Ptr& keyframe);

    SubmapBuilderOptions options_;
    std::vector<Submap> submaps_;
    SubmapId next_id_ = 0;
    mutable std::mutex mutex_;
};

}  // namespace n3mapping
