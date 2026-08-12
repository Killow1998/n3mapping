// Shadow-only projection of the reference keyframe graph onto submap owners.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "n3mapping/graph_optimizer.h"
#include "n3mapping/submap_builder.h"

namespace n3mapping {

enum class SubmapGraphEdgeClass {
    INTRA_SUBMAP,
    CROSS_SUBMAP,
    UNASSIGNED_ENDPOINT,
};

struct SubmapGraphNodeProjection {
    SubmapId submap_id = kInvalidSubmapId;
    MapSessionId session_id = kInvalidMapSessionId;
    int64_t anchor_keyframe_id = -1;
    std::size_t keyframe_count = 0;
    bool closed = false;
    std::uint64_t content_revision = 0;
    Eigen::Isometry3d T_map_submap = Eigen::Isometry3d::Identity();
};

struct SubmapGraphEdgeProjection {
    std::size_t source_edge_index = 0;
    // This is the exact reference keyframe edge. In particular, information
    // remains in the source keyframe-factor tangent space; it is deliberately
    // not advertised as a projected submap information matrix.
    EdgeInfo source_edge;
    SubmapGraphEdgeClass classification =
        SubmapGraphEdgeClass::UNASSIGNED_ENDPOINT;
    SubmapId from_submap_id = kInvalidSubmapId;
    SubmapId to_submap_id = kInvalidSubmapId;
    Eigen::Isometry3d T_from_submap_from_keyframe =
        Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_to_submap_to_keyframe =
        Eigen::Isometry3d::Identity();
    bool has_projected_measurement = false;
    Eigen::Isometry3d T_from_submap_to_submap_measurement =
        Eigen::Isometry3d::Identity();
    double reference_translation_residual_m = 0.0;
    double reference_rotation_residual_rad = 0.0;
};

struct SubmapGraphFloorProjection {
    std::size_t source_constraint_index = 0;
    FloorAttitudeConstraint source_constraint;
    bool assigned = false;
    SubmapId submap_id = kInvalidSubmapId;
    Eigen::Isometry3d T_submap_keyframe =
        Eigen::Isometry3d::Identity();
};

// Immutable, optimizer-free snapshot. It preserves every source edge/floor
// observation and makes omissions explicit. No value in this structure is
// written back to the keyframe graph, submaps, ranking, lock, or map output.
struct SubmapGraphSnapshot {
    bool valid = false;
    std::string failure_reason;
    std::vector<SubmapGraphNodeProjection> nodes;
    std::map<int64_t, SubmapId> keyframe_ownership;
    std::vector<int64_t> unassigned_keyframe_ids;
    std::vector<SubmapGraphEdgeProjection> edge_projections;
    std::vector<SubmapGraphFloorProjection> floor_projections;

    std::size_t source_edge_count = 0;
    std::size_t intra_submap_edge_count = 0;
    std::size_t cross_submap_edge_count = 0;
    std::size_t unassigned_endpoint_edge_count = 0;
    std::size_t cross_submap_odometry_edge_count = 0;
    std::size_t cross_submap_loop_edge_count = 0;
    std::size_t cross_submap_session_anchor_edge_count = 0;

    std::size_t source_floor_constraint_count = 0;
    std::size_t assigned_floor_constraint_count = 0;
    std::size_t unassigned_floor_constraint_count = 0;

    double mean_cross_edge_translation_residual_m = 0.0;
    double max_cross_edge_translation_residual_m = 0.0;
    double mean_cross_edge_rotation_residual_rad = 0.0;
    double max_cross_edge_rotation_residual_rad = 0.0;
};

struct SubmapGraphTopologyComponent {
    std::size_t component_index = 0;
    SubmapId anchor_submap_id = kInvalidSubmapId;
    std::vector<SubmapId> submap_ids;
    std::vector<MapSessionId> session_ids;
    // Indices into SubmapGraphSnapshot::edge_projections. Parallel source
    // constraints remain separate entries and are never fused here.
    std::vector<std::size_t> cross_edge_projection_indices;
};

// Structural qualification for a future shadow optimizer. shadow_graph_ready
// is necessary, not sufficient: it says nothing about gauge handling,
// information transport, numerical observability, or integration authority.
struct SubmapGraphTopologyDiagnostics {
    bool valid = false;
    std::string failure_reason;
    bool keyframe_ownership_complete = false;
    bool constraint_coverage_complete = false;
    bool connected = false;
    bool shadow_graph_ready = false;
    std::size_t node_count = 0;
    std::size_t cross_edge_count = 0;
    std::size_t component_count = 0;
    std::size_t isolated_submap_count = 0;
    std::size_t cross_session_edge_count = 0;
    std::vector<SubmapId> isolated_submap_ids;
    std::map<SubmapId, std::size_t> component_by_submap;
    std::vector<SubmapGraphTopologyComponent> components;
};

SubmapGraphSnapshot buildSubmapGraphSnapshot(
    const std::vector<Submap>& submaps,
    const std::vector<Keyframe::Ptr>& keyframes,
    const std::vector<EdgeInfo>& edges,
    const std::vector<FloorAttitudeConstraint>& floor_constraints = {});

SubmapGraphTopologyDiagnostics evaluateSubmapGraphTopology(
    const SubmapGraphSnapshot& snapshot);

}  // namespace n3mapping
