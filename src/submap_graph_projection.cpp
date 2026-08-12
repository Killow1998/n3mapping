#include "n3mapping/submap_graph_projection.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

namespace n3mapping {
namespace {

struct KeyframePoseSnapshot {
    MapSessionId session_id = kInvalidMapSessionId;
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
};

bool finiteRigidPose(const Eigen::Isometry3d& pose,
                     double tolerance = 1e-6) {
    if (!pose.matrix().allFinite()) return false;
    const Eigen::Matrix3d rotation = pose.linear();
    return (rotation.transpose() * rotation)
               .isApprox(Eigen::Matrix3d::Identity(), tolerance) &&
           std::abs(rotation.determinant() - 1.0) <= tolerance &&
           pose.matrix().row(3).isApprox(
               Eigen::RowVector4d(0.0, 0.0, 0.0, 1.0), tolerance);
}

bool validEdgeType(EdgeType type) {
    switch (type) {
        case EdgeType::ODOMETRY:
        case EdgeType::LOOP:
        case EdgeType::SESSION_ANCHOR:
            return true;
    }
    return false;
}

bool validConstraintMode(EdgeConstraintMode mode) {
    switch (mode) {
        case EdgeConstraintMode::FULL_6DOF:
        case EdgeConstraintMode::XY_YAW:
            return true;
    }
    return false;
}

SubmapGraphSnapshot invalidSnapshot(const std::string& reason) {
    SubmapGraphSnapshot snapshot;
    snapshot.failure_reason = reason;
    return snapshot;
}

SubmapGraphTopologyDiagnostics invalidTopology(const std::string& reason) {
    SubmapGraphTopologyDiagnostics diagnostics;
    diagnostics.failure_reason = reason;
    return diagnostics;
}

}  // namespace

SubmapGraphSnapshot buildSubmapGraphSnapshot(
    const std::vector<Submap>& submaps,
    const std::vector<Keyframe::Ptr>& keyframes,
    const std::vector<EdgeInfo>& edges,
    const std::vector<FloorAttitudeConstraint>& floor_constraints) {
    std::map<int64_t, KeyframePoseSnapshot> keyframe_by_id;
    for (const auto& keyframe : keyframes) {
        if (!keyframe || keyframe->id < 0 ||
            keyframe->session_id == kInvalidMapSessionId ||
            !finiteRigidPose(keyframe->pose_odom) ||
            !finiteRigidPose(keyframe->pose_optimized)) {
            return invalidSnapshot("invalid_keyframe_pose");
        }
        KeyframePoseSnapshot pose;
        pose.session_id = keyframe->session_id;
        pose.pose_odom = keyframe->pose_odom;
        if (!keyframe_by_id.emplace(keyframe->id, std::move(pose)).second) {
            return invalidSnapshot("duplicate_keyframe_id");
        }
    }

    SubmapGraphSnapshot snapshot;
    std::map<SubmapId, SubmapGraphNodeProjection> node_by_id;
    std::map<int64_t, Eigen::Isometry3d> T_submap_keyframe_by_id;
    for (const auto& submap : submaps) {
        if (submap.id == kInvalidSubmapId ||
            submap.session_id == kInvalidMapSessionId ||
            submap.keyframe_ids.empty() ||
            !finiteRigidPose(submap.T_session_submap) ||
            !finiteRigidPose(submap.T_map_submap)) {
            return invalidSnapshot("invalid_submap_pose");
        }
        if (node_by_id.find(submap.id) != node_by_id.end()) {
            return invalidSnapshot("duplicate_submap_id");
        }

        const int64_t anchor_id = submap.descriptor_keyframe_id >= 0
            ? submap.descriptor_keyframe_id
            : submap.keyframe_ids.front();
        if (std::find(submap.keyframe_ids.begin(), submap.keyframe_ids.end(),
                      anchor_id) == submap.keyframe_ids.end()) {
            return invalidSnapshot("invalid_submap_anchor");
        }
        const auto anchor = keyframe_by_id.find(anchor_id);
        if (anchor == keyframe_by_id.end() ||
            anchor->second.session_id != submap.session_id) {
            return invalidSnapshot("missing_submap_anchor");
        }

        SubmapGraphNodeProjection node;
        node.submap_id = submap.id;
        node.session_id = submap.session_id;
        node.anchor_keyframe_id = anchor_id;
        node.keyframe_count = submap.keyframe_ids.size();
        node.closed = submap.closed;
        node.content_revision = submap.content_revision;
        node.T_map_submap = submap.T_map_submap;
        node_by_id.emplace(node.submap_id, std::move(node));

        for (const int64_t keyframe_id : submap.keyframe_ids) {
            const auto keyframe = keyframe_by_id.find(keyframe_id);
            if (keyframe == keyframe_by_id.end() ||
                keyframe->second.session_id != submap.session_id) {
                return invalidSnapshot("missing_submap_keyframe");
            }
            if (!snapshot.keyframe_ownership
                     .emplace(keyframe_id, submap.id).second) {
                return invalidSnapshot("duplicate_submap_membership");
            }
            const Eigen::Isometry3d T_submap_keyframe =
                submap.T_session_submap.inverse() *
                keyframe->second.pose_odom;
            if (!finiteRigidPose(T_submap_keyframe)) {
                return invalidSnapshot("invalid_internal_keyframe_pose");
            }
            T_submap_keyframe_by_id.emplace(
                keyframe_id, T_submap_keyframe);
        }
    }

    snapshot.nodes.reserve(node_by_id.size());
    for (const auto& [id, node] : node_by_id) {
        (void)id;
        snapshot.nodes.push_back(node);
    }
    for (const auto& [keyframe_id, pose] : keyframe_by_id) {
        (void)pose;
        if (snapshot.keyframe_ownership.find(keyframe_id) ==
            snapshot.keyframe_ownership.end()) {
            snapshot.unassigned_keyframe_ids.push_back(keyframe_id);
        }
    }

    snapshot.source_edge_count = edges.size();
    snapshot.edge_projections.reserve(edges.size());
    double translation_residual_sum = 0.0;
    double rotation_residual_sum = 0.0;
    for (std::size_t index = 0; index < edges.size(); ++index) {
        const auto& edge = edges[index];
        if (edge.from_id < 0 || edge.to_id < 0 ||
            edge.from_id == edge.to_id ||
            !finiteRigidPose(edge.measurement) ||
            !edge.information.allFinite() ||
            !validEdgeType(edge.type) ||
            !validConstraintMode(edge.constraint_mode)) {
            return invalidSnapshot("invalid_source_edge");
        }
        if (keyframe_by_id.find(edge.from_id) == keyframe_by_id.end() ||
            keyframe_by_id.find(edge.to_id) == keyframe_by_id.end()) {
            return invalidSnapshot("missing_edge_keyframe");
        }

        SubmapGraphEdgeProjection projection;
        projection.source_edge_index = index;
        projection.source_edge = edge;
        const auto from_owner = snapshot.keyframe_ownership.find(edge.from_id);
        const auto to_owner = snapshot.keyframe_ownership.find(edge.to_id);
        if (from_owner != snapshot.keyframe_ownership.end()) {
            projection.from_submap_id = from_owner->second;
            projection.T_from_submap_from_keyframe =
                T_submap_keyframe_by_id.at(edge.from_id);
        }
        if (to_owner != snapshot.keyframe_ownership.end()) {
            projection.to_submap_id = to_owner->second;
            projection.T_to_submap_to_keyframe =
                T_submap_keyframe_by_id.at(edge.to_id);
        }
        if (from_owner == snapshot.keyframe_ownership.end() ||
            to_owner == snapshot.keyframe_ownership.end()) {
            projection.classification =
                SubmapGraphEdgeClass::UNASSIGNED_ENDPOINT;
            ++snapshot.unassigned_endpoint_edge_count;
            snapshot.edge_projections.push_back(std::move(projection));
            continue;
        }
        if (from_owner->second == to_owner->second) {
            projection.classification = SubmapGraphEdgeClass::INTRA_SUBMAP;
            ++snapshot.intra_submap_edge_count;
            snapshot.edge_projections.push_back(std::move(projection));
            continue;
        }

        projection.classification = SubmapGraphEdgeClass::CROSS_SUBMAP;
        projection.T_from_submap_to_submap_measurement =
            projection.T_from_submap_from_keyframe * edge.measurement *
            projection.T_to_submap_to_keyframe.inverse();
        const Eigen::Isometry3d predicted =
            node_by_id.at(projection.from_submap_id)
                    .T_map_submap.inverse() *
            node_by_id.at(projection.to_submap_id).T_map_submap;
        const Eigen::Isometry3d residual =
            projection.T_from_submap_to_submap_measurement.inverse() *
            predicted;
        if (!finiteRigidPose(projection.T_from_submap_to_submap_measurement) ||
            !finiteRigidPose(predicted) || !finiteRigidPose(residual)) {
            return invalidSnapshot("invalid_projected_edge");
        }
        projection.has_projected_measurement = true;
        projection.reference_translation_residual_m =
            residual.translation().norm();
        projection.reference_rotation_residual_rad = std::abs(
            Eigen::AngleAxisd(residual.rotation()).angle());
        if (!std::isfinite(projection.reference_translation_residual_m) ||
            !std::isfinite(projection.reference_rotation_residual_rad)) {
            return invalidSnapshot("nonfinite_projected_edge_residual");
        }
        translation_residual_sum +=
            projection.reference_translation_residual_m;
        rotation_residual_sum += projection.reference_rotation_residual_rad;
        snapshot.max_cross_edge_translation_residual_m = std::max(
            snapshot.max_cross_edge_translation_residual_m,
            projection.reference_translation_residual_m);
        snapshot.max_cross_edge_rotation_residual_rad = std::max(
            snapshot.max_cross_edge_rotation_residual_rad,
            projection.reference_rotation_residual_rad);
        ++snapshot.cross_submap_edge_count;
        switch (edge.type) {
            case EdgeType::ODOMETRY:
                ++snapshot.cross_submap_odometry_edge_count;
                break;
            case EdgeType::LOOP:
                ++snapshot.cross_submap_loop_edge_count;
                break;
            case EdgeType::SESSION_ANCHOR:
                ++snapshot.cross_submap_session_anchor_edge_count;
                break;
        }
        snapshot.edge_projections.push_back(std::move(projection));
    }
    if (snapshot.cross_submap_edge_count > 0) {
        const double denominator =
            static_cast<double>(snapshot.cross_submap_edge_count);
        snapshot.mean_cross_edge_translation_residual_m =
            translation_residual_sum / denominator;
        snapshot.mean_cross_edge_rotation_residual_rad =
            rotation_residual_sum / denominator;
    }

    snapshot.source_floor_constraint_count = floor_constraints.size();
    snapshot.floor_projections.reserve(floor_constraints.size());
    for (std::size_t index = 0; index < floor_constraints.size(); ++index) {
        const auto& constraint = floor_constraints[index];
        if (constraint.node_id < 0 || !constraint.normal_body.allFinite() ||
            constraint.normal_body.norm() < 1e-6 ||
            !std::isfinite(constraint.sigma_rad) ||
            constraint.sigma_rad <= 0.0) {
            return invalidSnapshot("invalid_floor_constraint");
        }
        if (keyframe_by_id.find(constraint.node_id) == keyframe_by_id.end()) {
            return invalidSnapshot("missing_floor_keyframe");
        }

        SubmapGraphFloorProjection projection;
        projection.source_constraint_index = index;
        projection.source_constraint = constraint;
        const auto owner =
            snapshot.keyframe_ownership.find(constraint.node_id);
        if (owner == snapshot.keyframe_ownership.end()) {
            ++snapshot.unassigned_floor_constraint_count;
        } else {
            projection.assigned = true;
            projection.submap_id = owner->second;
            projection.T_submap_keyframe =
                T_submap_keyframe_by_id.at(constraint.node_id);
            ++snapshot.assigned_floor_constraint_count;
        }
        snapshot.floor_projections.push_back(std::move(projection));
    }

    snapshot.valid = true;
    return snapshot;
}

SubmapGraphTopologyDiagnostics evaluateSubmapGraphTopology(
    const SubmapGraphSnapshot& snapshot) {
    if (!snapshot.valid) {
        return invalidTopology("invalid_snapshot");
    }

    std::map<SubmapId, MapSessionId> session_by_submap;
    std::map<SubmapId, std::set<SubmapId>> adjacency;
    std::size_t expected_owned_keyframes = 0;
    for (const auto& node : snapshot.nodes) {
        if (node.submap_id == kInvalidSubmapId ||
            node.session_id == kInvalidMapSessionId ||
            node.anchor_keyframe_id < 0 || node.keyframe_count == 0 ||
            !session_by_submap
                 .emplace(node.submap_id, node.session_id).second) {
            return invalidTopology("invalid_topology_node");
        }
        adjacency.emplace(node.submap_id, std::set<SubmapId>{});
        expected_owned_keyframes += node.keyframe_count;
    }
    if (expected_owned_keyframes != snapshot.keyframe_ownership.size()) {
        return invalidTopology("inconsistent_keyframe_ownership_count");
    }
    for (const auto& node : snapshot.nodes) {
        const auto anchor_owner =
            snapshot.keyframe_ownership.find(node.anchor_keyframe_id);
        if (anchor_owner == snapshot.keyframe_ownership.end() ||
            anchor_owner->second != node.submap_id) {
            return invalidTopology("invalid_topology_anchor_ownership");
        }
    }
    for (const auto& [keyframe_id, submap_id] :
         snapshot.keyframe_ownership) {
        if (keyframe_id < 0 ||
            session_by_submap.find(submap_id) == session_by_submap.end()) {
            return invalidTopology("invalid_topology_keyframe_ownership");
        }
    }

    std::set<int64_t> unassigned_keyframes;
    for (const int64_t keyframe_id : snapshot.unassigned_keyframe_ids) {
        if (keyframe_id < 0 ||
            snapshot.keyframe_ownership.find(keyframe_id) !=
                snapshot.keyframe_ownership.end() ||
            !unassigned_keyframes.insert(keyframe_id).second) {
            return invalidTopology("invalid_unassigned_keyframe_set");
        }
    }

    if (snapshot.source_edge_count != snapshot.edge_projections.size()) {
        return invalidTopology("inconsistent_source_edge_count");
    }
    std::size_t intra_edge_count = 0;
    std::size_t cross_edge_count = 0;
    std::size_t unassigned_edge_count = 0;
    std::vector<std::size_t> cross_projection_indices;
    std::size_t cross_session_edge_count = 0;
    for (std::size_t index = 0;
         index < snapshot.edge_projections.size(); ++index) {
        const auto& projection = snapshot.edge_projections[index];
        const auto& edge = projection.source_edge;
        if (projection.source_edge_index != index || edge.from_id < 0 ||
            edge.to_id < 0 || edge.from_id == edge.to_id) {
            return invalidTopology("invalid_topology_source_edge");
        }
        const auto from_owner =
            snapshot.keyframe_ownership.find(edge.from_id);
        const auto to_owner = snapshot.keyframe_ownership.find(edge.to_id);
        const bool from_assigned =
            from_owner != snapshot.keyframe_ownership.end();
        const bool to_assigned =
            to_owner != snapshot.keyframe_ownership.end();

        switch (projection.classification) {
            case SubmapGraphEdgeClass::INTRA_SUBMAP:
                if (!from_assigned || !to_assigned ||
                    from_owner->second != to_owner->second ||
                    projection.from_submap_id != from_owner->second ||
                    projection.to_submap_id != to_owner->second ||
                    projection.has_projected_measurement) {
                    return invalidTopology("invalid_intra_submap_projection");
                }
                ++intra_edge_count;
                break;
            case SubmapGraphEdgeClass::CROSS_SUBMAP: {
                if (!from_assigned || !to_assigned ||
                    from_owner->second == to_owner->second ||
                    projection.from_submap_id != from_owner->second ||
                    projection.to_submap_id != to_owner->second ||
                    !projection.has_projected_measurement ||
                    !finiteRigidPose(
                        projection.T_from_submap_to_submap_measurement)) {
                    return invalidTopology("invalid_cross_submap_projection");
                }
                const auto from_session =
                    session_by_submap.find(projection.from_submap_id);
                const auto to_session =
                    session_by_submap.find(projection.to_submap_id);
                if (from_session == session_by_submap.end() ||
                    to_session == session_by_submap.end()) {
                    return invalidTopology("missing_topology_edge_node");
                }
                adjacency.at(projection.from_submap_id)
                    .insert(projection.to_submap_id);
                adjacency.at(projection.to_submap_id)
                    .insert(projection.from_submap_id);
                cross_projection_indices.push_back(index);
                ++cross_edge_count;
                if (from_session->second != to_session->second) {
                    ++cross_session_edge_count;
                }
                break;
            }
            case SubmapGraphEdgeClass::UNASSIGNED_ENDPOINT:
                if ((from_assigned && to_assigned) ||
                    projection.has_projected_measurement ||
                    (from_assigned &&
                     projection.from_submap_id != from_owner->second) ||
                    (!from_assigned &&
                     projection.from_submap_id != kInvalidSubmapId) ||
                    (to_assigned &&
                     projection.to_submap_id != to_owner->second) ||
                    (!to_assigned &&
                     projection.to_submap_id != kInvalidSubmapId) ||
                    (!from_assigned &&
                     unassigned_keyframes.find(edge.from_id) ==
                         unassigned_keyframes.end()) ||
                    (!to_assigned &&
                     unassigned_keyframes.find(edge.to_id) ==
                         unassigned_keyframes.end())) {
                    return invalidTopology("invalid_unassigned_edge_projection");
                }
                ++unassigned_edge_count;
                break;
            default:
                return invalidTopology("invalid_edge_projection_class");
        }
    }
    if (intra_edge_count != snapshot.intra_submap_edge_count ||
        cross_edge_count != snapshot.cross_submap_edge_count ||
        unassigned_edge_count !=
            snapshot.unassigned_endpoint_edge_count) {
        return invalidTopology("inconsistent_projected_edge_counts");
    }

    if (snapshot.source_floor_constraint_count !=
        snapshot.floor_projections.size()) {
        return invalidTopology("inconsistent_source_floor_count");
    }
    std::size_t assigned_floor_count = 0;
    std::size_t unassigned_floor_count = 0;
    for (std::size_t index = 0;
         index < snapshot.floor_projections.size(); ++index) {
        const auto& projection = snapshot.floor_projections[index];
        const auto& constraint = projection.source_constraint;
        if (projection.source_constraint_index != index ||
            constraint.node_id < 0) {
            return invalidTopology("invalid_topology_floor_constraint");
        }
        const auto owner =
            snapshot.keyframe_ownership.find(constraint.node_id);
        if (projection.assigned) {
            if (owner == snapshot.keyframe_ownership.end() ||
                projection.submap_id != owner->second ||
                session_by_submap.find(projection.submap_id) ==
                    session_by_submap.end()) {
                return invalidTopology("invalid_assigned_floor_projection");
            }
            ++assigned_floor_count;
        } else {
            if (owner != snapshot.keyframe_ownership.end() ||
                projection.submap_id != kInvalidSubmapId ||
                unassigned_keyframes.find(constraint.node_id) ==
                    unassigned_keyframes.end()) {
                return invalidTopology("invalid_unassigned_floor_projection");
            }
            ++unassigned_floor_count;
        }
    }
    if (assigned_floor_count !=
            snapshot.assigned_floor_constraint_count ||
        unassigned_floor_count !=
            snapshot.unassigned_floor_constraint_count) {
        return invalidTopology("inconsistent_projected_floor_counts");
    }

    SubmapGraphTopologyDiagnostics diagnostics;
    diagnostics.node_count = snapshot.nodes.size();
    diagnostics.cross_edge_count = cross_edge_count;
    diagnostics.cross_session_edge_count = cross_session_edge_count;
    for (const auto& [submap_id, neighbors] : adjacency) {
        if (neighbors.empty()) {
            diagnostics.isolated_submap_ids.push_back(submap_id);
        }
    }
    diagnostics.isolated_submap_count =
        diagnostics.isolated_submap_ids.size();

    std::set<SubmapId> unvisited;
    for (const auto& [submap_id, neighbors] : adjacency) {
        (void)neighbors;
        unvisited.insert(submap_id);
    }
    while (!unvisited.empty()) {
        SubmapGraphTopologyComponent component;
        component.component_index = diagnostics.components.size();
        std::vector<SubmapId> pending{*unvisited.begin()};
        unvisited.erase(pending.front());
        std::set<MapSessionId> component_sessions;
        while (!pending.empty()) {
            const SubmapId submap_id = pending.back();
            pending.pop_back();
            component.submap_ids.push_back(submap_id);
            component_sessions.insert(session_by_submap.at(submap_id));
            for (const SubmapId neighbor : adjacency.at(submap_id)) {
                if (unvisited.erase(neighbor) > 0) {
                    pending.push_back(neighbor);
                }
            }
        }
        std::sort(component.submap_ids.begin(), component.submap_ids.end());
        component.anchor_submap_id = component.submap_ids.front();
        component.session_ids.assign(component_sessions.begin(),
                                     component_sessions.end());
        for (const SubmapId submap_id : component.submap_ids) {
            diagnostics.component_by_submap.emplace(
                submap_id, component.component_index);
        }
        diagnostics.components.push_back(std::move(component));
    }
    for (const std::size_t projection_index : cross_projection_indices) {
        const auto& projection =
            snapshot.edge_projections[projection_index];
        const std::size_t from_component =
            diagnostics.component_by_submap.at(projection.from_submap_id);
        const std::size_t to_component =
            diagnostics.component_by_submap.at(projection.to_submap_id);
        if (from_component != to_component) {
            return invalidTopology("inconsistent_topology_component");
        }
        diagnostics.components[from_component]
            .cross_edge_projection_indices.push_back(projection_index);
    }

    diagnostics.component_count = diagnostics.components.size();
    diagnostics.keyframe_ownership_complete =
        snapshot.unassigned_keyframe_ids.empty();
    diagnostics.constraint_coverage_complete =
        snapshot.unassigned_endpoint_edge_count == 0 &&
        snapshot.unassigned_floor_constraint_count == 0;
    diagnostics.connected =
        diagnostics.node_count > 0 && diagnostics.component_count == 1;
    diagnostics.shadow_graph_ready =
        diagnostics.connected && diagnostics.keyframe_ownership_complete &&
        diagnostics.constraint_coverage_complete;
    diagnostics.valid = true;
    return diagnostics;
}

}  // namespace n3mapping
