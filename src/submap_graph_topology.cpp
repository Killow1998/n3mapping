#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/rotation_utils.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

namespace n3mapping {
namespace {

SubmapGraphTopologyDiagnostics invalidTopology(const std::string& reason) {
    SubmapGraphTopologyDiagnostics diagnostics;
    diagnostics.failure_reason = reason;
    return diagnostics;
}

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
    if (expected_owned_keyframes != snapshot.keyframe_ownership.size() ||
        expected_owned_keyframes != snapshot.keyframe_projections.size()) {
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
    int64_t previous_projection_id = -1;
    std::map<int64_t, const SubmapGraphKeyframeProjection*>
        projection_by_keyframe;
    for (const auto& projection : snapshot.keyframe_projections) {
        const auto owner = snapshot.keyframe_ownership.find(
            projection.keyframe_id);
        const auto session = session_by_submap.find(projection.submap_id);
        if (projection.keyframe_id < 0 ||
            projection.keyframe_id <= previous_projection_id ||
            projection.submap_id == kInvalidSubmapId ||
            projection.session_id == kInvalidMapSessionId ||
            owner == snapshot.keyframe_ownership.end() ||
            owner->second != projection.submap_id ||
            session == session_by_submap.end() ||
            session->second != projection.session_id ||
            !isFiniteRigidPose(projection.T_submap_keyframe) ||
            !isFiniteRigidPose(projection.T_map_keyframe_reference)) {
            return invalidTopology("invalid_keyframe_projection");
        }
        projection_by_keyframe.emplace(projection.keyframe_id, &projection);
        previous_projection_id = projection.keyframe_id;
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
        if ((from_assigned &&
             (!isFiniteRigidPose(
                  projection.T_from_submap_from_keyframe) ||
              !projection.T_from_submap_from_keyframe.matrix().isApprox(
                  projection_by_keyframe.at(edge.from_id)
                      ->T_submap_keyframe.matrix(),
                  1e-12))) ||
            (to_assigned &&
             (!isFiniteRigidPose(
                  projection.T_to_submap_to_keyframe) ||
              !projection.T_to_submap_to_keyframe.matrix().isApprox(
                  projection_by_keyframe.at(edge.to_id)
                      ->T_submap_keyframe.matrix(),
                  1e-12)))) {
            return invalidTopology("inconsistent_edge_keyframe_projection");
        }

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
                    !isFiniteRigidPose(
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
                    session_by_submap.end() ||
                !isFiniteRigidPose(projection.T_submap_keyframe) ||
                !projection.T_submap_keyframe.matrix().isApprox(
                    projection_by_keyframe.at(constraint.node_id)
                        ->T_submap_keyframe.matrix(),
                    1e-12)) {
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

}
