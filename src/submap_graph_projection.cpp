#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/rotation_utils.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace n3mapping {
namespace {

struct KeyframePoseSnapshot {
    MapSessionId session_id = kInvalidMapSessionId;
    Eigen::Isometry3d pose_odom = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d pose_optimized = Eigen::Isometry3d::Identity();
};

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

}

SubmapGraphSnapshot buildSubmapGraphSnapshot(
    const std::vector<Submap>& submaps,
    const std::vector<Keyframe::Ptr>& keyframes,
    const std::vector<EdgeInfo>& edges,
    const std::vector<FloorAttitudeConstraint>& floor_constraints) {
    std::map<int64_t, KeyframePoseSnapshot> keyframe_by_id;
    for (const auto& keyframe : keyframes) {
        if (!keyframe || keyframe->id < 0 ||
            keyframe->session_id == kInvalidMapSessionId ||
            !isFiniteRigidPose(keyframe->pose_odom) ||
            !isFiniteRigidPose(keyframe->pose_optimized)) {
            return invalidSnapshot("invalid_keyframe_pose");
        }
        KeyframePoseSnapshot pose;
        pose.session_id = keyframe->session_id;
        pose.pose_odom = keyframe->pose_odom;
        pose.pose_optimized = keyframe->pose_optimized;
        if (!keyframe_by_id.emplace(keyframe->id, std::move(pose)).second) {
            return invalidSnapshot("duplicate_keyframe_id");
        }
    }

    SubmapGraphSnapshot snapshot;
    std::map<SubmapId, SubmapGraphNodeProjection> node_by_id;
    std::map<int64_t, Eigen::Isometry3d> T_submap_keyframe_by_id;
    std::map<int64_t, SubmapGraphKeyframeProjection>
        keyframe_projection_by_id;
    for (const auto& submap : submaps) {
        if (submap.id == kInvalidSubmapId ||
            submap.session_id == kInvalidMapSessionId ||
            submap.keyframe_ids.empty() ||
            !isFiniteRigidPose(submap.T_session_submap) ||
            !isFiniteRigidPose(submap.T_map_submap)) {
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
            if (!isFiniteRigidPose(T_submap_keyframe)) {
                return invalidSnapshot("invalid_internal_keyframe_pose");
            }
            T_submap_keyframe_by_id.emplace(
                keyframe_id, T_submap_keyframe);
            SubmapGraphKeyframeProjection projection;
            projection.keyframe_id = keyframe_id;
            projection.submap_id = submap.id;
            projection.session_id = submap.session_id;
            projection.T_submap_keyframe = T_submap_keyframe;
            projection.T_map_keyframe_reference =
                keyframe->second.pose_optimized;
            if (!keyframe_projection_by_id
                     .emplace(keyframe_id, std::move(projection)).second) {
                return invalidSnapshot("duplicate_keyframe_projection");
            }
        }
    }

    snapshot.nodes.reserve(node_by_id.size());
    for (const auto& [id, node] : node_by_id) {
        (void)id;
        snapshot.nodes.push_back(node);
    }
    for (const auto& [keyframe_id, pose] : keyframe_by_id) {
        (void)pose;
        const auto projection = keyframe_projection_by_id.find(keyframe_id);
        if (projection == keyframe_projection_by_id.end()) {
            snapshot.unassigned_keyframe_ids.push_back(keyframe_id);
        } else {
            snapshot.keyframe_projections.push_back(projection->second);
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
            !isFiniteRigidPose(edge.measurement) ||
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
        if (!isFiniteRigidPose(projection.T_from_submap_to_submap_measurement) ||
            !isFiniteRigidPose(predicted) || !isFiniteRigidPose(residual)) {
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

}
