#include "n3mapping/mapping_resuming.h"
#include "n3mapping/rotation_utils.h"
#include <glog/logging.h>
#include "n3mapping/submap_graph_projection.h"
#include "n3mapping/submap_graph_factor.h"
#include "n3mapping/submap_graph_trial_runtime.h"

namespace n3mapping {

bool MappingResuming::refreshSubmapPosesNoLock(const char* context) {
    if (!submap_builder_ || !submap_builder_->enabled()) {
        return true;
    }
    const auto keyframes = keyframe_manager_.getAllKeyframes();
    const auto diagnostics = submap_builder_->refreshMapPoses(keyframes);
    if (!diagnostics.valid) {
        LOG(WARNING) << "[MappingResuming][SubmapShadow] pose projection refresh failed context="
                     << (context ? context : "unknown")
                     << " reason=" << diagnostics.failure_reason;
        return false;
    }
    VLOG(1) << "[MappingResuming][SubmapShadow] pose projection context="
            << (context ? context : "unknown")
            << " submaps=" << diagnostics.submap_count
            << " projected_keyframes="
            << diagnostics.projected_keyframe_count
            << " unassigned_keyframes="
            << diagnostics.unassigned_keyframe_count
            << " refreshed_submaps="
            << diagnostics.refreshed_submap_count
            << " max_translation_residual_m="
            << diagnostics.max_translation_residual_m
            << " max_rotation_residual_rad="
            << diagnostics.max_rotation_residual_rad;

    const auto graph_snapshot = buildSubmapGraphSnapshot(
        submap_builder_->getSubmaps(), keyframes, optimizer_.getEdges(),
        optimizer_.floorAttitudeConstraints());
    if (!graph_snapshot.valid) {
        // Shadow graph construction cannot veto the established keyframe graph.
        LOG(WARNING) << "[MappingResuming][SubmapGraphShadow] snapshot failed context="
                     << (context ? context : "unknown")
                     << " reason=" << graph_snapshot.failure_reason;
    } else {
        VLOG(1) << "[MappingResuming][SubmapGraphShadow] snapshot context="
                << (context ? context : "unknown")
                << " nodes=" << graph_snapshot.nodes.size()
                << " source_edges=" << graph_snapshot.source_edge_count
                << " intra_edges="
                << graph_snapshot.intra_submap_edge_count
                << " cross_edges="
                << graph_snapshot.cross_submap_edge_count
                << " unassigned_edges="
                << graph_snapshot.unassigned_endpoint_edge_count
                << " floor_assigned="
                << graph_snapshot.assigned_floor_constraint_count
                << " floor_unassigned="
                << graph_snapshot.unassigned_floor_constraint_count
                << " max_cross_translation_residual_m="
                << graph_snapshot.max_cross_edge_translation_residual_m
                << " max_cross_rotation_residual_rad="
                << graph_snapshot.max_cross_edge_rotation_residual_rad;
        const auto topology = evaluateSubmapGraphTopology(graph_snapshot);
        if (!topology.valid) {
            LOG(WARNING) << "[MappingResuming][SubmapGraphShadow] topology failed context="
                         << (context ? context : "unknown")
                         << " reason=" << topology.failure_reason;
        } else {
            VLOG(1) << "[MappingResuming][SubmapGraphShadow] topology context="
                    << (context ? context : "unknown")
                    << " nodes=" << topology.node_count
                    << " components=" << topology.component_count
                    << " isolated_submaps="
                    << topology.isolated_submap_count
                    << " cross_edges=" << topology.cross_edge_count
                    << " cross_session_edges="
                    << topology.cross_session_edge_count
                    << " keyframe_ownership_complete="
                    << topology.keyframe_ownership_complete
                    << " constraint_coverage_complete="
                    << topology.constraint_coverage_complete
                    << " connected=" << topology.connected
                    << " shadow_graph_ready="
                    << topology.shadow_graph_ready;
        }
        const auto factors = evaluateSubmapGraphFactors(graph_snapshot);
        if (!factors.valid) {
            LOG(WARNING) << "[MappingResuming][SubmapGraphShadow] factor semantics failed context="
                         << (context ? context : "unknown")
                         << " reason=" << factors.failure_reason;
        } else {
            VLOG(1) << "[MappingResuming][SubmapGraphShadow] factor semantics context="
                    << (context ? context : "unknown")
                    << " cross_edges=" << factors.cross_edge_count
                    << " full_6d_edges=" << factors.full_6d_edge_count
                    << " xy_yaw_lifted_only="
                    << factors.xy_yaw_exact_lifted_only_count
                    << " direct_between_residual_equivalent="
                    << factors.direct_between_residual_equivalent_count
                    << " information_transportable="
                    << factors.direct_between_information_transportable_count
                    << " information_fallback_required="
                    << factors.information_fallback_required_count
                    << " assigned_floor_factors="
                    << factors.assigned_floor_factor_count
                    << " max_residual_transport_error="
                    << factors.max_residual_transport_error_norm
                    << " max_mahalanobis_squared_delta="
                    << factors.max_mahalanobis_squared_delta
                    << " max_floor_residual_error="
                    << factors.max_floor_residual_error_norm;
        }
    }

    const std::string context_name = context ? context : "unknown";
    const auto trial = runSubmapGraphTrialCheckpoint(
        graph_snapshot, config_, "mapping_resuming", context_name);
    if (trial.checkpoint) {
        if (!trial.persisted) {
            LOG(WARNING) << "[MappingResuming][SubmapGraphShadow] trial evidence write failed context="
                         << context_name << " path=" << trial.output_path;
        }
        LOG(INFO) << "[MappingResuming][SubmapGraphShadow] trial checkpoint context="
                  << context_name
                  << " valid=" << trial.diagnostics.valid
                  << " attempted=" << trial.diagnostics.attempted
                  << " solved=" << trial.diagnostics.solved
                  << " nodes=" << trial.diagnostics.node_count
                  << " active_factors="
                  << trial.diagnostics.active_edge_factor_count
                  << " initial_error="
                  << trial.diagnostics.initial_nonlinear_error
                  << " final_error="
                  << trial.diagnostics.final_nonlinear_error
                  << " max_translation_delta_m="
                  << trial.diagnostics.max_translation_delta_m
                  << " max_rotation_delta_rad="
                  << trial.diagnostics.max_rotation_delta_rad
                  << " optimized_keyframe_p95_translation_error_m="
                  << trial.diagnostics.optimized_keyframe_comparison
                         .p95_translation_error_m
                  << " optimized_keyframe_max_translation_error_m="
                  << trial.diagnostics.optimized_keyframe_comparison
                         .max_translation_error_m
                  << " optimized_keyframe_p95_rotation_error_rad="
                  << trial.diagnostics.optimized_keyframe_comparison
                         .p95_rotation_error_rad
                  << " persisted=" << trial.persisted
                  << " failure_reason="
                  << (trial.diagnostics.failure_reason.empty()
                          ? "none"
                          : trial.diagnostics.failure_reason);
    }
    return true;
}

}  // namespace n3mapping
