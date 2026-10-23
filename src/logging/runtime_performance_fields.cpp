#include "n3mapping/runtime_performance_debug_logger.h"
#include "n3mapping/core/n3mapping_core.h"
#include "n3mapping/ros_output_contract.h"

namespace n3mapping {
void populateFramePerformance(RuntimePerformanceDebugEvent& event, const core::BackendOutput& output) {
  event.relocalization_performance = output.relocalization_performance;
  event.core_success = output.success;
  event.accepted_keyframe =
    output.accepted_keyframe;
  event.keyframe_id = output.keyframe_id;
  event.matched_keyframe_id =
    output.matched_keyframe_id;
  event.relocalization_state =
    relocalizationStateName(output.relocalization_state);
  event.pose_source =
    poseSourceName(output.pose_source);
  event.relocalization_decision =
    output.relocalization_decision;
  event.relocalization_locked =
    output.relocalization_locked;
  event.tracking_attempted =
    output.performance.tracking_attempted;
  event.initial_relocalization_ms =
    output.performance.initial_relocalization_ms;
  event.loaded_map_tracking_ms =
    output.performance.loaded_map_tracking_ms;
  event.keyframe_gate_ms =
    output.performance.keyframe_gate_ms;
  event.keyframe_commit_ms =
    output.performance.keyframe_commit_ms;
  event.graph_update_ms =
    output.performance.graph_update_ms;
  event.descriptor_update_ms =
    output.performance.descriptor_update_ms;
  event.post_commit_refresh_ms =
    output.performance.post_commit_refresh_ms;
}
void populateLoopPerformance(RuntimePerformanceLoopEvent& event, const CoreLoopClosureResult& result) {
  event.queued_keyframe_count =
    result.queued_keyframe_count;
  event.detected_candidate_count =
    result.detected_candidate_count;
  event.place_candidate_count =
    result.place_candidate_count;
  event.accepted_loop_count =
    result.accepted_loops.size();
  event.edge_count = result.edge_count;
  event.optimized = result.optimized;
  event.async_pending = result.async_pending;
  event.pending_query_count = result.pending_query_count;
  event.graph_rebase_count = result.graph_rebase_count;
  event.stale_job_count = result.stale_job_count;
  event.worker_wall_ms = result.worker_wall_ms;
  event.oldest_pending_age_ms = result.oldest_pending_age_ms;
  event.snapshot_ms = result.snapshot_ms;
  event.commit_ms = result.commit_ms;
  event.geometry_planes = result.geometry.planes;
  event.geometry_observations = result.geometry.observations;
  event.geometry_association_ms = result.geometry.association_ms;
  event.geometry_solve_ms = result.geometry.solve_ms;
  event.atlas_prepare_ms = result.atlas_prepare_ms;
}
} // namespace n3mapping
