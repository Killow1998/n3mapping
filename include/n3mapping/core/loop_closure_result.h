#pragma once

#include <cstddef>
#include <vector>
#include "n3mapping/loop_types.h"
#include "n3mapping/planar_geometry.h"

namespace n3mapping {

struct CoreLoopClosureResult {
    bool optimized = false;
    std::size_t queued_keyframe_count = 0;
    std::size_t detected_candidate_count = 0;
    std::size_t place_candidate_count = 0;
    std::size_t edge_count = 0;
    std::size_t graph_edge_count = 0;
    std::size_t pose_update_count = 0;
    double loop_residual_translation_before = 0.0;
    double loop_residual_translation_after = 0.0;
    double loop_residual_rotation_before = 0.0;
    double loop_residual_rotation_after = 0.0;
    double mean_pose_update_translation = 0.0;
    double max_pose_update_translation = 0.0;
    double mean_pose_update_rotation = 0.0;
    double max_pose_update_rotation = 0.0;
    std::vector<VerifiedLoop> accepted_loops;
    bool async_pending = false;
    std::size_t pending_query_count = 0;
    std::size_t graph_rebase_count = 0;
    std::size_t stale_job_count = 0;
    double worker_wall_ms = 0.0;
    double oldest_pending_age_ms = 0.0;
    double snapshot_ms = 0.0;
    double commit_ms = 0.0;
    GeometryOptimizationStats geometry;
    double atlas_prepare_ms = 0.0;
};

} // namespace n3mapping
