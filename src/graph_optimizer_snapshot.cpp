#include "n3mapping/graph_optimizer.h"
#include <stdexcept>

namespace n3mapping {

std::unique_ptr<GraphOptimizer> GraphOptimizer::cloneCommitted() const {
    if (!new_factors_.empty() || !new_values_.empty() ||
        !pending_edges_.empty() || !pending_node_ids_.empty())
        throw std::logic_error("Cannot snapshot an uncommitted graph");
    auto result = std::make_unique<GraphOptimizer>(config_);
    result->graph_ = graph_;
    result->initial_values_ = initial_values_;
    result->current_estimate_ = current_estimate_;
    result->last_good_estimate_ = last_good_estimate_;
    result->edges_ = edges_;
    result->shared_planes_ = shared_planes_;
    result->node_ids_ = node_ids_;
    result->has_loop_closure_ = has_loop_closure_;
    result->has_session_anchor_ = has_session_anchor_;
    result->floor_attitude_factor_count_ = floor_attitude_factor_count_;
    result->committed_floor_attitude_constraints_ =
        committed_floor_attitude_constraints_;
    result->needs_optimization_ = needs_optimization_;
    // incrementalOptimize reconstructs a trial ISAM2 from graph_ and the
    // committed estimates. No mutable solver/Bayes-tree state is shared.
    return result;
}

} // namespace n3mapping
