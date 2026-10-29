#include "graph_optimizer_state.h"
#include <stdexcept>

namespace n3mapping {

std::unique_ptr<GraphOptimizer> GraphOptimizer::cloneCommitted() const {
    if (!gtsam_->new_factors_.empty() || !gtsam_->new_values_.empty() ||
        !pending_edges_.empty() || !pending_node_ids_.empty())
        throw std::logic_error("Cannot snapshot an uncommitted graph");
    auto result = std::make_unique<GraphOptimizer>(config_);
    result->gtsam_->graph_ = gtsam_->graph_;
    result->gtsam_->initial_values_ = gtsam_->initial_values_;
    result->gtsam_->current_estimate_ = gtsam_->current_estimate_;
    result->gtsam_->last_good_estimate_ = gtsam_->last_good_estimate_;
    result->edges_ = edges_;
    result->shared_planes_ = shared_planes_;
    result->node_ids_ = node_ids_;
    result->has_loop_closure_ = has_loop_closure_;
    result->has_session_anchor_ = has_session_anchor_;
    result->floor_attitude_factor_count_ = floor_attitude_factor_count_;
    result->committed_floor_attitude_constraints_ =
        committed_floor_attitude_constraints_;
    result->needs_optimization_ = needs_optimization_;
    // incrementalOptimize reconstructs a trial ISAM2 from gtsam_->graph_ and the
    // committed estimates. No mutable solver/Bayes-tree state is shared.
    return result;
}

} // namespace n3mapping
