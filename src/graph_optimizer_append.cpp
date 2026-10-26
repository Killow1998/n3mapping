#include <glog/logging.h>
#include "n3mapping/graph_optimizer.h"
#include "planar_geometry/moment_factor.h"
#include <iostream>
#include "graph_solver_execution.h"

namespace n3mapping {
namespace {
// Eliminate committed endpoints exactly from the small foreground solve.
// The original factor is retained unchanged in the global graph for the worker.
class ConditionedFactor : public gtsam::NoiseModelFactor {
  boost::shared_ptr<gtsam::NoiseModelFactor> factor_;
  gtsam::Values fixed_;
  std::vector<size_t> columns_;
public:
  ConditionedFactor(const gtsam::NonlinearFactor::shared_ptr& factor,
                    const gtsam::Values& committed) {
    factor_ = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factor);
    if (!factor_) throw std::invalid_argument("Cannot condition this graph factor");
    noiseModel_ = factor_->noiseModel();
    for (size_t i = 0; i < factor_->keys().size(); ++i) {
      const auto key = factor_->keys()[i];
      if (committed.exists(key)) fixed_.insert(key, committed.at(key));
      else { keys_.push_back(key); columns_.push_back(i); }
    }
  }
  gtsam::Vector unwhitenedError(const gtsam::Values& values,
      boost::optional<std::vector<gtsam::Matrix>&> H = boost::none) const override {
    auto full = fixed_;
    for (const auto key : keys_) full.insert(key, values.at(key));
    std::vector<gtsam::Matrix> jacobians(factor_->size());
    auto error = factor_->unwhitenedError(full, H ? boost::optional<std::vector<gtsam::Matrix>&>(jacobians) : boost::none);
    if (H) {
      H->resize(columns_.size());
      for (size_t i = 0; i < columns_.size(); ++i) (*H)[i] = jacobians[columns_[i]];
    }
    return error;
  }
};
} // namespace

bool GraphOptimizer::optimizeGeometryUpdate() {
  try {
    return internal::runGraphSolve(config_.num_threads, [&] {
      gtsam::LevenbergMarquardtParams params;
      params.maxIterations = 20;
      params.diagonalDamping = true;
      auto result = current_estimate_;
      if (!pending_node_ids_.empty()) {
        gtsam::NonlinearFactorGraph local;
        for (const auto& factor : new_factors_) {
          auto conditioned = boost::make_shared<ConditionedFactor>(factor, current_estimate_);
          // An old-to-old update belongs to the asynchronous global solve.
          if (conditioned->size() == 0)
            throw std::runtime_error("Old-node update in append transaction");
          local.push_back(conditioned);
        }
        const auto tail = gtsam::LevenbergMarquardtOptimizer(local, new_values_, params).optimize();
        if (!std::isfinite(local.error(tail))) throw std::runtime_error("Non-finite append cost");
        result.insert(tail);
      } else {
        auto candidate = graph_;
        appendPendingTo(&candidate, &result);
        result = gtsam::LevenbergMarquardtOptimizer(candidate, result, params).optimize();
        if (!std::isfinite(candidate.error(result))) throw std::runtime_error("Non-finite graph cost");
      }
      for (const auto id : pending_node_ids_)
        if (!result.at<gtsam::Pose3>(gtsam::Symbol('x', id)).matrix().allFinite())
          throw std::runtime_error("Non-finite appended pose");
      commitPending(result, createISAM2());
      return true;
    });
  } catch (const std::exception& e) {
    LOG(ERROR) << "Geometry graph update failed: " << e.what() << '\n';
    clearPending();
    needs_optimization_ = false;
    return false;
  }
}

bool GraphOptimizer::appendCommittedTail(const GraphOptimizer& live,
    const GraphOptimizer& snapshot,
    const std::map<int64_t, std::vector<PlaneMoment>>& body_moments) {
  if (snapshot.node_ids_.empty() || !new_factors_.empty() || !live.new_factors_.empty() ||
      live.graph_.size() < snapshot.graph_.size()) return false;
  for (size_t i = 0; i < snapshot.graph_.size(); ++i)
    if (snapshot.graph_[i] != live.graph_[i]) return false;
  for (int64_t id : snapshot.node_ids_) {
    if (!live.node_ids_.count(id) || !node_ids_.count(id) ||
        !live.getOptimizedPose(id).matrix().isApprox(
            snapshot.getOptimizedPose(id).matrix(), 1e-12)) return false;
  }
  for (int64_t id : live.node_ids_)
    if (!snapshot.node_ids_.count(id) && id <= *snapshot.node_ids_.rbegin()) return false;
  // Stage only new factors, retaining their noise and measurement semantics.
  for (size_t i = snapshot.graph_.size(); i < live.graph_.size(); ++i) {
    bool touches_new = false;
    for (auto key : live.graph_[i]->keys())
      touches_new |= !snapshot.current_estimate_.exists(key);
    if (!touches_new) return false;
  }
  std::map<int64_t, gtsam::NonlinearFactorGraph> factors;
  for (size_t i = snapshot.graph_.size(); i < live.graph_.size(); ++i) {
    const auto& factor = live.graph_[i];
    if (dynamic_cast<const planar_geometry::MomentPlaneFactor*>(factor.get())) continue;
    int64_t last = -1;
    for (auto key : factor->keys()) if (!snapshot.current_estimate_.exists(key)) {
      const gtsam::Symbol symbol(key);
      if (symbol.chr() != 'x' || !live.node_ids_.count(symbol.index())) return false;
      last = std::max(last, static_cast<int64_t>(symbol.index()));
    }
    factors[last].push_back(factor);
  }
  if (factors.empty()) return live.node_ids_ == snapshot.node_ids_;

  // Replay the tail with the same causal conditioning as foreground append.
  // A batch solve would let a later frame alter an earlier tail pose solely
  // because the worker finished later. Keep failure transactional as well.
  auto staged = cloneCommitted();
  const int64_t anchor = *snapshot.node_ids_.rbegin();
  const auto correction = getOptimizedPose(anchor) * live.getOptimizedPose(anchor).inverse();
  for (int64_t id : live.node_ids_) if (!snapshot.node_ids_.count(id)) {
    auto initial = correction * live.getOptimizedPose(id);
    for (size_t i = snapshot.edges_.size(); i < live.edges_.size(); ++i) {
      const auto& edge = live.edges_[i];
      if (edge.type == EdgeType::ODOMETRY && edge.to_id == id && staged->hasNode(edge.from_id)) {
        initial = staged->getOptimizedPose(edge.from_id) * edge.measurement;
        break;
      }
    }
    staged->pending_node_ids_.insert(id);
    staged->new_values_.insert(gtsam::Symbol('x', id), eigenToGtsam(initial));
    staged->new_factors_ = factors[id];
    for (size_t i = snapshot.edges_.size(); i < live.edges_.size(); ++i) {
      const auto& edge = live.edges_[i];
      if (std::max(edge.from_id, edge.to_id) == id) staged->pending_edges_.push_back(edge);
    }
    for (size_t i = snapshot.committed_floor_attitude_constraints_.size();
         i < live.committed_floor_attitude_constraints_.size(); ++i) {
      const auto& floor = live.committed_floor_attitude_constraints_[i];
      if (floor.node_id == id) staged->pending_floor_attitude_constraints_.push_back(floor);
    }
    // Re-evaluate all raw body patches, including old-map rejects. Reusing
    // admitted/merged patches or their corrected pose carries obsolete geometry.
    const auto moments = body_moments.find(id);
    if (moments != body_moments.end()) staged->appendPlaneObservations(id, moments->second);
    staged->pending_has_loop_closure_ = live.has_loop_closure_;
    staged->pending_has_session_anchor_ = live.has_session_anchor_;
    if (!staged->optimizeGeometryUpdate()) return false;
  }
  swapWith(*staged);
  return true;
}
} // namespace n3mapping
