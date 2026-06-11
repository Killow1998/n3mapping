#include "n3mapping/loop_closure_manager.h"

#include <algorithm>
#include <cmath>

namespace n3mapping {

namespace {

bool preferLoopCandidate(const VerifiedLoop& candidate, const VerifiedLoop& current, const Config& config)
{
    // Keep ICP fitness dominant; use vertical consistency only to break near ties.
    constexpr double kFitnessTieRatio = 1.05;
    if (candidate.fitness_score < current.fitness_score / kFitnessTieRatio) {
        return true;
    }
    if (candidate.fitness_score > current.fitness_score * kFitnessTieRatio) {
        return false;
    }
    if (config.loop_max_candidate_residual_z <= 0.0) {
        return candidate.fitness_score < current.fitness_score;
    }
    const double candidate_residual_z = std::abs(candidate.candidate_residual.translation().z());
    const double current_residual_z = std::abs(current.candidate_residual.translation().z());
    if (std::isfinite(candidate_residual_z) && std::isfinite(current_residual_z) &&
        candidate_residual_z != current_residual_z) {
        return candidate_residual_z < current_residual_z;
    }
    return candidate.fitness_score < current.fitness_score;
}

} // namespace

LoopClosureManager::LoopClosureManager(const Config& config)
  : config_(config)
{
}

std::vector<VerifiedLoop>
LoopClosureManager::filterValidLoops(const std::vector<VerifiedLoop>& loops) const
{
    std::vector<VerifiedLoop> valid;
    valid.reserve(loops.size());

    for (const auto& loop : loops) {
        if (!loop.verified) continue;
        if (loop.query_id < 0 || loop.match_id < 0) continue;
        if (loop.inlier_ratio < config_.loop_min_inlier_ratio) continue;
        if (loop.fitness_score > config_.loop_fitness_threshold) continue;
        const double residual_z = loop.candidate_residual.translation().z();
        if (config_.loop_max_candidate_residual_z > 0.0 &&
            (!std::isfinite(residual_z) ||
             std::abs(residual_z) > config_.loop_max_candidate_residual_z)) {
            continue;
        }
        valid.push_back(loop);
    }

    return valid;
}

std::vector<VerifiedLoop>
LoopClosureManager::selectBestPerQuery(const std::vector<VerifiedLoop>& loops) const
{
    std::unordered_map<int64_t, VerifiedLoop> best;
    for (const auto& loop : loops) {
        auto it = best.find(loop.query_id);
        if (it == best.end() || preferLoopCandidate(loop, it->second, config_)) {
            best[loop.query_id] = loop;
        }
    }

    std::vector<VerifiedLoop> result;
    result.reserve(best.size());
    for (const auto& kv : best) {
        result.push_back(kv.second);
    }

    return result;
}

std::vector<EdgeInfo>
LoopClosureManager::buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const
{
    std::vector<EdgeInfo> edges;
    edges.reserve(loops.size());

    for (const auto& loop : loops) {
        EdgeInfo edge;
        if (direction == LoopEdgeDirection::QueryToMatch) {
            edge.from_id = loop.query_id;
            edge.to_id = loop.match_id;
            edge.measurement = loop.T_match_query.inverse();
        } else {
            edge.from_id = loop.match_id;
            edge.to_id = loop.query_id;
            edge.measurement = loop.T_match_query;
        }
        edge.information = loop.information;
        edge.type = EdgeType::LOOP;
        edges.push_back(edge);
    }

    return edges;
}

bool
LoopClosureManager::applyEdges(const std::vector<EdgeInfo>& edges, LoopOptimizerInterface& optimizer) const
{
    if (edges.empty()) {
        return false;
    }

    for (const auto& edge : edges) {
        optimizer.addLoopEdge(edge);
    }

    return optimizer.incrementalOptimize();
}

} // namespace n3mapping
