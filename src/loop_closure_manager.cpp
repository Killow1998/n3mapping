// LoopClosureManager: filter, select best, build edges, and apply to optimizer.
#include "n3mapping/loop_closure_manager.h"

#include <algorithm>

namespace n3mapping {

LoopClosureManager::LoopClosureManager(const Config& config) : config_(config) {}

std::vector<VerifiedLoop> LoopClosureManager::filterValidLoops(const std::vector<VerifiedLoop>& loops) const {
    std::vector<VerifiedLoop> valid;
    for (const auto& loop : loops) {
        if (!loop.verified || loop.query_id < 0 || loop.match_id < 0) continue;
        if (loop.inlier_ratio < config_.loop_min_inlier_ratio) continue;
        if (loop.fitness_score > config_.loop_fitness_threshold) continue;
        valid.push_back(loop);
    }
    return valid;
}

std::vector<VerifiedLoop> LoopClosureManager::selectBestPerQuery(const std::vector<VerifiedLoop>& loops) const {
    std::unordered_map<int64_t, VerifiedLoop> best;
    for (const auto& loop : loops) {
        auto it = best.find(loop.query_id);
        if (it == best.end() || loop.fitness_score < it->second.fitness_score)
            best[loop.query_id] = loop;
    }
    std::vector<VerifiedLoop> result;
    result.reserve(best.size());
    for (const auto& kv : best) result.push_back(kv.second);
    return result;
}

std::vector<EdgeInfo> LoopClosureManager::buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const {
    std::vector<EdgeInfo> edges;
    for (const auto& loop : loops) {
        EdgeInfo edge;
        if (direction == LoopEdgeDirection::QueryToMatch) {
            edge.from_id = loop.query_id;
            edge.to_id = loop.match_id;
            edge.measurement = loop.T_match_query;
        } else {
            edge.from_id = loop.match_id;
            edge.to_id = loop.query_id;
            edge.measurement = loop.T_match_query.inverse();
        }
        edge.information = loop.information;
        edge.type = EdgeType::LOOP;
        edges.push_back(edge);
    }
    return edges;
}

bool LoopClosureManager::applyEdges(const std::vector<EdgeInfo>& edges, LoopOptimizerInterface& optimizer) const {
    if (edges.empty()) return false;
    for (const auto& edge : edges) optimizer.addLoopEdge(edge);
    optimizer.incrementalOptimize();
    return true;
}

} // namespace n3mapping
