#include "n3mapping/loop_closure_manager.h"

#include <cmath>

namespace n3mapping {

namespace {

bool preferLoopCandidate(const VerifiedLoop& candidate, const VerifiedLoop& current, const Config& config)
{
    (void)config;
    if (std::isfinite(candidate.loop_referee_energy) &&
        std::isfinite(current.loop_referee_energy) &&
        candidate.loop_referee_energy != current.loop_referee_energy) {
        return candidate.loop_referee_energy > current.loop_referee_energy;
    }
    if (candidate.fitness_score != current.fitness_score) {
        return candidate.fitness_score < current.fitness_score;
    }
    return candidate.match_id < current.match_id;
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

VerifiedLoop
LoopClosureManager::applyEdgeModel(const VerifiedLoop& loop) const
{
    VerifiedLoop modeled = loop;
    modeled.edge_mode = LoopEdgeMode::Full6Dof;
    modeled.vertical_downweighted = false;
    modeled.vertical_observability_score = 1.0;
    return modeled;
}

std::vector<EdgeInfo>
LoopClosureManager::buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const
{
    std::vector<EdgeInfo> edges;
    edges.reserve(loops.size());

    for (const auto& loop : loops) {
        if (!loop.verified) {
            continue;
        }
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
