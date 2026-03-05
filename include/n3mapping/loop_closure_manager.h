// LoopClosureManager: loop filtering, edge construction, and optimization triggering.
#pragma once

#include <unordered_map>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {

enum class LoopEdgeDirection { QueryToMatch, MatchToQuery };

class LoopClosureManager {
public:
    explicit LoopClosureManager(const Config& config);

    std::vector<VerifiedLoop> filterValidLoops(const std::vector<VerifiedLoop>& loops) const;
    std::vector<VerifiedLoop> selectBestPerQuery(const std::vector<VerifiedLoop>& loops) const;
    std::vector<EdgeInfo> buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const;
    bool applyEdges(const std::vector<EdgeInfo>& edges, LoopOptimizerInterface& optimizer) const;

private:
    Config config_;
};

} // namespace n3mapping
