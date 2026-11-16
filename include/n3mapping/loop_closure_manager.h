#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/graph_types.h"
#include "n3mapping/loop_types.h"
#include "n3mapping/keyframe.h"

namespace n3mapping {
enum class LoopEdgeDirection
{
    QueryToMatch,
    MatchToQuery
};

struct SameQueryLoopSelection
{
    std::vector<VerifiedLoop> selected;
    std::map<std::pair<int64_t, int64_t>, std::string> rejected;
};

class LoopClosureManager
{
  public:
    explicit LoopClosureManager(const Config& config);
    std::vector<VerifiedLoop> filterValidLoops(const std::vector<VerifiedLoop>& loops) const;

    std::vector<VerifiedLoop> selectBestPerQuery(const std::vector<VerifiedLoop>& loops) const;

    SameQueryLoopSelection selectSameQueryConsensus(
        const std::vector<VerifiedLoop>& loops,
        const std::map<int64_t, Keyframe::Ptr>& keyframes) const;
    std::vector<EdgeInfo> buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const;

    VerifiedLoop applyEdgeModel(const VerifiedLoop& loop) const;

    bool applyEdges(const std::vector<EdgeInfo>& edges, LoopOptimizerInterface& optimizer) const;

  private:
    Config config_;
};

}
