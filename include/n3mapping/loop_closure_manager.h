#pragma once

#include <unordered_map>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/loop_detector.h"

namespace n3mapping {

/**
 * @brief 回环边方向
 */
enum class LoopEdgeDirection
{
    QueryToMatch,
    MatchToQuery
};

/**
 * @brief 回环优化管理器
 *
 * 负责回环验证结果的筛选、回环边构建与优化触发
 */
class LoopClosureManager
{
  public:
    explicit LoopClosureManager(const Config& config);

    /**
     * @brief 过滤并保留有效回环（阈值筛选）
     */
    std::vector<VerifiedLoop> filterValidLoops(const std::vector<VerifiedLoop>& loops) const;

    /**
     * @brief 每个 query 仅保留 fitness 最优的回环
     */
    std::vector<VerifiedLoop> selectBestPerQuery(const std::vector<VerifiedLoop>& loops) const;

    /**
     * @brief 构建回环边
     */
    std::vector<EdgeInfo> buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const;

    /**
     * @brief 将回环边应用到图优化器，并触发增量优化
     * @return true 若触发优化
     */
    bool applyEdges(const std::vector<EdgeInfo>& edges, LoopOptimizerInterface& optimizer) const;

  private:
    Config config_;
};

} // namespace n3mapping
