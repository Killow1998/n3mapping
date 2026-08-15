#pragma once

#include <map>
#include <string>
#include <utility>
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

struct SameQueryLoopSelection
{
    std::vector<VerifiedLoop> selected;
    std::map<std::pair<int64_t, int64_t>, std::string> rejected;
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
     * @brief 保留同一 query 推导位姿的严格多数一致集
     *
     * 单候选保持不变；没有严格多数时回退到 selectBestPerQuery 的排序，
     * 避免在两个互相矛盾的小簇之间猜测。
     */
    SameQueryLoopSelection selectSameQueryConsensus(
        const std::vector<VerifiedLoop>& loops,
        const std::map<int64_t, Keyframe::Ptr>& keyframes) const;

    /**
     * @brief 构建回环边
     */
    std::vector<EdgeInfo> buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const;

    /**
     * @brief 根据垂直观测性塑形回环边信息矩阵
     */
    VerifiedLoop applyEdgeModel(const VerifiedLoop& loop) const;

    /**
     * @brief 将回环边应用到图优化器，并触发增量优化
     * @return true 若优化成功提交；false 表示无边或优化失败并已回滚
     */
    bool applyEdges(const std::vector<EdgeInfo>& edges, LoopOptimizerInterface& optimizer) const;

  private:
    Config config_;
};

} // namespace n3mapping
