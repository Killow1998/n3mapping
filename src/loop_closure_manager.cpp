#include "n3mapping/loop_closure_manager.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

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

bool isFiniteTransform(const Eigen::Isometry3d& pose)
{
    return pose.matrix().allFinite();
}

double rotationDistance(const Eigen::Isometry3d& lhs,
                        const Eigen::Isometry3d& rhs)
{
    const Eigen::Matrix3d relative = lhs.linear().transpose() * rhs.linear();
    const double cosine = std::max(-1.0, std::min(1.0,
        0.5 * (relative.trace() - 1.0)));
    return std::acos(cosine);
}

struct ImpliedQueryPose
{
    VerifiedLoop loop;
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
};

bool consensusCompatible(const ImpliedQueryPose& lhs,
                         const ImpliedQueryPose& rhs,
                         const Config& config)
{
    const double translation =
        (lhs.pose.translation() - rhs.pose.translation()).norm();
    const double rotation = rotationDistance(lhs.pose, rhs.pose);
    return std::isfinite(translation) && std::isfinite(rotation) &&
           translation <= config.loop_same_query_consensus_translation_m &&
           rotation <= config.loop_same_query_consensus_rotation_rad;
}

double normalizedConsensusResidual(const ImpliedQueryPose& lhs,
                                   const ImpliedQueryPose& rhs,
                                   const Config& config)
{
    const double translation =
        (lhs.pose.translation() - rhs.pose.translation()).norm();
    const double rotation = rotationDistance(lhs.pose, rhs.pose);
    return translation / config.loop_same_query_consensus_translation_m +
           rotation / config.loop_same_query_consensus_rotation_rad;
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

SameQueryLoopSelection
LoopClosureManager::selectSameQueryConsensus(
    const std::vector<VerifiedLoop>& loops,
    const std::map<int64_t, Keyframe::Ptr>& keyframes) const
{
    SameQueryLoopSelection result;
    std::map<int64_t, std::vector<ImpliedQueryPose>> by_query;

    for (const auto& loop : loops) {
        const auto key = std::make_pair(loop.query_id, loop.match_id);
        if (!loop.isValid()) {
            result.rejected[key] = "same_query_consensus_invalid_loop";
            continue;
        }
        const auto query_it = keyframes.find(loop.query_id);
        if (query_it == keyframes.end() || !query_it->second ||
            !isFiniteTransform(query_it->second->pose_optimized)) {
            result.rejected[key] = "same_query_consensus_missing_query_pose";
            continue;
        }
        const auto match_it = keyframes.find(loop.match_id);
        if (match_it == keyframes.end() || !match_it->second ||
            !isFiniteTransform(match_it->second->pose_optimized)) {
            result.rejected[key] = "same_query_consensus_missing_match_pose";
            continue;
        }
        if (!isFiniteTransform(loop.T_match_query)) {
            result.rejected[key] = "same_query_consensus_nonfinite_measurement";
            continue;
        }

        const Eigen::Isometry3d implied_query_pose =
            match_it->second->pose_optimized * loop.T_match_query;
        if (!isFiniteTransform(implied_query_pose)) {
            result.rejected[key] = "same_query_consensus_nonfinite_implied_pose";
            continue;
        }
        by_query[loop.query_id].push_back({loop, implied_query_pose});
    }

    for (const auto& group : by_query) {
        const auto& candidates = group.second;
        if (candidates.size() == 1) {
            result.selected.push_back(candidates.front().loop);
            continue;
        }

        std::size_t best_anchor = 0;
        std::vector<std::size_t> best_support;
        double best_residual = std::numeric_limits<double>::infinity();

        for (std::size_t anchor = 0; anchor < candidates.size(); ++anchor) {
            std::vector<std::size_t> support;
            double residual = 0.0;
            for (std::size_t candidate = 0; candidate < candidates.size();
                 ++candidate) {
                if (!consensusCompatible(candidates[anchor],
                                         candidates[candidate], config_)) {
                    continue;
                }
                support.push_back(candidate);
                residual += normalizedConsensusResidual(
                    candidates[anchor], candidates[candidate], config_);
            }

            const bool larger_support = support.size() > best_support.size();
            const bool lower_residual =
                support.size() == best_support.size() &&
                residual + 1e-12 < best_residual;
            const bool preferred_anchor =
                support.size() == best_support.size() &&
                std::abs(residual - best_residual) <= 1e-12 &&
                preferLoopCandidate(candidates[anchor].loop,
                                    candidates[best_anchor].loop, config_);
            if (larger_support || lower_residual || preferred_anchor) {
                best_anchor = anchor;
                best_support = std::move(support);
                best_residual = residual;
            }
        }

        const bool has_strict_majority =
            best_support.size() * 2 > candidates.size();
        if (!has_strict_majority || best_support.size() < 2) {
            std::size_t best = 0;
            for (std::size_t i = 1; i < candidates.size(); ++i) {
                if (preferLoopCandidate(candidates[i].loop,
                                        candidates[best].loop, config_)) {
                    best = i;
                }
            }
            result.selected.push_back(candidates[best].loop);
            const char* reason = best_support.size() < 2
                                     ? "same_query_consensus_no_support"
                                     : "same_query_consensus_ambiguous";
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                if (i != best) {
                    result.rejected[{candidates[i].loop.query_id,
                                     candidates[i].loop.match_id}] = reason;
                }
            }
            continue;
        }

        const std::set<std::size_t> selected_indices(best_support.begin(),
                                                     best_support.end());
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (selected_indices.count(i) != 0) {
                result.selected.push_back(candidates[i].loop);
            } else {
                result.rejected[{candidates[i].loop.query_id,
                                 candidates[i].loop.match_id}] =
                    "same_query_consensus_outlier";
            }
        }
    }

    return result;
}

VerifiedLoop
LoopClosureManager::applyEdgeModel(const VerifiedLoop& loop) const
{
    VerifiedLoop modeled = loop;
    modeled.edge_mode = LoopEdgeMode::Full6Dof;
    modeled.vertical_downweighted = false;
    // Left as the verifier computed it. This used to be overwritten with 1.0
    // here, which meant a field named for a measurement reported the same
    // constant for every loop while being written into the debug stream as
    // though it were data.
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
        if (loop.edge_mode == LoopEdgeMode::XYYaw || loop.edge_mode == LoopEdgeMode::VerticalNeutral) {
            edge.constraint_mode = EdgeConstraintMode::XY_YAW;
        }
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
