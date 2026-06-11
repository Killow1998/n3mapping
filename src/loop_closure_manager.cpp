#include "n3mapping/loop_closure_manager.h"

#include <algorithm>
#include <cmath>

namespace n3mapping {

namespace {

double positiveDiagOrOne(const Eigen::Matrix<double, 6, 6>& information, int index)
{
    const double value = information(index, index);
    return std::isfinite(value) && value > 0.0 ? value : 1.0;
}

double normalizeAngle(double angle)
{
    constexpr double kPi = 3.14159265358979323846;
    while (angle > kPi) {
        angle -= 2.0 * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0 * kPi;
    }
    return angle;
}

double angleDistance(double a, double b)
{
    return std::abs(normalizeAngle(a - b));
}

double residualYawForDiagnostics(const Eigen::Isometry3d& transform)
{
    return transform.rotation().eulerAngles(0, 1, 2).z();
}

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

Eigen::Matrix<double, 6, 6> makePlanarInformation(const Eigen::Matrix<double, 6, 6>& input, double weak_weight)
{
    const double clamped_weight = std::max(1e-6, std::min(1.0, weak_weight));
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(0, 0) = positiveDiagOrOne(input, 0);
    information(1, 1) = positiveDiagOrOne(input, 1);
    information(2, 2) = positiveDiagOrOne(input, 2) * clamped_weight;
    information(3, 3) = positiveDiagOrOne(input, 3) * clamped_weight;
    information(4, 4) = positiveDiagOrOne(input, 4) * clamped_weight;
    information(5, 5) = positiveDiagOrOne(input, 5);
    return information;
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

VerifiedLoop
LoopClosureManager::applyEdgeModel(const VerifiedLoop& loop) const
{
    VerifiedLoop modeled = loop;
    modeled.edge_mode = LoopEdgeMode::Full6Dof;
    modeled.vertical_downweighted = false;
    modeled.vertical_observability_score = 1.0;

    if (!modeled.verified || !config_.loop_observability_edge_model_enable) {
        return modeled;
    }

    const double residual_z = modeled.candidate_residual.translation().z();
    const double abs_residual_z = std::abs(residual_z);
    const double z_threshold =
        config_.loop_max_candidate_residual_z > 0.0
        ? std::max(1e-6, config_.loop_max_candidate_residual_z * 0.36)
        : std::max(1e-6, config_.loop_noise_position * 5.0);
    const double vertical_error_score = abs_residual_z / z_threshold;
    modeled.vertical_observability_score = 1.0 / (1.0 + vertical_error_score);

    if (!std::isfinite(vertical_error_score) || !std::isfinite(residual_z)) {
        modeled.verified = false;
        modeled.edge_mode = LoopEdgeMode::RejectedVerticalInconsistent;
        return modeled;
    }
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kYawHypothesisNearPiRad = 0.35;
    constexpr double kIcpYawNearZeroRad = 0.35;
    const double candidate_yaw = normalizeAngle(modeled.candidate_yaw_diff_rad);
    const double residual_yaw = normalizeAngle(residualYawForDiagnostics(modeled.candidate_residual));
    if (angleDistance(std::abs(candidate_yaw), kPi) < kYawHypothesisNearPiRad &&
        std::abs(residual_yaw) < kIcpYawNearZeroRad) {
        modeled.verified = false;
        modeled.edge_mode = LoopEdgeMode::RejectedYawInconsistent;
        return modeled;
    }
    if (config_.loop_max_candidate_residual_z > 0.0 &&
        abs_residual_z > config_.loop_max_candidate_residual_z * 0.8) {
        modeled.verified = false;
        modeled.edge_mode = LoopEdgeMode::RejectedVerticalInconsistent;
        return modeled;
    }
    if (vertical_error_score > 1.0) {
        modeled.edge_mode = LoopEdgeMode::PlanarXYYaw;
        modeled.vertical_downweighted = true;
        modeled.information = makePlanarInformation(modeled.information, config_.loop_planar_vertical_weight);
    }
    return modeled;
}

std::vector<EdgeInfo>
LoopClosureManager::buildLoopEdges(const std::vector<VerifiedLoop>& loops, LoopEdgeDirection direction) const
{
    std::vector<EdgeInfo> edges;
    edges.reserve(loops.size());

    for (const auto& loop : loops) {
        if (!loop.verified ||
            loop.edge_mode == LoopEdgeMode::RejectedVerticalInconsistent ||
            loop.edge_mode == LoopEdgeMode::RejectedYawInconsistent) {
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
