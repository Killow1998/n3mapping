#include "n3mapping/core/n3mapping_session.h"

namespace n3mapping {
namespace core {

N3MappingSession::N3MappingSession(const Config& config)
    : config_(config)
    , keyframe_manager_(config_)
    , loop_detector_(config_)
    , loop_closure_manager_(config_)
    , matcher_(config_)
    , optimizer_(config_)
    , serializer_(config_)
    , world_localizing_(config_, keyframe_manager_, loop_detector_, matcher_)
    , mapping_resuming_(config_,
                        keyframe_manager_,
                        loop_detector_,
                        matcher_,
                        optimizer_,
                        serializer_,
                        world_localizing_)
    , mapping_mode_processor_(config_, keyframe_manager_, loop_detector_, optimizer_)
    , localization_mode_processor_(world_localizing_)
    , map_extension_mode_processor_(keyframe_manager_,
                                    optimizer_,
                                    world_localizing_,
                                    mapping_resuming_) {}

const Config& N3MappingSession::config() const {
    return config_;
}

KeyframeManager& N3MappingSession::keyframes() {
    return keyframe_manager_;
}

const KeyframeManager& N3MappingSession::keyframes() const {
    return keyframe_manager_;
}

LoopDetector& N3MappingSession::loopDetector() {
    return loop_detector_;
}

LoopClosureManager& N3MappingSession::loopClosureManager() {
    return loop_closure_manager_;
}

PointCloudMatcher& N3MappingSession::matcher() {
    return matcher_;
}

GraphOptimizer& N3MappingSession::graphOptimizer() {
    return optimizer_;
}

MapSerializer& N3MappingSession::mapSerializer() {
    return serializer_;
}

WorldLocalizing& N3MappingSession::worldLocalizing() {
    return world_localizing_;
}

MappingResuming& N3MappingSession::mappingResuming() {
    return mapping_resuming_;
}

MappingModeProcessor& N3MappingSession::mappingModeProcessor() {
    return mapping_mode_processor_;
}

LocalizationModeProcessor& N3MappingSession::localizationModeProcessor() {
    return localization_mode_processor_;
}

MapExtensionModeProcessor& N3MappingSession::mapExtensionModeProcessor() {
    return map_extension_mode_processor_;
}

N3MappingSession::LoopProcessingResult
N3MappingSession::processLoopClosureForKeyframe(int64_t query_id) {
    LoopProcessingResult result;

    auto query_kf = keyframe_manager_.getKeyframe(query_id);
    if (!query_kf) {
        return result;
    }
    result.query_exists = true;

    std::vector<LoopCandidate> candidates = loop_detector_.detectLoopCandidates(query_id);
    if (candidates.empty()) {
        return result;
    }
    result.candidates_found = true;

    std::vector<VerifiedLoop> verified_loops;
    verified_loops.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf || !query_kf->cloud || !match_kf->cloud ||
            query_kf->cloud->empty() || match_kf->cloud->empty()) {
            continue;
        }

        auto source = keyframe_manager_.buildSubmapInRootFrame(query_id, 0, candidate.match_id);
        auto target = keyframe_manager_.buildSubmapInRootFrame(
            candidate.match_id, config_.gicp_submap_size, candidate.match_id);
        if (!source || source->empty() || !target || target->empty()) {
            continue;
        }

        MatchResult match_result =
            matcher_.alignCloud(target, source, Eigen::Isometry3d::Identity());

        VerifiedLoop loop;
        loop.query_id = query_id;
        loop.match_id = candidate.match_id;
        loop.fitness_score = match_result.fitness_score;
        loop.inlier_ratio = match_result.inlier_ratio;
        loop.information = config_.loop_use_icp_information
            ? match_result.information
            : Eigen::Matrix<double, 6, 6>::Identity();

        const bool fitness_ok =
            match_result.fitness_score < config_.loop_fitness_threshold;
        const bool inlier_ok =
            match_result.inlier_ratio >= config_.loop_min_inlier_ratio;
        const double icp_translation =
            match_result.T_target_source.translation().norm();
        const double icp_rotation =
            Eigen::AngleAxisd(match_result.T_target_source.rotation()).angle();
        const bool geom_ok =
            icp_translation <= config_.loop_max_icp_translation &&
            icp_rotation <= config_.loop_max_icp_rotation;

        loop.verified = match_result.converged && fitness_ok && inlier_ok && geom_ok;
        if (loop.verified) {
            const Eigen::Isometry3d T_est_match_query =
                match_kf->pose_optimized.inverse() * query_kf->pose_optimized;
            const Eigen::Isometry3d T_residual = match_result.T_target_source;
            loop.T_match_query = T_residual * T_est_match_query;
        }
        verified_loops.push_back(loop);
    }

    if (verified_loops.empty()) {
        return result;
    }

    auto valid_loops = loop_closure_manager_.filterValidLoops(verified_loops);
    result.selected_loops = loop_closure_manager_.selectBestPerQuery(valid_loops);
    if (result.selected_loops.empty()) {
        return result;
    }

    auto edges = loop_closure_manager_.buildLoopEdges(result.selected_loops,
                                                     LoopEdgeDirection::MatchToQuery);
    if (edges.empty()) {
        return result;
    }
    if (loop_closure_manager_.applyEdges(edges, optimizer_)) {
        result.edges_added = edges.size();
        result.optimized = true;
        keyframe_manager_.updateOptimizedPoses(optimizer_.getOptimizedPoses());
    }
    return result;
}

bool N3MappingSession::loadMapForLocalization(const std::string& path) {
    return serializer_.loadMap(path, keyframe_manager_, loop_detector_, optimizer_);
}

bool N3MappingSession::loadMapForExtension(const std::string& path) {
    return mapping_resuming_.loadExistingMap(path);
}

bool N3MappingSession::saveCurrentMap(const std::string& path) {
    return serializer_.saveMap(path, keyframe_manager_, loop_detector_, optimizer_);
}

bool N3MappingSession::saveExtendedMap(const std::string& path) {
    return mapping_resuming_.saveExtendedMap(path);
}

bool N3MappingSession::saveGlobalMap(const std::string& path) {
    return serializer_.saveGlobalMap(path, keyframe_manager_);
}

}  // namespace core
}  // namespace n3mapping
