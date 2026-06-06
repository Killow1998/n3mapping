#include "n3mapping/core/n3mapping_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include "n3mapping/pcl_compat.h"

namespace n3mapping {

namespace {

double rotationAngle(const Eigen::Isometry3d& transform)
{
    return std::abs(Eigen::AngleAxisd(transform.rotation()).angle());
}

std::pair<double, double> meanLoopResidual(
    const std::vector<EdgeInfo>& edges,
    const std::map<int64_t, Eigen::Isometry3d>& poses)
{
    if (edges.empty()) {
        return {0.0, 0.0};
    }

    double translation_sum = 0.0;
    double rotation_sum = 0.0;
    std::size_t count = 0;
    for (const auto& edge : edges) {
        auto from_it = poses.find(edge.from_id);
        auto to_it = poses.find(edge.to_id);
        if (from_it == poses.end() || to_it == poses.end()) {
            continue;
        }
        const Eigen::Isometry3d predicted = from_it->second.inverse() * to_it->second;
        const Eigen::Isometry3d residual = edge.measurement.inverse() * predicted;
        translation_sum += residual.translation().norm();
        rotation_sum += rotationAngle(residual);
        ++count;
    }

    if (count == 0) {
        return {0.0, 0.0};
    }
    return {translation_sum / static_cast<double>(count), rotation_sum / static_cast<double>(count)};
}

void accumulatePoseUpdateStats(const std::map<int64_t, Eigen::Isometry3d>& before,
                               const std::map<int64_t, Eigen::Isometry3d>& after,
                               CoreLoopClosureResult* result)
{
    double translation_sum = 0.0;
    double rotation_sum = 0.0;
    std::size_t count = 0;

    for (const auto& [id, before_pose] : before) {
        auto after_it = after.find(id);
        if (after_it == after.end()) {
            continue;
        }
        const Eigen::Isometry3d delta = before_pose.inverse() * after_it->second;
        const double translation = delta.translation().norm();
        const double rotation = rotationAngle(delta);
        translation_sum += translation;
        rotation_sum += rotation;
        result->max_pose_update_translation = std::max(result->max_pose_update_translation, translation);
        result->max_pose_update_rotation = std::max(result->max_pose_update_rotation, rotation);
        ++count;
    }

    if (count == 0) {
        return;
    }
    const std::size_t previous_count = result->pose_update_count;
    const std::size_t total_count = previous_count + count;
    result->mean_pose_update_translation =
        (result->mean_pose_update_translation * static_cast<double>(previous_count) + translation_sum) /
        static_cast<double>(total_count);
    result->mean_pose_update_rotation =
        (result->mean_pose_update_rotation * static_cast<double>(previous_count) + rotation_sum) /
        static_cast<double>(total_count);
    result->pose_update_count = total_count;
}

Config validateOrThrow(const Config& config)
{
    std::string error;
    if (!config.validate(&error)) {
        throw std::invalid_argument("Invalid N3MappingCore config: " + error);
    }
    return config;
}

}  // namespace

N3MappingCore::N3MappingCore(const Config& config)
  : config_(validateOrThrow(config))
  , session_(std::make_unique<core::N3MappingSession>(config_))
{
}

N3MappingCore::~N3MappingCore() = default;

CoreRunMode parseCoreRunMode(const std::string& mode)
{
    if (mode == "localization") {
        return CoreRunMode::LOCALIZATION;
    }
    if (mode == "map_extension") {
        return CoreRunMode::MAP_EXTENSION;
    }
    return CoreRunMode::MAPPING;
}

const char* coreRunModeName(CoreRunMode mode)
{
    switch (mode) {
        case CoreRunMode::LOCALIZATION:
            return "localization";
        case CoreRunMode::MAP_EXTENSION:
            return "map_extension";
        case CoreRunMode::MAPPING:
        default:
            return "mapping";
    }
}

bool coreRunModeLoadsMap(CoreRunMode mode)
{
    return mode == CoreRunMode::LOCALIZATION || mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeSavesMap(CoreRunMode mode)
{
    return mode == CoreRunMode::MAPPING || mode == CoreRunMode::MAP_EXTENSION;
}

bool coreRunModeProcessesLoopClosures(CoreRunMode mode)
{
    return mode == CoreRunMode::MAPPING;
}

core::BackendOutput N3MappingCore::processFrame(CoreRunMode mode, const core::LioFrame& frame)
{
    switch (mode) {
        case CoreRunMode::LOCALIZATION:
            return processLocalizationFrame(frame);
        case CoreRunMode::MAP_EXTENSION:
            return processMapExtensionFrame(frame);
        case CoreRunMode::MAPPING:
        default:
            return processMappingFrame(frame);
    }
}

core::BackendOutput N3MappingCore::processMappingFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    auto output = makeOutput(true, frame.T_world_lidar, frame.undistorted_cloud);
    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    auto& keyframes = session_->keyframeManager();
    if (!keyframes.shouldAddKeyframe(frame.T_world_lidar)) {
        appendDenseTrajectorySampleWithLatestAnchor(timestamp, frame.T_world_lidar);
        return output;
    }

    const int64_t keyframe_id = keyframes.addKeyframe(timestamp, frame.T_world_lidar, frame.undistorted_cloud);
    session_->loopDetector().addDescriptor(keyframe_id, frame.undistorted_cloud);
    addRhpdDescriptorForKeyframe(keyframe_id, frame.undistorted_cloud);

    if (keyframe_id == 0) {
        session_->graphOptimizer().addPriorFactor(keyframe_id, frame.T_world_lidar);
    } else {
        addOdometryConstraint(keyframe_id, frame.T_world_lidar);
    }

    session_->graphOptimizer().incrementalOptimize();
    refreshOptimizedPoses();

    Eigen::Isometry3d optimized_pose = frame.T_world_lidar;
    if (session_->graphOptimizer().hasNode(keyframe_id)) {
        try {
            optimized_pose = session_->graphOptimizer().getOptimizedPose(keyframe_id);
        } catch (const std::exception&) {
            optimized_pose = frame.T_world_lidar;
        }
    }

    output.accepted_keyframe = true;
    output.keyframe_id = keyframe_id;
    output.T_world_lidar = optimized_pose;
    output.cloud_world = makeWorldCloud(frame.undistorted_cloud, optimized_pose);
    if (auto kf = keyframes.getKeyframe(keyframe_id)) {
        appendDenseTrajectorySample(timestamp, frame.T_world_lidar, keyframe_id, kf->pose_odom);
    }

    {
        std::lock_guard<std::mutex> lock(loop_queue_mutex_);
        loop_detection_queue_.push_back(keyframe_id);
    }
    return output;
}

core::BackendOutput N3MappingCore::processLocalizationFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty()) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    if (!map_loaded_) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    Eigen::Isometry3d pose_map = frame.T_world_lidar;
    bool success = false;
    bool relocalization_locked = false;
    int64_t matched_keyframe_id = -1;
    auto& localizer = session_->worldLocalizing();

    if (localizer.isRelocalized()) {
        auto result = localizer.trackLocalization(frame.undistorted_cloud, frame.T_world_lidar);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
            matched_keyframe_id = result.matched_keyframe_id;
        }
    }

    if (!localizer.isRelocalized() || !success) {
        auto result = localizer.relocalize(frame.undistorted_cloud, frame.T_world_lidar);
        if (result.success) {
            pose_map = result.pose_in_map;
            success = true;
            relocalization_locked = true;
            matched_keyframe_id = result.matched_keyframe_id;
        }
    }

    if (!success) {
        pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
    }

    auto output = makeOutput(success, pose_map, frame.undistorted_cloud);
    output.relocalization_locked = relocalization_locked;
    output.matched_keyframe_id = matched_keyframe_id;
    return output;
}

core::BackendOutput N3MappingCore::processMapExtensionFrame(const core::LioFrame& frame)
{
    if (!frame.pose_valid || !frame.undistorted_cloud || frame.undistorted_cloud->empty() || !map_loaded_) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    auto& resuming = session_->mappingResuming();
    auto& localizer = session_->worldLocalizing();
    const auto state = resuming.getState();

    if (state == MappingResumingState::MAP_LOADED) {
        const bool locked = resuming.performInitialRelocalization(frame.undistorted_cloud, frame.T_world_lidar);
        auto output = makeOutput(locked,
                                 localizer.getMapToOdomTransform() * frame.T_world_lidar,
                                 frame.undistorted_cloud);
        output.relocalization_locked = locked;
        if (locked) {
            const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
            const int64_t matched_id = localizer.getLastMatchedKeyframeId();
            auto matched_kf = session_->keyframeManager().getKeyframe(matched_id);
            if (matched_kf) {
                appendDenseTrajectorySample(timestamp, output.T_world_lidar, matched_id, matched_kf->pose_optimized, false);
            }
        }
        return output;
    }

    if (state != MappingResumingState::RELOCALIZED && state != MappingResumingState::EXTENDING) {
        return makeOutput(false, frame.T_world_lidar, frame.undistorted_cloud);
    }

    Eigen::Isometry3d pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
    if (!session_->keyframeManager().shouldAddKeyframe(pose_map)) {
        const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
        appendDenseTrajectorySampleWithLatestAnchor(timestamp, pose_map, false);
        return makeOutput(true, pose_map, frame.undistorted_cloud);
    }

    const double timestamp = static_cast<double>(frame.stamp.nsec) * 1e-9;
    const int64_t keyframe_id = resuming.processNewKeyframe(timestamp, frame.T_world_lidar, frame.undistorted_cloud);
    if (keyframe_id >= 0) {
        resuming.detectCrossLoops(keyframe_id);
        session_->graphOptimizer().incrementalOptimize();
        refreshOptimizedPoses();
        if (session_->graphOptimizer().hasNode(keyframe_id)) {
            try {
                pose_map = session_->graphOptimizer().getOptimizedPose(keyframe_id);
            } catch (const std::exception&) {
                pose_map = localizer.getMapToOdomTransform() * frame.T_world_lidar;
            }
        }
    }

    auto output = makeOutput(keyframe_id >= 0, pose_map, frame.undistorted_cloud);
    output.accepted_keyframe = keyframe_id >= 0;
    output.keyframe_id = keyframe_id;
    if (keyframe_id >= 0) {
        if (auto kf = session_->keyframeManager().getKeyframe(keyframe_id)) {
            appendDenseTrajectorySample(timestamp, kf->pose_odom, keyframe_id, kf->pose_odom, false);
        }
    }
    return output;
}

CoreLoopClosureResult N3MappingCore::processPendingLoopClosures()
{
    CoreLoopClosureResult result;
    std::vector<int64_t> keyframes_to_check;
    {
        std::lock_guard<std::mutex> lock(loop_queue_mutex_);
        keyframes_to_check.swap(loop_detection_queue_);
    }

    for (int64_t query_id : keyframes_to_check) {
        if (query_id - last_loop_check_id_ < config_.loop_kf_gap) {
            continue;
        }

        auto query_kf = session_->keyframeManager().getKeyframe(query_id);
        if (!query_kf) {
            continue;
        }

        std::vector<LoopCandidate> candidates = session_->loopDetector().detectLoopCandidates(query_id);
        if (candidates.empty()) {
            continue;
        }
        last_loop_check_id_ = query_id;

        std::vector<VerifiedLoop> verified_loops;
        verified_loops.reserve(candidates.size());

        for (const auto& candidate : candidates) {
            auto match_kf = session_->keyframeManager().getKeyframe(candidate.match_id);
            if (!match_kf || !query_kf->cloud || !match_kf->cloud || query_kf->cloud->empty() || match_kf->cloud->empty()) {
                continue;
            }

            auto source = session_->keyframeManager().buildSubmapInRootFrame(query_id, 0, candidate.match_id);
            auto target = session_->keyframeManager().buildSubmapInRootFrame(candidate.match_id, config_.gicp_submap_size, candidate.match_id);
            if (!source || source->empty() || !target || target->empty()) {
                continue;
            }

            MatchResult match_result =
                session_->pointCloudMatcher().alignCloud(target, source, Eigen::Isometry3d::Identity());

            VerifiedLoop loop;
            loop.query_id = query_id;
            loop.match_id = candidate.match_id;
            loop.fitness_score = match_result.fitness_score;
            loop.inlier_ratio = match_result.inlier_ratio;
            loop.information =
                config_.loop_use_icp_information ? match_result.information : Eigen::Matrix<double, 6, 6>::Identity();

            const bool fitness_ok = match_result.fitness_score < config_.loop_fitness_threshold;
            const bool inlier_ok = match_result.inlier_ratio >= config_.loop_min_inlier_ratio;
            const double icp_translation = match_result.T_target_source.translation().norm();
            const double icp_rotation = Eigen::AngleAxisd(match_result.T_target_source.rotation()).angle();
            const bool geom_ok =
                icp_translation <= config_.loop_max_icp_translation && icp_rotation <= config_.loop_max_icp_rotation;

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
            continue;
        }

        auto valid_loops = session_->loopClosureManager().filterValidLoops(verified_loops);
        auto best_loops = session_->loopClosureManager().selectBestPerQuery(valid_loops);
        if (best_loops.empty()) {
            continue;
        }

        auto edges = session_->loopClosureManager().buildLoopEdges(best_loops, LoopEdgeDirection::MatchToQuery);
        if (edges.empty()) {
            continue;
        }

        const auto poses_before = session_->graphOptimizer().getOptimizedPoses();
        const auto residual_before = meanLoopResidual(edges, poses_before);
        session_->loopClosureManager().applyEdges(edges, session_->graphOptimizer());
        loop_count_ += edges.size();
        refreshOptimizedPoses();
        const auto poses_after = session_->graphOptimizer().getOptimizedPoses();
        const auto residual_after = meanLoopResidual(edges, poses_after);

        result.optimized = true;
        result.edge_count += edges.size();
        result.loop_residual_translation_before = residual_before.first;
        result.loop_residual_rotation_before = residual_before.second;
        result.loop_residual_translation_after = residual_after.first;
        result.loop_residual_rotation_after = residual_after.second;
        accumulatePoseUpdateStats(poses_before, poses_after, &result);
        result.accepted_loops.insert(result.accepted_loops.end(), best_loops.begin(), best_loops.end());
    }

    return result;
}

bool N3MappingCore::loadMap(const std::string& map_path)
{
    std::vector<core::DenseTrajectoryPose> loaded_dense_optimized;
    core::DenseTrajectoryMetadata loaded_dense_metadata;
    const bool loaded_for_dense = session_->mapSerializer().loadMap(
        map_path,
        session_->keyframeManager(),
        session_->loopDetector(),
        session_->graphOptimizer(),
        &loaded_dense_optimized,
        &loaded_dense_metadata);
    if (!loaded_for_dense) {
        return false;
    }
    if (!session_->mappingResuming().initializeFromLoadedMap()) {
        return false;
    }
    session_->worldLocalizing().reset();
    dense_trajectory_metadata_ = loaded_dense_metadata;
    initializeDenseSamplesFromOptimized(loaded_dense_optimized);
    map_loaded_ = true;
    return true;
}

bool N3MappingCore::saveMap(const std::string& map_path)
{
    const auto dense_optimized_trajectory = buildDenseOptimizedTrajectory();
    core::DenseTrajectoryMetadata metadata = dense_trajectory_metadata_;
    if (!dense_optimized_trajectory.empty() && (metadata.source.empty() || metadata.source == "none")) {
        metadata.source = "native";
        metadata.degraded = false;
    }
    return session_->mapSerializer().saveMap(
        map_path,
        session_->keyframeManager(),
        session_->loopDetector(),
        session_->graphOptimizer(),
        dense_optimized_trajectory,
        metadata);
}

bool N3MappingCore::saveGlobalMap(const std::string& pcd_path)
{
    return session_->mapSerializer().saveGlobalMap(
        pcd_path, session_->keyframeManager(), config_.save_global_map_voxel_size);
}

bool N3MappingCore::saveMapSnapshot(std::string* error)
{
    if (session_->keyframeManager().size() < 1) {
        if (error) *error = "no_keyframes";
        return false;
    }

    const std::string map_file = config_.map_save_path + "/n3map.pbstream";
    if (!saveMap(map_file)) {
        if (error) *error = "save_pbstream_failed";
        return false;
    }

    if (config_.save_global_map_on_shutdown) {
        const std::string global_map_file = config_.map_save_path + "/global_map.pcd";
        if (!saveGlobalMap(global_map_file)) {
            if (error) *error = "save_global_map_failed";
            return false;
        }
    }

    return true;
}

core::LioFrame::PointCloud::Ptr N3MappingCore::buildGlobalMap() const
{
    return session_->mapSerializer().buildGlobalMap(
        session_->keyframeManager(), config_.global_map_voxel_size);
}

bool N3MappingCore::mapLoaded() const
{
    return map_loaded_;
}

Keyframe::Ptr N3MappingCore::getKeyframe(int64_t id) const
{
    return session_->keyframeManager().getKeyframe(id);
}

std::vector<Keyframe::Ptr> N3MappingCore::getAllKeyframes() const
{
    return session_->keyframeManager().getAllKeyframes();
}

std::map<int64_t, Eigen::Isometry3d> N3MappingCore::getOptimizedPoses() const
{
    return session_->graphOptimizer().getOptimizedPoses();
}

std::vector<core::DenseTrajectoryPose> N3MappingCore::getDenseOptimizedTrajectory() const
{
    return buildDenseOptimizedTrajectory();
}

core::BackendOutput N3MappingCore::makeOutput(bool success,
                                              const Eigen::Isometry3d& pose,
                                              const PointCloud::Ptr& cloud) const
{
    core::BackendOutput output;
    output.success = success;
    output.T_world_lidar = pose;
    output.cloud_body = cloud;
    output.cloud_world = makeWorldCloud(cloud, pose);
    return output;
}

N3MappingCore::PointCloud::Ptr N3MappingCore::makeWorldCloud(const PointCloud::Ptr& cloud,
                                                             const Eigen::Isometry3d& pose) const
{
    if (!cloud || cloud->empty()) {
        return pcl::make_shared<PointCloud>();
    }
    auto transformed = pcl::make_shared<PointCloud>();
    pcl::transformPointCloud(*cloud, *transformed, pose.matrix().cast<float>());
    return transformed;
}

void N3MappingCore::appendDenseTrajectorySample(double timestamp,
                                                const Eigen::Isometry3d& raw_pose,
                                                int64_t anchor_keyframe_id,
                                                const Eigen::Isometry3d& anchor_raw_pose,
                                                bool use_bracketing_correction)
{
    core::AnchoredDenseTrajectorySample sample;
    sample.seq = static_cast<uint64_t>(dense_trajectory_samples_.size());
    sample.timestamp = timestamp;
    sample.pose_world_lidar_raw = raw_pose;
    sample.anchor_keyframe_id = anchor_keyframe_id;
    sample.anchor_pose_world_lidar_raw = anchor_raw_pose;
    sample.has_anchor = anchor_keyframe_id >= 0;
    sample.use_bracketing_correction = use_bracketing_correction;
    if (dense_trajectory_metadata_.source == "keyframe_fallback") {
        dense_trajectory_metadata_.source = "mixed_keyframe_fallback_and_high_rate";
        dense_trajectory_metadata_.degraded = true;
    } else if (dense_trajectory_metadata_.source.empty() || dense_trajectory_metadata_.source == "none") {
        dense_trajectory_metadata_.source = "native";
        dense_trajectory_metadata_.degraded = false;
    }
    dense_trajectory_samples_.push_back(sample);
}

void N3MappingCore::appendDenseTrajectorySampleWithLatestAnchor(double timestamp,
                                                                const Eigen::Isometry3d& raw_pose,
                                                                bool use_bracketing_correction)
{
    auto latest = session_->keyframeManager().getLatestKeyframe();
    if (!latest) {
        appendDenseTrajectorySample(timestamp, raw_pose, -1, Eigen::Isometry3d::Identity(), use_bracketing_correction);
        return;
    }

    const Eigen::Isometry3d anchor_raw_pose =
        latest->is_from_loaded_map ? latest->pose_optimized : latest->pose_odom;
    appendDenseTrajectorySample(timestamp, raw_pose, latest->id, anchor_raw_pose, use_bracketing_correction);
}

void N3MappingCore::initializeDenseSamplesFromOptimized(
    const std::vector<core::DenseTrajectoryPose>& dense_optimized)
{
    dense_trajectory_samples_.clear();
    dense_trajectory_samples_.reserve(dense_optimized.size());

    const auto keyframes = session_->keyframeManager().getAllKeyframes();
    for (const auto& dense_pose : dense_optimized) {
        Keyframe::Ptr anchor;
        double best_time = -std::numeric_limits<double>::infinity();
        for (const auto& kf : keyframes) {
            if (!kf) {
                continue;
            }
            if (kf->timestamp <= dense_pose.timestamp && kf->timestamp >= best_time) {
                anchor = kf;
                best_time = kf->timestamp;
            }
        }
        if (!anchor && !keyframes.empty()) {
            anchor = keyframes.front();
        }

        core::AnchoredDenseTrajectorySample sample;
        sample.seq = dense_pose.seq;
        sample.timestamp = dense_pose.timestamp;
        sample.pose_world_lidar_raw = dense_pose.pose_world_lidar;
        if (anchor) {
            sample.anchor_keyframe_id = anchor->id;
            sample.anchor_pose_world_lidar_raw = anchor->pose_optimized;
            sample.has_anchor = true;
            sample.use_bracketing_correction = false;
        }
        dense_trajectory_samples_.push_back(sample);
    }
}

std::vector<core::DenseTrajectoryPose> N3MappingCore::buildDenseOptimizedTrajectory() const
{
    std::vector<core::DenseTrajectoryPose> dense_optimized;
    dense_optimized.reserve(dense_trajectory_samples_.size());
    for (const auto& sample : dense_trajectory_samples_) {
        core::DenseTrajectoryPose pose;
        pose.seq = sample.seq;
        pose.timestamp = sample.timestamp;
        pose.pose_world_lidar = sample.pose_world_lidar_raw;
        if (sample.use_bracketing_correction) {
            pose.pose_world_lidar = interpolateDenseCorrection(sample.timestamp) * sample.pose_world_lidar_raw;
        } else if (sample.has_anchor) {
            auto anchor = session_->keyframeManager().getKeyframe(sample.anchor_keyframe_id);
            if (anchor) {
                pose.pose_world_lidar =
                    anchor->pose_optimized * sample.anchor_pose_world_lidar_raw.inverse() * sample.pose_world_lidar_raw;
            }
        }
        dense_optimized.push_back(pose);
    }
    return dense_optimized;
}

Eigen::Isometry3d N3MappingCore::interpolateDenseCorrection(double timestamp) const
{
    const auto keyframes = session_->keyframeManager().getAllKeyframes();
    Keyframe::Ptr before;
    Keyframe::Ptr after;

    for (const auto& kf : keyframes) {
        if (!kf) {
            continue;
        }
        if (kf->timestamp <= timestamp && (!before || kf->timestamp > before->timestamp)) {
            before = kf;
        }
        if (kf->timestamp >= timestamp && (!after || kf->timestamp < after->timestamp)) {
            after = kf;
        }
    }

    if (!before && !after) {
        return Eigen::Isometry3d::Identity();
    }
    if (!before) {
        before = after;
    }
    if (!after) {
        after = before;
    }

    const Eigen::Isometry3d correction_before = before->pose_optimized * before->pose_odom.inverse();
    const Eigen::Isometry3d correction_after = after->pose_optimized * after->pose_odom.inverse();
    double alpha = 0.0;
    const double dt = after->timestamp - before->timestamp;
    if (std::isfinite(dt) && dt > 1e-9) {
        alpha = std::clamp((timestamp - before->timestamp) / dt, 0.0, 1.0);
    }

    Eigen::Isometry3d correction = Eigen::Isometry3d::Identity();
    correction.translation() =
        (1.0 - alpha) * correction_before.translation() + alpha * correction_after.translation();
    Eigen::Quaterniond qb(correction_before.rotation());
    Eigen::Quaterniond qa(correction_after.rotation());
    qb.normalize();
    qa.normalize();
    correction.linear() = qb.slerp(alpha, qa).toRotationMatrix();
    return correction;
}

void N3MappingCore::addRhpdDescriptorForKeyframe(int64_t keyframe_id, const PointCloud::Ptr& fallback_cloud)
{
    auto kf = session_->keyframeManager().getKeyframe(keyframe_id);
    if (!kf) {
        return;
    }

    const int submap_radius = std::max(0, config_.rhpd_submap_kf_radius);
    PointCloud::Ptr rhpd_cloud = fallback_cloud;
    if (submap_radius > 0) {
        rhpd_cloud = session_->keyframeManager().buildCausalSubmapInRootFrame(keyframe_id, submap_radius, keyframe_id);
    }

    if (rhpd_cloud && !rhpd_cloud->empty() && config_.rhpd_submap_voxel_size > 1e-4) {
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setLeafSize(config_.rhpd_submap_voxel_size,
                          config_.rhpd_submap_voxel_size,
                          config_.rhpd_submap_voxel_size);
        voxel.setInputCloud(rhpd_cloud);
        auto filtered = pcl::make_shared<PointCloud>();
        voxel.filter(*filtered);
        if (!filtered->empty()) {
            rhpd_cloud = filtered;
        }
    }

    kf->rhpd_descriptor = session_->loopDetector().addRHPD(keyframe_id, rhpd_cloud);
}

bool N3MappingCore::addOdometryConstraint(int64_t keyframe_id, const Eigen::Isometry3d& pose)
{
    auto prev_kf = session_->keyframeManager().getKeyframe(keyframe_id - 1);
    if (!prev_kf) {
        return false;
    }

    EdgeInfo edge;
    edge.from_id = keyframe_id - 1;
    edge.to_id = keyframe_id;
    edge.measurement = prev_kf->pose_odom.inverse() * pose;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
    edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
    edge.type = EdgeType::ODOMETRY;
    session_->graphOptimizer().addOdometryEdge(edge);
    return true;
}

void N3MappingCore::refreshOptimizedPoses()
{
    session_->keyframeManager().updateOptimizedPoses(session_->graphOptimizer().getOptimizedPoses());
}

}  // namespace n3mapping
