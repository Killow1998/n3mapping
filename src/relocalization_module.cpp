#include "n3mapping/relocalization_module.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <pcl/common/transforms.h>

namespace n3mapping {

RelocalizationModule::RelocalizationModule(const Config& config,
                                           KeyframeManager& keyframe_manager,
                                           LoopDetector& loop_detector,
                                           PointCloudMatcher& matcher)
  : config_(config)
  , keyframe_manager_(keyframe_manager)
  , loop_detector_(loop_detector)
  , matcher_(matcher)
  , is_relocalized_(false)
  , T_map_odom_(Eigen::Isometry3d::Identity())
  , last_matched_id_(-1)
  , last_odom_pose_(Eigen::Isometry3d::Identity())
  , consecutive_track_failures_(0)
{
}

RelocResult
RelocalizationModule::relocalize(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose)
{
    RelocResult result;
    result.success = false;

    if (!cloud || cloud->empty()) {
        LOG(WARNING) << "Empty point cloud provided for relocalization.";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 检查地图是否有关键帧
    if (keyframe_manager_.size() == 0) {
        LOG(WARNING) << "No keyframes available in the map for relocalization.";
        return result;
    }

    // Step 1: 使用 ScanContext 搜索候选帧
    std::vector<LoopCandidate> candidates = searchCandidates(cloud);

    if (candidates.empty()) {
        LOG(WARNING) << "No relocalization candidates found using ScanContext.";
        return result;
    }

    // Step 2: 使用 small_gicp 验证候选帧
    result = verifyCandidates(cloud, candidates);

    // Step 3: 如果验证成功，更新状态
    if (result.success) {
        T_map_odom_ = result.pose_in_map * odom_pose.inverse();
        // 更新状态
        is_relocalized_ = true;
        last_matched_id_ = result.matched_keyframe_id;
        last_odom_pose_ = odom_pose;
        consecutive_track_failures_ = 0;
        LOG(INFO) << "Relocalization successful! Matched Keyframe ID: " << result.matched_keyframe_id << ", Confidence: " << result.confidence;
    }

    return result;
}

RelocResult
RelocalizationModule::trackLocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose)
{
    // 使用里程计增量预测当前位姿
    Eigen::Isometry3d predicted_pose = T_map_odom_ * odom_pose;

    RelocResult result;
    result.success = false;
    result.pose_in_map = predicted_pose;

    // 如果连续失败次数过多，重置重定位状态
    if (consecutive_track_failures_ > config_.reloc_max_track_failures) {
        is_relocalized_ = false;
        return result; // 返回失败，需要重新全局重定位
    }

    if (!cloud || cloud->empty()) {
        LOG(WARNING) << "Empty point cloud provided for tracking localization.";
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 查找最近的关键帧进行配准验证
    int64_t nearest_kf_id = findNearestKeyframe(predicted_pose);

    if (nearest_kf_id < 0) {
        LOG(WARNING) << "No nearby keyframe found for tracking localization.";
        // 没有找到近邻关键帧，使用里程计预测
        consecutive_track_failures_++;
        result.success = true;
        result.matched_keyframe_id = last_matched_id_;
        result.pose_in_map = predicted_pose;
        result.confidence = 0.5; // 较低置信度
        last_odom_pose_ = odom_pose;
        return result;
    }

    // 构建局部子图 (前后 N 帧叠加，在世界坐标系下)
    auto submap = keyframe_manager_.buildLocalSubmap(nearest_kf_id, config_.gicp_submap_size);

    if (!submap || submap->empty()) {
        LOG(WARNING) << "Failed to build local submap around keyframe " << nearest_kf_id;
        consecutive_track_failures_++;
        result.success = true;
        result.matched_keyframe_id = last_matched_id_;
        result.pose_in_map = predicted_pose;
        result.confidence = 0.5;
        last_odom_pose_ = odom_pose;
        return result;
    }

    // 使用 alignCloud 进行配准
    // submap 已经在世界坐标系下，cloud 在 body 坐标系下
    // init_guess 是预测的 body 在世界坐标系下的位姿
    MatchResult match_result = matcher_.alignCloud(submap, cloud, predicted_pose);

    // 计算置信度：使用指数衰减函数，避免负值
    // fitness_score 越小越好，0 表示完美匹配
    double scale = config_.gicp_fitness_threshold * 0.5;
    double current_confidence = std::exp(-match_result.fitness_score / scale);
    current_confidence = std::max(0.0, std::min(1.0, current_confidence));

    const bool icp_basic_ok = match_result.converged && match_result.fitness_score < config_.gicp_fitness_threshold &&
                              match_result.inlier_ratio >= config_.reloc_min_inlier_ratio;

    double delta_translation = std::numeric_limits<double>::infinity();
    double delta_rotation = std::numeric_limits<double>::infinity();
    if (match_result.converged) {
        const Eigen::Isometry3d delta = predicted_pose.inverse() * match_result.T_target_source;
        delta_translation = delta.translation().norm();
        delta_rotation = Eigen::AngleAxisd(delta.rotation()).angle();
    }

    const bool icp_delta_ok =
      (delta_translation <= config_.reloc_track_max_translation) && (delta_rotation <= config_.reloc_track_max_rotation);

    if (icp_basic_ok && icp_delta_ok) {
        // 配准成功
        // T_target_source 就是 source (body) 在 target (world) 坐标系下的位姿
        Eigen::Isometry3d refined_pose = match_result.T_target_source;

        // 融合里程计预测和 ICP 结果
        double w_icp = std::max(std::min(current_confidence, 0.92), 0.08);

        Eigen::Vector3d t_fused = w_icp * refined_pose.translation() + (1.0 - w_icp) * predicted_pose.translation();

        Eigen::Quaterniond q_icp(refined_pose.rotation());
        Eigen::Quaterniond q_odom(predicted_pose.rotation());
        Eigen::Quaterniond q_fused = q_odom.slerp(w_icp, q_icp);

        result.success = true;
        result.matched_keyframe_id = nearest_kf_id;
        result.pose_in_map = Eigen::Isometry3d::Identity();
        result.pose_in_map.translation() = t_fused;
        result.pose_in_map.linear() = q_fused.toRotationMatrix();
        result.fitness_score = match_result.fitness_score;
        result.confidence = current_confidence;

        // 更新 T_map_odom
        T_map_odom_ = result.pose_in_map * odom_pose.inverse();
        last_matched_id_ = nearest_kf_id;
        last_odom_pose_ = odom_pose;

        // 只有置信度很低时才增加失败计数
        if (current_confidence < 0.3) {
            consecutive_track_failures_++;
        } else {
            consecutive_track_failures_ = 0;
        }

        VLOG(1) << "Tracking: fitness=" << match_result.fitness_score << ", confidence=" << current_confidence << ", w_icp=" << w_icp;

    } else {
        // 配准失败，使用里程计预测
        consecutive_track_failures_++;
        LOG(WARNING) << "Tracking localization failed. converged=" << match_result.converged << " fitness=" << match_result.fitness_score
                     << " (threshold: " << config_.gicp_fitness_threshold << ")"
                     << " inlier_ratio=" << match_result.inlier_ratio << " (min: " << config_.reloc_min_inlier_ratio << ")"
                     << " delta_t=" << delta_translation << " (max: " << config_.reloc_track_max_translation << ")"
                     << " delta_R=" << delta_rotation << " (max: " << config_.reloc_track_max_rotation << ")"
                     << " Consecutive failures: " << consecutive_track_failures_;
        // 使用里程计预测
        result.success = true;
        result.matched_keyframe_id = last_matched_id_;
        result.pose_in_map = predicted_pose;
        result.confidence = 0.2;
        last_odom_pose_ = odom_pose;
    }

    return result;
}

bool
RelocalizationModule::isRelocalized() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return is_relocalized_;
}

Eigen::Isometry3d
RelocalizationModule::getMapToOdomTransform() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return T_map_odom_;
}

void
RelocalizationModule::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    is_relocalized_ = false;
    T_map_odom_ = Eigen::Isometry3d::Identity();
    last_matched_id_ = -1;
    last_odom_pose_ = Eigen::Isometry3d::Identity();
    consecutive_track_failures_ = 0;
}

void
RelocalizationModule::setMapToOdomTransform(const Eigen::Isometry3d& T_map_odom)
{
    std::lock_guard<std::mutex> lock(mutex_);
    T_map_odom_ = T_map_odom;
    is_relocalized_ = true;
}

int64_t
RelocalizationModule::getLastMatchedKeyframeId() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_matched_id_;
}

std::vector<LoopCandidate>
RelocalizationModule::searchCandidates(const PointCloudT::Ptr& cloud)
{
    std::vector<LoopCandidate> candidates;

    // 生成当前点云的 ScanContext 描述子
    Eigen::MatrixXd descriptor = loop_detector_.makeScanContext(cloud);

    if (descriptor.size() == 0) {
        return candidates;
    }

    // 获取所有地图关键帧的描述子
    auto all_descriptors = loop_detector_.getDescriptors();

    if (all_descriptors.empty()) {
        return candidates;
    }

    // 计算与所有关键帧的 ScanContext 距离
    std::vector<std::tuple<double, int, int64_t>> distances; // (distance, yaw_shift, keyframe_id)

    for (const auto& [kf_id, kf_desc] : all_descriptors) {
        if (kf_desc.size() == 0) continue;

        // 计算距离
        Eigen::MatrixXd desc_copy = descriptor;
        Eigen::MatrixXd kf_desc_copy = kf_desc;
        auto [dist, yaw_shift] = loop_detector_.computeDistance(desc_copy, kf_desc_copy);

        distances.emplace_back(dist, yaw_shift, kf_id);
    }

    // 按距离排序
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    // 取 Top-K 候选
    int num_candidates = std::min(config_.reloc_num_candidates, static_cast<int>(distances.size()));

    for (int i = 0; i < num_candidates; ++i) {
        double dist = std::get<0>(distances[i]);
        int yaw_shift = std::get<1>(distances[i]);
        int64_t kf_id = std::get<2>(distances[i]);

        // 只返回距离小于阈值的候选
        if (dist < config_.reloc_sc_dist_threshold) {
            LoopCandidate candidate;
            candidate.query_id = -1; // 当前帧没有 ID
            candidate.match_id = kf_id;
            candidate.sc_distance = dist;
            // 计算 yaw 差异 (弧度)
            auto [rows, cols] = loop_detector_.getDescriptorDimensions();
            double sector_angle = 360.0 / static_cast<double>(cols);
            candidate.yaw_diff_rad = static_cast<float>(yaw_shift) * static_cast<float>(sector_angle) * static_cast<float>(M_PI / 180.0);
            candidates.push_back(candidate);
        }
    }

    return candidates;
}

RelocResult
RelocalizationModule::verifyCandidates(const PointCloudT::Ptr& cloud, const std::vector<LoopCandidate>& candidates)
{
    RelocResult best_result;
    double best_score = std::numeric_limits<double>::max();

    for (const auto& candidate : candidates) {
        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf) {
            continue;
        }

        // 构建局部子图 (前后 N 帧叠加，在世界坐标系下)
        auto submap = keyframe_manager_.buildLocalSubmap(candidate.match_id, config_.gicp_submap_size);

        if (!submap || submap->empty()) {
            // 如果子图构建失败，回退到单帧
            if (!match_kf->cloud || match_kf->cloud->empty()) {
                continue;
            }
            // 将单帧变换到世界坐标系
            submap = std::make_shared<PointCloudT>();
            Eigen::Matrix4f transform = match_kf->pose_optimized.matrix().cast<float>();
            pcl::transformPointCloud(*match_kf->cloud, *submap, transform);
        }

        // 构建初始猜测：使用 ScanContext 估计的 yaw 差异
        // 初始猜测是 source (body) 在世界坐标系下的位姿
        Eigen::Isometry3d init_guess = match_kf->pose_optimized;
        init_guess.rotate(Eigen::AngleAxisd(candidate.yaw_diff_rad, Eigen::Vector3d::UnitZ()));

        // 执行配准 (submap 在世界坐标系，cloud 在 body 坐标系)
        MatchResult match_result = matcher_.alignCloud(submap, cloud, init_guess);

        // 计算置信度：使用指数衰减函数
        double scale = config_.gicp_fitness_threshold * 0.5;
        double current_confidence = std::exp(-match_result.fitness_score / scale);
        current_confidence = std::max(0.0, std::min(1.0, current_confidence));

        if (match_result.converged && match_result.fitness_score < config_.gicp_fitness_threshold &&
            match_result.inlier_ratio >= config_.reloc_min_inlier_ratio && current_confidence >= config_.reloc_min_confidence &&
            match_result.fitness_score < best_score) {

            best_score = match_result.fitness_score;

            // T_target_source 就是 source (body) 在 target (world) 坐标系下的位姿
            best_result.success = true;
            best_result.matched_keyframe_id = candidate.match_id;
            best_result.pose_in_map = match_result.T_target_source;
            best_result.fitness_score = match_result.fitness_score;
            best_result.confidence = current_confidence;
        }
    }

    return best_result;
}

int64_t
RelocalizationModule::findNearestKeyframe(const Eigen::Isometry3d& pose) const
{
    int64_t nearest_id = -1;
    double min_distance = std::numeric_limits<double>::max();

    auto all_keyframes = keyframe_manager_.getAllKeyframes();

    for (const auto& kf : all_keyframes) {
        if (!kf) continue;

        // 计算位置距离
        double distance = (kf->pose_optimized.translation() - pose.translation()).norm();

        // 只考虑一定范围内的关键帧
        if (distance < config_.reloc_search_radius && distance < min_distance) {
            min_distance = distance;
            nearest_id = kf->id;
        }
    }

    return nearest_id;
}

} // namespace n3mapping
