#include "n3mapping/map_extension_module.h"

#include <algorithm>

namespace n3mapping {

MapExtensionModule::MapExtensionModule(const Config& config,
                                       KeyframeManager& keyframe_manager,
                                       LoopDetector& loop_detector,
                                       PointCloudMatcher& matcher,
                                       GraphOptimizer& optimizer,
                                       MapSerializer& serializer,
                                       RelocalizationModule& relocalization)
  : config_(config)
  , keyframe_manager_(keyframe_manager)
  , loop_detector_(loop_detector)
  , matcher_(matcher)
  , optimizer_(optimizer)
  , serializer_(serializer)
  , relocalization_(relocalization)
  , state_(MapExtensionState::NOT_INITIALIZED)
  , original_keyframe_count_(0)
  , original_max_keyframe_id_(-1)
  , cross_loop_count_(0)
  , last_keyframe_pose_(Eigen::Isometry3d::Identity())
  , last_keyframe_id_(-1)
{
}

bool
MapExtensionModule::loadExistingMap(const std::string& map_path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Requirements: 12.1 - 加载已有地图
    if (!serializer_.loadMap(map_path, keyframe_manager_, loop_detector_, optimizer_)) {
        return false;
    }

    // 记录原始地图信息
    original_keyframe_count_ = keyframe_manager_.size();

    // 获取原始地图最大关键帧 ID
    auto all_keyframes = keyframe_manager_.getAllKeyframes();
    original_max_keyframe_id_ = -1;
    for (const auto& kf : all_keyframes) {
        if (kf && kf->id > original_max_keyframe_id_) {
            original_max_keyframe_id_ = kf->id;
        }
    }

    // 标记所有已加载的关键帧为来自原始地图
    for (auto& kf : all_keyframes) {
        if (kf) {
            kf->is_from_loaded_map = true;
        }
    }

    state_ = MapExtensionState::MAP_LOADED;
    return true;
}

bool
MapExtensionModule::performInitialRelocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ != MapExtensionState::MAP_LOADED) {
        return false;
    }

    RelocResult result = relocalization_.relocalize(cloud, odom_pose);

    if (!result.success) {
        return false;
    }

    // 计算 T_map_odom
    Eigen::Isometry3d T_map_odom = result.pose_in_map * odom_pose.inverse();
    relocalization_.setMapToOdomTransform(T_map_odom);

    // 记录初始位姿
    last_keyframe_pose_ = result.pose_in_map;
    last_keyframe_id_ = result.matched_keyframe_id;

    state_ = MapExtensionState::RELOCALIZED;
    return true;
}

int64_t
MapExtensionModule::processNewKeyframe(double timestamp, const Eigen::Isometry3d& odom_pose, const PointCloudT::Ptr& cloud)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ != MapExtensionState::RELOCALIZED && state_ != MapExtensionState::EXTENDING) {
        return -1;
    }

    // 将里程计位姿转换到地图坐标系
    Eigen::Isometry3d T_map_odom = relocalization_.getMapToOdomTransform();
    Eigen::Isometry3d pose_in_map = T_map_odom * odom_pose;

    // 检查是否需要添加新关键帧
    if (!keyframe_manager_.shouldAddKeyframe(pose_in_map)) {
        return -1;
    }

    // Requirements: 12.3 - 新关键帧 ID 从原始地图最大 ID + 1 开始
    int64_t new_kf_id = keyframe_manager_.addKeyframe(timestamp, pose_in_map, cloud);

    // 添加描述子
    loop_detector_.addDescriptor(new_kf_id, cloud);

    // 添加里程计边
    if (last_keyframe_id_ >= 0) {
        auto last_kf = keyframe_manager_.getKeyframe(last_keyframe_id_);
        if (last_kf) {
            EdgeInfo edge;
            edge.from_id = last_keyframe_id_;
            edge.to_id = new_kf_id;
            edge.measurement = last_kf->pose_optimized.inverse() * pose_in_map;
            edge.information = Eigen::Matrix<double, 6, 6>::Identity();
            edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.odom_noise_position * config_.odom_noise_position);
            edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.odom_noise_rotation * config_.odom_noise_rotation);
            edge.type = EdgeType::ODOMETRY;

            optimizer_.addOdometryEdge(edge);
        }
    } else {
        // 第一个新关键帧，添加与重定位匹配帧的约束
        int64_t matched_id = relocalization_.getLastMatchedKeyframeId();
        if (matched_id >= 0) {
            auto matched_kf = keyframe_manager_.getKeyframe(matched_id);
            if (matched_kf) {
                addRelocalizationConstraint(new_kf_id, matched_id, matched_kf->pose_optimized.inverse() * pose_in_map);
            }
        }
    }

    // 更新状态
    last_keyframe_pose_ = pose_in_map;
    last_keyframe_id_ = new_kf_id;
    state_ = MapExtensionState::EXTENDING;

    return new_kf_id;
}

int
MapExtensionModule::detectCrossLoops(int64_t new_keyframe_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (state_ != MapExtensionState::EXTENDING) {
        return 0;
    }

    auto new_kf = keyframe_manager_.getKeyframe(new_keyframe_id);
    if (!new_kf || !new_kf->cloud) {
        return 0;
    }

    // Requirements: 12.4 - 检测新旧关键帧之间的回环
    std::vector<LoopCandidate> candidates = loop_detector_.detectLoopCandidates(new_keyframe_id);

    int cross_loops_found = 0;

    for (const auto& candidate : candidates) {
        // 只处理与原始地图关键帧的回环
        if (!isFromOriginalMap(candidate.match_id)) {
            continue;
        }

        auto match_kf = keyframe_manager_.getKeyframe(candidate.match_id);
        if (!match_kf || !match_kf->cloud) {
            continue;
        }

        // 验证回环
        VerifiedLoop verified = loop_detector_.verifyLoopCandidate(candidate, new_kf, match_kf, matcher_);

        if (verified.verified) {
            // Requirements: 12.5 - 添加回环边并触发优化
            EdgeInfo loop_edge;
            loop_edge.from_id = candidate.match_id;
            loop_edge.to_id = new_keyframe_id;
            loop_edge.measurement = verified.T_match_query;
            loop_edge.information = verified.information;
            loop_edge.type = EdgeType::LOOP;

            optimizer_.addLoopEdge(loop_edge);
            cross_loops_found++;
            cross_loop_count_++;
        }
    }

    // 如果检测到回环，执行优化
    if (cross_loops_found > 0) {
        optimizer_.incrementalOptimize();

        // 更新所有关键帧位姿
        auto optimized_poses = optimizer_.getOptimizedPoses();
        keyframe_manager_.updateOptimizedPoses(optimized_poses);
    }

    return cross_loops_found;
}

bool
MapExtensionModule::saveExtendedMap(const std::string& map_path)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Requirements: 12.6, 12.7 - 保存扩展后的地图，保留原有关键帧和约束完整性
    return serializer_.saveMap(map_path, keyframe_manager_, loop_detector_, optimizer_);
}

MapExtensionState
MapExtensionModule::getState() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

size_t
MapExtensionModule::getOriginalKeyframeCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return original_keyframe_count_;
}

size_t
MapExtensionModule::getNewKeyframeCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframe_manager_.size() - original_keyframe_count_;
}

size_t
MapExtensionModule::getCrossLoopCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return cross_loop_count_;
}

bool
MapExtensionModule::isFromOriginalMap(int64_t keyframe_id) const
{
    // 原始地图的关键帧 ID <= original_max_keyframe_id_
    return keyframe_id <= original_max_keyframe_id_;
}

void
MapExtensionModule::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);

    state_ = MapExtensionState::NOT_INITIALIZED;
    original_keyframe_count_ = 0;
    original_max_keyframe_id_ = -1;
    cross_loop_count_ = 0;
    last_keyframe_pose_ = Eigen::Isometry3d::Identity();
    last_keyframe_id_ = -1;

    relocalization_.reset();
}

void
MapExtensionModule::addRelocalizationConstraint(int64_t new_keyframe_id, int64_t matched_keyframe_id, const Eigen::Isometry3d& T_match_new)
{
    EdgeInfo edge;
    edge.from_id = matched_keyframe_id;
    edge.to_id = new_keyframe_id;
    edge.measurement = T_match_new;
    edge.information = Eigen::Matrix<double, 6, 6>::Identity();
    edge.information.block<3, 3>(0, 0) *= 1.0 / (config_.loop_noise_position * config_.loop_noise_position);
    edge.information.block<3, 3>(3, 3) *= 1.0 / (config_.loop_noise_rotation * config_.loop_noise_rotation);
    edge.type = EdgeType::LOOP;

    optimizer_.addLoopEdge(edge);
}

} // namespace n3mapping
