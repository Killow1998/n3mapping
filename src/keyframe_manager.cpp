#include "n3mapping/keyframe_manager.h"
#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <pcl/common/transforms.h>

namespace n3mapping {

KeyframeManager::KeyframeManager(const Config& config)
  : config_(config)
  , next_id_(0)
  , last_keyframe_(nullptr)
{
}

bool
KeyframeManager::shouldAddKeyframe(const Eigen::Isometry3d& current_pose) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    // 如果没有关键帧，应该添加第一个
    if (!last_keyframe_) {
        return true;
    }

    // 计算与上一关键帧的距离和角度差
    double distance = computeTranslationDistance(current_pose, last_keyframe_->pose_odom);
    double angle = computeRotationAngle(current_pose, last_keyframe_->pose_odom);

    if (distance >= config_.keyframe_distance_threshold) {
        return true;
    }

    if (angle >= config_.keyframe_angle_threshold) {
        return true;
    }

    return false;
}

int64_t
KeyframeManager::addKeyframe(double timestamp, const Eigen::Isometry3d& pose, const Keyframe::PointCloudT::Ptr& cloud)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // 创建新关键帧
    auto keyframe = Keyframe::create(next_id_, timestamp, pose, cloud);

    // 存储关键帧
    keyframes_[next_id_] = keyframe;
    last_keyframe_ = keyframe;

    VLOG(1) << "[KeyframeManager] Added keyframe " << next_id_ << " at timestamp " << timestamp << ", position (" << pose.translation().x()
            << ", " << pose.translation().y() << ", " << pose.translation().z() << ")";
    return next_id_++;
}

Keyframe::Ptr
KeyframeManager::getKeyframe(int64_t id) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = keyframes_.find(id);
    if (it != keyframes_.end()) {
        return it->second;
    }
    return nullptr;
}

Keyframe::Ptr
KeyframeManager::getLatestKeyframe() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return last_keyframe_;
}

std::vector<Keyframe::Ptr>
KeyframeManager::getAllKeyframes() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Keyframe::Ptr> result;
    result.reserve(keyframes_.size());

    for (const auto& pair : keyframes_) {
        result.push_back(pair.second);
    }

    return result;
}

size_t
KeyframeManager::size() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.size();
}

bool
KeyframeManager::empty() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.empty();
}

void
KeyframeManager::updateOptimizedPoses(const std::map<int64_t, Eigen::Isometry3d>& poses)
{
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& pair : poses) {
        auto it = keyframes_.find(pair.first);
        if (it != keyframes_.end()) {
            it->second->pose_optimized = pair.second;
        }
    }
}

void
KeyframeManager::loadKeyframes(const std::vector<Keyframe::Ptr>& keyframes)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // 清空现有关键帧
    keyframes_.clear();
    last_keyframe_ = nullptr;
    next_id_ = 0;

    // 加载关键帧
    for (const auto& kf : keyframes) {
        if (kf) {
            kf->is_from_loaded_map = true;
            keyframes_[kf->id] = kf;

            // 更新 next_id 为最大 ID + 1
            if (kf->id >= next_id_) {
                next_id_ = kf->id + 1;
            }

            // 更新 last_keyframe 为 ID 最大的关键帧
            if (!last_keyframe_ || kf->id > last_keyframe_->id) {
                last_keyframe_ = kf;
            }
        }
    }
    LOG(INFO) << "[KeyframeManager] Loaded " << keyframes_.size() << " keyframes, next_id = " << next_id_;
}

int64_t
KeyframeManager::getNextKeyframeId() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return next_id_;
}

void
KeyframeManager::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    last_keyframe_ = nullptr;
    next_id_ = 0;
}

Keyframe::Ptr
KeyframeManager::findNearestByTimestamp(double timestamp) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (keyframes_.empty()) {
        return nullptr;
    }

    Keyframe::Ptr nearest = nullptr;
    double min_diff = std::numeric_limits<double>::max();

    for (const auto& pair : keyframes_) {
        double diff = std::abs(pair.second->timestamp - timestamp);
        if (diff < min_diff) {
            min_diff = diff;
            nearest = pair.second;
        }
    }

    return nearest;
}

Keyframe::Ptr
KeyframeManager::findNearestByPosition(const Eigen::Vector3d& position) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (keyframes_.empty()) {
        return nullptr;
    }

    Keyframe::Ptr nearest = nullptr;
    double min_dist = std::numeric_limits<double>::max();

    for (const auto& pair : keyframes_) {
        double dist = (pair.second->getPosition() - position).norm();
        if (dist < min_dist) {
            min_dist = dist;
            nearest = pair.second;
        }
    }

    return nearest;
}

double
KeyframeManager::computeTranslationDistance(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2)
{
    return (pose1.translation() - pose2.translation()).norm();
}

double
KeyframeManager::computeRotationAngle(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2)
{
    // 计算相对旋转
    Eigen::Matrix3d R_rel = pose1.rotation().transpose() * pose2.rotation();

    // 使用 AngleAxis 提取旋转角度
    Eigen::AngleAxisd angle_axis(R_rel);

    return std::abs(angle_axis.angle());
}

bool
KeyframeManager::updateDescriptor(int64_t id, const Eigen::MatrixXd& descriptor)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = keyframes_.find(id);
    if (it == keyframes_.end()) {
        return false;
    }

    it->second->sc_descriptor = descriptor;
    return true;
}

Keyframe::PointCloudT::Ptr
KeyframeManager::buildLocalSubmap(int64_t center_id, int submap_size) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto submap = std::make_shared<Keyframe::PointCloudT>();
    
    // 获取中心关键帧
    auto center_it = keyframes_.find(center_id);
    if (center_it == keyframes_.end()) {
        LOG(WARNING) << "[KeyframeManager] Center keyframe " << center_id << " not found for submap";
        return submap;
    }
    
    // 收集前后 N 帧的 ID
    std::vector<int64_t> kf_ids;
    for (int64_t id = center_id - submap_size; id <= center_id + submap_size; ++id) {
        if (keyframes_.find(id) != keyframes_.end()) {
            kf_ids.push_back(id);
        }
    }
    
    // 叠加所有帧的点云 (变换到世界坐标系)
    for (int64_t id : kf_ids) {
        auto kf = keyframes_.at(id);
        if (!kf->cloud || kf->cloud->empty()) {
            continue;
        }
        
        // 使用优化后的位姿变换到世界坐标系
        Eigen::Matrix4f transform = kf->pose_optimized.matrix().cast<float>();
        
        // 变换点云并添加到子图
        Keyframe::PointCloudT transformed;
        pcl::transformPointCloud(*kf->cloud, transformed, transform);
        *submap += transformed;
    }
    
    VLOG(1) << "[KeyframeManager] Built submap around keyframe " << center_id 
            << " with " << kf_ids.size() << " frames, " << submap->size() << " points";
    
    return submap;
}

} // namespace n3mapping
