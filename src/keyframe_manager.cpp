// KeyframeManager: keyframe CRUD, submap building (world frame and virtual root frame).
#include "n3mapping/keyframe_manager.h"

#include <limits>
#include <boost/make_shared.hpp>
#include <pcl/common/transforms.h>

namespace n3mapping {

KeyframeManager::KeyframeManager(const Config& config)
    : config_(config), next_id_(0), last_keyframe_(nullptr) {}

bool KeyframeManager::shouldAddKeyframe(const Eigen::Isometry3d& current_pose) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_keyframe_) return true;
    double distance = computeTranslationDistance(current_pose, last_keyframe_->pose_odom);
    double angle = computeRotationAngle(current_pose, last_keyframe_->pose_odom);
    return distance >= config_.keyframe_distance_threshold || angle >= config_.keyframe_angle_threshold;
}

int64_t KeyframeManager::addKeyframe(double timestamp, const Eigen::Isometry3d& pose, const Keyframe::PointCloudT::Ptr& cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto keyframe = Keyframe::create(next_id_, timestamp, pose, cloud);
    keyframes_[next_id_] = keyframe;
    last_keyframe_ = keyframe;
    return next_id_++;
}

Keyframe::Ptr KeyframeManager::getKeyframe(int64_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = keyframes_.find(id);
    return (it != keyframes_.end()) ? it->second : nullptr;
}

Keyframe::Ptr KeyframeManager::getLatestKeyframe() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_keyframe_;
}

std::vector<Keyframe::Ptr> KeyframeManager::getAllKeyframes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Keyframe::Ptr> result;
    result.reserve(keyframes_.size());
    for (const auto& pair : keyframes_) result.push_back(pair.second);
    return result;
}

size_t KeyframeManager::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.size();
}

bool KeyframeManager::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return keyframes_.empty();
}

void KeyframeManager::updateOptimizedPoses(const std::map<int64_t, Eigen::Isometry3d>& poses) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : poses) {
        auto it = keyframes_.find(pair.first);
        if (it != keyframes_.end()) it->second->pose_optimized = pair.second;
    }
}

void KeyframeManager::loadKeyframes(const std::vector<Keyframe::Ptr>& keyframes) {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    last_keyframe_ = nullptr;
    next_id_ = 0;
    for (const auto& kf : keyframes) {
        if (!kf) continue;
        kf->is_from_loaded_map = true;
        keyframes_[kf->id] = kf;
        if (kf->id >= next_id_) next_id_ = kf->id + 1;
        if (!last_keyframe_ || kf->id > last_keyframe_->id) last_keyframe_ = kf;
    }
}

int64_t KeyframeManager::getNextKeyframeId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_id_;
}

void KeyframeManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    last_keyframe_ = nullptr;
    next_id_ = 0;
}

Keyframe::Ptr KeyframeManager::findNearestByTimestamp(double timestamp) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Keyframe::Ptr nearest = nullptr;
    double min_diff = std::numeric_limits<double>::max();
    for (const auto& pair : keyframes_) {
        double diff = std::abs(pair.second->timestamp - timestamp);
        if (diff < min_diff) { min_diff = diff; nearest = pair.second; }
    }
    return nearest;
}

Keyframe::Ptr KeyframeManager::findNearestByPosition(const Eigen::Vector3d& position) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Keyframe::Ptr nearest = nullptr;
    double min_dist = std::numeric_limits<double>::max();
    for (const auto& pair : keyframes_) {
        double dist = (pair.second->getPosition() - position).norm();
        if (dist < min_dist) { min_dist = dist; nearest = pair.second; }
    }
    return nearest;
}

bool KeyframeManager::updateDescriptor(int64_t id, const Eigen::MatrixXd& descriptor) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = keyframes_.find(id);
    if (it == keyframes_.end()) return false;
    it->second->sc_descriptor = descriptor;
    return true;
}

double KeyframeManager::computeTranslationDistance(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2) {
    return (pose1.translation() - pose2.translation()).norm();
}

double KeyframeManager::computeRotationAngle(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2) {
    Eigen::Matrix3d R_rel = pose1.rotation().transpose() * pose2.rotation();
    return std::abs(Eigen::AngleAxisd(R_rel).angle());
}

Keyframe::PointCloudT::Ptr KeyframeManager::buildLocalSubmap(int64_t center_id, int submap_size) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto submap = boost::make_shared<Keyframe::PointCloudT>();
    auto center_it = keyframes_.find(center_id);
    if (center_it == keyframes_.end()) return submap;

    for (int64_t id = center_id - submap_size; id <= center_id + submap_size; ++id) {
        auto it = keyframes_.find(id);
        if (it == keyframes_.end()) continue;
        auto kf = it->second;
        if (!kf->cloud || kf->cloud->empty()) continue;
        Eigen::Matrix4f transform = kf->pose_optimized.matrix().cast<float>();
        Keyframe::PointCloudT transformed;
        pcl::transformPointCloud(*kf->cloud, transformed, transform);
        *submap += transformed;
    }
    return submap;
}

Keyframe::PointCloudT::Ptr KeyframeManager::buildSubmapInRootFrame(int64_t center_id, int range, int64_t root_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto submap = boost::make_shared<Keyframe::PointCloudT>();

    auto root_it = keyframes_.find(root_id);
    if (root_it == keyframes_.end()) return submap;

    Eigen::Matrix4f T_root_inv = root_it->second->pose_optimized.matrix().cast<float>().inverse();

    for (int64_t id = center_id - range; id <= center_id + range; ++id) {
        auto it = keyframes_.find(id);
        if (it == keyframes_.end()) continue;
        auto kf = it->second;
        if (!kf->cloud || kf->cloud->empty()) continue;
        Eigen::Matrix4f T_kf = kf->pose_optimized.matrix().cast<float>();
        Eigen::Matrix4f T_root_kf = T_root_inv * T_kf;
        Keyframe::PointCloudT transformed;
        pcl::transformPointCloud(*kf->cloud, transformed, T_root_kf);
        *submap += transformed;
    }
    return submap;
}

} // namespace n3mapping
