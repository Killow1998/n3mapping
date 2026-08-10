// KeyframeManager: keyframe CRUD, submap building (world frame and virtual root frame).
#include "n3mapping/keyframe_manager.h"

#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <pcl/common/transforms.h>
#include "n3mapping/pcl_compat.h"

namespace n3mapping {
namespace {

std::uint64_t nextMapGeneration() {
    static std::atomic<std::uint64_t> generation{0};
    return generation.fetch_add(1, std::memory_order_relaxed) + 1;
}

bool validSessionInfo(const MapSessionInfo& session) {
    return session.id != kInvalidMapSessionId &&
           std::isfinite(session.start_timestamp) &&
           session.T_map_session_initial.matrix().allFinite();
}

}  // namespace

KeyframeManager::KeyframeManager(const Config& config)
    : config_(config), next_id_(0), last_keyframe_(nullptr) {
    revision_.generation = nextMapGeneration();
}

bool KeyframeManager::shouldAddKeyframe(const Eigen::Isometry3d& current_pose) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_keyframe_) return true;
    double distance = computeTranslationDistance(current_pose, last_keyframe_->pose_odom);
    double angle = computeRotationAngle(current_pose, last_keyframe_->pose_odom);
    return distance >= config_.keyframe_distance_threshold || angle >= config_.keyframe_angle_threshold;
}

bool KeyframeManager::shouldAddKeyframeInSession(
    MapSessionId session_id,
    const Eigen::Isometry3d& current_pose_in_session) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = keyframes_.rbegin(); it != keyframes_.rend(); ++it) {
        const auto& keyframe = it->second;
        if (!keyframe || keyframe->session_id != session_id) continue;
        const double distance = computeTranslationDistance(
            current_pose_in_session, keyframe->pose_odom);
        const double angle = computeRotationAngle(
            current_pose_in_session, keyframe->pose_odom);
        return distance >= config_.keyframe_distance_threshold ||
               angle >= config_.keyframe_angle_threshold;
    }
    return true;
}

int64_t KeyframeManager::addKeyframe(double timestamp, const Eigen::Isometry3d& pose, const Keyframe::PointCloudT::Ptr& cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sessions_.find(0) == sessions_.end()) {
        MapSessionInfo session;
        session.start_timestamp = timestamp;
        sessions_.emplace(0, std::move(session));
    }
    auto keyframe = Keyframe::create(next_id_, timestamp, pose, cloud);
    keyframes_[next_id_] = keyframe;
    last_keyframe_ = keyframe;
    ++revision_.structure_revision;
    return next_id_++;
}

int64_t KeyframeManager::addKeyframe(
    double timestamp, const Eigen::Isometry3d& pose_in_session,
    const Eigen::Isometry3d& initial_pose_in_map,
    const Keyframe::PointCloudT::Ptr& cloud, MapSessionId session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sessions_.find(session_id) == sessions_.end()) return -1;
    auto keyframe = Keyframe::create(
        next_id_, timestamp, pose_in_session, initial_pose_in_map, cloud,
        session_id);
    keyframes_[next_id_] = keyframe;
    last_keyframe_ = keyframe;
    ++revision_.structure_revision;
    return next_id_++;
}

bool KeyframeManager::removeLatestKeyframe(int64_t expected_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_keyframe_ || last_keyframe_->id != expected_id ||
        next_id_ != expected_id + 1) {
        return false;
    }

    const auto erased = keyframes_.erase(expected_id);
    if (erased != 1u) return false;

    next_id_ = expected_id;
    last_keyframe_ = keyframes_.empty() ? nullptr : keyframes_.rbegin()->second;
    ++revision_.structure_revision;
    return true;
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

KeyframeMapRevision KeyframeManager::revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revision_;
}

void KeyframeManager::updateOptimizedPoses(const std::map<int64_t, Eigen::Isometry3d>& poses) {
    std::lock_guard<std::mutex> lock(mutex_);
    bool changed = false;
    for (const auto& pair : poses) {
        auto it = keyframes_.find(pair.first);
        if (it != keyframes_.end() &&
            !it->second->pose_optimized.matrix().isApprox(
                pair.second.matrix(), 1e-12)) {
            it->second->pose_optimized = pair.second;
            changed = true;
        }
    }
    if (changed) {
        ++revision_.pose_revision;
    }
}

bool KeyframeManager::loadKeyframes(
    const std::vector<Keyframe::Ptr>& keyframes) {
    MapSessionInfo legacy;
    legacy.source_frame_id = "legacy";
    legacy.loaded = true;
    bool have_timestamp = false;
    for (const auto& keyframe : keyframes) {
        if (!keyframe) continue;
        keyframe->session_id = 0;
        if (!have_timestamp || keyframe->timestamp < legacy.start_timestamp) {
            legacy.start_timestamp = keyframe->timestamp;
            have_timestamp = true;
        }
    }
    return loadKeyframes(keyframes, {legacy});
}

bool KeyframeManager::loadKeyframes(
    const std::vector<Keyframe::Ptr>& keyframes,
    const std::vector<MapSessionInfo>& sessions) {
    std::map<MapSessionId, MapSessionInfo> next_sessions;
    for (const auto& session : sessions) {
        if (!validSessionInfo(session) ||
            !next_sessions.emplace(session.id, session).second) {
            return false;
        }
    }
    if (next_sessions.empty() && !keyframes.empty()) return false;

    std::map<int64_t, Keyframe::Ptr> next_keyframes;
    Keyframe::Ptr next_last_keyframe;
    int64_t next_id = 0;
    for (const auto& kf : keyframes) {
        if (!kf) continue;
        if (next_sessions.find(kf->session_id) == next_sessions.end()) {
            return false;
        }
        kf->is_from_loaded_map = true;
        next_keyframes[kf->id] = kf;
        if (kf->id >= next_id) next_id = kf->id + 1;
        if (!next_last_keyframe || kf->id > next_last_keyframe->id) {
            next_last_keyframe = kf;
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_ = std::move(next_keyframes);
    sessions_ = std::move(next_sessions);
    last_keyframe_ = std::move(next_last_keyframe);
    next_id_ = next_id;
    revision_.generation = nextMapGeneration();
    ++revision_.structure_revision;
    ++revision_.pose_revision;
    return true;
}

void KeyframeManager::swapWith(KeyframeManager& other) {
    if (this == &other) return;
    std::lock(mutex_, other.mutex_);
    std::lock_guard<std::mutex> lock_this(mutex_, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.mutex_, std::adopt_lock);
    std::swap(config_, other.config_);
    std::swap(keyframes_, other.keyframes_);
    std::swap(sessions_, other.sessions_);
    std::swap(next_id_, other.next_id_);
    std::swap(last_keyframe_, other.last_keyframe_);
    revision_.generation = nextMapGeneration();
    other.revision_.generation = nextMapGeneration();
    ++revision_.structure_revision;
    ++revision_.pose_revision;
    ++other.revision_.structure_revision;
    ++other.revision_.pose_revision;
}

int64_t KeyframeManager::getNextKeyframeId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_id_;
}

void KeyframeManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    keyframes_.clear();
    sessions_.clear();
    last_keyframe_ = nullptr;
    next_id_ = 0;
    revision_.generation = nextMapGeneration();
    ++revision_.structure_revision;
    ++revision_.pose_revision;
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

bool KeyframeManager::addSession(const MapSessionInfo& session) {
    if (!validSessionInfo(session)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.emplace(session.id, session).second;
}

bool KeyframeManager::removeSessionIfEmpty(MapSessionId session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& item : keyframes_) {
        if (item.second && item.second->session_id == session_id) return false;
    }
    return sessions_.erase(session_id) == 1u;
}

bool KeyframeManager::hasSession(MapSessionId session_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.find(session_id) != sessions_.end();
}

std::vector<MapSessionInfo> KeyframeManager::getSessions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MapSessionInfo> result;
    result.reserve(sessions_.size());
    for (const auto& item : sessions_) result.push_back(item.second);
    return result;
}

MapSessionId KeyframeManager::getNextSessionId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (sessions_.empty()) return 0;
    if (sessions_.rbegin()->first == kInvalidMapSessionId - 1u) {
        return kInvalidMapSessionId;
    }
    return sessions_.rbegin()->first + 1u;
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
    auto submap = pcl::make_shared<Keyframe::PointCloudT>();
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
    auto submap = pcl::make_shared<Keyframe::PointCloudT>();

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

Keyframe::PointCloudT::Ptr KeyframeManager::buildCausalSubmapInRootFrame(int64_t center_id, int range, int64_t root_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto submap = pcl::make_shared<Keyframe::PointCloudT>();

    auto root_it = keyframes_.find(root_id);
    if (root_it == keyframes_.end()) return submap;

    Eigen::Matrix4f T_root_inv = root_it->second->pose_optimized.matrix().cast<float>().inverse();

    for (int64_t id = center_id - range; id <= center_id; ++id) {
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
