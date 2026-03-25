// KeyframeManager: keyframe selection, storage, retrieval, and local submap construction.
#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"

namespace n3mapping {

class KeyframeManager {
public:
    using Ptr = std::shared_ptr<KeyframeManager>;

    explicit KeyframeManager(const Config& config);

    bool shouldAddKeyframe(const Eigen::Isometry3d& current_pose) const;
    int64_t addKeyframe(double timestamp, const Eigen::Isometry3d& pose, const Keyframe::PointCloudT::Ptr& cloud);
    Keyframe::Ptr getKeyframe(int64_t id) const;
    Keyframe::Ptr getLatestKeyframe() const;
    std::vector<Keyframe::Ptr> getAllKeyframes() const;
    size_t size() const;
    bool empty() const;
    void updateOptimizedPoses(const std::map<int64_t, Eigen::Isometry3d>& poses);
    void loadKeyframes(const std::vector<Keyframe::Ptr>& keyframes);
    int64_t getNextKeyframeId() const;
    void clear();
    Keyframe::Ptr findNearestByTimestamp(double timestamp) const;
    Keyframe::Ptr findNearestByPosition(const Eigen::Vector3d& position) const;
    bool updateDescriptor(int64_t id, const Eigen::MatrixXd& descriptor);

    Keyframe::PointCloudT::Ptr buildLocalSubmap(int64_t center_id, int submap_size) const;

    Keyframe::PointCloudT::Ptr buildSubmapInRootFrame(int64_t center_id, int range, int64_t root_id) const;

private:
    Config config_;
    std::map<int64_t, Keyframe::Ptr> keyframes_;
    int64_t next_id_;
    Keyframe::Ptr last_keyframe_;
    mutable std::mutex mutex_;

    static double computeTranslationDistance(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2);
    static double computeRotationAngle(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2);
};

} // namespace n3mapping
