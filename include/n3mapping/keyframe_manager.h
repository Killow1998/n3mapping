#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/keyframe.h"
#include "n3mapping/keyframe_map_revision.h"

namespace n3mapping {

class KeyframeManager {
public:
    using Ptr = std::shared_ptr<KeyframeManager>;

    explicit KeyframeManager(const Config& config);

    bool shouldAddKeyframe(const Eigen::Isometry3d& current_pose) const;
    bool shouldAddKeyframeInSession(
        MapSessionId session_id,
        const Eigen::Isometry3d& current_pose_in_session) const;
    int64_t addKeyframe(double timestamp, const Eigen::Isometry3d& pose, const Keyframe::PointCloudT::Ptr& cloud);
    int64_t addKeyframe(
        double timestamp, const Eigen::Isometry3d& pose_in_session,
        const Eigen::Isometry3d& initial_pose_in_map,
        const Keyframe::PointCloudT::Ptr& cloud, MapSessionId session_id);

    bool removeLatestKeyframe(int64_t expected_id);
    Keyframe::Ptr getKeyframe(int64_t id) const;
    Keyframe::Ptr getLatestKeyframe() const;
    std::vector<Keyframe::Ptr> getAllKeyframes() const;
    size_t size() const;
    bool empty() const;
    KeyframeMapRevision revision() const;
    void updateOptimizedPoses(const std::map<int64_t, Eigen::Isometry3d>& poses);
    bool loadKeyframes(const std::vector<Keyframe::Ptr>& keyframes);
    bool loadKeyframes(const std::vector<Keyframe::Ptr>& keyframes,
                       const std::vector<MapSessionInfo>& sessions);
    void swapWith(KeyframeManager& other);
    int64_t getNextKeyframeId() const;
    void clear();
    Keyframe::Ptr findNearestByTimestamp(double timestamp) const;
    Keyframe::Ptr findNearestByPosition(const Eigen::Vector3d& position) const;

    bool addSession(const MapSessionInfo& session);
    bool removeSessionIfEmpty(MapSessionId session_id);
    bool hasSession(MapSessionId session_id) const;
    std::vector<MapSessionInfo> getSessions() const;
    MapSessionId getNextSessionId() const;

    Keyframe::PointCloudT::Ptr buildLocalSubmap(int64_t center_id, int submap_size) const;

    Keyframe::PointCloudT::Ptr buildSubmapInRootFrame(
        int64_t center_id, int range, int64_t root_id,
        int64_t excluded_keyframe_id = -1) const;
    enum class PoseSource { Optimized, Odometry };

    Keyframe::PointCloudT::Ptr buildCausalSubmapInRootFrame(
        int64_t center_id, int range, int64_t root_id,
        PoseSource poses = PoseSource::Optimized) const;

private:
    Config config_;
    std::map<int64_t, Keyframe::Ptr> keyframes_;
    std::map<MapSessionId, MapSessionInfo> sessions_;
    int64_t next_id_;
    Keyframe::Ptr last_keyframe_;
    KeyframeMapRevision revision_;
    mutable std::mutex mutex_;

    static double computeTranslationDistance(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2);
    static double computeRotationAngle(const Eigen::Isometry3d& pose1, const Eigen::Isometry3d& pose2);
};

}
