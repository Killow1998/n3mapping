// MappingResuming: map extension — load existing map, relocalize, add new keyframes, detect cross-loops.
#pragma once

#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping {

enum class MappingResumingState {
    NOT_INITIALIZED,
    MAP_LOADED,
    RELOCALIZED,
    EXTENDING
};

class MappingResuming {
public:
    using PointCloudT = pcl::PointCloud<pcl::PointXYZI>;

    MappingResuming(const Config& config, KeyframeManager& keyframe_manager,
                    LoopDetector& loop_detector, PointCloudMatcher& matcher,
                    GraphOptimizer& optimizer, MapSerializer& serializer,
                    WorldLocalizing& world_localizing);

    bool loadExistingMap(const std::string& map_path);
    bool performInitialRelocalization(const PointCloudT::Ptr& cloud, const Eigen::Isometry3d& odom_pose);
    int64_t processNewKeyframe(double timestamp, const Eigen::Isometry3d& odom_pose, const PointCloudT::Ptr& cloud);
    int detectCrossLoops(int64_t new_keyframe_id);
    bool saveExtendedMap(const std::string& map_path);
    MappingResumingState getState() const;
    size_t getOriginalKeyframeCount() const;
    size_t getNewKeyframeCount() const;
    size_t getCrossLoopCount() const;
    bool isFromOriginalMap(int64_t keyframe_id) const;
    void reset();

private:
    void addRelocalizationConstraint(int64_t new_keyframe_id, int64_t matched_keyframe_id, const Eigen::Isometry3d& T_match_new);

    Config config_;
    KeyframeManager& keyframe_manager_;
    LoopDetector& loop_detector_;
    PointCloudMatcher& matcher_;
    GraphOptimizer& optimizer_;
    LoopClosureManager loop_closure_manager_;
    MapSerializer& serializer_;
    WorldLocalizing& world_localizing_;

    MappingResumingState state_;
    size_t original_keyframe_count_;
    int64_t original_max_keyframe_id_;
    size_t cross_loop_count_;
    Eigen::Isometry3d last_keyframe_pose_;
    int64_t last_keyframe_id_;
    mutable std::mutex mutex_;
};

} // namespace n3mapping
