// Owns ROS-free backend components for a single n3mapping session.
#pragma once

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping {
namespace core {

class N3MappingSession {
  public:
    explicit N3MappingSession(const Config& config);

    const Config& config() const { return config_; }
    KeyframeManager& keyframeManager() { return keyframe_manager_; }
    const KeyframeManager& keyframeManager() const { return keyframe_manager_; }
    PointCloudMatcher& pointCloudMatcher() { return point_cloud_matcher_; }
    LoopDetector& loopDetector() { return loop_detector_; }
    const LoopDetector& loopDetector() const { return loop_detector_; }
    LoopClosureManager& loopClosureManager() { return loop_closure_manager_; }
    GraphOptimizer& graphOptimizer() { return graph_optimizer_; }
    const GraphOptimizer& graphOptimizer() const { return graph_optimizer_; }
    MapSerializer& mapSerializer() { return map_serializer_; }
    WorldLocalizing& worldLocalizing() { return world_localizing_; }
    MappingResuming& mappingResuming() { return mapping_resuming_; }

  private:
    Config config_;
    KeyframeManager keyframe_manager_;
    PointCloudMatcher point_cloud_matcher_;
    LoopDetector loop_detector_;
    LoopClosureManager loop_closure_manager_;
    GraphOptimizer graph_optimizer_;
    MapSerializer map_serializer_;
    WorldLocalizing world_localizing_;
    MappingResuming mapping_resuming_;
};

}  // namespace core
}  // namespace n3mapping
