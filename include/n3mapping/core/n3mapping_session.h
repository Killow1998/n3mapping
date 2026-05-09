// ROS-independent owner for the n3mapping core component graph.
#pragma once

#include <vector>

#include "n3mapping/config.h"
#include "n3mapping/core/localization_mode_processor.h"
#include "n3mapping/core/map_extension_mode_processor.h"
#include "n3mapping/core/mapping_mode_processor.h"
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
    struct LoopProcessingResult {
        bool query_exists = false;
        bool candidates_found = false;
        std::vector<VerifiedLoop> selected_loops;
        size_t edges_added = 0;
        bool optimized = false;
    };

    explicit N3MappingSession(const Config& config);

    const Config& config() const;
    KeyframeManager& keyframes();
    const KeyframeManager& keyframes() const;
    LoopDetector& loopDetector();
    LoopClosureManager& loopClosureManager();
    PointCloudMatcher& matcher();
    GraphOptimizer& graphOptimizer();
    MapSerializer& mapSerializer();
    WorldLocalizing& worldLocalizing();
    MappingResuming& mappingResuming();
    MappingModeProcessor& mappingModeProcessor();
    LocalizationModeProcessor& localizationModeProcessor();
    MapExtensionModeProcessor& mapExtensionModeProcessor();
    LoopProcessingResult processLoopClosureForKeyframe(int64_t query_id);
    bool loadMapForLocalization(const std::string& path);
    bool loadMapForExtension(const std::string& path);
    bool saveCurrentMap(const std::string& path);
    bool saveExtendedMap(const std::string& path);
    bool saveGlobalMap(const std::string& path);

private:
    Config config_;
    KeyframeManager keyframe_manager_;
    LoopDetector loop_detector_;
    LoopClosureManager loop_closure_manager_;
    PointCloudMatcher matcher_;
    GraphOptimizer optimizer_;
    MapSerializer serializer_;
    WorldLocalizing world_localizing_;
    MappingResuming mapping_resuming_;
    MappingModeProcessor mapping_mode_processor_;
    LocalizationModeProcessor localization_mode_processor_;
    MapExtensionModeProcessor map_extension_mode_processor_;
};

}  // namespace core
}  // namespace n3mapping
