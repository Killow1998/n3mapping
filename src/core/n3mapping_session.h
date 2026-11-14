#pragma once

#include "n3mapping/config.h"
#include "n3mapping/graph_optimizer.h"
#include "n3mapping/keyframe_manager.h"
#include "n3mapping/loop_closure_manager.h"
#include "n3mapping/loop_detector.h"
#include "n3mapping/map_serializer.h"
#include "n3mapping/mapping_resuming.h"
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/submap_builder.h"
#include "n3mapping/world_localizing.h"

namespace n3mapping::core {

// Replaced as one unit when loading a map; references are wired only here.
struct N3MappingSession {
  explicit N3MappingSession(const Config& config)
      : keyframes(config), matcher(config), detector(config), loops(config),
        optimizer(config), serializer(config), submaps(config),
        localizer(config, keyframes, detector, matcher),
        resuming(config, keyframes, detector, matcher, optimizer, serializer,
                 localizer, &submaps) {}

  KeyframeManager keyframes;
  PointCloudMatcher matcher;
  LoopDetector detector;
  LoopClosureManager loops;
  GraphOptimizer optimizer;
  MapSerializer serializer;
  SubmapBuilder submaps;
  WorldLocalizing localizer;
  MappingResuming resuming;
};

} // namespace n3mapping::core
