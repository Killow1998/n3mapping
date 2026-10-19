#pragma once

#include <chrono>
#include <glog/logging.h>
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_performance.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

// Registration and visibility share one immutable observation throughout a
// search call. Preparing it lazily avoids work when retrieval has no support.
struct RelocalizationObservation {
  PointCloudMatcher::PreparedSource registration;
  PreparedVisibilityObservation visibility;

  void prepare(const PointCloudMatcher::PointCloudT::Ptr& cloud,
               PointCloudMatcher& matcher, const Config& config,
               RelocalizationPerformance& cost) {
    if (prepared_) return;
    const auto source_started = std::chrono::steady_clock::now();
    registration = matcher.prepareSourceCloud(cloud);
    const auto visibility_started = std::chrono::steady_clock::now();
    visibility = prepareVisibilityObservation(*cloud, visibilityOptionsFromConfig(config));
    cost.source_preparations = 1;
    cost.visibility_preparations = 1;
    cost.source_prepare_ms = std::chrono::duration<double, std::milli>(
        visibility_started - source_started).count();
    cost.visibility_prepare_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - visibility_started).count();
    prepared_ = true;
    VLOG(1) << "[RelocQueryCost] source_preparations=1 source_prepare_ms="
            << cost.source_prepare_ms << " visibility_preparations=1 visibility_prepare_ms="
            << cost.visibility_prepare_ms;
  }

private:
  bool prepared_ = false;
};
} // namespace n3mapping
