#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include "point_cloud_sampling.h"
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
    if (!registration.plane_levels.empty()) {
      auto& proposal = registration.plane_levels.front();
      // Only the broad search proposal is sparse; final geometry is intact.
      // Limit extra sampling displacement to half the correspondence radius.
      const double resolution = std::min(proposal.resolution * 2.0,
          config.gicp_max_correspondence_distance / (2.0 * std::sqrt(3.0)));
      if (resolution > proposal.resolution) {
        auto points = registration_detail::sampleCoordinates(
            *proposal.cloud, resolution, config.num_threads);
        if (points->size() >= 10) proposal.cloud = std::move(points);
      }
    }
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
