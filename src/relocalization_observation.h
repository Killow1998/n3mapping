#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include "point_cloud_sampling.h"
#include "compact_kdtree_builder.h"
#include <small_gicp/ann/kdtree_omp.hpp>
#include "n3mapping/point_cloud_matcher.h"
#include "n3mapping/relocalization_performance.h"
#include "n3mapping/visibility_consistency.h"

namespace n3mapping {

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

      const double resolution = std::min(proposal.resolution * 2.0,
          config.gicp_max_correspondence_distance / (2.0 * std::sqrt(3.0)));
      if (resolution > proposal.resolution) {
        auto points = registration_detail::sampleRepresentativeCoordinates(
            *proposal.cloud, resolution);
        if (points->size() >= 10) proposal.cloud = std::move(points);
      }
    }

    for (const auto& level : registration.plane_levels) {
      if (level.cloud == registration.refine_cloud || level.cloud->empty()) continue;
      const PointCloudMatcher::SmallGicpKdTree order(level.cloud,
          registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilderOMP>(
              *level.cloud, false, config.num_threads));
    }
    if (registration.refine_cloud && !registration.refine_cloud->empty()) {
      const PointCloudMatcher::SmallGicpKdTree order(registration.refine_cloud,
          registration_detail::CompactKdTreeBuilder<small_gicp::KdTreeBuilderOMP>(
              *registration.refine_cloud, true, config.num_threads));
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
}
