#include "n3mapping/world_localizing.h"
#include "point_cloud_target_view.h"
#include "registration_search.h"
#include "tracking_geometry.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <glog/logging.h>

namespace n3mapping {
void WorldLocalizing::recoverWeakTranslation(
    const PointCloudMatcher::PreparedTarget &target,
    const PointCloudMatcher::PreparedSource &prepared_source,
    const Eigen::Isometry3d &predicted_pose,
    const std::optional<Eigen::Vector3d> &previous_proposal,
    MatchResult &match_result, WeakTranslationAxes &weak_axes,
    Eigen::Vector3d &confirmed_shift) {
  using namespace registration_detail;
  const auto started = std::chrono::steady_clock::now();
  const auto &level = localization_atlas_->preparedTarget().refine_level;
  const auto &source = *prepared_source.refine_cloud;
  const Eigen::Vector3d axis = weak_axes->front();
  const auto &endpoint = match_result.T_target_source;
  const Eigen::Vector3d local_axis = endpoint.rotation().transpose() * axis;
  std::vector<Eigen::Vector4d> points;
  for (std::size_t i = 0; i < source.size(); ++i) {
    if (local_axis.dot(source.cov(i).topLeftCorner<3, 3>() * local_axis) < 0.5)
      points.push_back(endpoint * source.point(i));
  }
  const double radius = config_.icp_refine_max_correspondence_distance;
  const double spacing = config_.gicp_downsampling_resolution;
  const int steps =
      static_cast<int>(config_.reloc_track_max_translation / spacing);
  double center_support = 0;
  const auto recover = [&](const auto &tree, const auto &surface) {
    const auto score = [&](int step) {
      const Eigen::Vector3d shift = axis * (step * spacing);
      double total = 0.0;
      for (const auto &point : points) {
        auto query = point;
        query.head<3>() += shift;
        std::size_t i;
        double distance_sq;
        if (tree.nearest(query, &i, &distance_sq) &&
            std::abs(axis.dot(surface.normal(i).template head<3>())) >
                std::sqrt(0.5))
          total += 1.0 - distance_sq / (radius * radius);
      }
      return total;
    };
    center_support = score(0);
    std::vector<double> scores(2 * steps + 1);
    for (int step = -steps; step <= steps; ++step)
      scores[step + steps] = step ? score(step) : center_support;
    const auto peak = std::max_element(scores.begin(), scores.end());
    const int best_step = static_cast<int>(peak - scores.begin()) - steps;
    const int same_peak_steps =
        static_cast<int>(std::lround(2 * radius / spacing));
    if (std::abs(best_step) <= same_peak_steps)
      return;
    double alternative = center_support;
    for (int step = -steps; step <= steps; ++step)
      if (std::abs(step - best_step) > same_peak_steps)
        alternative = std::max(alternative, scores[step + steps]);
    if (*peak - alternative < config_.gicp_num_neighbors)
      return;
    const double shift = best_step * spacing;
    auto seed = endpoint;
    seed.translation() += axis * shift;
    auto local = matcher_.getSettings();
    local.type = small_gicp::RegistrationSetting::GICP;
    local.downsampling_resolution = config_.icp_refine_downsampling_resolution;
    local.max_correspondence_distance = radius;
    local.max_iterations = config_.icp_refine_max_iterations;
    auto recovered =
        matcher_.alignPrepared(target, prepared_source, seed, local);
    const auto delta = seed.inverse() * recovered.T_target_source;
    const bool within_search =
        delta.translation().norm() <= radius &&
        (predicted_pose.inverse() * recovered.T_target_source)
                .translation()
                .norm() <= config_.reloc_track_max_translation &&
        Eigen::AngleAxisd(delta.rotation()).angle() <=
            config_.reloc_track_max_rotation;
    if (!within_search ||
        !evaluateRelocMatchQuality(recovered, config_).accepted)
      return;
    WeakTranslationAxes recovered_axes;
    auto pose = supportedTrackingPose(source, config_.gicp_num_neighbors, seed,
                                      recovered, recovered_axes);
    if (!pose)
      return;
    auto checked = matcher_.evaluatePreparedPose(target, prepared_source, *pose,
                                                 recovered.metric);
    if (!evaluateRelocMatchQuality(checked, config_).accepted)
      return;
    const Eigen::Vector3d proposal =
        axis * axis.dot(checked.T_target_source.translation() -
                        endpoint.translation());
    if (!(previous_proposal &&
          (proposal - *previous_proposal).norm() <= radius)) {
      directional_proposal_ = proposal;
      return;
    }
    double gain = 0.0;
    const double limit = radius * radius;
    for (std::size_t i = 0; i < source.size(); ++i) {
      const auto &point = source.point(i);
      std::size_t nearest;
      double here, there;
      const bool hit_here = tree.nearest(endpoint * point, &nearest, &here);
      const bool hit_there = tree.nearest(*pose * point, &nearest, &there);
      gain += (hit_here ? here : limit) - (hit_there ? there : limit);
    }
    if (gain > 0.0) {
      confirmed_shift = proposal;
      match_result = std::move(checked);
      weak_axes = std::move(recovered_axes);
      VLOG(1) << "Directional recovery shift=" << shift
              << " center_support=" << center_support;
    }
    VLOG(1) << "Whole scan support gain=" << gain << " proposal=" << shift;
  };
  if (points.size() >= static_cast<std::size_t>(config_.gicp_num_neighbors)) {
    if (level.mapped)
      recover(MappedCorrespondenceSearch{*level.mapped, radius * radius,
                                         level.use_occupied_bounds},
              MappedTargetView{*level.mapped});
    else
      recover(CorrespondenceSearch{*level.kdtree, radius * radius,
                                   level.search_bounds.get()},
              TargetView{*level.cloud, level.lazy_normals.get()});
  }
  const auto search_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  VLOG(1) << "Directional search ms=" << search_ms
          << " points=" << points.size() << " center=" << center_support;
}
} // namespace n3mapping
