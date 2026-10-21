#include "n3mapping/graph_optimizer.h"
#include "n3mapping/graph_factor_noise.h"
#include "planar_geometry/association.h"

namespace n3mapping {
size_t GraphOptimizer::appendPlaneObservations(
    int64_t id, const std::vector<PlaneMoment>& moments) {
  if (!pending_node_ids_.count(id) || shared_planes_.empty()) return 0;
  for (const auto& [plane, observation] : pending_plane_observations_)
    if (observation.node_id == id) return 0;
  const auto pose = new_values_.at<gtsam::Pose3>(gtsam::Symbol('x', id));
  const auto planes = sharedPlanes();
  const auto matches = planar_geometry::associateToPlanes(
      moments, pose, planes, getOptimizedPoses());
  for (const auto& [index, moment] : matches) {
    const auto& plane = planes[index];
    const auto noise = planar_geometry::momentNoise(config_, plane.sigma);
    new_factors_.add(planar_geometry::MomentPlaneFactor(
        gtsam::Symbol('x', id), gtsam::Symbol('p', index), moment, plane.origin, noise));
    pending_plane_observations_.push_back({index, {id, moment}});
  }
  return matches.size();
}
}  // namespace n3mapping
