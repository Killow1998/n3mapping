#include "association.h"
#include <chrono>

namespace n3mapping {
std::vector<SharedPlane> buildSharedPlanes(
    const std::vector<Keyframe::Ptr>& keyframes,
    GeometryOptimizationStats& stats) {
  const auto start = std::chrono::steady_clock::now();
  std::vector<gtsam::Pose3> poses;
  poses.reserve(keyframes.size());
  for (const auto& frame : keyframes) {
    if (!frame || !frame->pose_optimized.matrix().allFinite())
      throw std::invalid_argument("Invalid geometry keyframe");
    poses.emplace_back(frame->pose_optimized.matrix());
  }
  Json::Value audit;
  auto planes = planar_geometry::mergePlanes(
      planar_geometry::associate(keyframes, poses, audit), poses, audit);
  std::vector<SharedPlane> result;
  for (const auto& plane : planes) {
    SharedPlane shared{plane.normal, plane.origin, plane.offset, plane.sigma, {}};
    for (const auto& patch : plane.patches)
      shared.observations.push_back({keyframes.at(patch.index)->id, patch.moment});
    stats.observations += shared.observations.size();
    result.push_back(std::move(shared));
  }
  stats.planes = result.size();
  stats.association_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}
}  // namespace n3mapping
