#include "association.h"
#include <chrono>
#include <map>

namespace n3mapping {
std::vector<size_t> geometrySourceIds(const std::vector<Keyframe::Ptr>& keyframes) {
  std::map<std::pair<double, size_t>, std::vector<size_t>> buckets;
  std::vector<size_t> ids;
  for (size_t i = 0; i < keyframes.size(); ++i) {
    const auto& frame = keyframes[i];
    size_t source = i;
    if (frame && frame->cloud) {
      auto& candidates = buckets[{frame->timestamp, frame->cloud->size()}];
      for (size_t index : candidates) {
        const auto& other = keyframes[index];
        // Quaternion serialization can perturb a raw rotation at roundoff.
        if (!frame->pose_odom.matrix().isApprox(other->pose_odom.matrix(), 1e-12)) continue;
        bool identical = true;
        for (size_t p = 0; p < frame->cloud->size(); ++p) {
          const auto& a = frame->cloud->at(p); const auto& b = other->cloud->at(p);
          if (a.x != b.x || a.y != b.y || a.z != b.z || a.intensity != b.intensity) {
            identical = false; break;
          }
        }
        if (identical) { source = index; break; }
      }
      if (source == i) candidates.push_back(i);
    }
    ids.push_back(source);
  }
  return ids;
}

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
  const auto sources = geometrySourceIds(keyframes);
  std::vector<SharedPlane> result;
  for (const auto& plane : planes) {
    std::map<size_t, size_t> multiplicity;
    for (const auto& patch : plane.patches) ++multiplicity[sources.at(patch.index)];
    if (multiplicity.size() < 3) continue;
    SharedPlane shared{plane.normal, plane.origin, plane.offset, plane.sigma, {}};
    for (const auto& patch : plane.patches)
      shared.observations.push_back({keyframes.at(patch.index)->id, patch.moment,
          1. / multiplicity.at(sources.at(patch.index))});
    stats.observations += shared.observations.size();
    result.push_back(std::move(shared));
  }
  stats.planes = result.size();
  stats.association_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}
}  // namespace n3mapping
