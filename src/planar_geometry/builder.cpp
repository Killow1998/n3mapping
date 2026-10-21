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
        if (frame->sameAcquisition(*other)) { source = index; break; }
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
  const auto sources = geometrySourceIds(keyframes);
  std::map<size_t, std::vector<size_t>> copies;
  auto acquisitions = keyframes;
  for (size_t i = 0; i < sources.size(); ++i) {
    copies[sources[i]].push_back(i);
    if (sources[i] != i) acquisitions[i].reset();
  }
  Json::Value audit;
  // Replays cannot change association, plane fitting or the noise estimate.
  // Normalize before fitting, then retain every graph node with shared weight.
  auto planes = planar_geometry::mergePlanes(
      planar_geometry::associate(acquisitions, poses, audit), poses, audit);
  std::vector<SharedPlane> result;
  for (const auto& plane : planes) {
    SharedPlane shared{plane.normal, plane.origin, plane.offset, plane.sigma, {}};
    for (const auto& patch : plane.patches)
      for (size_t i : copies.at(patch.index))
        shared.observations.push_back({keyframes.at(i)->id, patch.moment,
            1. / copies.at(patch.index).size()});
    stats.observations += shared.observations.size();
    result.push_back(std::move(shared));
  }
  stats.planes = result.size();
  stats.association_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  return result;
}
}  // namespace n3mapping
