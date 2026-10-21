#include "plane_partition.h"
#include <algorithm>
#include <numeric>

namespace n3mapping::planar_geometry {
Plane planeFrame(const std::vector<Plane>& planes, const std::vector<int>& ids) {
  Plane frame{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0, 0, {}};
  for (int id : ids) {
    frame.normal += planes[id].normal;
    frame.origin += planes[id].origin;
  }
  frame.normal.normalize();
  frame.origin /= ids.size();
  std::vector<double> offsets;
  for (int id : ids)
    offsets.push_back(frame.normal.dot(planes[id].origin +
        planes[id].normal * planes[id].offset - frame.origin));
  frame.offset = median(offsets);
  return frame;
}

bool coherentPlane(const std::vector<Plane>& planes, const std::vector<int>& ids,
                   const Plane& frame) {
  for (int id : ids)
    if (frame.normal.dot(planes[id].normal) < std::cos(2 * M_PI / 180.) ||
        std::abs(frame.normal.dot(planes[id].origin +
            planes[id].normal * planes[id].offset - frame.origin) - frame.offset) > .1)
      return false;
  return true;
}

std::vector<std::vector<int>> partitionPlane(
    const std::vector<Plane>& planes, const std::vector<std::vector<int>>& neighbors,
    const std::vector<int>& ids) {
  std::vector<std::vector<int>> output, pending{ids};
  std::vector<int> side(planes.size(), -1);
  while (!pending.empty()) {
    auto group = std::move(pending.back());
    pending.pop_back();
    const auto frame = planeFrame(planes, group);
    if (group.size() < 2 || coherentPlane(planes, group, frame)) {
      output.push_back(std::move(group));
      continue;
    }
    // Split the largest normalized discrepancy, then restore spatial
    // connectivity. A chain of locally similar cells cannot fuse two surfaces.
    const auto feature = [&](int id, int axis) {
      const auto& p = planes[id];
      return axis < 3 ? p.normal[axis] / std::sin(2 * M_PI / 180.)
          : frame.normal.dot(p.origin + p.normal * p.offset - frame.origin) / .1;
    };
    int axis = 0;
    double extent = -1;
    for (int a = 0; a < 4; ++a) {
      double lo = feature(group.front(), a), hi = lo;
      for (int id : group) {
        lo = std::min(lo, feature(id, a));
        hi = std::max(hi, feature(id, a));
      }
      if (hi - lo > extent) { extent = hi - lo; axis = a; }
    }
    std::stable_sort(group.begin(), group.end(), [&](int a, int b) {
      return feature(a, axis) < feature(b, axis);
    });
    for (size_t i = 0; i < group.size(); ++i)
      side[group[i]] = i < group.size() / 2 ? 0 : 1;
    for (int seed : group) {
      if (side[seed] < 0) continue;
      const int label = side[seed];
      std::vector<int> connected{seed};
      side[seed] = -1;
      for (size_t i = 0; i < connected.size(); ++i)
        for (int next : neighbors[connected[i]])
          if (side[next] == label) {
            side[next] = -1;
            connected.push_back(next);
          }
      std::sort(connected.begin(), connected.end());
      pending.push_back(std::move(connected));
    }
  }
  std::sort(output.begin(), output.end(), [](const auto& a, const auto& b) {
    return a.front() < b.front();
  });
  return output;
}
} // namespace n3mapping::planar_geometry
