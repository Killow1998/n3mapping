#include "plane_partition.h"
#include <cmath>
#include <map>
#include <numeric>
namespace n3mapping::planar_geometry {
std::vector<Plane> mergePlanes(std::vector<Plane> input,
                               const std::vector<gtsam::Pose3> &poses,
                               Json::Value &audit) {
  std::map<Cell, std::vector<int>> by_cell;
  for (int i = 0; i < int(input.size()); ++i) {
    const auto &p = input[i];
    Cell c;
    for (int a = 0; a < 3; ++a)
      c[a] = std::floor(p.origin[a]);
    by_cell[c].push_back(i);
  }
  std::vector<std::vector<int>> neighbors(input.size());
  std::vector<int> parent(input.size());
  std::iota(parent.begin(), parent.end(), 0);
  const auto root = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (const auto &[cell, ids] : by_cell)
    for (int id : ids) {
      const auto &p = input[id];
      for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
          for (int z = -1; z <= 1; ++z) {
            auto it = by_cell.find({cell[0] + x, cell[1] + y, cell[2] + z});
            if (it == by_cell.end())
              continue;
            for (int other : it->second) {
              if (other <= id)
                continue;
              const auto &q = input[other];
              if (p.normal.dot(q.normal) < std::cos(2 * M_PI / 180.))
                continue;
              const Eigen::Vector3d pc = p.origin + p.normal * p.offset,
                                    qc = q.origin + q.normal * q.offset;
              if (std::abs(p.normal.dot(qc - pc)) > .1 ||
                  std::abs(q.normal.dot(pc - qc)) > .1)
                continue;
              parent[root(other)] = root(id);
              neighbors[id].push_back(other);
              neighbors[other].push_back(id);
            }
          }
    }
  std::map<int, std::vector<int>> components;
  for (int i = 0; i < int(input.size()); ++i)
    components[root(i)].push_back(i);
  std::vector<Plane> output;
  int refused = 0, observations = 0, largest = 0, partitions = 0;
  for (const auto &[unused, component] : components) {
    const bool original_valid = coherentPlane(input, component, planeFrame(input, component));
    if (!original_valid) ++refused;
    auto groups = partitionPlane(input, neighbors, component);
    partitions += int(groups.size()) - 1;
    for (const auto& ids : groups) {
      const auto frame = planeFrame(input, ids);
      const auto& normal = frame.normal;
      const auto& origin = frame.origin;
      const double offset = frame.offset;
      std::map<int, Moment> scans;
      for (int id : ids)
        for (const auto &patch : input[id].patches) {
          auto &combined = scans[patch.index];
          combined.count += patch.moment.count;
          combined.sum += patch.moment.sum;
          combined.square += patch.moment.square;
        }
      std::vector<Patch> accepted;
      std::vector<double> scatter;
      for (const auto &[index, moment] : scans) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(moment.covariance());
        if (eigen.eigenvalues()[0] > .0004 ||
            eigen.eigenvalues()[0] > .02 * eigen.eigenvalues()[1])
          continue;
        const auto center = poses[index].transformFrom(moment.mean());
        Eigen::Vector3d n =
            poses[index].rotation().matrix() * eigen.eigenvectors().col(0);
        if (n.dot(poses[index].translation() - center) < 0)
          n = -n;
        if (normal.dot(n) < std::cos(5 * M_PI / 180.))
          continue;
        const double variance = std::max(0., eigen.eigenvalues()[0]);
        const double delta = normal.dot(center - origin) - offset;
        scatter.push_back(delta * delta + variance);
        accepted.push_back({index, moment, center, n, variance});
      }
      if (!original_valid && accepted.size() != scans.size()) {
        // A recovered subgroup must retain every original observation. If scan
        // moments cannot be combined without changing the gates, keep its cells.
        for (int id : ids) {
          observations += input[id].patches.size();
          output.push_back(std::move(input[id]));
        }
        continue;
      }
      if (accepted.size() < 3) continue;
      largest = std::max(largest, int(ids.size()));
      observations += accepted.size();
      output.push_back({normal, origin, offset,
                        std::sqrt(.0001 + median(scatter)), std::move(accepted)});
    }
  }
  audit["partitioned_subgroups"] = partitions;
  audit["unmerged_landmarks"] = int(input.size());
  audit["landmarks"] = int(output.size());
  audit["merged_components_refused"] = refused;
  audit["largest_merged_component"] = largest;
  audit["observations"] = observations;
  return output;
}

} // namespace n3mapping::planar_geometry
