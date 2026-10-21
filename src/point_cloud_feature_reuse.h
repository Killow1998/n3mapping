#pragma once

#include "point_cloud_features.h"
#include <small_gicp/ann/kdtree.hpp>
#include <atomic>
#include <limits>
#include <memory>
#include <vector>

namespace n3mapping::registration_detail {

// Reuse only when the query coordinate and its entire KNN point multiset are
// unchanged. Deleted, inserted and duplicated coordinates invalidate nearby
// supports. All metadata is temporary; nothing survives a map revision.
inline std::size_t reuseTargetFeatures(
    const small_gicp::PointCloud& source,
    const small_gicp::KdTree<small_gicp::PointCloud>& source_tree,
    const std::vector<double>& support_radius,
    small_gicp::PointCloud& target,
    const small_gicp::KdTree<small_gicp::PointCloud>& target_tree,
    int neighbors, int threads) {
  using Cloud = small_gicp::PointCloud;
  using Tree = small_gicp::KdTree<Cloud>;
  constexpr auto missing = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> mapping(target.size(), missing);
  std::vector<std::atomic<unsigned>> counts(source.size());
  for (auto& count : counts) count.store(0, std::memory_order_relaxed);
#pragma omp parallel for num_threads(threads)
  for (std::size_t i = 0; i < target.size(); ++i) {
    std::size_t index;
    double distance;
    if (source_tree.nearest_neighbor_search(target.point(i), &index, &distance) &&
        distance == 0.0) {
      mapping[i] = index;
      counts[index].fetch_add(1, std::memory_order_relaxed);
    }
  }
  auto changed = std::make_shared<Cloud>();
  for (std::size_t i = 0; i < source.size(); ++i)
    if (counts[i].load(std::memory_order_relaxed) != 1)
      changed->points.push_back(source.point(i));
  for (std::size_t i = 0; i < target.size(); ++i)
    if (mapping[i] == missing) changed->points.push_back(target.point(i));
  std::unique_ptr<Tree> changed_tree;
  if (!changed->empty()) changed_tree = std::make_unique<Tree>(changed);

  target.normals.resize(target.size());
  std::vector<Eigen::Matrix4d>().swap(target.covs);
  std::size_t reused = 0;
#pragma omp parallel num_threads(threads) reduction(+:reused)
  {
    std::vector<std::size_t> indices(neighbors);
    std::vector<double> distances(neighbors);
#pragma omp for
    for (std::size_t i = 0; i < target.size(); ++i) {
      const auto from = mapping[i];
      bool safe = from != missing && support_radius[from] >= 0.0;
      if (safe && changed_tree) {
        std::size_t index;
        double distance;
        changed_tree->nearest_neighbor_search(target.point(i), &index, &distance);
        safe = distance > support_radius[from] * (1 + 1e-12) + 1e-16;
      }
      if (safe) {
        target.normal(i) = source.normal(from);
        ++reused;
      } else {
        estimateFeature<small_gicp::NormalSetter<Cloud>>(
            target, target_tree.kdtree, neighbors, i, indices.data(), distances.data());
      }
    }
  }
  return reused;
}

}  // namespace n3mapping::registration_detail
