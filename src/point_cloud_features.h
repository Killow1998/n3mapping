// SPDX-FileCopyrightText: Copyright 2024 Kenji Koide
// SPDX-License-Identifier: MIT
// Adapted from small_gicp::estimate_local_features: retain its search,
// accumulation and eigen decomposition, but allocate scratch once per worker.
#pragma once
#include <small_gicp/util/normal_estimation.hpp>
#include <algorithm>
#include <type_traits>

namespace n3mapping::registration_detail {
template <typename Setter, typename Tree>
void estimateFeature(small_gicp::PointCloud& cloud, Tree& tree, int neighbors,
                     size_t i, size_t* indices, double* distances,
                     double* support_radius = nullptr) {
  const int requested = neighbors + (support_radius ? 1 : 0);
  const auto found = tree.knn_search(cloud.point(i), requested, indices, distances);
  if (support_radius && found == static_cast<size_t>(requested) &&
      distances[neighbors] > distances[neighbors - 1] * (1 + 1e-12) + 1e-16)
    *support_radius = distances[neighbors - 1];
  const auto n = std::min<size_t>(found, neighbors);
  if (n < 5) { Setter::set_invalid(cloud, i); return; }
  Eigen::Vector4d sum = Eigen::Vector4d::Zero();
  Eigen::Matrix4d square = Eigen::Matrix4d::Zero();
  for (size_t k = 0; k < n; ++k) {
    const auto& point = cloud.point(indices[k]);
    sum += point;
    square += point * point.transpose();
  }
  const Eigen::Vector4d mean = sum / n;
  const Eigen::Matrix4d covariance = (square - mean * sum.transpose()) / n;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen;
  eigen.computeDirect(covariance.block<3, 3>(0, 0));
  Setter::set(cloud, i, eigen.eigenvectors());
}

template <typename Setter, typename Tree>
void estimateFeatures(small_gicp::PointCloud& cloud, Tree& tree,
                      int neighbors, int threads,
                      std::vector<double>* support_radius = nullptr) {
  if constexpr (!std::is_same_v<Setter, small_gicp::CovarianceSetter<small_gicp::PointCloud>>)
    cloud.normals.resize(cloud.size());
  if constexpr (!std::is_same_v<Setter, small_gicp::NormalSetter<small_gicp::PointCloud>>)
    cloud.covs.resize(cloud.size());
  // One extra neighbor certifies a unique boundary for cross-level reuse.
  // Ambiguous ties and undersized neighborhoods always require recomputation.
  if (support_radius) support_radius->assign(cloud.size(), -1.0);
  const int requested = neighbors + (support_radius ? 1 : 0);
#pragma omp parallel num_threads(threads)
  {
    std::vector<size_t> indices(requested);
    std::vector<double> distances(requested);
#pragma omp for
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(cloud.size()); ++i)
      estimateFeature<Setter>(cloud, tree, neighbors, i, indices.data(), distances.data(),
                              support_radius ? &(*support_radius)[i] : nullptr);
  }
}
} // namespace n3mapping::registration_detail
