#pragma once

#include "point_cloud_features.h"
#include <small_gicp/ann/kdtree.hpp>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

namespace n3mapping::registration_detail {
// Coordinates and the full KNN tree never change. Each requested normal uses
// the eager estimator verbatim, and is published once for concurrent readers.
class LazyTargetNormals {
public:
  using Cloud = small_gicp::PointCloud;
  using Tree = small_gicp::KdTree<Cloud>;
  LazyTargetNormals(Cloud::Ptr cloud, std::shared_ptr<Tree> tree, int neighbors)
      : cloud_(std::move(cloud)), tree_(std::move(tree)), neighbors_(neighbors),
        state_(cloud_->size()) {
    cloud_->normals.resize(cloud_->size());
    for (auto& state : state_) state.store(0, std::memory_order_relaxed);
  }

  const Eigen::Vector4d& normal(std::size_t i) const {
    if (state_[i].load(std::memory_order_acquire) != 2) {
      // Allocate scratch before claiming a point, so allocation failure cannot
      // leave its readers waiting for an owner that has unwound.
      thread_local std::vector<std::size_t> indices;
      thread_local std::vector<double> distances;
      indices.resize(neighbors_);
      distances.resize(neighbors_);
      std::uint8_t empty = 0;
      if (state_[i].compare_exchange_strong(empty, 1, std::memory_order_acquire)) {
        estimateFeature<small_gicp::NormalSetter<Cloud>>(
            *cloud_, tree_->kdtree, neighbors_, i, indices.data(), distances.data());
        state_[i].store(2, std::memory_order_release);
      } else {
        while (state_[i].load(std::memory_order_acquire) != 2) std::this_thread::yield();
      }
    }
    return cloud_->normal(i);
  }

  std::size_t memoryBytes() const { return sizeof(*this) + state_.capacity() * sizeof(state_[0]); }
  std::size_t computedCount() const {
    std::size_t count = 0;
    for (const auto& state : state_) count += state.load(std::memory_order_acquire) == 2;
    return count;
  }

private:
  Cloud::Ptr cloud_;
  std::shared_ptr<Tree> tree_;
  int neighbors_;
  mutable std::vector<std::atomic<std::uint8_t>> state_;
};
} // namespace n3mapping::registration_detail
