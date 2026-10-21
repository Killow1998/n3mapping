#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <small_gicp/registration/registration_helper.hpp>
#include "registration_bounds.h"

namespace n3mapping::registration_detail {

// Exact nearest-neighbor search within the registration's existing rejection
// distance. The immutable tree, point order and near-first traversal are shared
// with small_gicp; only branches containing exclusively rejected points vanish.
class CorrespondenceSearch {
 public:
  using Tree = small_gicp::KdTree<small_gicp::PointCloud>;

  CorrespondenceSearch(const Tree& tree, double max_distance_squared,
                       const SearchBounds* bounds = nullptr)
      : tree_(tree), bounds_(bounds),
        search_limit_(std::isfinite(max_distance_squared)
            ? std::min(std::numeric_limits<double>::max(),
                std::nextafter(max_distance_squared,
                               std::numeric_limits<double>::infinity()))
            : max_distance_squared) {}

  struct Hint {
    const CorrespondenceSearch& search;
    std::size_t index;

    std::size_t nearest(const Eigen::Vector4d& query, std::size_t* result,
                        double* distance_squared) const {
      return search.nearest(query, result, distance_squared, index);
    }
  };

  Hint withHint(std::size_t index) const { return {*this, index}; }

  std::size_t nearest(const Eigen::Vector4d& query, std::size_t* index,
                      double* distance_squared,
                      std::size_t hint_index = std::numeric_limits<std::size_t>::max()) const {
    if (!query.allFinite() || !std::isfinite(search_limit_))
      return tree_.nearest_neighbor_search(query, index, distance_squared);
    *index = std::numeric_limits<std::size_t>::max();
    // DistanceRejector accepts equality. One representable step above the
    // gate permits that boundary while retaining the original strict tie rule.
    *distance_squared = search_limit_;
    // A previous correspondence is only an upper bound, never the answer.
    // Keep one representable step of slack so equal-distance points are still
    // visited in the tree's original order, including the hint itself.
    if (hint_index < tree_.points->size()) {
      const double hint_distance =
          (tree_.points->point(hint_index) - query).squaredNorm();
      if (hint_distance < *distance_squared)
        *distance_squared = std::nextafter(
            hint_distance, std::numeric_limits<double>::infinity());
    }
    if (!tree_.points->empty())
      visit(tree_.kdtree.root, query, index, distance_squared, {});
    return *index != std::numeric_limits<std::size_t>::max();
  }

 private:
  void visit(small_gicp::NodeIndexType node_index, const Eigen::Vector4d& query,
             std::size_t* index, double* distance_squared,
             const std::array<double, 3>& cell_distance_squared) const {
    if (bounds_ && bounds_->outside(node_index, query, *distance_squared)) return;
    const auto& tree = tree_.kdtree;
    const auto& node = tree.nodes[node_index];
    if (node.left == small_gicp::INVALID_NODE) {
      for (auto i = node.node_type.lr.first; i < node.node_type.lr.last; ++i) {
        const auto candidate = tree.indices[i];
        const double distance = (tree.points.point(candidate) - query).squaredNorm();
        if (distance >= *distance_squared) continue;
        *index = candidate;
        *distance_squared = distance;
      }
      return;
    }
    const double offset = node.node_type.sub.proj(query) - node.node_type.sub.thresh;
    const auto near_child = offset < 0.0 ? node.left : node.right;
    const auto far_child = offset < 0.0 ? node.right : node.left;
    visit(near_child, query, index, distance_squared, cell_distance_squared);
    if (*distance_squared > offset * offset) {
      // Splits on different axes jointly bound this subtree's cell. Testing
      // only the latest split revisits cells that are already outside the
      // nearest-neighbor sphere because of an earlier orthogonal split.
      auto far_cell_distance = cell_distance_squared;
      far_cell_distance[node.node_type.sub.proj.axis] = offset * offset;
      const double lower_bound = far_cell_distance[0] + far_cell_distance[1] +
                                 far_cell_distance[2];
      const double rounding_guard = 16.0 * std::numeric_limits<double>::epsilon() *
                                    (1.0 + *distance_squared);
      if (lower_bound <= *distance_squared + rounding_guard)
        visit(far_child, query, index, distance_squared, far_cell_distance);
    }
  }

  const Tree& tree_;
  const SearchBounds* bounds_;
  double search_limit_;
};

small_gicp::RegistrationResult align(
    const small_gicp::PointCloud& target, const small_gicp::PointCloud& source,
    const CorrespondenceSearch::Tree& tree, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting,
    const SearchBounds* bounds = nullptr);

}  // namespace n3mapping::registration_detail

namespace small_gicp::traits {
template <>
struct Traits<n3mapping::registration_detail::CorrespondenceSearch> {
  static std::size_t nearest_neighbor_search(
      const n3mapping::registration_detail::CorrespondenceSearch& search,
      const Eigen::Vector4d& query, std::size_t* index, double* distance_squared) {
    return search.nearest(query, index, distance_squared);
  }
};

template <>
struct Traits<n3mapping::registration_detail::CorrespondenceSearch::Hint> {
  static std::size_t nearest_neighbor_search(
      const n3mapping::registration_detail::CorrespondenceSearch::Hint& hint,
      const Eigen::Vector4d& query, std::size_t* index, double* distance_squared) {
    return hint.nearest(query, index, distance_squared);
  }
};
}  // namespace small_gicp::traits
