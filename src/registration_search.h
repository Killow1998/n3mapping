#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <small_gicp/registration/registration_helper.hpp>
#include "registration_tree_view.h"

namespace n3mapping::registration_detail {

// Exact nearest-neighbor search within the registration's existing rejection
// distance. The immutable tree, point order and near-first traversal are shared
// with small_gicp; only branches containing exclusively rejected points vanish.
template <typename Search>
struct CorrespondenceHint {
  const Search& search;
  std::size_t index;
  std::size_t nearest(const Eigen::Vector4d& query, std::size_t* result,
                      double* distance_squared) const {
    return search.nearest(query, result, distance_squared, index);
  }
};

template <typename View>
class TypedCorrespondenceSearch {
 public:
  using Tree = small_gicp::KdTree<small_gicp::PointCloud>;
  TypedCorrespondenceSearch(const Tree& tree, double maximum,
                            const SearchBounds* bounds = nullptr)
      : TypedCorrespondenceSearch(View{tree, bounds}, maximum) {}
  TypedCorrespondenceSearch(const MappedTargetLevel& level, double maximum,
                            bool bounds = true)
      : TypedCorrespondenceSearch(View{level, bounds}, maximum) {}
  using Hint = CorrespondenceHint<TypedCorrespondenceSearch>;
  Hint withHint(std::size_t index) const { return {*this, index}; }

  std::size_t nearest(const Eigen::Vector4d& query, std::size_t* index,
                      double* distance_squared,
                      std::size_t hint_index = std::numeric_limits<std::size_t>::max()) const {
    if (!query.allFinite() || !std::isfinite(search_limit_))
      return reference(view_, query, index, distance_squared);
    *index = std::numeric_limits<std::size_t>::max();
    // DistanceRejector accepts equality. One representable step above the
    // gate permits that boundary while retaining the original strict tie rule.
    *distance_squared = search_limit_;
    // A previous correspondence is only an upper bound, never the answer.
    // Keep one representable step of slack so equal-distance points are still
    // visited in the tree's original order, including the hint itself.
    if (hint_index < view_.size()) {
      const double hint_distance =
          (view_.point(hint_index) - query).squaredNorm();
      if (hint_distance < *distance_squared)
        *distance_squared = std::nextafter(
            hint_distance, std::numeric_limits<double>::infinity());
    }
    bool rounding_boundary = false;
    if (view_.size() != 0) {
      if (view_.bounds)
        visit<true>(view_.root(), query, index, distance_squared, {}, rounding_boundary);
      else
        visit<false>(view_.root(), query, index, distance_squared, {}, rounding_boundary);
    }
    // ARM contraction can round the hint and leaf differently. Resolve only
    // ambiguous misses through the reference search, keeping its gate exact.
    if (*index == std::numeric_limits<std::size_t>::max() && rounding_boundary)
      return reference(view_, query, index, distance_squared) &&
             *distance_squared <= max_distance_squared_;
    return *index != std::numeric_limits<std::size_t>::max();
  }

 private:
  template <bool OccupiedBounds>
  void visit( small_gicp::NodeIndexType node_index, const Eigen::Vector4d& query,
             std::size_t* index, double* distance_squared,
             const std::array<double, 3>& cell_distance_squared,
             bool& rounding_boundary) const {
    if constexpr (OccupiedBounds)
      if (view_.bounds->outside(node_index, query, *distance_squared)) return;
    const auto& node = view_.node(node_index);
    if (node.left == small_gicp::INVALID_NODE) {
      for (auto i = node.node_type.lr.first; i < node.node_type.lr.last; ++i) {
        const auto candidate = view_.index(i);
        const double distance = (view_.point(candidate) - query).squaredNorm();
        if (distance >= *distance_squared) {
          rounding_boundary = rounding_boundary ||
              (*index == std::numeric_limits<std::size_t>::max() && distance - *distance_squared <=
              16.0 * std::numeric_limits<double>::epsilon() * (1.0 + *distance_squared));
          continue;
        }
        *index = candidate;
        *distance_squared = distance;
      }
      return;
    }
    const double offset = (node.node_type.sub.proj(query) - node.node_type.sub.thresh);
    const auto near_child = offset < 0.0 ? node.left : node.right;
    const auto far_child = offset < 0.0 ? node.right : node.left;
    visit<OccupiedBounds>(near_child, query, index, distance_squared, cell_distance_squared, rounding_boundary);
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
        visit<OccupiedBounds>(far_child, query, index, distance_squared, far_cell_distance, rounding_boundary);
    } else {
      rounding_boundary = rounding_boundary ||
          (*index == std::numeric_limits<std::size_t>::max() && offset * offset - *distance_squared <=
          16.0 * std::numeric_limits<double>::epsilon() * (1.0 + *distance_squared));
    }
  }

  static std::size_t reference(const NativeTreeView& view, const Eigen::Vector4d& query,
      std::size_t* index, double* distance) {
    return view.tree.nearest_neighbor_search(query, index, distance);
  }
  static std::size_t reference(const MappedTreeView& view, const Eigen::Vector4d& query,
      std::size_t* index, double* distance) {
    *index = std::numeric_limits<std::size_t>::max();
    *distance = std::numeric_limits<double>::max();
    const auto visit_reference = [&](auto&& self, std::uint32_t id) -> void {
      const auto& node = view.node(id);
      if (node.left == UINT32_MAX) {
        for (auto i = node.node_type.lr.first; i < node.node_type.lr.last; ++i) {
          const double squared = (view.point(i) - query).squaredNorm();
          if (squared < *distance) { *distance = squared; *index = i; }
        }
        return;
      }
      const auto offset = (node.node_type.sub.proj(query) - node.node_type.sub.thresh);
      self(self, offset < 0 ? node.left : node.right);
      if (*distance > offset * offset) self(self, offset < 0 ? node.right : node.left);
    };
    visit_reference(visit_reference, view.root());
    return *index != std::numeric_limits<std::size_t>::max();
  }
  TypedCorrespondenceSearch(View view, double maximum)
      : view_(view), max_distance_squared_(maximum),
        search_limit_(std::isfinite(maximum)
            ? std::min(std::numeric_limits<double>::max(),
                std::nextafter(maximum, std::numeric_limits<double>::infinity()))
            : maximum) {}
  View view_;
  double max_distance_squared_, search_limit_;
};
using CorrespondenceSearch = TypedCorrespondenceSearch<NativeTreeView>;
using MappedCorrespondenceSearch = TypedCorrespondenceSearch<MappedTreeView>;

extern template std::size_t TypedCorrespondenceSearch<NativeTreeView>::nearest(
    const Eigen::Vector4d&, std::size_t*, double*, std::size_t) const;
extern template std::size_t TypedCorrespondenceSearch<MappedTreeView>::nearest(
    const Eigen::Vector4d&, std::size_t*, double*, std::size_t) const;
extern template void TypedCorrespondenceSearch<NativeTreeView>::visit<true>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
extern template void TypedCorrespondenceSearch<NativeTreeView>::visit<false>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
extern template void TypedCorrespondenceSearch<MappedTreeView>::visit<true>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;
extern template void TypedCorrespondenceSearch<MappedTreeView>::visit<false>(
    small_gicp::NodeIndexType, const Eigen::Vector4d&, std::size_t*, double*,
    const std::array<double, 3>&, bool&) const;

small_gicp::RegistrationResult align(
    const PointCloudMatcher::PreparedTargetLevel& target, const small_gicp::PointCloud& source,
    const Eigen::Isometry3d& initial_pose, const small_gicp::RegistrationSetting& setting,
    bool reciprocal = false);

class LazyTargetNormals;
small_gicp::RegistrationResult align(
    const small_gicp::PointCloud& target, const small_gicp::PointCloud& source,
    const CorrespondenceSearch::Tree& tree, const Eigen::Isometry3d& initial_pose,
    const small_gicp::RegistrationSetting& setting,
    const SearchBounds* bounds = nullptr,
    const LazyTargetNormals* lazy_normals = nullptr, bool reciprocal = false);

}  // namespace n3mapping::registration_detail

namespace small_gicp::traits {
template <typename View>
struct Traits<n3mapping::registration_detail::TypedCorrespondenceSearch<View>> {
  static std::size_t nearest_neighbor_search(
      const n3mapping::registration_detail::TypedCorrespondenceSearch<View>& search,
      const Eigen::Vector4d& query, std::size_t* index, double* distance_squared) {
    return search.nearest(query, index, distance_squared);
  }
};
template <typename Search>
struct Traits<n3mapping::registration_detail::CorrespondenceHint<Search>> {
  static std::size_t nearest_neighbor_search(
      const n3mapping::registration_detail::CorrespondenceHint<Search>& hint,
      const Eigen::Vector4d& query, std::size_t* index, double* distance_squared) {
    return hint.nearest(query, index, distance_squared);
  }
};
} // namespace small_gicp::traits
