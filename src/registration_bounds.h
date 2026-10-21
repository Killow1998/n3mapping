#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <small_gicp/registration/registration_helper.hpp>

namespace n3mapping::registration_detail {
// Actual occupied bounds tighten the KD split cells without changing their
// traversal, leaf order or tie rule. Round outward so float storage can only
// cause extra visits, never exclude an accepted double-precision neighbor.
struct SearchBounds {
  struct Box { std::array<float, 3> lower, upper; };
  std::vector<Box> nodes;

  explicit SearchBounds(const small_gicp::KdTree<small_gicp::PointCloud>& tree)
      : nodes(tree.kdtree.nodes.size()) {
    if (!tree.points->empty()) build(tree, tree.kdtree.root);
  }

  bool outside(small_gicp::NodeIndexType index, const Eigen::Vector4d& point,
               double distance_squared) const {
    const auto& box = nodes[index];
    double lower_bound = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double offset = std::max({0.0, double(box.lower[k]) - point[k],
                                     point[k] - double(box.upper[k])});
      lower_bound += offset * offset;
    }
    const double guard = 16.0 * std::numeric_limits<double>::epsilon() *
                         (1.0 + distance_squared);
    return lower_bound > distance_squared + guard;
  }

 private:
  void build(const small_gicp::KdTree<small_gicp::PointCloud>& tree,
             small_gicp::NodeIndexType index) {
    const auto& node = tree.kdtree.nodes[index];
    auto& box = nodes[index];
    if (node.left != small_gicp::INVALID_NODE) {
      build(tree, node.left);
      build(tree, node.right);
      for (int k = 0; k < 3; ++k) {
        box.lower[k] = std::min(nodes[node.left].lower[k], nodes[node.right].lower[k]);
        box.upper[k] = std::max(nodes[node.left].upper[k], nodes[node.right].upper[k]);
      }
      return;
    }
    const float infinity = std::numeric_limits<float>::infinity();
    for (int k = 0; k < 3; ++k) {
      double lower = std::numeric_limits<double>::infinity(), upper = -lower;
      for (auto i = node.node_type.lr.first; i < node.node_type.lr.last; ++i) {
        const double value = tree.points->point(tree.kdtree.indices[i])[k];
        if (!std::isfinite(value)) { lower = -infinity; upper = infinity; break; }
        lower = std::min(lower, value); upper = std::max(upper, value);
      }
      box.lower[k] = std::nextafter(static_cast<float>(lower), -infinity);
      box.upper[k] = std::nextafter(static_cast<float>(upper), infinity);
    }
  }
};
}  // namespace n3mapping::registration_detail
