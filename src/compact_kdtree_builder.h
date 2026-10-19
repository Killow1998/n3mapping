#pragma once

#include <small_gicp/points/point_cloud.hpp>
#include <utility>

namespace n3mapping::registration_detail {

// Retained target trees need only their populated nodes. small_gicp reserves
// one node per point and reduces size after building, leaving that capacity
// alive with the immutable tree. Compact and reorder only during construction,
// before the target is published. Fresh target features are estimated afterward;
// decoded Atlas features must move with their points.
template <typename Builder>
struct CompactKdTreeBuilder : Builder {
  template <typename... Args>
  CompactKdTreeBuilder(small_gicp::PointCloud& points, bool features_ready,
                       Args&&... args)
      : Builder(std::forward<Args>(args)...), points_(points),
        features_ready_(features_ready) {}

  template <typename Tree, typename Cloud>
  void build_tree(Tree& tree, const Cloud& cloud) const {
    Builder::build_tree(tree, cloud);
    tree.nodes.shrink_to_fit();
    // Pack each leaf's points into consecutive addresses. Cycle permutation
    // uses the tree's own index array as scratch, without another cloud copy.
    // Geometry, split planes, leaf order and equal-distance choices stay fixed.
    for (std::size_t first = 0; first < tree.indices.size(); ++first) {
      if (tree.indices[first] == first) continue;
      const auto point = points_.point(first);
      Eigen::Vector4d normal;
      Eigen::Matrix4d covariance;
      if (features_ready_) {
        normal = points_.normal(first);
        covariance = points_.cov(first);
      }
      auto current = first;
      while (tree.indices[current] != first) {
        const auto next = tree.indices[current];
        points_.point(current) = points_.point(next);
        if (features_ready_) {
          points_.normal(current) = points_.normal(next);
          points_.cov(current) = points_.cov(next);
        }
        tree.indices[current] = current;
        current = next;
      }
      points_.point(current) = point;
      if (features_ready_) {
        points_.normal(current) = normal;
        points_.cov(current) = covariance;
      }
      tree.indices[current] = current;
    }
  }

 private:
  small_gicp::PointCloud& points_;
  bool features_ready_;
};

}  // namespace n3mapping::registration_detail
