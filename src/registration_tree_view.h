#pragma once
#include "mapped_target.h"
#include "registration_bounds.h"

namespace n3mapping::registration_detail {
struct NativeTreeView {
  const PointCloudMatcher::SmallGicpKdTree& tree;
  const SearchBounds* bounds;
  std::size_t size() const { return tree.points->size(); }
  std::uint32_t root() const { return tree.kdtree.root; }
  const Eigen::Vector4d& point(std::size_t i) const { return tree.points->point(i); }
  std::size_t index(std::size_t i) const { return tree.kdtree.indices[i]; }
  const auto& node(std::uint32_t i) const { return tree.kdtree.nodes[i]; }
};

struct MappedTreeView {
  const char* points;
  const MappedSearchIndex::Node* nodes;
  const SearchBounds* bounds;
  std::uint32_t count, root_index;
  MappedTreeView(const MappedTargetLevel& level, bool use_bounds)
      : points(level.points), nodes(level.search_index->nodes.data()),
        bounds(use_bounds ? &level.search_index->bounds : nullptr),
        count(level.header.points), root_index(level.header.root) {}
  std::size_t size() const { return count; }
  std::uint32_t root() const { return root_index; }
  Eigen::Vector4d point(std::size_t i) const {
    Eigen::Vector4d result;
    std::memcpy(result.data(), points + i * 32, 32);
    return result;
  }
  std::size_t index(std::size_t i) const { return i; }
  const auto& node(std::uint32_t i) const { return nodes[i]; }
};
} // namespace n3mapping::registration_detail
