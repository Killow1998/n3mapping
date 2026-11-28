#include "mapped_target.h"
#include "registration_bounds.h"
#include <ostream>

namespace n3mapping::registration_detail {
bool writeAtlasSearchIndex(std::ostream& output, const PointCloudMatcher::SmallGicpKdTree& tree) {
  AtlasSearchHeader header;
  header.points = tree.points->size();
  header.nodes = tree.kdtree.nodes.size();
  header.root = tree.kdtree.root;
  const SearchBounds bounds(tree);
  output.write(reinterpret_cast<const char*>(&header), sizeof(header));
  std::array<AtlasSearchNode, 1024> buffer;
  for (std::size_t i = 0; i < header.nodes; ++i) {
    const auto& input = tree.kdtree.nodes[i];
    AtlasSearchNode node;
    node.left = input.left;
    node.right = input.right;
    if (node.left == UINT32_MAX) {
      node.first = input.node_type.lr.first;
      node.last = input.node_type.lr.last;
    } else {
      // The median and both child ranges use the persisted float coordinates.
      node.threshold = static_cast<float>(input.node_type.sub.thresh);
      node.first = input.node_type.sub.proj.axis;
    }
    node.lower = bounds.nodes[i].lower;
    node.upper = bounds.nodes[i].upper;
    const auto slot = i % buffer.size();
    buffer[slot] = node;
    if (slot + 1 == buffer.size() || i + 1 == header.nodes)
      output.write(reinterpret_cast<const char*>(buffer.data()), (slot + 1) * sizeof(node));
  }
  return output.good();
}
}
