#include "mapped_target.h"
#include "registration_bounds.h"

namespace n3mapping::registration_detail {
std::string encodeAtlasSearchIndex(const PointCloudMatcher::SmallGicpKdTree& tree) {
  AtlasSearchHeader header;
  header.points = tree.points->size();
  header.nodes = tree.kdtree.nodes.size();
  header.root = tree.kdtree.root;
  const SearchBounds bounds(tree);
  std::string result(sizeof(header) + header.nodes * sizeof(AtlasSearchNode), '\0');
  std::memcpy(result.data(), &header, sizeof(header));
  for (std::size_t i = 0; i < header.nodes; ++i) {
    const auto& input = tree.kdtree.nodes[i];
    AtlasSearchNode node;
    node.left = input.left;
    node.right = input.right;
    if (node.left == UINT32_MAX) {
      node.first = input.node_type.lr.first;
      node.last = input.node_type.lr.last;
    } else {
      node.threshold = input.node_type.sub.thresh;
      node.first = input.node_type.sub.proj.axis;
    }
    node.lower = bounds.nodes[i].lower;
    node.upper = bounds.nodes[i].upper;
    std::memcpy(result.data() + sizeof(header) + i * sizeof(node), &node, sizeof(node));
  }
  return result;
}
} // namespace n3mapping::registration_detail
