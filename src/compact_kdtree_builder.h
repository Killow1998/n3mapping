#pragma once

namespace n3mapping::registration_detail {

// Retained target trees need only their populated nodes. small_gicp reserves
// one node per point and reduces size after building, leaving that capacity
// alive with the immutable tree. Compact before the wrapper makes it const.
template <typename Builder>
struct CompactKdTreeBuilder : Builder {
  using Builder::Builder;

  template <typename Tree, typename Cloud>
  void build_tree(Tree& tree, const Cloud& cloud) const {
    Builder::build_tree(tree, cloud);
    tree.nodes.shrink_to_fit();
  }
};

}  // namespace n3mapping::registration_detail
