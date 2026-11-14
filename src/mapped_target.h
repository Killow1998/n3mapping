#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include "n3mapping/point_cloud_matcher.h"
#include "registration_bounds.h"

namespace n3mapping {
class LocalizationAtlasData;
class LocalizationAtlasPreparedLevel;
namespace registration_detail {
inline constexpr const char* kMappedAtlasFormat = "n3mapping_localization_atlas_v4";

struct AtlasSearchHeader {
  std::uint64_t magic = 0x31584449334eULL;
  std::uint32_t points = 0, nodes = 0, root = 0, reserved = 0;
};
struct AtlasSearchNode {
  double threshold = 0;
  std::uint32_t left = UINT32_MAX, right = UINT32_MAX;
  // Leaves store [first,last); inner nodes store the split axis in first.
  std::uint32_t first = 0, last = 0;
  std::array<float, 3> lower{}, upper{};
};
static_assert(sizeof(AtlasSearchHeader) == 24);
static_assert(sizeof(AtlasSearchNode) == 48);

struct MappedSearchIndex {
  using Node = small_gicp::KdTreeNode<small_gicp::AxisAlignedProjection>;
  std::vector<Node> nodes;
  SearchBounds bounds;
  std::size_t memoryBytes() const {
    return sizeof(*this) + nodes.capacity() * sizeof(Node) +
        bounds.nodes.capacity() * sizeof(SearchBounds::Box);
  }
};

class AtlasMapping;
struct MappedTargetLevel {
  std::shared_ptr<AtlasMapping> owner;
  const char* points = nullptr;
  const char* normals = nullptr;
  const char* covariances = nullptr;
  std::shared_ptr<const MappedSearchIndex> search_index;
  AtlasSearchHeader header;
  double resolution = 0;
  bool covariance_from_normal = true;

  Eigen::Vector4d point(std::size_t index) const {
    Eigen::Vector4d result;
    std::memcpy(result.data(), points + index * 32, 32);
    return result;
  }
  Eigen::Vector4d normal(std::size_t index) const {
    Eigen::Vector4d result;
    std::memcpy(result.data(), normals + index * 32, 32);
    return result;
  }
  Eigen::Matrix4d covariance(std::size_t index) const;
  const MappedSearchIndex::Node& node(std::size_t index) const {
    return search_index->nodes[index];
  }
};

// Owns the read-only mapping and the existing reusable background worker.
// Paged levels retain their mapping while registration uses the typed views.
class AtlasMapping {
public:
  ~AtlasMapping();
  const char* data() const;
  std::size_t size() const;
  void releasePages();
  void prefetch(const std::vector<Eigen::AlignedBox3d>& regions);
  static std::shared_ptr<AtlasMapping> open(const std::string& path,
                                          std::string* error);
private:
  AtlasMapping();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend bool loadMappedAtlas(const std::string&, LocalizationAtlasData&,
      PointCloudMatcher::PreparedTarget&, std::shared_ptr<AtlasMapping>&,
      std::string*);
};

bool loadMappedAtlas(const std::string& path, LocalizationAtlasData& metadata,
    PointCloudMatcher::PreparedTarget& target,
    std::shared_ptr<AtlasMapping>& mapping, std::string* error);
std::string encodeAtlasSearchIndex(
    const PointCloudMatcher::SmallGicpKdTree& tree);

} // namespace registration_detail
} // namespace n3mapping
