#include "atlas_container.h"
#include "mapped_target_state.h"
#include "localization_atlas.pb.h"

#include <climits>

namespace n3mapping::registration_detail {
namespace {
using namespace atlas_codec;

std::shared_ptr<MappedTargetLevel> readLevel(
    AtlasSectionView section, const std::shared_ptr<AtlasMapping>& owner) {
  if (section.size < sizeof(LevelHeader)) return {};
  LevelHeader header;
  std::memcpy(&header, section.data, sizeof(header));
  if (header.points == 0 || header.points > UINT32_MAX ||
      !std::isfinite(header.resolution) || header.resolution <= 0 ||
      header.explicit_covariance > 1) return {};

  const auto feature_stride = align8(header.points * kAtlasVectorBytes);
  const auto feature_end = sizeof(header) + 2 * feature_stride;
  const auto index_offset = feature_end + (header.explicit_covariance ? header.points * 72 : 0);
  if (index_offset > section.size || header.search_bytes != section.size - index_offset ||
      header.search_bytes < sizeof(AtlasSearchHeader)) return {};

  auto level = std::make_shared<MappedTargetLevel>();
  level->owner = owner;
  level->resolution = header.resolution;
  level->covariance_from_normal = !header.explicit_covariance;
  level->points = section.data + sizeof(header);
  level->normals = level->points + feature_stride;
  if (header.explicit_covariance) level->covariances = section.data + feature_end;

  const char* index = section.data + index_offset;
  std::memcpy(&level->header, index, sizeof(level->header));
  const AtlasSearchHeader expected;
  if (level->header.magic != expected.magic || level->header.points != header.points ||
      level->header.nodes == 0 || level->header.root != 0 || level->header.reserved ||
      level->header.nodes > 2 * header.points - 1 ||
      header.search_bytes != sizeof(AtlasSearchHeader) +
          std::uint64_t(level->header.nodes) * sizeof(AtlasSearchNode)) return {};

  auto restored = std::make_shared<MappedSearchIndex>();
  restored->nodes.resize(level->header.nodes);
  restored->bounds.nodes.resize(level->header.nodes);
  level->search_index = restored;
  std::vector<uint8_t> parents(level->header.nodes, 0);
  std::vector<SearchBounds::Box> occupied(level->header.nodes);
  std::uint32_t next_point = 0;
  for (std::uint32_t i = 0; i < level->header.nodes; ++i) {
    AtlasSearchNode node;
    std::memcpy(&node, index + sizeof(AtlasSearchHeader) + i * sizeof(node), sizeof(node));
    auto& output = restored->nodes[i];
    output.left = node.left;
    output.right = node.right;
    if (node.left == UINT32_MAX) {
      output.node_type.lr.first = node.first;
      output.node_type.lr.last = node.last;
    } else {
      output.node_type.sub.proj.axis = node.first;
      output.node_type.sub.thresh = node.threshold;
    }
    restored->bounds.nodes[i] = {node.lower, node.upper};
    for (int axis = 0; axis < 3; ++axis)
      if (std::isnan(node.lower[axis]) || std::isnan(node.upper[axis]) ||
          node.lower[axis] > node.upper[axis]) return {};
    if (node.left == UINT32_MAX) {
      if (node.right != UINT32_MAX || node.first != next_point ||
          node.last <= node.first || node.last > header.points) return {};
      auto& box = occupied[i];
      box.lower.fill(std::numeric_limits<float>::infinity());
      box.upper.fill(-std::numeric_limits<float>::infinity());
      for (auto p = node.first; p < node.last; ++p) {
        const Eigen::Vector4d position = level->point(p);
        if (!position.allFinite() || !level->normal(p).allFinite() ||
            (header.explicit_covariance && !level->covariance(p).allFinite())) return {};
        for (int axis = 0; axis < 3; ++axis) {
          const float value = static_cast<float>(position[axis]);
          if (value < node.lower[axis] || value > node.upper[axis]) return {};
          box.lower[axis] = std::min(box.lower[axis], value);
          box.upper[axis] = std::max(box.upper[axis], value);
        }
      }
      next_point = node.last;
    } else if (node.first >= 3 || !std::isfinite(node.threshold) ||
               node.left <= i || node.right <= node.left ||
               node.left >= level->header.nodes || node.right >= level->header.nodes) {
      return {};
    } else if (++parents[node.left] != 1 || ++parents[node.right] != 1) {
      return {};
    }
  }
  if (next_point != header.points ||
      std::find(parents.begin() + 1, parents.end(), 0) != parents.end()) return {};
  for (std::size_t i = restored->nodes.size(); i-- > 0;) {
    const auto& node = restored->nodes[i];
    if (node.left == UINT32_MAX) continue;
    const auto& left = occupied[node.left];
    const auto& right = occupied[node.right];
    const auto& parent = restored->bounds.nodes[i];
    const auto axis = node.node_type.sub.proj.axis;
    if (left.upper[axis] > node.node_type.sub.thresh ||
        right.lower[axis] < node.node_type.sub.thresh) return {};
    for (int k = 0; k < 3; ++k) {
      for (auto child : {node.left, node.right}) {
        const auto& box = restored->bounds.nodes[child];
        if (box.lower[k] < parent.lower[k] || box.upper[k] > parent.upper[k]) return {};
      }
      occupied[i].lower[k] = std::min(left.lower[k], right.lower[k]);
      occupied[i].upper[k] = std::max(left.upper[k], right.upper[k]);
    }
  }
  return level;
}
}

bool loadMappedAtlas(const std::string& path, LocalizationAtlasData& metadata,
    PointCloudMatcher::PreparedTarget& target,
    std::shared_ptr<AtlasMapping>& mapping, std::string* error) {
  const auto file = AtlasMapping::open(path, error);
  if (!file) return false;
  const auto fail = [&](const char* message) {
    if (error) *error = message;
    return false;
  };
  ContainerHeader header;
  if (file->size() < sizeof(header) || std::memcmp(file->data(), kMagic.data(), kMagic.size()) != 0)
    return fail("atlas format version mismatch; rebuild from the source map with n3mapping_localization_atlas_compile");
  std::memcpy(&header, file->data(), sizeof(header));
  if (header.sections < 4 || header.sections > kMaxSections ||
      header.sections * sizeof(SectionEntry) > file->size() - sizeof(header))
    return fail("atlas section directory is malformed");

  AtlasSectionView meta, global, planes[2], refine, free;
  std::uint64_t end = sizeof(header) + header.sections * sizeof(SectionEntry);
  for (std::size_t i = 0; i < header.sections; ++i) {
    SectionEntry entry;
    std::memcpy(&entry, file->data() + sizeof(header) + i * sizeof(entry), sizeof(entry));
    if (entry.offset % 8 || entry.offset < end || entry.offset > file->size() ||
        entry.size == 0 || entry.size > file->size() - entry.offset)
      return fail("atlas section is out of bounds, misaligned or overlapping");
    if (entry.flags > (entry.tag == SectionTag::PlaneLevel ? 1u : 0u))
      return fail("atlas section flags are invalid");
    AtlasSectionView* dest = nullptr;
    switch (entry.tag) {
      case SectionTag::Metadata: dest = &meta; break;
      case SectionTag::GlobalPoints: dest = &global; break;
      case SectionTag::PlaneLevel: dest = &planes[entry.flags]; break;
      case SectionTag::RefineLevel: dest = &refine; break;
      case SectionTag::FreeSpace: dest = &free; break;
    }
    if (!dest || dest->data) return fail("atlas section tag is invalid or duplicated");
    *dest = {file->data() + entry.offset, static_cast<std::size_t>(entry.size)};
    end = entry.offset + entry.size;
  }
  if (!meta.data || !global.data || !planes[0].data || !planes[1].data)
    return fail("atlas is missing required sections");
  if (meta.size > INT_MAX || !metadata.ParseFromArray(meta.data, static_cast<int>(meta.size)))
    return fail("failed to decode atlas metadata");
  if (metadata.format_version() != kMappedAtlasFormat)
    return fail("atlas format version mismatch; rebuild from the source map with n3mapping_localization_atlas_compile");
  if (global.size % 16 || global.size / 16 != metadata.global_point_count() ||
      metadata.global_point_count() > UINT32_MAX)
    return fail("atlas global point section size mismatch");

  const double shared = metadata.shared_refine_resolution();
  if (!std::isfinite(shared) || shared < 0 || (shared > 0 && refine.data))
    return fail("invalid shared Atlas refine level");
  PointCloudMatcher::PreparedTarget decoded;
  const auto install = [&](AtlasSectionView section, PointCloudMatcher::PreparedTargetLevel& output) {
    const auto level = readLevel(section, file);
    if (!level) return false;
    output.resolution = level->resolution;
    output.mapped = level;
    auto unowned = *level;
    unowned.owner.reset();
    file->impl_->levels.push_back(std::move(unowned));
    return true;
  };
  decoded.plane_levels.resize(2);
  if (!install(planes[0], decoded.plane_levels[0]) ||
      !install(planes[1], decoded.plane_levels[1]) ||
      (refine.data && !install(refine, decoded.refine_level)))
    return fail("atlas level array size mismatch or invalid search index");
  if (shared > 0) {
    decoded.refine_level = decoded.plane_levels.back();
    decoded.refine_level.resolution = shared;
  }
  decoded.has_refine_level = shared > 0 || refine.data;
  file->impl_->global_points = global;
  file->impl_->free_space = free;
  target = std::move(decoded);
  mapping = file;
  return true;
}
}
