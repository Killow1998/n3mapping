#include "mapped_target_state.h"
#include "localization_atlas.pb.h"
#include <cmath>
#include <climits>
#include <numeric>
#include <string_view>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/wire_format_lite.h>

namespace n3mapping::registration_detail {
namespace {
using google::protobuf::io::CodedInputStream;
using Wire = google::protobuf::internal::WireFormatLite;
struct RestoredAtlasTreeBuilder {
  const MappedTargetLevel& level;
  template <typename Tree, typename Cloud>
  void build_tree(Tree& tree, const Cloud& points) const {
    tree.root = level.header.root;
    tree.nodes = level.search_index->nodes;
    tree.indices.resize(points.size());
    std::iota(tree.indices.begin(), tree.indices.end(), 0);
  }
};


bool slice(CodedInputStream& input, const char* base, const char** pointer,
           std::size_t* bytes) {
  std::uint32_t length;
  if (!input.ReadVarint32(&length) || length > INT_MAX) return false;
  *pointer = base + input.CurrentPosition();
  *bytes = length;
  return input.Skip(length);
}

std::shared_ptr<MappedTargetLevel> readLevel(
    const char* bytes, std::size_t size, std::shared_ptr<AtlasMapping> owner) {
  CodedInputStream input(reinterpret_cast<const std::uint8_t*>(bytes), size);
  auto level = std::make_shared<MappedTargetLevel>();
  level->owner = std::move(owner);
  std::uint64_t count = 0;
  std::size_t feature_bytes = 0, covariance_bytes = 0, index_bytes = 0;
  const char* features = nullptr;
  const char* index = nullptr;
  bool compact = false;
  for (std::uint32_t tag; (tag = input.ReadTag()) != 0;) {
    std::uint64_t scalar;
    switch (tag) {
      case 9:
        if (!input.ReadLittleEndian64(&scalar)) return {};
        std::memcpy(&level->resolution, &scalar, sizeof(scalar));
        break;
      case 16:
        if (!input.ReadVarint64(&count)) return {};
        break;
      case 42:
        if (!slice(input, bytes, &level->covariances, &covariance_bytes)) return {};
        break;
      case 48:
        if (!input.ReadVarint64(&scalar)) return {};
        compact = scalar != 0;
        break;
      case 58:
        if (!slice(input, bytes, &index, &index_bytes)) return {};
        break;
      case 66:
        if (!slice(input, bytes, &features, &feature_bytes)) return {};
        break;
      default:
        if (!Wire::SkipField(&input, tag)) return {};
    }
  }
  if (input.CurrentPosition() != static_cast<int>(size) || count == 0 ||
      count > static_cast<std::uint64_t>(INT_MAX / 9) ||
      !std::isfinite(level->resolution) || level->resolution <= 0 ||
      feature_bytes != count * 64 ||
      covariance_bytes != (compact ? 0 : count * 72) ||
      index_bytes < sizeof(AtlasSearchHeader)) return {};
  std::memcpy(&level->header, index, sizeof(level->header));
  const AtlasSearchHeader expected;
  if (level->header.magic != expected.magic || level->header.points != count ||
      level->header.nodes == 0 || level->header.root >= level->header.nodes ||
      index_bytes != sizeof(AtlasSearchHeader) +
          std::uint64_t(level->header.nodes) * sizeof(AtlasSearchNode)) return {};
  auto restored = std::make_shared<MappedSearchIndex>();
  restored->nodes.resize(level->header.nodes);
  restored->bounds.nodes.resize(level->header.nodes);
  level->search_index = restored;
  level->covariance_from_normal = compact;
  level->points = features;
  level->normals = features + count * 32;
  for (std::size_t i = 0; i < count; ++i) {
    if (!level->point(i).allFinite() || !level->normal(i).allFinite() ||
        level->point(i)[3] != 1 || level->normal(i)[3] != 0) return {};
    if (!compact) {
      double values[9];
      std::memcpy(values, level->covariances + i * 72, sizeof(values));
      for (const auto value : values) if (!std::isfinite(value)) return {};
    }
  }
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
          node.last <= node.first || node.last > count) return {};
      next_point = node.last;
    } else if (node.first >= 3 || !std::isfinite(node.threshold) ||
               node.left <= i || node.right <= node.left ||
               node.left >= level->header.nodes || node.right >= level->header.nodes) {
      return {};
    }
  }
  if (next_point != count) return {};
  return level;
}
} // namespace

bool loadMappedAtlas(const std::string& path, LocalizationAtlasData& metadata,
    PointCloudMatcher::PreparedTarget& target,
    std::shared_ptr<AtlasMapping>& mapping, std::string* error) {
  const auto file = AtlasMapping::open(path, error);
  if (!file) return false;
  CodedInputStream probe(reinterpret_cast<const std::uint8_t*>(file->data()), file->size());
  std::string_view format;
  for (std::uint32_t tag; (tag = probe.ReadTag()) != 0;) {
    if (tag == 10) {
      const char* text;
      std::size_t length;
      if (!slice(probe, file->data(), &text, &length)) return false;
      format = std::string_view(text, length);
      break;
    }
    if (!Wire::SkipField(&probe, tag)) return false;
  }
  if (format != kMappedAtlasFormat) return false;
  mapping = file;
  const auto fail = [&](const char* message) {
    if (error) *error = message;
    return false;
  };
  CodedInputStream input(reinterpret_cast<const std::uint8_t*>(file->data()), file->size());
  std::vector<std::unique_ptr<google::protobuf::io::ArrayInputStream>> pieces;
  std::vector<google::protobuf::io::ZeroCopyInputStream*> inputs;
  std::vector<std::shared_ptr<MappedTargetLevel>> planes;
  std::shared_ptr<MappedTargetLevel> refine;
  std::size_t segment = 0;
  for (;;) {
    const auto field_start = input.CurrentPosition();
    const auto tag = input.ReadTag();
    if (!tag) break;
    if (tag == 58 || tag == 74) {
      if (field_start > static_cast<int>(segment))
        pieces.push_back(std::make_unique<google::protobuf::io::ArrayInputStream>(
            file->data() + segment, field_start - segment));
      const char* bytes;
      std::size_t count;
      if (!slice(input, file->data(), &bytes, &count)) return fail("truncated mapped atlas level");
      auto level = readLevel(bytes, count, file);
      if (!level) return fail("atlas level array size mismatch or invalid search index");
      file->releasePages();
      if (tag == 58) planes.push_back(std::move(level));
      else refine = std::move(level);
      segment = input.CurrentPosition();
    } else if (!Wire::SkipField(&input, tag)) {
      return fail("malformed mapped atlas metadata");
    }
  }
  if (input.CurrentPosition() != static_cast<int>(file->size()) || planes.size() != 2)
    return fail("mapped atlas level metadata is malformed");
  if (segment < file->size())
    pieces.push_back(std::make_unique<google::protobuf::io::ArrayInputStream>(
        file->data() + segment, file->size() - segment));
  for (auto& piece : pieces) inputs.push_back(piece.get());
  google::protobuf::io::ConcatenatingInputStream joined(inputs.data(), inputs.size());
  CodedInputStream metadata_input(&joined);
  metadata_input.SetTotalBytesLimit(1 << 30);
  if (!metadata.ParseFromCodedStream(&metadata_input) || !metadata_input.ConsumedEntireMessage())
    return fail("failed to decode mapped atlas metadata");
  file->releasePages();
  target = {};
  const auto install = [&](const std::shared_ptr<MappedTargetLevel>& level,
                           PointCloudMatcher::PreparedTargetLevel& output,
                           LocalizationAtlasPreparedLevel* proto) {
    output.resolution = level->resolution;
    if (level != refine && (!refine || level->resolution > refine->resolution)) {
      auto cloud = std::make_shared<PointCloudMatcher::SmallGicpCloud>();
      cloud->points.resize(level->header.points);
      cloud->normals.resize(level->header.points);
      for (std::size_t i = 0; i < cloud->size(); ++i) {
        cloud->point(i) = level->point(i);
        cloud->normal(i) = level->normal(i);
      }
      output.cloud = cloud;
      output.kdtree = std::make_shared<PointCloudMatcher::SmallGicpKdTree>(
          cloud, RestoredAtlasTreeBuilder{*level});
      output.search_bounds = std::make_shared<SearchBounds>(level->search_index->bounds);
      file->releasePages();
    } else {
      output.mapped = level;
      auto unowned = *level;
      unowned.owner.reset();
      file->impl_->levels.push_back(std::move(unowned));
    }
    proto->set_resolution(level->resolution);
    proto->set_num_points(level->header.points);
  };
  for (const auto& level : planes) {
    target.plane_levels.emplace_back();
    install(level, target.plane_levels.back(), metadata.add_plane_levels());
  }
  target.has_refine_level = bool(refine);
  if (refine) install(refine, target.refine_level, metadata.mutable_refine_level());
  return true;
}
} // namespace n3mapping::registration_detail
